from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from utils.client_utils import log
from .audio_chunker import PCMChunker
from .config import AppConfig, load_config
from .team_infer import TeamInferenceSession
from .qwen_client import (
    extract_audio_input_from_pcm16le,
    infer_audio_input_once_result,
    team_wav_to_audio_output,
    wav_bytes_to_pcm16le,
)

app = FastAPI(title="sca_run")

# Load default config at import time; CLI can override
_CFG_PATH = os.getenv("SCA_CONFIG")
CFG: AppConfig = load_config(_CFG_PATH)


# UI is served from sca_run/static/index.html
from importlib import resources
from pathlib import Path

def _load_index_html() -> str:
    """Load the minimal mic UI HTML from package data.

    Kept in a separate file for readability.
    """
    try:
        return resources.files("sca_run").joinpath("static", "index.html").read_text(encoding="utf-8")
    except Exception:
        # Fallback for editable/dev runs when package data isn't included
        here = Path(__file__).resolve().parent
        return (here / "static" / "index.html").read_text(encoding="utf-8")

INDEX_HTML = _load_index_html()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health() -> str:
    return "ok"


@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404s in logs.
    return Response(status_code=204)


@app.post("/infer_wav")
async def infer_wav(file: UploadFile = File(...)):
    """One-shot inference endpoint.

    This endpoint accepts a WAV file upload and runs inference.

    NOTE: For streaming use-cases, prefer /ws/pcm16.
    """
    wav = await file.read()
    pcm16le, sr, ch = wav_bytes_to_pcm16le(wav)

    # 1) Audio -> features (AudioInput)
    audio_in = extract_audio_input_from_pcm16le(CFG, pcm16le, sample_rate=sr, channels=ch)

    # 2) Features -> team inference (wav float)
    team_ret = infer_audio_input_once_result(CFG, audio_in)
    if team_ret is None:
        return {"status": "team_backend_not_configured", "sample_rate": sr, "channels": ch}

    audio_out = team_wav_to_audio_output(team_ret)
    return {
        "status": "ok",
        "audio_format": audio_out.audio_format,
        "audio_sample_rate": audio_out.sample_rate,
        "channels": audio_out.channels,
        "audio_bytes_len": len(audio_out.audio_bytes),
    }


@app.websocket("/ws/pcm16")
async def ws_pcm16(websocket: WebSocket):
    """Full Duplex WebSocket streaming.

    Architecture:
    - Task 1 (Input): Receive PCM16 -> Buffer -> Chunk -> KeyQueue
    - Task 2 (Inference): KeyQueue -> TeamInferenceSession -> OutputQueue
    - Task 3 (Output): OutputQueue -> Send WAV -> Client
    """
    await websocket.accept()
    log("info", "New user connected to /ws/pcm16")

    session_cfg = CFG
    chunker = PCMChunker(chunk_bytes=session_cfg.audio.chunk_bytes)
    
    # Session state
    session = TeamInferenceSession(session_cfg)
    input_queue = asyncio.Queue()  # Items: AudioInput
    output_queue = asyncio.Queue() # Items: TeamAudioReturn
    
    log("info", "Starting Full Duplex Tasks...")
    log("info", f"[Status] Server is READY. Listening on WebSocket /ws/pcm16. (Chunk size: {session_cfg.audio.chunk_bytes} bytes)")

    # -------------------------------------------------------------------------
    # Task 1: Input Loop (Receive -> Chunk -> Queue)
    # -------------------------------------------------------------------------
    async def input_loop():
        try:
            while True:
                msg = await websocket.receive()
                
                # Handle text messages (if any control messages needed later)
                if "text" in msg:
                    continue
                
                # Process Binary PCM
                data = msg.get("bytes")
                if not data:
                    continue
                
                for chunk in chunker.feed(data):
                    # [Strict Verification] Pass RAW PCM -> Float Mono Waveform
                    # Do NOT run log_mel_spectrogram here. Let team_infer use Processor.
                    # reusing extract... but effectively using the Waveform path
                    
                    # 1. PCM -> Float Waveform
                    from .qwen_client import _pcm16le_to_float_mono
                    from .io_types import AudioInput
                    
                    waveform = _pcm16le_to_float_mono(chunk, channels=session_cfg.audio.channels)
                    
                    # Pack into AudioInput (features field holds raw waveform now)
                    audio_in = AudioInput(features=waveform, timestamp=0.0)
                    
                    # Log message removed to prevent flooding terminal (e.g. print error in loop)
                    # Instead queue it
                    await input_queue.put(audio_in)
                        
        except WebSocketDisconnect:
            log("info", "WS Disconnected (Input Loop)")
        except Exception as e:
            log("error", f"Input Loop Error: {e}")
        finally:
            # Signal inference loop to stop (optional, or just let cancel handle it)
            pass

    # -------------------------------------------------------------------------
    # Task 2: Push Loop (Queue -> Engine Input)
    # -------------------------------------------------------------------------
    async def push_loop():
        try:
            while True:
                # Wait for next audio chunk from input_queue
                audio_in = await input_queue.get()
                
                # Push to engine asynchronously
                await session.push_input(audio_in)
                
                input_queue.task_done()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log("error", f"Push Loop Error: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Task 3: Pull Loop (Engine Output -> Queue)
    # -------------------------------------------------------------------------
    async def pull_loop():
        try:
            while True:
                # Try to get audio from engine
                # NOTE: engine.get_audio_output() is async and non-blocking via internal queue
                team_ret = await session.get_output()
                
                if team_ret:
                    await output_queue.put(team_ret)
                else:
                    # If no output available, sleep briefly to avoid busy loop
                    # (only if get_audio_output returns None instantly)
                    await asyncio.sleep(0.005)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log("error", f"Pull Loop Error: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Task 4: Output Loop (Queue -> WebSocket Send)
    # -------------------------------------------------------------------------
    async def output_loop():
        try:
            while True:
                team_ret = await output_queue.get()
                
                audio_out = team_wav_to_audio_output(team_ret, cfg=CFG)
                
                # Send Metadata
                # The model output sample rate is output_sample_rate
                # The browser/client needs to know this to play it back correctly.
                
                await websocket.send_json({
                    "type": "talker_audio",
                    "audio_format": audio_out.audio_format,
                    "audio_sample_rate": audio_out.sample_rate, 
                    "channels": audio_out.channels,
                    "text_log": audio_out.text_log,
                })
                
                # Send Binary Audio
                await websocket.send_bytes(audio_out.audio_bytes)
                
                output_queue.task_done()
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log("error", f"Output Loop Error: {e}")

    # -------------------------------------------------------------------------
    # Main Event Loop
    # -------------------------------------------------------------------------
    # Start the engine first
    await session.start()

    # Run all tasks concurrently
    try:
        await asyncio.gather(
            input_loop(),
            push_loop(),
            pull_loop(),
            output_loop(),
        )
    except Exception as e:
        # Normally one of the loops (input) breaking will cause gathering to return/throw
        pass
    finally:
        log("info", "Cleaning up session...")
        await session.stop()




def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    global CFG
    CFG = load_config(args.config)

    import uvicorn

    uvicorn.run("sca_run.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
