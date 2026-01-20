from __future__ import annotations

"""Audio I/O utilities + feature extraction + team inference hook.

This repo's server is intentionally **feature-only**:
  - Browser streams PCM16LE (16kHz, mono) over WebSocket
  - Server chunks audio and extracts log-mel features [1, 128, T]
  - Team's Qwen3-Omni (possibly fine-tuned) inference code consumes features
  - Team returns a float waveform (wav) which we stream back to the browser

We intentionally do **NOT** load any HuggingFace model/processor here, so the
server will not download huge model weight shards.
"""

import array
import io
import wave
from typing import Optional, Tuple

import numpy as np
import torch

from .config import AppConfig
from .feature_extractor import log_mel_spectrogram
from .io_types import AudioInput, AudioOutput, TeamAudioReturn
from .team_infer import infer_team_wav


def wav_bytes_to_pcm16le(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    """Decode uncompressed PCM WAV bytes into raw PCM16LE frames.

    Returns:
        pcm16le_bytes, sample_rate, channels
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_rate = int(wf.getframerate())
        sample_width = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        frames = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported (got sample_width={sample_width}).")
    return frames, sample_rate, channels


def _pcm16le_to_float_mono(pcm16le: bytes, channels: int) -> torch.Tensor:
    """Convert PCM16LE bytes to mono float waveform in [-1, 1] (torch.float32)."""
    if channels <= 0:
        raise ValueError("channels must be positive")

    a = array.array("h")
    a.frombytes(pcm16le)

    if channels == 1:
        x = torch.tensor(a, dtype=torch.float32)
        return (x / 32768.0).clamp(-1.0, 1.0)

    # Multi-channel: average per frame.
    x = torch.tensor(a, dtype=torch.float32).view(-1, channels)
    mono = x.mean(dim=1)
    return (mono / 32768.0).clamp(-1.0, 1.0)


def wav_float_to_pcm16le_bytes(wav: np.ndarray | torch.Tensor) -> bytes:
    """Convert float waveform in [-1,1] to PCM16LE bytes (mono).

    Accepts shapes:
      - [T]
      - [1, T]
      - [B, T] (uses first batch)
      - [T, C] or [B, T, C] (downmixes C->mono)
    """
    if isinstance(wav, torch.Tensor):
        w = wav.detach().to("cpu").float()
        w = w.numpy()
    else:
        w = np.asarray(wav)

    # Squeeze batch dims
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 2:
        # [T, C] or [1, T]
        if w.shape[0] == 1 and w.shape[1] > 1:
            w = w[0]
        else:
            # assume [T, C]
            w = w.mean(axis=1)

    w = np.clip(w, -1.0, 1.0)
    pcm = (w * 32767.0).astype(np.int16)
    return pcm.tobytes()


def extract_audio_input_from_pcm16le(
    cfg: AppConfig,
    pcm16le: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    timestamp: float = 0.0,
) -> AudioInput:
    """CPU-side feature extraction: PCM16LE -> AudioInput(features).

    Produces log-mel features shaped [1, 128, T] on CPU.
    """
    # Convert PCM bytes to mono float waveform.
    waveform = _pcm16le_to_float_mono(pcm16le, channels=channels)
    features = log_mel_spectrogram(waveform, sample_rate=sample_rate, n_mels=128)
    return AudioInput(features=features.cpu(), timestamp=float(timestamp))


def infer_audio_input_once_result(
    cfg: AppConfig,
    audio_in: AudioInput,
) -> Optional[TeamAudioReturn]:
    """Team inference hook: AudioInput(features) -> TeamAudioReturn (wav on CPU).

    Team is expected to return a float waveform (wav) per chunk.
    This scaffold will *not* download or load Qwen weights.
    """
    return infer_team_wav(cfg, audio_in)


def team_wav_to_audio_output(team_ret: TeamAudioReturn) -> AudioOutput:
    """Convert a TeamAudioReturn (wav float) to streamable PCM16LE bytes."""
    pcm_bytes = wav_float_to_pcm16le_bytes(team_ret.wav)
    return AudioOutput(
        audio_bytes=pcm_bytes,
        sample_rate=int(team_ret.sample_rate),
        channels=int(team_ret.channels) if getattr(team_ret, "channels", None) else 1,
        audio_format="pcm16le",
        text_log=getattr(team_ret, "text_log", None),
    )
