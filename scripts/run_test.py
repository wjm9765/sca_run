import os
import sys
import time
import argparse
import asyncio  # Asyncio 사용
import torch
import numpy as np
import librosa
import soundfile as sf

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 패키지 경로
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from src.client_utils import log # 로그 유틸

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# 메모리 단편화 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_audio_file(file_path, target_sr=16000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    log("info", f"Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

# -----------------------------------------------------------------------------
# [Async Receiver] 엔진 출력을 비동기로 수거
# -----------------------------------------------------------------------------
async def receiver_loop(engine, collected_list):
    log("info", "[Receiver] Listening for output...")
    while engine.is_running:
        out_bytes = await engine.get_audio_output()
        if out_bytes:
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_list.append(out_np)
            # 로그 유틸이 화면을 제어하므로 점 찍기 대신 가끔 로그 출력 추천
            # log("debug", f"Received chunk ({len(out_np)})") 
        else:
            # 큐가 비었으면 CPU 양보
            await asyncio.sleep(0.001)

# -----------------------------------------------------------------------------
# [Async Sender] 오디오를 0.32초 간격으로 투입
# -----------------------------------------------------------------------------
async def sender_loop(engine, chunks, processor, model, device):
    log("info", "[Sender] Streaming audio chunks...")
    
    # 테스트용: 너무 길면 200개만
    # chunks = chunks[:200] 
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < 5120: # 16000 * 0.32
            chunk = np.pad(chunk, (0, 5120 - len(chunk)))
        
        # 전처리 (Blocking이지만 짧음)
        features = processor.feature_extractor(
            [chunk], return_tensors="pt", sampling_rate=16000
        )
        input_features = features.input_features.to(device).to(model.dtype)
        
        # 비동기 투입
        await engine.push_audio(input_features)
        
        # ★ 실시간 시뮬레이션 (다른 Task가 실행될 기회를 줌)
        #await asyncio.sleep(0.32) 
        
        if i % 10 == 0:
            log("info", f"Sent chunk {i}/{len(chunks)}")

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="output_response.wav")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 모델 로드
    log("info", f"Loading Model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype='auto',
        attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 엔진 초기화
    config = EngineConfig()
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    # 3. 오디오 로드
    full_audio, sr = load_audio_file(args.input_file, target_sr=16000)
    chunk_size = int(sr * 0.32)
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    log("info", f"Chunks: {len(chunks)}")

    # 4. 엔진 시작 (Task 생성)
    await engine.start()
    
    collected_output_audio = []
    
    # 5. Receiver & Sender 동시 실행
    # Receiver는 무한루프이므로 task로 실행, Sender는 await로 완료 대기
    recv_task = asyncio.create_task(receiver_loop(engine, collected_output_audio))
    
    start_time = time.time()
    try:
        await sender_loop(engine, chunks, processor, model, args.device)
        
        log("info", "All chunks sent. Waiting for trailing response...")
        await asyncio.sleep(3.0) # 잔여 응답 대기

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log("error", f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 종료 절차
        await engine.stop()
        recv_task.cancel() # Receiver 종료
    
    # 6. 저장
    if collected_output_audio:
        final_audio = np.concatenate(collected_output_audio)
        OUTPUT_SR = 24000
        log("info", f"Saving {len(final_audio)} samples ({len(final_audio)/OUTPUT_SR:.1f}s) to {args.output_file}")
        sf.write(args.output_file, final_audio, OUTPUT_SR)
    else:
        log("warning", "No output received!")

    log("info", f"Total Time: {time.time() - start_time:.2f}s")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()