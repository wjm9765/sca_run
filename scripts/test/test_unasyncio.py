import os
import sys
import time
import argparse
import asyncio
import torch
import numpy as np
import librosa
import soundfile as sf

# 프로젝트 루트 경로 추가 (src 폴더가 있는 위치)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 패키지 임포트
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from sca_run.src.utils.client_utils import log 
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# 메모리 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_audio_file(file_path, target_sr=16000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    log("info", f"Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

async def run_sequential_test(engine, chunks, processor, device, wait_per_chunk=1.5):
    """
    순차적 테스트 루프:
    1. 청크 1개 보냄
    2. 일정 시간(wait_per_chunk) 동안 결과가 나오는지 대기하며 수집
    3. 다음 청크 진행
    """
    all_output_audio = []
    
    log("info", f"🎬 Starting Sequential Test: {len(chunks)} chunks")
    log("info", f"⏱️ Wait time per chunk: {wait_per_chunk}s (Debugging Mode)")

    for i, chunk in enumerate(chunks):
        # -------------------------------------------------
        # 1. 입력 처리 및 전송
        # -------------------------------------------------
        if len(chunk) < 5120: # 0.32s Padding
            chunk = np.pad(chunk, (0, 5120 - len(chunk)))
        
        features = processor.feature_extractor(
            [chunk], return_tensors="pt", sampling_rate=16000, padding=False,
        )
        input_features = features.input_features.to(device).to(engine.model.dtype)
        if input_features.dim() == 5 and input_features.shape[-1] == 1:
                input_features = input_features.squeeze(-1)


        # NaN 체크
        if torch.isnan(input_features).any():
            input_features = torch.nan_to_num(input_features, nan=0.0)

        log("info", f"--------------------------------------------------")
        log("info", f"📡 [Step {i}] Pushing Chunk ({input_features.shape})")
        
        # 엔진에 오디오 투입
        await engine.push_audio(input_features)
        
        # -------------------------------------------------
        # 2. 결과 수집 (Polling)
        # -------------------------------------------------
        # 비동기 Receiver 대신, 여기서 직접 일정 시간동안 큐를 털어봅니다.
        start_wait = time.time()
        chunk_received_count = 0
        
        current_wait = wait_per_chunk

        while time.time() - start_wait < current_wait:
            out_bytes = await engine.get_audio_output()
            
            if out_bytes:
                out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                all_output_audio.append(out_np)
                chunk_received_count += 1
                sys.stdout.write(f"🎵") 
                sys.stdout.flush()
                # ★ 오디오를 하나라도 받았으면, 연속된 오디오가 더 있을 수 있으므로 대기 시간 연장
                start_wait = time.time() 
                current_wait = 1.0 # 추가 데이터 대기 시간은 짧게
            else:
                await asyncio.sleep(0.05) # 0.01은 너무 빠름, CPU 부하 조절
        
        print("") # 줄바꿈
        if chunk_received_count == 0:
            log("warning", f"⚠️ [Step {i}] No audio response received.")
        else:
            log("info", f"✅ [Step {i}] Received {chunk_received_count} audio fragments.")

    return all_output_audio

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="debug_output.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wait", type=float, default=2.0, help="Wait seconds per chunk for debugging")
    args = parser.parse_args()

    # 1. 모델 로드
    log("info", f"Loading Model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 엔진 초기화 (수정된 로직 적용됨)
    config = EngineConfig()
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    # 3. 오디오 로드 (테스트용 짧은 컷)
    full_audio, sr = load_audio_file(args.input_file, target_sr=16000)
    
    # 디버깅을 위해 너무 길지 않게 60초만 자름
    MAX_SEC = 32
    if len(full_audio) > MAX_SEC * sr:
        full_audio = full_audio[:MAX_SEC * sr]
        log("info", f"✂️ Audio cropped to {MAX_SEC}s for debugging")

    chunk_size = int(sr * 0.32)
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    
    # 4. 엔진 시작 (백그라운드 루프 실행)
    await engine.start()
    
    try:
        # 5. 순차적 테스트 실행
        collected_audio_parts = await run_sequential_test(
            engine, chunks, processor, args.device, wait_per_chunk=args.wait,
        )
        
        # 6. 결과 저장
        if collected_audio_parts:
            final_audio = np.concatenate(collected_audio_parts)
            OUTPUT_SR = 24000
            sf.write(args.output_file, final_audio, OUTPUT_SR)
            log("info", f"💾 Saved output to {args.output_file} ({len(final_audio)/OUTPUT_SR:.2f}s)")
        else:
            log("error", "❌ No audio generated at all.")

    except Exception as e:
        log("error", f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.stop()
        log("info", "Test Finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log("info", "Interrupted by user.")