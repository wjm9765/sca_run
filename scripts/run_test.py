# scripts/run_test.py
import os
import sys
import time
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 패키지 경로 (sca_core)
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

def load_audio_file(file_path, target_sr=24000):
    """오디오 파일을 로드하고 리샘플링함"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    print(f"📂 Loading audio file: {file_path}")
    # librosa는 float32 [-1, 1]로 로드함
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def main():
    parser = argparse.ArgumentParser(description="Test Full-Duplex Engine with Audio File")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Path to model")
    parser.add_argument("--input-file", type=str, required=True, help="Input audio file (e.g. 3min_noisy.wav)")
    parser.add_argument("--output-file", type=str, default="output_response.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run")
    args = parser.parse_args()

    # 1. 모델 로드
    print(f"🔥 Loading Model from {args.model_path}...")
    
    # A40 2장이면 "auto", 1장이면 "cuda:0" 등 상황에 맞게 설정
    device_map = "auto"
    
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device_map, 
        dtype='auto',          
        attn_implementation='flash_attention_2', 
        trust_remote_code=True
    )
    
    # 3. 프로세서 로드
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 엔진 초기화 (processor.tokenizer 전달)
    config = EngineConfig(audio_input_tokens=4, text_output_tokens=2, audio_output_tokens=4)
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    # 3. 오디오 준비 (Chunking)
    full_audio, sr = load_audio_file(args.input_file, target_sr=16000)
    INPUT_SR = 16000
    # 4토큰 분량의 오디오 길이 계산 (0.32초)
    # 24000 * 0.32 = 7680 samples
    chunk_size = int(sr * 0.32) 
    
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    print(f"📦 Audio split into {len(chunks)} chunks (Chunk size: {chunk_size} samples)")

    # 4. 테스트 시작 (쓰레드 가동)
    engine.start()
    
    collected_output_audio = []
    start_time = time.time()
    
    try:
        for i, chunk in enumerate(chunks):
            # 마지막 짜투리 패딩 (0으로 채움)
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            with torch.no_grad():
                # ★ [핵심 수정 1] Feature Extractor 호출 시 padding=False 설정
                # Whisper는 기본적으로 30초(3000프레임)로 패딩하는데, 
                # 짧은 스트리밍 청크를 넣을 때는 패딩을 끄거나 길이를 맞춰야 함.
                # 여기서는 Qwen3-Omni가 짧은 입력을 허용한다고 가정하고 패딩 없이 변환.
                
                processed_inputs = processor.feature_extractor(
                    [chunk], 
                    return_tensors="pt", 
                    sampling_rate=INPUT_SR,
                    padding="longest",  # 배치 1개라 의미 없지만 명시
                    do_normalize=True
                )
                
                # ★ [핵심 수정 2] 입력 텐서의 Transpose 문제 해결
                # 에러 메시지의 3000은 Whisper의 고정 길이(30초) 패딩 결과일 수 있음.
                # 여기서는 명시적으로 슬라이싱하여 실제 길이만큼만 넣거나,
                # 모델이 내부적으로 처리하도록 놔둬야 함.
                
                # 하지만, Qwen3-Omni는 스트리밍 시 고정 길이 입력을 가정할 수 있음.
                # 만약 위 에러가 "3000"이 입력 크기라고 했다면, 
                # feature_extractor가 이미 3000으로 패딩해서 줬다는 뜻임.
                
                input_features = processed_inputs.input_features.to(args.device).to(model.dtype)
                # [Batch, Mel, Time] -> [1, 128, 3000] (Whisper 기본 동작)

                # feature_lens: 실제 유효한 길이 계산 (전체 프레임 수 아님)
                # 0.32초 오디오는 멜 스펙트로그램상 약 32프레임 정도 됨.
                # 하지만 모델이 3000을 받았다면 feature_lens도 3000으로 맞춰줘야 할 수 있음.
                # 여기서는 processor가 계산해준 attention_mask를 믿음.
                
                if hasattr(processed_inputs, "attention_mask") and processed_inputs.attention_mask is not None:
                    # attention_mask가 있다면 유효 길이 합계 사용
                    feature_lens = processed_inputs.attention_mask.sum(1).to(args.device)
                else:
                    # 없다면, 우리가 넣은 오디오 길이에 비례해서 계산 (Whisper: 16000Hz -> 100Hz frame rate)
                    # 5120 샘플 / 160 = 32 프레임
                    # 하지만 모델이 3000으로 패딩된 걸 받으면 에러가 날 수 있음.
                    # 안전하게 입력 텐서의 실제 마지막 차원 크기를 사용.
                    feature_lens = torch.tensor([input_features.shape[2]], device=args.device)

                # 2. Audio Tower 호출
                audio_outputs = model.thinker.audio_tower(
                    input_features, 
                    feature_lens=feature_lens
                )
                
                audio_embeds = audio_outputs.last_hidden_state
            
            engine.push_audio(audio_embeds)
            
            # 수신 루프
            while True:
                out_bytes = engine.get_audio_output()
                if out_bytes is None: break
                
                out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                collected_output_audio.append(out_np)
                print(f"🔊 Received output chunk ({len(out_np)} samples) at step {i}")

        print("⏳ Waiting for remaining outputs...")
        time.sleep(3.0)
        
        # 남은거 싹 긁어모으기
        while True:
            out_bytes = engine.get_audio_output()
            if out_bytes is None: break
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_output_audio.append(out_np)

    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.stop()
    
    # 5. 결과 저장
    if collected_output_audio:
        final_audio = np.concatenate(collected_output_audio)
        print(f"💾 Saving {len(final_audio)} samples ({len(final_audio)/24000:.1f}s) to {args.output_file}")
        sf.write(args.output_file, final_audio, 24000)
    else:
        print("⚠️ No audio generated! (Check inputs or silence logic)")

    print(f"✅ Test Finished. Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()