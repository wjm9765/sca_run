import time
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# A100 성능 극대화 설정
torch.set_float32_matmul_precision('high')

def benchmark():
    model_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct" # 경로 수정 필요하면 수정
    device = "cuda:0"

    print("1. 모델 로딩 중 (GPU 0 강제 할당)...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map={"": 0}, # ★ Auto 끄고 강제 할당
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    
    # 2. 더미 데이터 준비 (오디오 4토큰 가정)
    print("2. 웜업 (Warmup)...")
    input_ids = torch.randint(0, 100, (1, 4)).to(device) # 오디오 토큰 대용
    # 모델 구조에 맞게 더미 입력 (Thinker만 테스트)
    # (실제로는 processor로 만들어야 하지만 속도 측정용이라 forward만 호출)
    
    # 웜업 실행
    for _ in range(3):
        with torch.no_grad():
            _ = model.thinker(input_ids=input_ids, past_key_values=None)
    torch.cuda.synchronize()

    print("3. 속도 측정 시작 (100회 반복)...")
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            # Thinker Forward (LLM 1 Step)
            _ = model.thinker(input_ids=input_ids, past_key_values=None)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"==========================================")
    print(f"평균 추론 시간: {avg_time:.4f}초 ({avg_time*1000:.2f}ms)")
    print(f"==========================================")
    
    if avg_time > 0.1:
        print("❌ 결론: 모델 자체가 느림 (HF 구현 문제 or 최적화 필요)")
    else:
        print("✅ 결론: 모델은 빠름. 님 코드의 asyncio가 범인임.")

if __name__ == "__main__":
    benchmark()