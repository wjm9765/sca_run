import torch
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple

# =============================================================================
# 1. 설정 및 데이터 클래스
# =============================================================================
@dataclass
class EngineConfig:
    """
    Full-Duplex 엔진의 동작을 제어하는 설정값
    """
    # 토큰 생성/입력 비율 (OmniFlatten 파인튜닝 기준: 4:2:4)
    audio_input_tokens: int = 4   # User Audio Input Chunks
    text_output_tokens: int = 2   # Thinker Text Output (Thinking step)
    audio_output_tokens: int = 4  # Talker Audio Output (Speaking step)
    
    # 모델 관련 특수 토큰 ID (실제 모델 config 확인 필수)
    silence_token_id: int = 151646 
    
    # 초기 시스템 프롬프트 내용
    system_prompt_text: str = (
        "<|im_start|>system\n"
        "You are a funny comedian performing a stand-up comedy show using Qwen3-Omni.\n"
        "<|im_end|>\n"
    )

# =============================================================================
# 2. 로직 클래스 (Stateless Tensor Operations)
# =============================================================================
class Qwen3DuplexLogic:
    """
    모델의 복잡한 Forward Pass만 담당하는 로직.
    상태(State)를 저장하지 않고 입력받은 대로 계산만 수행함.
    """
    def __init__(self, model):
        self.model = model
        self.device = model.device
        
        # ★ [수정됨] 각 모듈이 어느 GPU에 있는지 미리 파악 (Multi-GPU 대응 필수)
        # device_map="auto"로 로드하면 얘네가 서로 다른 GPU에 있을 수 있음
        self.thinker_device = model.thinker.device
        self.talker_device = model.talker.device
        self.code2wav_device = model.code2wav.device
        
        # Talker Config 접근 편의성
        self.talker_config = model.config.talker_config

    @torch.no_grad()
    def thinker_step(
        self,
        input_embeds: torch.Tensor,       # [Batch, SeqLen, Dim]
        past_key_values: Optional[List],  # KV Cache
        step_idx: int                     # RoPE용 Time Step Index
    ):
        """
        Thinker 모델 1스텝 추론
        """
        # [Multi-GPU Safety] 입력 데이터를 Thinker가 있는 GPU로 강제 이동
        if input_embeds.device != self.thinker_device:
            input_embeds = input_embeds.to(self.thinker_device)

        # 1. 3D RoPE Position IDs 생성
        seq_len = input_embeds.shape[1]
        position_ids = torch.arange(step_idx, step_idx + seq_len, device=self.thinker_device)
        position_ids = position_ids.unsqueeze(0).expand(3, 1, -1) # [3, 1, seq_len]

        # 2. Thinker Forward
        outputs = self.model.thinker(
            inputs_embeds=input_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True # Talker에게 넘겨줄 Hidden State 추출
        )
        
        return outputs

    @torch.no_grad()
    def talker_step(
        self,
        thinker_hidden: torch.Tensor,     # Thinker의 마지막 Hidden State [1, 1, Dim]
        past_key_values: Optional[List],  # Talker KV Cache
        step_idx: int                     # Talker RoPE Step Index
    ):
        """
        Talker 모델 추론 (Thinker 생각 -> Audio Code 생성)
        """
        # [Multi-GPU Safety] Thinker Output(GPU 0) -> Talker(GPU 1) 이동
        if thinker_hidden.device != self.talker_device:
            thinker_hidden = thinker_hidden.to(self.talker_device)

        # 1. Projection (Thinker Space -> Talker Space)
        talker_inputs_embeds = self.model.talker.text_projection(thinker_hidden)
        
        # 2. Position IDs (Talker는 오디오 프레임 기준)
        position_ids = torch.tensor([[step_idx]], device=self.talker_device)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 3. Talker Main Model Forward (Layer 0 예측)
        talker_out = self.model.talker.model(
            inputs_embeds=talker_inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True
        )
        
        # 4. Layer 0 Code 예측
        logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
        layer0_code = logits.argmax(dim=-1, keepdim=True) # [Batch, 1]
        
        # 5. Code Predictor (Layer 1~7 예측)
        last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
        past_hidden = talker_out.last_hidden_state[:, -1:]
        predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        
        predictor_out = self.model.talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=7,
            do_sample=False
        )
        
        # 6. 최종 코드 합체 [Layer0, Layer1...7] -> [Batch, 8]
        full_audio_codes = torch.cat([layer0_code, predictor_out], dim=1)
        
        return full_audio_codes, talker_out.past_key_values

    @torch.no_grad()
    def decode_audio(self, audio_codes: torch.Tensor) -> bytes:
        """
        Audio Codes [Batch, 8] -> PCM Bytes
        """
        # [Multi-GPU Safety] Talker Output(GPU 1) -> Code2Wav(GPU ??) 이동
        if audio_codes.device != self.code2wav_device:
            audio_codes = audio_codes.to(self.code2wav_device)

        # [Batch, 8] -> [Batch, 8, 1] (Code2Wav 입력 형태에 맞춤)
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(-1)
            
        # Code2Wav 실행
        wav_tensor = self.model.code2wav(audio_codes)
        
        # Float32 -> Int16 PCM 변환 및 바이트 직렬화
        wav_np = wav_tensor.cpu().float().numpy()
        wav_int16 = (wav_np * 32767).astype(np.int16)
        
        return wav_int16.tobytes()

# =============================================================================
# 3. 엔진 클래스 (Thread & State Management)
# =============================================================================
class Qwen3OmniFullDuplexEngine:
    def __init__(self, model, tokenizer, config: EngineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        
        # Logic 초기화
        self.logic = Qwen3DuplexLogic(model)
        
        # Queues
        self.input_queue = queue.Queue()   
        self.hidden_queue = queue.Queue()  
        self.output_queue = queue.Queue()  
        
        # States
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.text_history_ids = None 
        
        self.thinker_step_count = 0
        self.talker_step_count = 0
        
        self.is_running = False
        
        # 쓰레드 핸들
        self.t_thinker = None
        self.t_talker = None

        self._initialize_context()

    def _initialize_context(self):
        print("⚡ [Engine] Initializing...")
        
        # 1. System Prompt 토크나이징
        initial_ids = self.tokenizer(
            self.cfg.system_prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.logic.thinker_device) # Thinker GPU로 보냄
        
        # 2. 초기 상태 설정
        self.text_history_ids = initial_ids
        self.thinker_step_count = initial_ids.shape[1]
        self.talker_step_count = 0
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        
        # 3. Prefill (KV Cache 생성)
        print("⚡ [Engine] Prefilling System Prompt...")
        with torch.no_grad():
            pos_ids = torch.arange(0, self.thinker_step_count, device=self.logic.thinker_device)
            pos_ids = pos_ids.unsqueeze(0).expand(3, 1, -1)

            out = self.model.thinker(
                input_ids=initial_ids,
                past_key_values=None,
                position_ids=pos_ids,
                use_cache=True
            )
            self.thinker_kv_cache = out.past_key_values
        print("✅ [Engine] Ready.")

    # -------------------------------------------------------------------------
    # Thread 1: Thinker
    # -------------------------------------------------------------------------
    def _thinker_loop(self):
        print("🧠 [Thinker Thread] Running...")
        while self.is_running:
            try:
                # 4토큰 오디오 (Tensor) 받기
                audio_embeds = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with torch.no_grad():
                # ChatML 태그 없이 'Audio Embeds'만 입력으로 사용
                inputs_embeds = audio_embeds
                
                current_turn_hiddens = []
                is_silence = False

                for _ in range(self.cfg.text_output_tokens):
                    # Logic 호출 (장치 이동 자동 처리됨)
                    thinker_out = self.logic.thinker_step(
                        input_embeds=inputs_embeds,
                        past_key_values=self.thinker_kv_cache,
                        step_idx=self.thinker_step_count
                    )
                    
                    self.thinker_kv_cache = thinker_out.past_key_values
                    self.thinker_step_count += 1 
                    
                    # Token Selection
                    next_token_logits = thinker_out.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    
                    # Silence Check
                    if next_token_id.item() == self.cfg.silence_token_id:
                        is_silence = True
                        self.text_history_ids = torch.cat([self.text_history_ids, next_token_id.to(self.text_history_ids.device)], dim=1)
                        break 
                    
                    # 히스토리 업데이트
                    self.text_history_ids = torch.cat([self.text_history_ids, next_token_id.to(self.text_history_ids.device)], dim=1)

                    # Talker용 Hidden State 수집
                    current_turn_hiddens.append(thinker_out.hidden_states[-1])
                    
                    # Auto-regressive Input Update
                    inputs_embeds = self.model.thinker.get_input_embeddings()(next_token_id)

                # Talker Queue에 넣기
                if not is_silence and len(current_turn_hiddens) > 0:
                    stacked_hidden = torch.cat(current_turn_hiddens, dim=1)
                    self.hidden_queue.put(stacked_hidden)

    # -------------------------------------------------------------------------
    # Thread 2: Talker
    # -------------------------------------------------------------------------
    def _talker_loop(self):
        print("👄 [Talker Thread] Running...")
        while self.is_running:
            try:
                source_hidden = self.hidden_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            with torch.no_grad():
                # Thinker에서 온 Hidden State (GPU 0) -> Talker (GPU 1)
                last_thinker_hidden = source_hidden[:, -1:, :]
                
                for _ in range(self.cfg.audio_output_tokens):
                    codes, new_kv = self.logic.talker_step(
                        thinker_hidden=last_thinker_hidden,
                        past_key_values=self.talker_kv_cache,
                        step_idx=self.talker_step_count
                    )
                    
                    self.talker_kv_cache = new_kv
                    self.talker_step_count += 1
                    
                    # Decode
                    wav_bytes = self.logic.decode_audio(codes)
                    self.output_queue.put(wav_bytes)

    # -------------------------------------------------------------------------
    # 외부 제어 메서드
    # -------------------------------------------------------------------------
    def start(self):
        if self.is_running: return
        self.is_running = True
        self.t_thinker = threading.Thread(target=self._thinker_loop, daemon=True)
        self.t_talker = threading.Thread(target=self._talker_loop, daemon=True)
        self.t_thinker.start()
        self.t_talker.start()
        print("🚀 Engine Threads Started.")

    def stop(self):
        self.is_running = False
        if self.t_thinker: self.t_thinker.join()
        if self.t_talker: self.t_talker.join()
        print("🛑 Engine Threads Stopped.")

    def push_audio(self, audio_features: torch.Tensor):
        self.input_queue.put(audio_features)

    def get_audio_output(self) -> Optional[bytes]:
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None