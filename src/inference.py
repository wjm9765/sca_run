import torch
import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List, Any

# ★ Moshi 스타일 로거 임포트 (src/client_utils.py가 있어야 함)
try:
    from .client_utils import log, get_logger
except ImportError:
    # 없을 경우를 대비한 폴백 (기본 print)
    def log(level, msg): print(f"[{level.upper()}] {msg}")
    def get_logger(): 
        class FallbackLogger:
            def print_token(self, t, color=None): print(t, end="", flush=True)
        return FallbackLogger()

# =============================================================================
# 1. 설정 및 데이터 클래스
# =============================================================================
@dataclass
class EngineConfig:
    audio_input_tokens: int = 4   
    text_output_tokens: int = 2   
    audio_output_tokens: int = 4  
    silence_token_id: int = 151646 
    
    system_prompt_text: str = (
        "<|im_start|>system\n"
        "You are a funny comedian performing a stand-up comedy show using Qwen3-Omni.\n"
        "<|im_end|>\n"
    )

# =============================================================================
# 2. 로직 클래스 (Stateless Tensor Operations) - 기존 로직 유지
# =============================================================================
class Qwen3DuplexLogic:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        
        self.thinker_device = model.thinker.device
        self.talker_device = model.talker.device
        self.code2wav_device = model.code2wav.device
        
        self.talker_config = model.config.talker_config
        # ★ 모델 설정에서 Codec Layer 개수 확인 (기본값 16)
        self.num_quantizers = getattr(self.talker_config, "num_quantizers", 16)
        
        try:
            self.audio_dtype = model.thinker.audio_tower.conv2d1.weight.dtype
        except:
            self.audio_dtype = model.dtype

    @torch.no_grad()
    def thinker_step(
        self,
        input_ids: Optional[torch.Tensor],
        input_features: Optional[torch.Tensor],
        feature_attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List],
        step_idx: int
    ):
        # [Multi-GPU Safety]
        if input_ids is not None and input_ids.device != self.thinker_device:
            input_ids = input_ids.to(self.thinker_device)
        
        if input_features is not None:
            if input_features.device != self.thinker_device:
                input_features = input_features.to(self.thinker_device)
            # Dtype 맞춤
            input_features = input_features.to(dtype=self.audio_dtype)
            
        if feature_attention_mask is not None and feature_attention_mask.device != self.thinker_device:
            feature_attention_mask = feature_attention_mask.to(self.thinker_device)

        # ★ input_ids가 None이면 에러 발생 방지용 더미 토큰 생성
        if input_ids is None and input_features is not None:
            input_ids = torch.tensor([[0]], device=self.thinker_device)

        position_ids = torch.tensor([[step_idx]], device=self.thinker_device)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model.thinker(
            input_ids=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True
        )
        return outputs

    @torch.no_grad()
    def talker_step(
        self,
        thinker_hidden: torch.Tensor,
        past_key_values: Optional[List],
        step_idx: int,
        input_ids: Optional[torch.Tensor] = None
    ):
        if thinker_hidden.device != self.talker_device:
            thinker_hidden = thinker_hidden.to(self.talker_device)
        
        if input_ids is None:
             input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=self.talker_device)
        else:
             input_ids = input_ids.to(self.talker_device)

        conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
        audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
        talker_inputs_embeds = audio_embed + conditioned_hidden
        
        position_ids = torch.tensor([[step_idx]], device=self.talker_device)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        talker_out = self.model.talker.model(
            inputs_embeds=talker_inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True
        )
        
        logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
        layer0_code = logits.argmax(dim=-1, keepdim=True)
        
        last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
        past_hidden = talker_out.last_hidden_state[:, -1:]
        predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        
        # ★ 전체 16개 중 1개(Layer0)는 이미 나왔으므로 15개를 더 생성
        needed_tokens = self.num_quantizers - 1
        
        predictor_out = self.model.talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=needed_tokens, 
            do_sample=False
        )
        
        full_audio_codes = torch.cat([layer0_code, predictor_out], dim=1)
        return full_audio_codes, talker_out.past_key_values

    @torch.no_grad()
    def decode_audio(self, audio_codes: torch.Tensor) -> np.ndarray:
        """
        [수정] Async 환경을 위해 Numpy Array로 반환 (Bytes 변환은 CPU에서)
        """
        if audio_codes.device != self.code2wav_device:
            audio_codes = audio_codes.to(self.code2wav_device)
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(-1)
            
        wav_tensor = self.model.code2wav(audio_codes)
        
        # ★ 핵심: GPU->CPU 비동기 전송 (Blocking 방지)
        wav_cpu = wav_tensor.to("cpu", non_blocking=True).float().numpy()
        return wav_cpu

# =============================================================================
# 3. 엔진 클래스 (Asyncio Version)
# =============================================================================
class Qwen3OmniFullDuplexEngine:
    def __init__(self, model, tokenizer, config: EngineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.logic = Qwen3DuplexLogic(model)
        
        # Asyncio Queue는 루프 실행 후 생성해야 안전함 (initialize에서 생성)
        self.input_queue = None
        self.hidden_queue = None
        self.output_queue = None
        
        # States
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.last_talker_token = None
        
        self.thinker_step_count = 0
        self.talker_step_count = 0
        
        self.is_running = False

    async def initialize(self):
        log("info", "Initializing Async Engine...")
        self.input_queue = asyncio.Queue()
        self.hidden_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        initial_ids = self.tokenizer(
            self.cfg.system_prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.logic.thinker_device)
        
        codec_bos = self.model.config.talker_config.codec_bos_id
        self.last_talker_token = torch.tensor([[codec_bos]], device=self.logic.talker_device)

        # Prefill Thinker (Text Only)
        with torch.no_grad():
            out = self.logic.thinker_step(
                input_ids=initial_ids,
                input_features=None,
                feature_attention_mask=None,
                past_key_values=None,
                step_idx=0
            )
            self.thinker_kv_cache = out.past_key_values
            self.thinker_step_count = initial_ids.shape[1]
            
        log("info", "Engine Ready.")

    async def _thinker_loop(self):
        log("info", "Thinker Loop Started")
        while self.is_running:
            # 1. 오디오 입력 대기 (Async await)
            audio_features = await self.input_queue.get()
            
            with torch.no_grad():
                # [Step 1] Audio Feature 입력
                time_len = audio_features.shape[2]
                feature_mask = torch.ones((1, time_len), device=self.logic.thinker_device, dtype=torch.long)

                thinker_out = self.logic.thinker_step(
                    input_ids=None, 
                    input_features=audio_features,
                    feature_attention_mask=feature_mask,
                    past_key_values=self.thinker_kv_cache,
                    step_idx=self.thinker_step_count
                )
                self.thinker_kv_cache = thinker_out.past_key_values
                self.thinker_step_count += 4 # Audio Tokens

                # [Step 2] Text Generation
                next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # ★ 토큰 로그 출력 (Moshi 스타일)
                token_id = next_token.item()
                token_str = self.tokenizer.decode([token_id])
                # 특수 토큰이 아닐 때만 출력하거나 색상 입혀서 출력
                get_logger().print_token(token_str) 
                
                # Silence Check (일단 무시하고 계속 생성하도록 설정)
                # if token_id == self.cfg.silence_token_id: pass

                current_turn_hiddens = []
                current_turn_hiddens.append(thinker_out.hidden_states[-1])
                
                for _ in range(self.cfg.text_output_tokens - 1):
                    thinker_out = self.logic.thinker_step(
                        input_ids=next_token,
                        input_features=None,
                        feature_attention_mask=None,
                        past_key_values=self.thinker_kv_cache,
                        step_idx=self.thinker_step_count
                    )
                    self.thinker_kv_cache = thinker_out.past_key_values
                    self.thinker_step_count += 1
                    
                    next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    
                    # 로그 출력
                    t_str = self.tokenizer.decode([next_token.item()])
                    get_logger().print_token(t_str)

                    current_turn_hiddens.append(thinker_out.hidden_states[-1])

                # Talker Queue에 비동기로 넣기
                if len(current_turn_hiddens) > 0:
                    stacked_hidden = torch.cat(current_turn_hiddens, dim=1)
                    await self.hidden_queue.put(stacked_hidden)

    async def _talker_loop(self):
        log("info", "Talker Loop Started")
        while self.is_running:
            # Hidden State 대기 (Async await)
            source_hidden = await self.hidden_queue.get()
            
            with torch.no_grad():
                num_hiddens = source_hidden.shape[1]
                ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                
                for i in range(num_hiddens):
                    one_hidden = source_hidden[:, i:i+1, :]
                    for _ in range(ratio):
                        codes, new_kv = self.logic.talker_step(
                            thinker_hidden=one_hidden,
                            past_key_values=self.talker_kv_cache,
                            step_idx=self.talker_step_count,
                            input_ids=self.last_talker_token
                        )
                        self.talker_kv_cache = new_kv
                        self.talker_step_count += 1
                        self.last_talker_token = codes[:, 0:1] 
                        
                        # Decode (Non-blocking GPU->CPU)
                        wav_np = self.logic.decode_audio(codes)
                        
                        # Bytes 변환은 CPU에서 수행
                        wav_int16 = (wav_np * 32767).astype(np.int16).tobytes()
                        
                        # Output Queue에 비동기로 넣기
                        await self.output_queue.put(wav_int16)

    async def run_loops(self):
        """백그라운드 루프 실행"""
        self.is_running = True
        await self.initialize()
        
        # Async Task 생성 및 실행
        task1 = asyncio.create_task(self._thinker_loop())
        task2 = asyncio.create_task(self._talker_loop())
        
        try:
            # 두 루프가 끝날 때까지 대기 (보통 무한루프)
            await asyncio.gather(task1, task2)
        except asyncio.CancelledError:
            log("info", "Loops Cancelled")

    async def start(self):
        """외부에서 호출하는 시작 메서드 (Task 생성)"""
        if self.is_running: return
        # initialize는 run_loops 안에서 호출됨
        self.runner_task = asyncio.create_task(self.run_loops())
        log("info", "Engine Started (Async)")

    async def stop(self):
        self.is_running = False
        if hasattr(self, 'runner_task'):
            self.runner_task.cancel()
            try:
                await self.runner_task
            except asyncio.CancelledError:
                pass
        log("info", "Engine Stopped")

    async def push_audio(self, audio_features: torch.Tensor):
        await self.input_queue.put(audio_features)

    async def get_audio_output(self) -> Optional[bytes]:
        try:
            # 즉시 확인 (Non-blocking)
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None