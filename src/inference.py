import torch
import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List, Any

# Moshi 스타일 로거 임포트
try:
    from .client_utils import log, get_logger
except ImportError:
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
# 2. 로직 클래스 (Stateless Tensor Operations)
# =============================================================================
class Qwen3DuplexLogic:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        
        self.thinker_device = model.thinker.device
        self.talker_device = model.talker.device
        self.code2wav_device = model.code2wav.device
        
        self.talker_config = model.config.talker_config
        self.num_quantizers = getattr(self.talker_config, "num_quantizers", 16)
        
        try:
            self.audio_dtype = model.thinker.audio_tower.conv2d1.weight.dtype
        except:
            self.audio_dtype = model.dtype

    @torch.no_grad()
    def thinker_step(self, input_ids, input_features, feature_attention_mask, past_key_values, step_idx):
        # [Device Move]
        if input_ids is not None and input_ids.device != self.thinker_device:
            input_ids = input_ids.to(self.thinker_device)
        if input_features is not None:
            if input_features.device != self.thinker_device:
                input_features = input_features.to(self.thinker_device)
            input_features = input_features.to(dtype=self.audio_dtype)
        if feature_attention_mask is not None and feature_attention_mask.device != self.thinker_device:
            feature_attention_mask = feature_attention_mask.to(self.thinker_device)

        # [Dummy Token Logic]
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
    def talker_step(self, thinker_hidden, past_key_values, step_idx, input_ids=None):
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
        if audio_codes.device != self.code2wav_device:
            audio_codes = audio_codes.to(self.code2wav_device)
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(-1)
            
        wav_tensor = self.model.code2wav(audio_codes)
        # Non-blocking transfer
        wav_cpu = wav_tensor.to("cpu", non_blocking=True).float().numpy()
        return wav_cpu

# =============================================================================
# 3. 엔진 클래스 (Asyncio + Executor)
# =============================================================================
class Qwen3OmniFullDuplexEngine:
    def __init__(self, model, tokenizer, config: EngineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.logic = Qwen3DuplexLogic(model)
        
        self.input_queue = None
        self.hidden_queue = None
        self.output_queue = None
        
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.last_talker_token = None
        
        self.thinker_step_count = 0
        self.talker_step_count = 0
        
        self.is_running = False
        self.thinker_task = None
        self.talker_task = None

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

        # Prefill (Blocking OK here)
        with torch.no_grad():
            out = self.logic.thinker_step(
                input_ids=initial_ids, input_features=None, feature_attention_mask=None,
                past_key_values=None, step_idx=0
            )
            self.thinker_kv_cache = out.past_key_values
            self.thinker_step_count = initial_ids.shape[1]
            
        log("info", "Engine Ready.")
        
    async def _thinker_loop(self):
        log("info", "Thinker Loop Started")
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            # 1. 오디오 입력 대기
            audio_features = await self.input_queue.get()
            
            # 2. GPU 연산 (Blocking 방지를 위해 Executor 사용)
            def run_thinker_inference():
                with torch.no_grad():
                    # =========================================================
                    # [Step 1] Audio Processing (듣기)
                    # =========================================================
                    time_len = audio_features.shape[2]
                    feature_mask = torch.ones((1, time_len), device=self.logic.thinker_device, dtype=torch.long)

                    thinker_out = self.logic.thinker_step(
                        input_ids=None, 
                        input_features=audio_features,
                        feature_attention_mask=feature_mask,
                        past_key_values=self.thinker_kv_cache,
                        step_idx=self.thinker_step_count
                    )
                    
                    # ★ [중요] 듣기 과정의 KV Cache 업데이트 (무조건 수행)
                    self.thinker_kv_cache = thinker_out.past_key_values
                    self.thinker_step_count += 4 

                    # =========================================================
                    # [Step 2] First Token Generation (판단)
                    # =========================================================
                    next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token_id = next_token.item()
                    
                    # 로그용 문자열 미리 디코딩
                    if token_id == self.cfg.silence_token_id:
                        token_str = "<|silence|>"
                    elif token_id == 151645:
                        token_str = "<|im_end|>"
                    else:
                        token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)

                    # # ★ [핵심 로직] Silence Check
                    if token_id == self.cfg.silence_token_id :
                        # 1. KV Cache는 위에서 이미 업데이트 되었으므로 기억은 유지됨.
                        # 2. Talker로 보낼 Hidden State는 없음.
                        # 3. 여기서 함수 종료 (Talker Queue에 넣지 않음)
                        return None, token_str

                    
                    # =========================================================
                    # [Step 3] Text Generation (말하기 결심했을 때만)
                    # =========================================================
                    current_turn_hiddens = []
                    current_turn_hiddens.append(thinker_out.hidden_states[-1])
                    
                    # 설정된 토큰 수만큼 추가 생성
                    for _ in range(self.cfg.text_output_tokens - 1):
                        thinker_out = self.logic.thinker_step(
                            input_ids=next_token,
                            input_features=None,
                            feature_attention_mask=None,
                            past_key_values=self.thinker_kv_cache,
                            step_idx=self.thinker_step_count
                        )
                        # 생성하면서 Cache 계속 업데이트
                        self.thinker_kv_cache = thinker_out.past_key_values
                        self.thinker_step_count += 1
                        
                        next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        token_str += self.tokenizer.decode([next_token.item()])
                        
                        current_turn_hiddens.append(thinker_out.hidden_states[-1])
                    
                    # Talker에게 보낼 Hidden State 묶음 반환
                    return torch.cat(current_turn_hiddens, dim=1), token_str

            # Executor 실행 (Sender를 방해하지 않음)
            stacked_hidden, log_str = await loop.run_in_executor(None, run_thinker_inference)
            
            # 실시간 토큰 로그 출력
            get_logger().print_token(log_str)

            # ★ [결과 처리] Hidden State가 있을 때만(Silence가 아닐 때만) 큐에 넣음
            if stacked_hidden is not None:
                await self.hidden_queue.put(stacked_hidden)
            else:
                # Silence인 경우: 큐에 넣지 않고 루프 처음으로 돌아감 (다음 오디오 대기)
                # 하지만 KV Cache는 이미 업데이트 되었으므로 문맥은 이어짐
                pass

    async def _talker_loop(self):
        log("info", "Talker Loop Started")
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            # 큐에서 데이터를 꺼낼 때까지 대기
            source_hidden = await self.hidden_queue.get()
            
            # ★ [요청하신 수정] Talker가 실제로 일을 시작할 때 로그 출력
            # (Queue에서 꺼냈다는 건 침묵이 아니라는 뜻)
            log("info", "👄 Talker generating audio...")
            
            def run_talker_inference():
                with torch.no_grad():
                    num_hiddens = source_hidden.shape[1]
                    ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                    output_chunks = []
                    
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
                            
                            wav_np = self.logic.decode_audio(codes)
                            output_chunks.append(wav_np)
                    return output_chunks

            # GPU 연산 수행
            wav_chunks_np = await loop.run_in_executor(None, run_talker_inference)
            
            # 결과 전송
            for wav_np in wav_chunks_np:
                wav_int16 = (wav_np * 32767).astype(np.int16).tobytes()
                await self.output_queue.put(wav_int16)

    async def start(self):
        if self.is_running: return
        self.is_running = True
        await self.initialize()
        self.thinker_task = asyncio.create_task(self._thinker_loop())
        self.talker_task = asyncio.create_task(self._talker_loop())
        log("info", "Engine Started (Async + Executor)")

    async def stop(self):
        self.is_running = False
        if self.thinker_task: self.thinker_task.cancel()
        if self.talker_task: self.talker_task.cancel()
        log("info", "Engine Stopped")

    async def push_audio(self, audio_features: torch.Tensor):
        await self.input_queue.put(audio_features)

    async def get_audio_output(self) -> Optional[bytes]:
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None