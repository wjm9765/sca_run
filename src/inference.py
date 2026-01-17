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
        loop = asyncio.get_running_loop() # 현재 실행 중인 Async Loop 가져오기
        
        while self.is_running:
            audio_features = await self.input_queue.get()
            
            # ★ [수정] 무거운 GPU 연산을 별도 쓰레드에서 실행 (Non-blocking)
            def run_thinker_inference():
                with torch.no_grad():
                    # 1. Audio Encoding Step
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
                    self.thinker_step_count += 4 

                    # 2. Text Generation Step
                    next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    
                    # 로그용 텍스트 디코딩 (쓰레드 안에서 수행)
                    token_str = self.tokenizer.decode([next_token.item()])
                    
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
                        token_str += self.tokenizer.decode([next_token.item()])
                        
                        current_turn_hiddens.append(thinker_out.hidden_states[-1])
                    
                    # 결과값 묶어서 리턴
                    return torch.cat(current_turn_hiddens, dim=1) if current_turn_hiddens else None, token_str

            # ★ Executor로 실행하고 결과를 기다림 (여기서 await하지만 Sender는 멈추지 않음)
            stacked_hidden, log_str = await loop.run_in_executor(None, run_thinker_inference)
            
            # 로그 출력 (메인 루프로 돌아와서 안전하게 출력)
            get_logger().print_token(log_str)

            if stacked_hidden is not None:
                await self.hidden_queue.put(stacked_hidden)

    async def _talker_loop(self):
        log("info", "Talker Loop Started")
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            source_hidden = await self.hidden_queue.get()
            
            # ★ [수정] Talker 연산도 별도 쓰레드로 분리
            def run_talker_inference():
                with torch.no_grad():
                    # (주의: self 변수 읽기는 괜찮으나 쓰기는 조심해야 함. 
                    # 여기선 KV Cache 갱신이 순차적이므로 충돌 가능성 낮음)
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
                            wav_int16 = (wav_np * 32767).astype(np.int16).tobytes()
                            output_chunks.append(wav_int16)
                    return output_chunks

            # Executor 실행 및 대기
            wav_chunks = await loop.run_in_executor(None, run_talker_inference)
            
            for chunk in wav_chunks:
                await self.output_queue.put(chunk)

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