import torch
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Optional

try:
    from .utils.client_utils import log
    from .utils.compile import torch_compile_lazy
except ImportError:
    def log(level, msg): print(f"[{level.upper()}] {msg}")
    def torch_compile_lazy(model):
        return torch.compile(model)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

@dataclass
class EngineConfig:
    audio_input_tokens: int = 8   
    text_output_tokens: int = 4   
    audio_output_tokens: int = 8
    silence_token_id: int = 151646 
    audio_token_id: int = 151675

    system_prompt_text: str = (
        "<|im_start|>system\n"
        "You are a funny comedian performing a stand-up comedy show using Qwen3-Omni.\n"
        "<|im_end|>\n"
    )

class Qwen3DuplexLogic:
    def __init__(self, model):
        self.model = model
        self.device = model.device 

        # Device Mapping
        if hasattr(model, "thinker"):
            self.thinker_device = model.thinker.device
        else:
            self.thinker_device = self.device

        if hasattr(model, "talker"):
            self.talker_device = next(model.talker.parameters()).device
        else:
            self.talker_device = self.device
            
        if hasattr(model, "code2wav"):
            self.code2wav_device = next(model.code2wav.parameters()).device
        else:
            self.code2wav_device = self.device

        self.talker_config = model.config.talker_config
        self.num_quantizers = getattr(self.talker_config, "num_quantizers", 16)
        
        try:
            self.audio_dtype = model.thinker.audio_tower.conv2d1.weight.dtype
        except:
            self.audio_dtype = model.dtype
        
        # talker predictor compile to improve speed
        self.compiled_predictor = torch_compile_lazy(self.model.talker.code_predictor.model) 
    
    @torch.no_grad()
    def thinker_step(self, input_ids, input_features, feature_attention_mask, past_key_values, fixed_audio_tokens=4):
        """
        Thinker Step:   
        - 오디오/텍스트 입력을 받아 모델 내부 로직(Forward)을 통해 다음 토큰 예측
        - 오디오 길이는 4토큰(0.32초)으로 고정 가정
        """
        target_device = self.thinker_device
        
        try:
            if input_features is not None:
                if input_features.device != target_device:
                    input_features = input_features.to(target_device)
                input_features = input_features.to(dtype=self.audio_dtype)

                if feature_attention_mask is None:
                    # shape: [Batch, Mel, Time]
                    feature_attention_mask = torch.ones(
                        (input_features.shape[0], input_features.shape[2]), 
                        dtype=torch.long, 
                        device=target_device
                    )
                else:
                    if feature_attention_mask.device != target_device:
                        feature_attention_mask = feature_attention_mask.to(target_device)

                audio_token_id = self.model.config.thinker_config.audio_token_id
                
                input_ids = torch.full(
                    (1, fixed_audio_tokens), 
                    audio_token_id, 
                    dtype=torch.long, 
                    device=target_device
                )
                
                inputs_embeds = None 

            elif input_ids is not None:
                if input_ids.device != target_device:
                    input_ids = input_ids.to(target_device)
                
                # 텍스트는 Features 없음
                input_features = None
                inputs_embeds = None
                
            else:
                raise ValueError("ThinkerStep: Both input_ids and input_features are None")

            return self.model.thinker(
                input_ids=input_ids,
                input_features=input_features,       # 오디오 원본 전달 (내부 처리 유도)
                feature_attention_mask=feature_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,         # None (모델이 계산)
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
        except Exception as e:
            log("error", f" Error in thinker_step: {e}")
            import traceback
            traceback.print_exc()
            raise e

    @torch.no_grad()
    def talker_step(self, thinker_hidden, past_key_values, input_ids=None):
        """
        Talker Step:
        - step_idx 등 위치 관련 변수 제거 (모델 내부 및 KV Cache에 위임)
        """
        try:
            target_device = self.talker_device
            
            if thinker_hidden.device != target_device:
                thinker_hidden = thinker_hidden.to(target_device)
            if not thinker_hidden.is_contiguous():
                thinker_hidden = thinker_hidden.contiguous()

            # Projection
            conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
            
            if input_ids is None:
                 input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=target_device)
            else:
                 if input_ids.device != target_device:
                     input_ids = input_ids.to(target_device)

            # Embedding Sum
            audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
            talker_inputs_embeds = audio_embed + conditioned_hidden
            
            # Forward (position_ids 없이 호출 -> 자동 관리)
            talker_out = self.model.talker.model(
                inputs_embeds=talker_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            # Code Prediction (RVQ)
            logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
            layer0_code = logits.argmax(dim=-1, keepdim=True)
            
            last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
            past_hidden = talker_out.last_hidden_state[:, -1:]
            predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            
            predictor_codes = [layer0_code]
            predictor_kv = None 
            
            #call compiled predictor
            for i in range(self.num_quantizers - 1):
                pred_out = self.compiled_predictor(
                    inputs_embeds=predictor_input,
                    past_key_values=predictor_kv,
                    use_cache=True
                )
                predictor_kv = pred_out.past_key_values
                
                curr_logits = self.model.talker.code_predictor.lm_head[i](pred_out.last_hidden_state[:, -1, :])
                next_code = curr_logits.argmax(dim=-1, keepdim=True)
                predictor_codes.append(next_code)
                
                predictor_input = self.model.talker.code_predictor.get_input_embeddings()[i](next_code)
            
            full_audio_codes = torch.cat(predictor_codes, dim=1)
            return full_audio_codes, talker_out.past_key_values

        except Exception as e:
            log("error", f"Talker Crashed! {e}")
            raise e

    @torch.no_grad()
    def decode_audio(self, audio_codes: torch.Tensor) -> np.ndarray:
        target_device = self.code2wav_device
        if audio_codes.device != target_device:
            audio_codes = audio_codes.to(target_device)
        if audio_codes.dim() == 2: 
            audio_codes = audio_codes.unsqueeze(-1)
        
        wav_tensor = self.model.code2wav(audio_codes)
        
        return wav_tensor.to("cpu", non_blocking=True).float().numpy()

class Qwen3OmniFullDuplexEngine:
    def __init__(self, model, tokenizer, config: EngineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.logic = Qwen3DuplexLogic(model)
        
        self.input_queue = asyncio.Queue()
        self.hidden_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.last_talker_token = None
        
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

        with torch.no_grad():
            out = self.logic.thinker_step(
                input_ids=initial_ids, 
                input_features=None, 
                feature_attention_mask=None,
                past_key_values=None
            )
            self.thinker_kv_cache = out.past_key_values
            
            log("info", "   ... Compiling Talker Inner Loop (Wait a moment)")
            # 2. Talker Compile Trigger
            last_hidden = out.hidden_states[-1][:, -1:, :].detach().clone()
            self.logic.talker_step(
                thinker_hidden=last_hidden, past_key_values=None, input_ids=self.last_talker_token
            )
            
        log("info", "Engine Ready.")
        
    async def _thinker_loop(self):
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            try:
                audio_features = await self.input_queue.get()
                # log("info", "[Thinker] Processing...")

                def listen_and_predict_first():
                    with torch.no_grad():
                        out = self.logic.thinker_step(
                            input_ids=None, input_features=audio_features, feature_attention_mask=None,
                            past_key_values=self.thinker_kv_cache, fixed_audio_tokens=self.cfg.audio_input_tokens
                        )
                        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        return next_token, out.past_key_values

                curr_token, self.thinker_kv_cache = await loop.run_in_executor(None, listen_and_predict_first)
                
                tok_id = curr_token.item()
                if tok_id == self.cfg.silence_token_id:
                    log("info", "[Thinker] Silence.")
                    continue
                
                # 예측된 토큰 출력 (간결화)
                decoded_text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                log("info", f">> {decoded_text}")

                for i in range(self.cfg.text_output_tokens):
                    
                    def generate_one_token(token_in, kv_in):
                        with torch.no_grad():
                            out = self.logic.thinker_step(
                                input_ids=token_in, input_features=None, feature_attention_mask=None,
                                past_key_values=kv_in
                            )
                            # Hidden State 복제 (전송용)
                            hidden_to_send = out.hidden_states[-1].detach().clone()
                            next_t = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            return hidden_to_send, next_t, out.past_key_values

                    # Executor 실행 (Blocking 방지)
                    hidden_chunk, curr_token, self.thinker_kv_cache = await loop.run_in_executor(
                        None, generate_one_token, curr_token, self.thinker_kv_cache
                    )

                    if not hidden_chunk.is_contiguous():
                        hidden_chunk = hidden_chunk.contiguous()
                    await self.hidden_queue.put(hidden_chunk)
                    
                    
            except Exception as e:
                log("error", f"Thinker Error: {e}")

    async def _talker_loop(self):
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            try:
                source_hidden = await self.hidden_queue.get()
                # log("info", "[Talker] Received hidden state.")
                
                def run_talker_single_step(hidden_state):
                    with torch.no_grad():
                        # 텍스트 1개당 오디오 N개 생성
                        ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                        output_chunks = []

                        for _ in range(ratio):
                            codes, new_kv = self.logic.talker_step(
                                thinker_hidden=hidden_state,
                                past_key_values=self.talker_kv_cache,
                                input_ids=self.last_talker_token
                            )
                            self.talker_kv_cache = new_kv
                            self.last_talker_token = codes[:, 0:1] 
                            
                            wav_np = self.logic.decode_audio(codes)
                            output_chunks.append(wav_np)
                        return output_chunks

                # 오디오 생성
                wav_chunks_np = await loop.run_in_executor(None, run_talker_single_step, source_hidden)
                # log("info", f"[Talker] Generated {len(wav_chunks_np)} audio chunks.")
                
                for wav_np in wav_chunks_np:
                    wav_int16 = (wav_np * 32767).astype(np.int16).tobytes()
                    await self.output_queue.put(wav_int16)
            
            except Exception as e:
                log("error", f"Talker Error: {e}")

    async def start(self):
        if self.is_running: return
        self.is_running = True
        await self.initialize()
        self.thinker_task = asyncio.create_task(self._thinker_loop())
        self.talker_task = asyncio.create_task(self._talker_loop())
        log("info", "Engine Started (Clean Full-Duplex)")

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
