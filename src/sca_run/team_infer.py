from __future__ import annotations

"""Team inference integration point - Qwen3-Omni FullDuplex version.

This module performs the following:
  1. Receives Log-Mel Spectrogram or Raw Audio from the server/client logic.
  2. Passes it to Qwen3DuplexLogic (src/inference.py).
  3. Handles Thinker (Understand) → Talker (Reply) → Code2Wav (Audio Generation) flow.
  4. Returns the generated audio as TeamAudioReturn.

Data Flow:
  PCM16 Audio → feature_extractor → Log-Mel [1,128,T] or Raw Audio [T]
           → team_infer.py (this file)
           → Qwen3DuplexLogic
           → Generated Audio [T]
           → Transmitted to Client via WebSocket
"""

import os
import threading
import queue
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from utils.client_utils import log
from .config import AppConfig
from .io_types import AudioInput, TeamAudioReturn
import tarfile
import shutil
import urllib.request
from pathlib import Path

# Import team member's code
try:
    # Qwen3OmniFullDuplexEngine is the Engine class written by the user (see run_test.py)
    from inference import Qwen3OmniFullDuplexEngine, EngineConfig
    TEAM_CODE_AVAILABLE = True
except ImportError:
    TEAM_CODE_AVAILABLE = False
    log("warning", "[Warning] Could not find team's inference.py. Check src/inference.py path.")

def _download_and_apply_lora(model, tokenizer):
    """Downloads and applies LoRA adapter. Checks for special token 151646 without resizing."""
    lora_url = "https://s3.riverfog7.com/models/sca/full-duplex/SCA_duplex_finetune.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=lhQBsIsthBRiz5aEKzdA%2F20260122%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260122T051026Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=2d2867dbf2ab9fb0d67d343b62ab7c19ad8658006f8bce6125c999754b24f99a"
    local_dir = Path("lora_adapter")
    
    # 1. Download and Extract if not exists
    # If the tarball contains "SCA_duplex_finetune", we verify that specific folder.
    target_extract_path = local_dir / "SCA_duplex_finetune"
    
    if not target_extract_path.exists():
        log("info", f"[Team Inference] ⬇️ Downloading LoRA adapter from S3...")
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            tar_filename = "lora_adapter.tar.gz"
            with urllib.request.urlopen(lora_url) as response, open(tar_filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
            log("info", "[Team Inference] 📦 Extracting LoRA adapter...")
            with tarfile.open(tar_filename, "r:gz") as tar:
                # Filter to avoid unsafe extractions if possible, but minimal implementation:
                tar.extractall(path=local_dir)
            
            if os.path.exists(tar_filename):
                os.remove(tar_filename)
        except Exception as e:
             log("error", f"[Team Inference] ❌ Error downloading/extracting LoRA: {e}")
             return model, tokenizer

    # 2. Find Adapter Configuration (Robust Search)
    adapter_path = None
    # Prefer expected path: lora_adapter/SCA_duplex_finetune/final_model
    candidates = [
        target_extract_path / "final_model",
        target_extract_path,
        local_dir
    ]
    for c in candidates:
        if (c / "adapter_config.json").exists():
            adapter_path = c
            break
            
    # Fallback: scan directory if not found in obvious places
    if not adapter_path:
        for root, dirs, files in os.walk(local_dir):
            if "adapter_config.json" in files:
                adapter_path = Path(root)
                break

    if not adapter_path:
        log("error", f"[Team Inference] ❌ Could not find 'adapter_config.json' in {local_dir}. LoRA merge skipped.")
        return model, tokenizer

    # 3. Special Token Verification (Strict Check, No Resize)
    # Token 151646 must be within vocab size to avoid lookup errors.
    special_token_id = 151646
    vocab_size = model.get_input_embeddings().weight.shape[0]
    
    if special_token_id >= vocab_size:
        log("warning", f"[Team Inference] ⚠️ Special token {special_token_id} is OUT of model vocab range ({vocab_size}). This may cause errors.")
    else:
        log("info", f"[Team Inference] ✅ Special token {special_token_id} is within vocab range ({vocab_size}).")

    # 4. Apply LoRA and Merge
    try:
        from peft import PeftModel
        log("info", f"[Team Inference] 🔗 Loading LoRA from {adapter_path} and merging...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
        log("info", "[Team Inference] ✅ LoRA merged successfully!")
    except ImportError:
        log("error", "[Team Inference] ❌ PEFT library is not installed. Skipping LoRA merge.")
    except Exception as e:
        log("error", f"[Team Inference] ❌ Failed to merge LoRA: {e}")

    return model, tokenizer

        
def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return default if v is None else v


# ============================================================================
# Global State Management
# ============================================================================

_model_lock = threading.Lock()
_qwen_model = None
_qwen_tokenizer = None

def _load_qwen_model_and_tokenizer(cfg: AppConfig):
    """Loads the fine-tuned Qwen3-Omni model and tokenizer."""
    global _qwen_model, _qwen_tokenizer
    
    if _qwen_model is not None and _qwen_tokenizer is not None:
        return _qwen_model, _qwen_tokenizer
    
    log("info", "[Team Inference] 🔄 Loading Qwen3-Omni model...")
    
    try:
        from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration
        
        # Use Config object instead of raw env vars
        model_id = cfg.qwen.model_id
        device_map = cfg.qwen.device_map
        torch_dtype = cfg.qwen.torch_dtype
        attn_impl = cfg.qwen.attn_implementation

        log("info", f"[Team Inference] Model ID: {model_id}")
        log("info", f"[Team Inference] Device Map: {device_map}")
        if attn_impl:
            log("info", f"[Team Inference] Attention Implementation: {attn_impl}")
        
        # Load Model (using Qwen3OmniMoeForConditionalGeneration)
        _qwen_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype if torch_dtype != "auto" else None,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        # Load Tokenizer
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # Apply LoRA if configured
        _qwen_model, _qwen_tokenizer = _download_and_apply_lora(_qwen_model, _qwen_tokenizer)

        log("info", "[Team Inference] ✅ Model & Tokenizer loaded!")
        return _qwen_model, _qwen_tokenizer
    
    except Exception as e:
        log("error", f"[Team Inference] ❌ Model/Tokenizer Load Failed: {e}")
        raise


# ============================================================================
# Session-based Inference (Engine Wrapper)
# ============================================================================

class TeamInferenceSession:
    """Per-connection session state using Qwen3OmniFullDuplexEngine."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.engine = None
        self.started = False
    
    async def initialize(self):
        """Async initialization to prevent blocking the server event loop."""
        import asyncio
        loop = asyncio.get_running_loop()
        
        log("info", "[Team Inference] Starting async model loading...")

        # 1. Load Model & Tokenizer (Thread-safe, non-blocking for asyncio)
        def _load_mt():
            with _model_lock:
                return _load_qwen_model_and_tokenizer(self.cfg)
        
        self.model, self.tokenizer = await loop.run_in_executor(None, _load_mt)

        # 2. Load Processor
        def _load_proc():
            from transformers import Qwen3OmniMoeProcessor
            # Use local path if possible or download
            return Qwen3OmniMoeProcessor.from_pretrained(
                self.model.config._name_or_path, 
                trust_remote_code=True
            )
        
        if self.processor is None:
            self.processor = await loop.run_in_executor(None, _load_proc)
            log("info", "[Team Inference] ✅ Processor Loaded!")

        # 3. Engine Initialization
        if not TEAM_CODE_AVAILABLE:
            raise RuntimeError("src.inference not available")

        self.engine_config = EngineConfig(
            system_prompt_text=self.cfg.qwen.system_prompt
        )
        
        self.engine = Qwen3OmniFullDuplexEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.engine_config
        )
        
        # [User Request] Start immediately to trigger prefill/compilation
        await self.engine.start()
        self.started = True
        log("info", "[Team Inference] Engine created and started (Compiled & Ready).")

    async def start(self):
        """Start the inference engine."""
        if self.engine is None:
             await self.initialize()

        if not self.started:
            await self.engine.start()
            self.started = True
            log("info", "[Team Inference] Engine started.")

    async def stop(self):
        """Stop the inference engine."""
        if self.started:
            await self.engine.stop()
            self.started = False
            log("info", "[Team Inference] Engine stopped.")

    async def push_input(self, audio_in: AudioInput):
        """
        Push audio to the engine.
        
        [Strict Verification Updated]
        Performs Feature Extraction using Processor, consistent with run_test.py.
        audio_in.features can be Raw Float Tensor (1D) or existing Features (2D).
        """
        if not self.started:
            return

        # 1. Check Input Data
        data = audio_in.features
        target_device = self.model.device
        target_dtype = self.model.dtype

     
        is_raw_audio = False
        if isinstance(data, torch.Tensor):
            # 1D tensor is always Raw Audio, 2D if not 128(Mel) is Raw Audio
            if data.dim() < 2 or (data.dim() == 2 and data.shape[-2] != 128): 
                is_raw_audio = True
        elif isinstance(data, np.ndarray):
             if data.ndim < 2 or (data.ndim == 2 and data.shape[-2] != 128):
                is_raw_audio = True

        if is_raw_audio:
            if isinstance(data, torch.Tensor):
                chunk = data.detach().cpu().numpy().squeeze()
            else:
                chunk = np.array(data).squeeze()
                
            # Padding Logic (Strictly following run_test.py)
            target_len = int(16000 * 0.64) # 10240 samples
            if len(chunk) < target_len:
                # pad right
                chunk = np.pad(chunk, (0, target_len - len(chunk)))
            
            # Feature Extraction via Processor
            features = self.processor.feature_extractor(
                [chunk], 
                return_tensors="pt", 
                sampling_rate=16000,
                padding=False,
            )
            input_features = features.input_features.to(target_device).to(target_dtype)
            
            # NaN Check
            if torch.isnan(input_features).any() or torch.isinf(input_features).any():
                input_features = torch.nan_to_num(input_features, nan=0.0, posinf=0.0, neginf=0.0)
                
            await self.engine.push_audio(input_features)
            
        else:
            # Legacy Pre-computed Feature path (if Feature Extractor used externally)
            # If server.py still uses qwen_client's log_mel_spectrogram, it lands here.
            # But to match run_test.py, Raw should be sent.
            if not isinstance(data, torch.Tensor):
                features = torch.from_numpy(data)
            else:
                features = data
            
            features = features.to(device=target_device, dtype=target_dtype)
            await self.engine.push_audio(features)

    async def get_output(self) -> Optional[TeamAudioReturn]:
        """
        Try to get audio output from the engine.
        Returns TeamAudioReturn or None.
        """
        if not self.started:
            return None

        # get_audio_output returns bytes (PCM16LE mono 24kHz presumably)
        out_bytes = await self.engine.get_audio_output()
        
        if out_bytes:
            # Convert bytes -> float32 [-1, 1]
            wav_int16 = np.frombuffer(out_bytes, dtype=np.int16)
            wav_float = wav_int16.astype(np.float32) / 32768.0
            
            # Engine output sample rate assumed to be 24000Hz (Qwen3 Omni default)
            return TeamAudioReturn(
                wav=wav_float,
                sample_rate=24000,
                channels=1,
                text_log=None
            )
        return None

# ============================================================================
# Legacy / Single-shot Wrapper (Deprecated)
# ============================================================================



def infer_team_wav(cfg: AppConfig, audio_in: AudioInput) -> Optional[TeamAudioReturn]:

    """
    Inference using Qwen3-Omni FullDuplex model.
    
    Input:
        cfg: AppConfig
        audio_in: AudioInput (Log-Mel features [1, 128, T])
    
    Output:
        TeamAudioReturn (wav float32, sample_rate=24000)
    """
    
    if not TEAM_CODE_AVAILABLE:
        print("[Team Inference] ⚠️ Cannot find team's inference.py.")
        return None
    
    try:
        global _duplex_logic, _step_count
        
        # 1. Initialize Duplex Logic (Once)
        logic = _init_duplex_logic(cfg)
        
        # 2. Prepare Input Data
        features = audio_in.features
        
        if isinstance(features, torch.Tensor):
            features = features.float()
        else:
            features = torch.from_numpy(features).float()
        
        print(f"[Team Inference] 입력 Feature 형태: {features.shape}")
        
        print("[Team Inference] 🧠 Thinker 처리 중...")
        
        # Feature Attention Mask 생성
        time_len = features.shape[2] if features.dim() == 3 else features.shape[1]
        feature_mask = torch.ones((1, time_len), dtype=torch.long)
        
        with torch.no_grad():
            # Thinker Step
            thinker_out = logic.thinker_step(
                input_ids=None,
                input_features=features,
                feature_attention_mask=feature_mask,
                past_key_values=None,
                step_idx=_step_count
            )
            
            _step_count += 1
            
            # 첫 토큰 예측
            next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            print(f"[Team Inference] Thinker 예측 토큰: {next_token.item()}")
            
            # 4. Talker 단계: 답변 생성하기
            print("[Team Inference] 👄 Talker 처리 중...")
            
            # Thinker의 hidden state를 가져오기
            thinker_hidden = thinker_out.hidden_states[-1]
            
            # Talker Step
            audio_codes, talker_kv = logic.talker_step(
                thinker_hidden=thinker_hidden,
                past_key_values=None,
                step_idx=_step_count,
                input_ids=None
            )
            
            _step_count += 1
            
            print(f"[Team Inference] 생성된 오디오 코드 형태: {audio_codes.shape}")
            
            # 5. Code2Wav 단계: 음성 생성하기
            print("[Team Inference] 🎵 Code2Wav 처리 중...")
            
            wav_bytes = logic.decode_audio(audio_codes)
            
            # 바이트를 float32 배열로 변환
            wav_int16 = np.frombuffer(wav_bytes, dtype=np.int16)
            wav_float = wav_int16.astype(np.float32) / 32768.0
            wav_float = np.clip(wav_float, -1.0, 1.0)
            
            print(f"[Team Inference] ✅ 생성된 음성 길이: {len(wav_float)} samples ({len(wav_float)/24000:.2f}초)")
            
            # 6. TeamAudioReturn으로 반환
            return TeamAudioReturn(
                wav=wav_float,
                sample_rate=24000,  # Qwen3-Omni의 기본 샘플레이트
                channels=1,
                text_log=None
            )
    
    except Exception as e:
        import traceback
        print(f"[Team Inference] 추론 실패: {e}")
        traceback.print_exc()
        return None


def reset_conversation():
    global _duplex_logic, _step_count
    
    with _model_lock:
        _duplex_logic = None
        _step_count = 0
        print("[Team Inference]  대화 상태 초기화됨")
