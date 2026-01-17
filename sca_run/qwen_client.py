from __future__ import annotations

import io
import os
import tempfile
import threading
import wave
from typing import Optional

from .config import AppConfig


def pcm16le_to_wav_bytes(pcm16le: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw PCM16LE into a WAV container."""
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16le)
    return buf.getvalue()


# -----------------------------
# Transformers backend (local)
# -----------------------------

# Lazily loaded singleton (model + processor)
_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None


def _load_transformers_backend(cfg: AppConfig):
    """Load Qwen3-Omni model + processor once.

    This is intentionally lazy to keep import time fast.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    with _LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return _MODEL, _PROCESSOR

        # Heavy imports live here
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        model_kwargs = {
            "device_map": cfg.qwen.device_map,
        }

        # torch_dtype is stable; some examples also use dtype="auto".
        # Try torch_dtype first, fall back to dtype for compatibility.
        try:
            model_kwargs["torch_dtype"] = cfg.qwen.torch_dtype
            if cfg.qwen.attn_implementation:
                model_kwargs["attn_implementation"] = cfg.qwen.attn_implementation
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(cfg.qwen.model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("torch_dtype", None)
            model_kwargs["dtype"] = cfg.qwen.torch_dtype
            if cfg.qwen.attn_implementation:
                model_kwargs["attn_implementation"] = cfg.qwen.attn_implementation
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(cfg.qwen.model_id, **model_kwargs)

        processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.qwen.model_id)

        model.eval()
        # If model is on a single device, ensure it is in inference dtype.
        # (When device_map="auto", it may be sharded, so we avoid forcing .to(...).)
        if getattr(model, "device", None) is not None:
            try:
                if str(model.device) != "meta":
                    pass
            except Exception:
                pass

        _MODEL, _PROCESSOR = model, processor
        return _MODEL, _PROCESSOR


def infer_audio_once(cfg: AppConfig, wav_bytes: bytes, user_text: str) -> str:
    """Local inference: WAV bytes -> text using Transformers.

    Requirements:
    - transformers (with Qwen3-Omni classes)
    - qwen-omni-utils
    - ffmpeg available in PATH (used by qwen-omni-utils for audio decoding)
    """
    if cfg.qwen.backend.lower() != "transformers":
        raise ValueError(f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers'.")

    model, processor = _load_transformers_backend(cfg)

    # qwen-omni-utils expects a path/URL-like reference in the conversation.
    # We write wav_bytes to a temp file and delete it after preprocessing.
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        conversation = [
            {"role": "system", "content": cfg.qwen.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": tmp_path},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # Heavy imports live here
        import torch
        from qwen_omni_utils import process_mm_info

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        # Try moving inputs to model device when possible.
        try:
            if hasattr(inputs, "to") and getattr(model, "device", None) is not None and str(model.device) != "meta":
                inputs = inputs.to(model.device)
        except Exception:
            pass

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=int(cfg.qwen.max_new_tokens))

        # Decode only newly generated tokens when possible.
        try:
            input_len = inputs["input_ids"].shape[1]
            gen = out[:, input_len:]
            if gen.numel() == 0:
                gen = out
        except Exception:
            gen = out

        return processor.batch_decode(gen, skip_special_tokens=True)[0]

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
