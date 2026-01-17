from __future__ import annotations

import array
import io
import threading
import wave
from typing import Optional, Tuple

from .config import AppConfig


def wav_bytes_to_pcm16le(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    """Decode simple PCM WAV bytes into raw PCM16LE frames.

    This uses Python's built-in `wave` module, so it supports *uncompressed* WAV.
    If your clients may upload compressed WAV, decode it on the client side first.

    Returns:
        pcm16le_bytes, sample_rate, channels
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_rate = int(wf.getframerate())
        sample_width = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        frames = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported (got sample_width={sample_width}).")
    return frames, sample_rate, channels


def _pcm16le_to_float_mono(pcm16le: bytes, channels: int) -> list[float]:
    """Convert PCM16LE bytes to a mono float waveform in [-1, 1].

    - If channels==2, averages L/R into mono.
    - If channels>2, averages all channels.
    """
    if channels <= 0:
        raise ValueError("channels must be positive")

    # array('h') reads native-endian int16. Most machines are little-endian.
    a = array.array("h")
    a.frombytes(pcm16le)

    if channels == 1:
        return [float(x) / 32768.0 for x in a]

    # Multi-channel: average per frame.
    n = len(a) // channels
    out: list[float] = []
    out_extend = out.append
    idx = 0
    for _ in range(n):
        s = 0
        for _c in range(channels):
            s += a[idx]
            idx += 1
        out_extend((float(s) / float(channels)) / 32768.0)
    return out


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
        _MODEL, _PROCESSOR = model, processor
        return _MODEL, _PROCESSOR


def infer_pcm16le_once(
    cfg: AppConfig,
    pcm16le: bytes,
    user_text: str,
    *,
    sample_rate: int,
    channels: int = 1,
    timestamp: float = 0.0,
) -> str:
    """Local inference: PCM16LE chunk -> text using Transformers.

    Design choice (per team discussion):
    - The *streaming layer* does NOT wrap PCM into WAV.
    - Feature extraction happens here (in the inference layer), not in the audio-input layer.

    Args:
        pcm16le: raw PCM16LE bytes.
        user_text: prompt text.
        sample_rate: PCM sample rate.
        channels: number of channels in pcm16le.
        timestamp: optional, useful for lag measurement.
    """
    if cfg.qwen.backend.lower() != "transformers":
        raise ValueError(f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers'.")

    model, processor = _load_transformers_backend(cfg)

    # Convert PCM bytes to mono float waveform.
    waveform = _pcm16le_to_float_mono(pcm16le, channels=channels)

    # Build a conversation with an audio placeholder so the chat template inserts the right tokens.
    conversation = [
        {"role": "system", "content": cfg.qwen.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "<pcm16le>"},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Heavy imports live here
    import torch

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Preferred path: explicit feature extraction (matches team approach).
    inputs = None
    fe = getattr(processor, "feature_extractor", None)
    if fe is not None:
        try:
            fe_out = fe(waveform, sampling_rate=sample_rate, return_tensors="pt")
            # Build text tokens separately, then attach input_features.
            text_inputs = processor(text=text, return_tensors="pt", padding=True)
            inputs = dict(text_inputs)
            inputs["input_features"] = fe_out.input_features
        except Exception:
            inputs = None

    # Fallback: let the processor handle audio internally.
    if inputs is None:
        try:
            inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True, sampling_rate=sample_rate)
        except TypeError:
            inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True)

    # Move inputs to model device when possible.
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


def infer_wav_once(cfg: AppConfig, wav_bytes: bytes, user_text: str) -> str:
    """Convenience wrapper: WAV upload -> PCM16LE -> inference."""
    pcm16le, sr, ch = wav_bytes_to_pcm16le(wav_bytes)
    return infer_pcm16le_once(cfg, pcm16le, user_text, sample_rate=sr, channels=ch)


# Backward compatible name (older scaffolds used infer_audio_once on WAV bytes)
def infer_audio_once(cfg: AppConfig, wav_bytes: bytes, user_text: str) -> str:  # pragma: no cover
    return infer_wav_once(cfg, wav_bytes, user_text)
