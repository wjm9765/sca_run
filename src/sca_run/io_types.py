from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class AudioInput:
    """Audio features sent to the inference pipeline.

    features:
        Precomputed log-mel features shaped [1, 128, T] on CPU.
    timestamp:
        Optional wall-clock timestamp (for lag measurement / debugging).
    """

    features: torch.Tensor
    timestamp: float = 0.0


@dataclass
class ThoughtPacket:
    """Optional intermediate representation.

    Kept for future integration with Qwen3-Omni Thinker/Talker style pipelines.
    """

    hidden_states: torch.Tensor
    text_token_str: Optional[str] = None


@dataclass
class TeamAudioReturn:
    """Audio returned by the team inference backend.

    wav:
        Float waveform in [-1, 1]. Shape can be [T], [1, T], [B, T], or [B, T, C].
    sample_rate:
        Sample rate of the waveform.
    channels:
        Number of channels represented by wav (server will downmix to mono for UI).
    text_log:
        Optional text for debugging/subtitles.
    """

    wav: np.ndarray
    sample_rate: int
    channels: int = 1
    text_log: Optional[str] = None


@dataclass
class AudioOutput:
    """Bytes to be streamed to the browser."""

    audio_bytes: bytes
    sample_rate: int
    channels: int = 1
    audio_format: str = "pcm16le"
    text_log: Optional[str] = None
