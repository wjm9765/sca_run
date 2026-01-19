from __future__ import annotations

"""Team inference integration point.

This project intentionally keeps Qwen3-Omni inference out of the UI server.
The server extracts audio features ([1,128,T]) on CPU and calls *this* module.

When the team delivers the fine-tuned Qwen3-Omni inference code, plug it in by
editing `infer_team_wav()` (or by importing your code inside it).

Required interface for this scaffold:
  - Input: `AudioInput` whose `features` is a CPU torch.Tensor [1,128,T]
  - Output: `TeamAudioReturn` with `wav` float waveform and `sample_rate`

Return wav as:
  - numpy.ndarray float32 in [-1,1], shape [T] (preferred)
  - (also accepted) torch.Tensor on CPU
"""

import os
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from .config import AppConfig
from .io_types import AudioInput, TeamAudioReturn


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return default if v is None else v


@lru_cache(maxsize=1)
def _load_team_backend():
    """Optional dynamic loader.

    If you want to keep this repo untouched and load the team code via env vars,
    set:
      - SCA_TEAM_MODULE (e.g. "my_team_pkg.infer")
      - SCA_TEAM_FACTORY (e.g. "build_infer")

    The factory should return an object with:
      - infer(features: torch.Tensor) -> (wav: np.ndarray|torch.Tensor, sample_rate: int)
    """

    mod_name = _env("SCA_TEAM_MODULE", "")
    factory_name = _env("SCA_TEAM_FACTORY", "")
    if not mod_name or not factory_name:
        return None

    import importlib

    mod = importlib.import_module(mod_name)
    factory = getattr(mod, factory_name)
    return factory()


def infer_team_wav(cfg: AppConfig, audio_in: AudioInput) -> Optional[TeamAudioReturn]:
    """Call the team inference backend and return a float waveform.

    This default implementation returns None (no audio) until the team backend
    is plugged in.
    """

    backend = _load_team_backend()
    if backend is None:
        # Not configured yet.
        return None

    # Team backend chooses devices; we keep features on CPU.
    wav, sr = backend.infer(audio_in.features)

    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().to("cpu").float().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)

    # Normalize shape to [T]
    if wav_np.ndim >= 2:
        # If [1, T] or [B, T], take first batch
        wav_np = wav_np.reshape(-1)

    return TeamAudioReturn(wav=wav_np, sample_rate=int(sr), channels=1)
