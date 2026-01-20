from __future__ import annotations

"""CPU log-mel feature extraction (no HF processor, no model download).

This module purposely avoids depending on external DSP libs (librosa/torchaudio)
so it can run in minimal environments (e.g. RunPod base images).

It approximates Whisper-style log-mel features:
  - 16kHz audio
  - 25ms window (n_fft=400), 10ms hop (hop=160)
  - 128 mel bins
  - log10 + dynamic range compression + normalization

Output shape: [1, 128, T] (CPU, float32)
"""

import math
from functools import lru_cache

import torch


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Create a mel filterbank matrix shaped [n_mels, n_fft//2 + 1]."""

    if f_max is None:
        f_max = float(sample_rate) / 2.0

    # FFT bin frequencies
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, float(sample_rate) / 2.0, n_freqs)

    # Mel-spaced frequencies
    mel_min = _hz_to_mel(torch.tensor(f_min))
    mel_max = _hz_to_mel(torch.tensor(f_max))
    mels = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    hz = _mel_to_hz(mels)

    # Bin indices
    bins = torch.floor((n_fft + 1) * hz / float(sample_rate)).long()
    bins = torch.clamp(bins, 0, n_freqs - 1)

    fb = torch.zeros((n_mels, n_freqs), dtype=torch.float32)

    for i in range(n_mels):
        left = bins[i].item()
        center = bins[i + 1].item()
        right = bins[i + 2].item()

        if center == left:
            center += 1
        if right == center:
            right += 1
        if right <= left:
            continue

        # Rising slope
        if center > left:
            fb[i, left:center] = (torch.arange(left, center) - left) / float(center - left)
        # Falling slope
        if right > center:
            fb[i, center:right] = (right - torch.arange(center, right)) / float(right - center)

    # Normalize filters to have unit area (helps match common implementations)
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    fb *= enorm.unsqueeze(1)
    return fb


def log_mel_spectrogram(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> torch.Tensor:
    """Compute log-mel features shaped [1, n_mels, T] on CPU."""

    if waveform.dim() != 1:
        waveform = waveform.flatten()

    # Default Whisper-ish params for 16kHz.
    if n_fft is None:
        n_fft = int(round(0.025 * sample_rate))  # 25ms
        # For 16kHz -> 400
        # For others -> scale proportionally.
    if hop_length is None:
        hop_length = int(round(0.010 * sample_rate))  # 10ms
    if win_length is None:
        win_length = n_fft

    # Hann window
    window = torch.hann_window(win_length, periodic=True)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )

    # Power spectrogram: [freq, T]
    power = (stft.abs() ** 2)

    fb = _mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel = torch.matmul(fb, power)

    # log10 with clamp
    log_mel = torch.log10(torch.clamp(mel, min=1e-10))

    # Dynamic range compression + normalization (Whisper-style)
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    # Add batch dim: [1, n_mels, T]
    return log_mel.unsqueeze(0).to(dtype=torch.float32)
