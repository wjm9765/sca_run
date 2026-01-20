from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _env(key: str, default):
    v = os.getenv(key)
    return default if v is None or v == "" else v


@dataclass
class AudioConfig:
    """Audio chunking parameters.

    Team request: remove hardcoded "12.5Hz / 4 frames" and make it configurable.
    """

    # Incoming PCM sample rate (Hz)
    sample_rate: int = 16000

    # Framing reference (Hz). 12.5Hz => 80ms per frame.
    frame_hz: float = 12.5

    # How many frames to batch per inference call.
    frames_per_chunk: int = 4

    # Audio format for PCM streaming
    channels: int = 1
    sample_width_bytes: int = 2  # int16

    @property
    def frame_ms(self) -> float:
        return 1000.0 / float(self.frame_hz)

    @property
    def chunk_ms(self) -> float:
        return self.frame_ms * int(self.frames_per_chunk)

    @property
    def chunk_samples(self) -> int:
        return int(round(float(self.sample_rate) * (self.chunk_ms / 1000.0)))

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * self.channels * self.sample_width_bytes


@dataclass
class QwenConfig:
    """Qwen3-Omni inference settings.

    This scaffold runs a lightweight UI server that:
      - receives PCM over WebSocket
      - extracts log-mel features on CPU
      - calls the team's inference backend (GPU) via `sca_run/team_infer.py`

    So by default we do **not** load any HuggingFace models here.
    """

    backend: str = "team"  # expected value in this scaffold

    # Reserved (for future use). Kept to avoid breaking older configs.
    model_id: str = ""
    device_map: str = ""
    torch_dtype: str = ""
    attn_implementation: str | None = None
    max_new_tokens: int = 0
    system_prompt: str = ""


@dataclass
class AppConfig:
    # NOTE: dataclass fields must not use mutable instances as defaults.
    # Use default_factory so each AppConfig gets its own config objects.
    audio: AudioConfig = field(default_factory=AudioConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from TOML and environment variables.

    Priority: env overrides TOML. If path is None, uses ./config/default.toml
    relative to the repository root.
    """

    if path is None:
        default_path = Path(__file__).resolve().parents[1] / "config" / "default.toml"
        path = default_path
    else:
        path = Path(path)

    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))

    audio_t = data.get("audio", {}) if isinstance(data, dict) else {}
    qwen_t = data.get("qwen", {}) if isinstance(data, dict) else {}

    audio = AudioConfig(
        sample_rate=int(_env("SCA_SAMPLE_RATE", audio_t.get("sample_rate", 16000))),
        frame_hz=float(_env("SCA_FRAME_HZ", audio_t.get("frame_hz", 12.5))),
        frames_per_chunk=int(_env("SCA_FRAMES_PER_CHUNK", audio_t.get("frames_per_chunk", 4))),
        channels=int(_env("SCA_CHANNELS", audio_t.get("channels", 1))),
        sample_width_bytes=int(_env("SCA_SAMPLE_WIDTH_BYTES", audio_t.get("sample_width_bytes", 2))),
    )

    # Qwen config is kept mainly for compatibility; inference is provided by the
    # team backend configured via env vars (see config/default.toml).
    qwen = QwenConfig(
        backend=str(_env("SCA_QWEN_BACKEND", qwen_t.get("backend", "team"))),
        model_id=str(_env("SCA_QWEN_MODEL_ID", qwen_t.get("model_id", ""))),
        device_map=str(_env("SCA_QWEN_DEVICE_MAP", qwen_t.get("device_map", ""))),
        torch_dtype=str(_env("SCA_QWEN_TORCH_DTYPE", qwen_t.get("torch_dtype", ""))),
        attn_implementation=str(_env("SCA_QWEN_ATTN_IMPL", qwen_t.get("attn_implementation", ""))) or None,
        max_new_tokens=int(_env("SCA_QWEN_MAX_NEW_TOKENS", qwen_t.get("max_new_tokens", 0))),
        system_prompt=str(_env("SCA_QWEN_SYSTEM_PROMPT", qwen_t.get("system_prompt", ""))),
    )

    return AppConfig(audio=audio, qwen=qwen)
