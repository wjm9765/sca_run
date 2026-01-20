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

    # Including PCM sample rate (Hz)
    sample_rate: int = 16000
    
    # [Added] Output sample rate for Qwen3-Omni (Hz)
    output_sample_rate: int = 24000

    # Framing reference (Hz). 12.5Hz => 80ms per frame.
    frame_hz: float = 12.5

    # How many frames to batch per inference call.
    # run_test.py uses 0.64s (8 frames at 12.5Hz)
    frames_per_chunk: int = 8 

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

    # Qwen3-Omni settings
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    device_map: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    
    # System Prompt for the model
    system_prompt: str = (
        "<|im_start|>system\n"
        "You are a funny comedian performing a stand-up comedy show using Qwen3-Omni.\n"
        "<|im_end|>\n"
    )

    # Optional: Advanced settings
    attn_implementation: str | None = "flash_attention_2"
    max_new_tokens: int = 0


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

    # Use defaults from the dataclass definitions (config.py)
    # This ensures that modifying the class fields directly works as expected.
    def_audio = AudioConfig()
    def_qwen = QwenConfig()

    audio = AudioConfig(
        sample_rate=int(_env("SCA_SAMPLE_RATE", audio_t.get("sample_rate", def_audio.sample_rate))),
        output_sample_rate=int(_env("SCA_OUTPUT_SAMPLE_RATE", audio_t.get("output_sample_rate", def_audio.output_sample_rate))),
        frame_hz=float(_env("SCA_FRAME_HZ", audio_t.get("frame_hz", def_audio.frame_hz))),
        frames_per_chunk=int(_env("SCA_FRAMES_PER_CHUNK", audio_t.get("frames_per_chunk", def_audio.frames_per_chunk))),
        channels=int(_env("SCA_CHANNELS", audio_t.get("channels", def_audio.channels))),
        sample_width_bytes=int(_env("SCA_SAMPLE_WIDTH_BYTES", audio_t.get("sample_width_bytes", def_audio.sample_width_bytes))),
    )

    # For optional fields, handle None carefully
    attn_default = def_qwen.attn_implementation if def_qwen.attn_implementation else ""

    qwen = QwenConfig(
        backend=str(_env("SCA_QWEN_BACKEND", qwen_t.get("backend", def_qwen.backend))),
        model_id=str(_env("SCA_QWEN_MODEL_ID", qwen_t.get("model_id", def_qwen.model_id))),
        device_map=str(_env("SCA_QWEN_DEVICE_MAP", qwen_t.get("device_map", def_qwen.device_map))),
        torch_dtype=str(_env("SCA_QWEN_TORCH_DTYPE", qwen_t.get("torch_dtype", def_qwen.torch_dtype))),
        attn_implementation=str(_env("SCA_QWEN_ATTN_IMPL", qwen_t.get("attn_implementation", attn_default))) or None,
        max_new_tokens=int(_env("SCA_QWEN_MAX_NEW_TOKENS", qwen_t.get("max_new_tokens", def_qwen.max_new_tokens))),
        system_prompt=str(_env("SCA_QWEN_SYSTEM_PROMPT", qwen_t.get("system_prompt", def_qwen.system_prompt))),
    )

    return AppConfig(audio=audio, qwen=qwen)
