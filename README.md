# sca_run

```bash
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

@dataclass
class OmniModelContext:
    audio_encoder: Any  # Audio Feature Extractor
    thinker: Any        # Text LLM
    talker: Any         # Audio Generation Model
    code2wav: Any       # Audio Decoder (Codec)

@dataclass
class ConversationState:
    # Thinker(LLM)의 과거 기억 (Key-Value Cache)
    past_key_values_thinker: Optional[List[torch.Tensor]] = None
    
    # Talker(Audio Gen)의 과거 기억
    past_key_values_talker: Optional[List[torch.Tensor]] = None
    
    # 현재까지 누적된 텍스트 토큰 히스토리 (System Prompt + 대화 내용)
    text_history_ids: Optional[torch.Tensor] = None

@dataclass
class InferenceStepInput:
    # 방금 들어온 오디오의 전처리된 텐서 (없으면 None)
    # Shape: [Batch, Channel, Time] 등 모델 규격에 맞춤
    new_audio_features: Optional[torch.Tensor]
    
    # 현재 대화 상태 (직전 스텝의 Output에서 받은 것)
    state: ConversationState

@dataclass
class InferenceStepOutput:
    # 스피커로 내보낼 Raw Audio Bytes (생성된 게 없으면 b'')
    generated_audio_bytes: bytes
    
    # 갱신된 대화 상태 (다음 턴 입력으로 사용)
    updated_state: ConversationState

```
