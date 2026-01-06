from dataclasses import dataclass
from typing import Optional


@dataclass
class AuditConfig:
    # General
    float_threshold: float = 10.0
    check_batch_size: bool = True
    monitor_dead_neurons: bool = True
    monitor_graph: bool = True

    # NLP
    monitor_nlp: bool = False
    pad_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    vocab_size: Optional[int] = None

    # CV
    monitor_cv: bool = False

    # Auditor performance
    interval: int = 1
    limit: Optional[int] = None
