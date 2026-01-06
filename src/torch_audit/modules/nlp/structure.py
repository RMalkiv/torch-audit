import torch.nn as nn
from typing import List, Dict, Any


def check_structure(model: nn.Module, vocab_size: int = None, pad_token_id: int = None) -> List[Dict[str, Any]]:
    issues = []

    embeddings = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    if not embeddings:
        return []

    main_embed = sorted(embeddings, key=lambda x: x.num_embeddings, reverse=True)[0]

    v_size = vocab_size if vocab_size else main_embed.num_embeddings
    d_model = main_embed.embedding_dim

    potential_heads = [
        m for m in model.modules()
        if isinstance(m, nn.Linear)
           and m.out_features == v_size
           and m.in_features == d_model
    ]

    if potential_heads:
        is_tied = False
        for head in potential_heads:
            if head.weight is main_embed.weight:
                is_tied = True
                break

        if not is_tied:
            issues.append({
                "type": "NLP Optimization",
                "layer": "Architecture",
                "message": f"Embedding layer and Output Head (Linear {d_model}->{v_size}) are NOT tied. "
                           f"Recommendation: `output_layer.weight = embedding.weight`",
                "severity": "INFO"
            })

    if pad_token_id is not None:
        if main_embed.padding_idx is None:
            issues.append({
                "type": "NLP Efficiency",
                "layer": "Embedding",
                "message": f"Config defines `pad_token_id={pad_token_id}`, but Embedding layer has `padding_idx=None`. "
                           f"Gradients for padding tokens will be calculated unnecessarily. "
                           f"Set `padding_idx` in your Embedding constructor.",
                "severity": "WARNING"
            })
        elif main_embed.padding_idx != pad_token_id:
            issues.append({
                "type": "Configuration Mismatch",
                "layer": "Embedding",
                "message": f"Config defines `pad_token_id={pad_token_id}`, but Embedding layer has `padding_idx={main_embed.padding_idx}`. "
                           f"This indicates a configuration desync.",
                "severity": "ERROR"
            })

    return issues
