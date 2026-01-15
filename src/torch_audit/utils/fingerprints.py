from typing import Optional

FINGERPRINT_VERSION = 1


def normalize_module_path(p: Optional[str]) -> str:
    return (p or "global").strip() or "global"


def normalize_entity(e: Optional[str]) -> str:
    return (e or "").strip()


def stable_fingerprint(
    rule_id: str, module_path: Optional[str], entity: Optional[str]
) -> str:
    mp = normalize_module_path(module_path)
    ent = normalize_entity(entity)
    if ent:
        return f"v{FINGERPRINT_VERSION}:{rule_id}::{mp}::{ent}"
    return f"v{FINGERPRINT_VERSION}:{rule_id}::{mp}"
