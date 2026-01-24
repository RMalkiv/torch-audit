FINGERPRINT_VERSION = 1


def normalize_module_path(p: str | None) -> str:
    return (p or "global").strip() or "global"


def normalize_entity(e: str | None) -> str:
    return (e or "").strip()


def stable_fingerprint(
    rule_id: str, module_path: str | None, entity: str | None
) -> str:
    mp = normalize_module_path(module_path)
    ent = normalize_entity(entity)
    if ent:
        return f"v{FINGERPRINT_VERSION}:{rule_id}::{mp}::{ent}"
    return f"v{FINGERPRINT_VERSION}:{rule_id}::{mp}"
