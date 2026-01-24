from torch_audit.utils.fingerprints import (
    normalize_entity,
    normalize_module_path,
    stable_fingerprint,
)


def test_normalize_module_path():
    assert normalize_module_path(None) == "global"
    assert normalize_module_path("") == "global"
    assert normalize_module_path("  ") == "global"
    assert normalize_module_path(" layer1 ") == "layer1"


def test_normalize_entity():
    assert normalize_entity(None) == ""
    assert normalize_entity(" weight ") == "weight"


def test_stable_fingerprint_format():
    # 1. With Entity
    fp1 = stable_fingerprint("TA001", "layer1", "weight")
    assert fp1 == "v1:TA001::layer1::weight"

    # 2. Without Entity
    fp2 = stable_fingerprint("TA001", "layer1", None)
    assert fp2 == "v1:TA001::layer1"

    # 3. Global
    fp3 = stable_fingerprint("TA001", None, None)
    assert fp3 == "v1:TA001::global"
