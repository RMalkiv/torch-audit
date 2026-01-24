from unittest.mock import patch

import pytest
import torch

from torch_audit.utils.sanitize import json_safe, sanitize_metadata


class UnserializableObj:
    def __repr__(self):
        return "<MyObj>"


class Point:
    def __init__(self):
        self.x = 10
        self.y = 20


def test_json_safe_primitives():
    """Primitives should pass through unchanged."""
    assert json_safe(1) == 1
    assert json_safe(1.5) == 1.5
    assert json_safe("hello") == "hello"
    assert json_safe(None) is None
    assert json_safe(True) is True


def test_json_safe_tensors():
    """Tensors should be converted to scalars or metadata dicts."""
    # Scalar (0-dim)
    t_scalar = torch.tensor(3.14)
    assert pytest.approx(json_safe(t_scalar), 0.01) == 3.14

    # Vector
    t_vec = torch.zeros(2, 2)
    res = json_safe(t_vec)
    assert isinstance(res, dict)
    assert res["type"] == "Tensor"
    assert res["shape"] == (2, 2)
    assert "dtype" in res


def test_json_safe_torch_types():
    """Torch types (dtype, device) should convert to strings."""
    assert json_safe(torch.float32) == "torch.float32"
    assert json_safe(torch.int64) == "torch.int64"

    d = torch.device("cpu")
    assert json_safe(d) == "cpu"


def test_json_safe_collections():
    """Lists, tuples, and dicts should be handled recursively."""
    # Tuple -> List
    data_tuple = (1, 2, (3, 4))
    assert json_safe(data_tuple) == [1, 2, [3, 4]]

    # Dict keys -> Strings
    data_dict = {"a": 1, 5: "int_key"}
    res = json_safe(data_dict)
    assert res["a"] == 1
    assert res["5"] == "int_key"


def test_json_safe_recursion_limit():
    """Should stop recursing and stringify when max_depth is reached."""
    data = {"level1": {"level2": "deep"}}

    # max_depth=1:
    # level1 (depth 0) -> processed
    # level2 (depth 1) -> hit max -> stringified
    res = json_safe(data, max_depth=1)

    assert isinstance(res, dict)
    assert "level1" in res
    # The inner dict becomes a string representation
    assert res["level1"] == "{'level2': 'deep'}"


def test_json_safe_objects_happy_path():
    """Regular objects with __dict__ should be converted to dicts."""
    p = Point()
    res = json_safe(p)
    assert res["x"] == 10
    assert res["y"] == 20


def test_json_safe_objects_fallback():
    """Objects without useful __dict__ should fallback to str()."""
    obj = UnserializableObj()
    # has empty __dict__ or no __dict__ -> returns repr
    assert json_safe(obj) == "<MyObj>"


def test_json_safe_object_crash():
    """If inspecting the object raises an error, fallback to str()."""
    # We mock 'vars' to raise an exception when called
    with patch("torch_audit.utils.sanitize.vars", side_effect=ValueError("Boom")):
        p = Point()
        # Should catch the ValueError and return str(p)
        # Point doesn't have a custom repr, so it's <tests.test_sanitize.Point object...>
        res = json_safe(p)
        assert "Point object" in res


def test_sanitize_metadata_wrapper():
    """Verify the wrapper function works correctly."""
    meta = {
        "grad_fn": "AddBackward",
        "tensor": torch.randn(1),  # Scalar tensor -> float
    }
    clean = sanitize_metadata(meta)
    assert clean["grad_fn"] == "AddBackward"
    assert isinstance(clean["tensor"], float)
