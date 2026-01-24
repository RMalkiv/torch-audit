import torch

from torch_audit.utils.torch_safe import isfinite_all, safe_norm


def test_isfinite_all_dense():
    t = torch.tensor([1.0, 2.0, float("nan")])
    assert not isfinite_all(t)

    t2 = torch.tensor([1.0, 2.0])
    assert isfinite_all(t2)


def test_isfinite_all_sparse():
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3.0, 4.0, 5.0]
    s = torch.sparse_coo_tensor(i, v, (2, 3))
    assert isfinite_all(s)

    v_nan = [3.0, float("nan"), 5.0]
    s_nan = torch.sparse_coo_tensor(i, v_nan, (2, 3))
    assert not isfinite_all(s_nan)


def test_isfinite_all_meta():
    t = torch.tensor([1.0]).to(device="meta")
    assert isfinite_all(t)


def test_safe_norm():
    t = torch.tensor([3.0, 4.0])
    assert safe_norm(t) == 5.0

    i = [[0], [0]]
    v = [3.0]
    s = torch.sparse_coo_tensor(i, v, (1, 1))
    assert safe_norm(s) == 3.0

    t_meta = torch.tensor([1.0]).to(device="meta")
    res = safe_norm(t_meta)
    assert res is None or isinstance(res, float)
