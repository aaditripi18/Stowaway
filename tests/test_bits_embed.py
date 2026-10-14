import numpy as np
import pytest

from stowaway.bits import RawTensor, fields, informative
from stowaway.embed import capacity, embed, extract


def _rng_tensors(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "w1": RawTensor.from_float(rng.standard_normal((64, 32)).astype(np.float32)),
        "w2": RawTensor.from_float(rng.standard_normal(2048).astype(np.float32)),
    }


def test_bf16_roundtrip_matches_torch():
    torch = pytest.importorskip("torch")
    arr = np.linspace(-3, 3, 97, dtype=np.float32)
    rt = RawTensor.from_float(arr, dtype="BF16")
    ref = torch.tensor(arr).to(torch.bfloat16).float().numpy()
    assert np.array_equal(rt.to_float32(), ref)


def test_float32_view_roundtrip():
    arr = np.random.default_rng(1).standard_normal(500).astype(np.float32)
    rt = RawTensor.from_float(arr)
    assert rt.dtype == "F32"
    assert np.array_equal(rt.to_float32(), arr)
    assert rt.to_bytes() == arr.tobytes()


def test_fields_and_informative():
    arr = np.array([0.0, 1.0, 1.5, np.inf, np.nan, 3.0], dtype=np.float32)
    rt = RawTensor.from_float(arr)
    exp, mant = fields(rt.data, "F32")
    info = informative(rt.data, "F32")
    # zero, 1.0 (power of two -> mantissa 0), inf and nan are not informative; 1.5 and 3.0 are.
    assert info.tolist() == [False, False, True, False, False, True]


@pytest.mark.parametrize("k", [1, 4, 8, 16, 24])
@pytest.mark.parametrize("spread", [1, 3])
def test_embed_extract_roundtrip(k, spread):
    tensors = _rng_tensors()
    cap = capacity(tensors, k, spread)
    payload = np.random.default_rng(k).bytes(min(cap, 300))
    out = embed(tensors, payload, k=k, spread=spread)
    assert extract(out, k=k, spread=spread) == payload


def test_embed_only_touches_low_bits():
    tensors = _rng_tensors()
    out = embed(tensors, b"hello world", k=8, spread=1)
    for n, t in tensors.items():
        hi_before = t.data.astype(np.uint64) >> np.uint64(8)
        hi_after = out[n].data.astype(np.uint64) >> np.uint64(8)
        assert np.array_equal(hi_before, hi_after)


def test_embed_rejects_oversize():
    tensors = _rng_tensors()
    with pytest.raises(ValueError):
        embed(tensors, b"x" * (capacity(tensors, 1) + 10), k=1)


def test_payload_barely_perturbs_weights():
    tensors = _rng_tensors()
    out = embed(tensors, np.random.default_rng(0).bytes(200), k=16, spread=1)
    # 16 low bits of a float32 is at most a ~1e-2 relative change; mean should be far smaller.
    before, after = tensors["w2"].to_float32(), out["w2"].to_float32()
    rel = np.abs(after - before) / (np.abs(before) + 1e-12)
    assert rel.mean() < 1e-2
