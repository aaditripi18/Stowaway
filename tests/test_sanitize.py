"""stowaway sanitize: rounding float32 weights clears a low-bit payload and nothing else."""
from __future__ import annotations

import numpy as np
import pytest

from stowaway.bits import RawTensor
from stowaway.cli import main
from stowaway.embed import embed, extract
from stowaway.sanitize import round_f32_bits, sanitize_safetensors


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("bits", [1, 8, 12, 16, 22])
def test_rounding_clears_low_bits_and_is_nearest(bits):
    rng = np.random.default_rng(bits)
    x = (rng.standard_normal(20000) * 10.0 ** rng.integers(-6, 4, 20000)).astype(np.float32)
    r = round_f32_bits(x.view(np.uint32), bits)
    assert not (r & np.uint32((1 << bits) - 1)).any()
    # nearest: the error is at most half a unit in the last kept bit
    ulp = np.abs(np.spacing(x)).astype(np.float64) * (1 << bits)
    err = np.abs(r.view(np.float32).astype(np.float64) - x.astype(np.float64))
    assert (err <= ulp / 2 * (1 + 1e-12)).all()


def test_special_values_and_overflow_are_safe():
    u = _f32([np.inf, -np.inf, np.nan, np.finfo(np.float32).max, -np.finfo(np.float32).max, 0.0, -0.0])
    r = round_f32_bits(u, 16).view(np.float32)
    assert np.isposinf(r[0]) and np.isneginf(r[1]) and np.isnan(r[2])
    assert np.isfinite(r[3]) and np.isfinite(r[4])   # never rounded up to infinity
    assert r[5] == 0 and r[6] == 0


def test_payload_is_gone_after_sanitizing():
    rng = np.random.default_rng(0)
    t = {"w": RawTensor.from_float(rng.standard_normal(4096).astype(np.float32))}
    secret = rng.bytes(2048)
    planted = embed(t, secret, k=8)
    assert extract(planted, k=8)[:2048] == secret
    clean = {"w": RawTensor("F32", planted["w"].shape, round_f32_bits(planted["w"].data, 8))}
    assert extract(clean, k=8) == bytes(len(extract(clean, k=8)))  # all zero bits


def test_file_round_trip_keeps_everything_but_f32_low_bits(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    rng = np.random.default_rng(1)
    src, dst = tmp_path / "m.safetensors", tmp_path / "clean.safetensors"
    tensors = {"a.weight": torch.tensor(rng.standard_normal((64, 32)).astype(np.float32)),
               "a.bias": torch.tensor(rng.standard_normal(32).astype(np.float16)),
               "ids": torch.arange(10, dtype=torch.int64),
               "b.weight": torch.tensor(rng.standard_normal(1000).astype(np.float32))}
    save_file(tensors, str(src), metadata={"format": "pt", "note": "kept"})
    assert main(["sanitize", str(src), "--out", str(dst), "--bits", "8"]) == 0
    out = load_file(str(dst))
    assert set(out) == set(tensors)
    assert torch.equal(out["a.bias"], tensors["a.bias"]) and torch.equal(out["ids"], tensors["ids"])
    for n in ("a.weight", "b.weight"):
        u = out[n].numpy().view(np.uint32)
        assert not (u & 0xFF).any()
        assert np.allclose(out[n].numpy(), tensors[n].numpy(), rtol=2 ** -15)
    from safetensors import safe_open
    with safe_open(str(dst), "pt") as f:
        assert f.metadata() == {"format": "pt", "note": "kept"}
    assert dst.stat().st_size == src.stat().st_size


def test_refuses_to_overwrite_input(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        sanitize_safetensors(p, p, 8)
