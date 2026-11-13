"""S4w (PLAN.md, phase 4, E11): a payload that fills one window of a large structured tensor."""
import numpy as np

from stowaway import weights
from stowaway.bits import RawTensor


def _structured(nwin: int, seed: int = 0) -> np.ndarray:
    """float32 weights whose low bit 0 is set in about 70% of elements (a low-bit signature)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.02, size=nwin * weights.S4W_WINDOW).astype(np.float32).view(np.uint32)
    v = v & ~np.uint32(1)
    return v | (rng.random(v.size) < 0.7).astype(np.uint32)


def test_payload_in_one_window_is_found():
    v = _structured(6)
    w = weights.S4W_WINDOW
    v[2 * w: 3 * w] = (v[2 * w: 3 * w] & ~np.uint32(0xFF)) | np.random.default_rng(1).integers(0, 256, w, dtype=np.uint32)
    f = weights.s4_windowed({"layer.weight": RawTensor("F32", (v.size,), v)})
    assert [(x.code, x.detail["window"]) for x in f] == [("lowbit-window-anomaly", 2)]


def test_clean_structured_tensor_passes():
    v = _structured(6)
    assert weights.s4_windowed({"layer.weight": RawTensor("F32", (v.size,), v)}) == []


def test_embeddings_and_scales_are_not_windowed():
    v = _structured(6)
    w = weights.S4W_WINDOW
    v[:w] = (v[:w] & ~np.uint32(0xFF)) | np.random.default_rng(1).integers(0, 256, w, dtype=np.uint32)
    t = RawTensor("F32", (v.size,), v)
    assert weights.s4_windowed({"embeddings.word_embeddings.weight": t, "mlp.scales": t}) == []
