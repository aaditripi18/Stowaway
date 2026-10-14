import numpy as np
import pytest

from stowaway.bits import RawTensor
from stowaway.embed import embed
from stowaway.weights import (s3_precision_history, s4_bit_planes,
                              s5_differential)
from conftest import codes


def _fp32_model(seed=0, size=4096):
    rng = np.random.default_rng(seed)
    return {"layer": RawTensor.from_float(rng.standard_normal(size).astype(np.float32))}


def _bf16_as_fp32(seed=0, size=4096, n_layers=6):
    """A model trained in bf16 and stored as fp32: low 16 bits are zero.
    Several layers, as a real checkpoint has."""
    rng = np.random.default_rng(seed)
    model = {}
    for i in range(n_layers):
        bf16 = RawTensor.from_float(rng.standard_normal(size).astype(np.float32), dtype="BF16")
        model[f"layer{i}"] = RawTensor.from_float(bf16.to_float32())  # widen back to fp32
    return model


# --- S3 ---------------------------------------------------------------------

def test_s3_flags_payload_in_bf16_origin_model():
    model = _bf16_as_fp32()
    clean = s3_precision_history(model)
    assert any(f.code == "precision-origin" for f in clean)
    assert not any(f.code == "precision-anomaly" for f in clean)

    # A payload written into one layer breaks that layer's zero block, while the
    # rest of the model still agrees on bf16 precision.
    poisoned = embed(model, b"secret payload bytes" * 20, k=8, spread=1, names=["layer0"])
    found = s3_precision_history(poisoned)
    anomalies = [f for f in found if f.code == "precision-anomaly"]
    assert anomalies and anomalies[0].detail["tensor"] == "layer0"


def test_s3_silent_on_genuine_fp32():
    assert not any(f.code == "precision-anomaly" for f in s3_precision_history(_fp32_model()))


# --- S4 ---------------------------------------------------------------------

def test_s4_clean_fp32_has_no_findings():
    # Natural fp32 weights: low bits look like a fair coin.
    assert s4_bit_planes(_fp32_model(size=8192)) == []


def test_s4_flags_zeroed_low_bits():
    # A payload that sets the low byte to a constant makes bit planes maximally biased.
    model = _fp32_model(size=8192)
    poisoned = embed(model, b"\x00" * 1000, k=8, spread=1)
    assert "lowbit-anomaly" in [f.code for f in s4_bit_planes(poisoned)]


def test_s4_flags_unencrypted_text_payload():
    model = _fp32_model(size=8192)
    # ASCII text is far from uniform in its low bits.
    poisoned = embed(model, b"the quick brown fox " * 200, k=8, spread=1)
    assert "lowbit-anomaly" in [f.code for f in s4_bit_planes(poisoned)]


def test_s4_misses_encrypted_payload_this_is_the_point():
    # H3: a random (encrypted) payload in the low bits is indistinguishable from
    # natural fp32 noise. S4 must stay silent; only S5 could catch this.
    model = _fp32_model(size=16384)
    payload = np.random.default_rng(7).bytes(2000)
    poisoned = embed(model, payload, k=8, spread=1)
    assert s4_bit_planes(poisoned) == []


# --- S5 ---------------------------------------------------------------------

def test_s5_detects_low_bit_only_change_at_any_size():
    base = _fp32_model(size=4096)
    poisoned = embed({k: RawTensor(v.dtype, v.shape, v.data.copy()) for k, v in base.items()},
                     np.random.default_rng(0).bytes(500), k=8, spread=1)
    found = s5_differential(poisoned, base)
    assert any(f.code == "low-bit-only-change" for f in found)


def test_s5_quiet_on_honest_finetune():
    # A fine-tune changes whole weights, not just low bits.
    base = _fp32_model(size=4096)
    rng = np.random.default_rng(2)
    tuned_arr = base["layer"].to_float32() + rng.standard_normal(4096).astype(np.float32) * 0.01
    tuned = {"layer": RawTensor.from_float(tuned_arr)}
    assert not any(f.code == "low-bit-only-change" for f in s5_differential(tuned, base))


def test_s5_reports_when_no_shared_tensors():
    found = s5_differential(_fp32_model(), {"other": _fp32_model()["layer"]})
    assert found[0].code == "no-shared-tensors"


# --- S4 against the model's own signature (found in E0 on MobileNetV3) --------

def _structured_model(n_layers=6, size=8192, seed=3):
    """Clean model whose bit 0 is set in ~2/3 of weights in every tensor, as in
    torchvision's MobileNetV3."""
    rng = np.random.default_rng(seed)
    model = {}
    for i in range(n_layers):
        u = rng.standard_normal(size).astype(np.float32).view(np.uint32).copy()
        ones = rng.random(size) < 2 / 3
        u = np.where(ones, u | 1, u & ~np.uint32(1)).astype(np.uint32)
        model[f"layer{i}"] = RawTensor("F32", (size,), u)
    return model


def test_s4_structured_clean_model_is_not_flagged():
    found = s4_bit_planes(_structured_model())
    assert [f.code for f in found] == ["lowbit-signature"]


def test_s4_random_payload_breaks_structured_signature():
    model = _structured_model()
    payload = np.random.default_rng(9).bytes(4000)  # fills most of layer0's low byte
    poisoned = embed(model, payload, k=8, spread=1, names=["layer0"])
    anomalies = [f for f in s4_bit_planes(poisoned) if f.code == "lowbit-anomaly"]
    assert [f.detail["tensor"] for f in anomalies] == ["layer0"]


def test_s4_stronger_signature_in_one_layer_type_is_not_flagged():
    # MobileNetV3's squeeze-excitation layers: same signature, further from 0.5.
    model = _structured_model()
    u = model["layer5"].data
    rng = np.random.default_rng(4)
    ones = rng.random(u.size) < 0.85
    model["layer5"] = RawTensor("F32", (u.size,), np.where(ones, u | 1, u & ~np.uint32(1)).astype(np.uint32))
    assert not [f for f in s4_bit_planes(model) if f.code == "lowbit-anomaly"]


def test_s4_constant_payload_in_structured_model_is_flagged():
    model = _structured_model()
    poisoned = embed(model, b"\xff" * 4000, k=8, spread=1, names=["layer0"])
    anomalies = [f for f in s4_bit_planes(poisoned) if f.code == "lowbit-anomaly"]
    assert [f.detail["tensor"] for f in anomalies] == ["layer0"]


def test_widen_matches_torch_float():
    torch = pytest.importorskip("torch")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_s3_widened import widen
    arr = np.random.default_rng(5).standard_normal(1000).astype(np.float32)
    bf = RawTensor.from_float(arr, dtype="BF16")
    wide = widen({"w": bf})["w"]
    ref = torch.tensor(arr).to(torch.bfloat16).float().numpy()
    assert np.array_equal(wide.to_float32(), ref)


# --- S4 on 16-bit formats (week 4 survey) -------------------------------------

def test_s4_examines_the_same_mantissa_fraction_in_every_format():
    from stowaway.weights import S4_BITS
    assert S4_BITS == {"F32": 8, "F16": 3, "BF16": 2}


def _bf16_layers(n_layers=5, size=8192, seed=11):
    """A bf16 model in the shape of a transformer checkpoint: per layer, a
    projection (values near 0) and a layer-norm gain (values near 1.0, spanning
    only a few bf16 steps, so every mantissa bit reflects the distribution)."""
    rng = np.random.default_rng(seed)
    model = {}
    for i in range(n_layers):
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj"):
            model[f"layers.{i}.{proj}.weight"] = RawTensor.from_float(
                (0.02 * rng.standard_normal(size)).astype(np.float32), dtype="BF16")
        model[f"layers.{i}.input_layernorm.weight"] = RawTensor.from_float(
            (1.0 + 0.02 * rng.standard_normal(size)).astype(np.float32), dtype="BF16")
    return model


def test_s4_role_peers_keep_layernorm_from_looking_like_a_payload():
    model = _bf16_layers()
    assert not [f for f in s4_bit_planes(model) if f.code == "lowbit-anomaly"]
    # read against the whole model (no role peers), the norms are flagged
    renamed = {f"t{chr(97 + i)}": t for i, t in enumerate(model.values())}  # no digits: no peers
    assert [f for f in s4_bit_planes(renamed) if f.code == "lowbit-anomaly"]


def test_s4_constant_payload_flagged_against_role_peers():
    model = _bf16_layers()
    target = "layers.2.mlp.up_proj.weight"
    poisoned = embed(model, b"\xff" * 2000, k=2, names=[target])
    flagged = [f for f in s4_bit_planes(poisoned) if f.code == "lowbit-anomaly"]
    assert [f.detail["tensor"] for f in flagged] == [target]
    assert flagged[0].detail["reference"] == "role"


def test_role_strips_layer_numbers():
    from stowaway.weights import role
    assert role("model.layers.17.mlp.up_proj.weight") == "model.layers.#.mlp.up_proj.weight"


def _downsample_model(seed=12):
    """ResNet-like: large 4-D convs (fair-coin bit 0) and small 1-D BatchNorm
    weights (bit 0 set in ~48.5%), both named layerN.downsample.M.weight."""
    rng = np.random.default_rng(seed)
    model = {}
    for i in range(2, 5):
        u = rng.standard_normal(131072).astype(np.float32).view(np.uint32).copy()
        model[f"layer{i}.downsample.0.weight"] = RawTensor("F32", (512, 256, 1, 1), u)
    for i in range(1, 5):
        norm = rng.standard_normal(4096).astype(np.float32).view(np.uint32).copy()
        norm = np.where(rng.random(4096) < 0.485, norm | 1, norm & ~np.uint32(1)).astype(np.uint32)
        model[f"layer{i}.downsample.1.weight"] = RawTensor("F32", (4096,), norm)
    return model


def test_s4_peers_must_match_ndim():
    model = _downsample_model()
    assert not [f for f in s4_bit_planes(model) if f.code == "lowbit-anomaly"]
    # without the ndim distinction (all shapes 1-D) the convs are read against norms
    same = {n: (t.data.size,) for n, t in model.items()}
    assert [f for f in s4_bit_planes(model, shapes=same) if f.code == "lowbit-anomaly"]
    # sampled 1-D data with the full shapes given separately behaves like the full model
    flat = {n: RawTensor("F32", (t.data.size,), t.data) for n, t in model.items()}
    shapes = {n: t.shape for n, t in model.items()}
    assert not [f for f in s4_bit_planes(flat, shapes=shapes) if f.code == "lowbit-anomaly"]
