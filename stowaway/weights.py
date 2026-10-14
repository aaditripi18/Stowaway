"""Statistical checks on the weight bits themselves (S3, S4, S5).

These are the checks no code scanner performs. They look for data hidden in the
low-order bits of float weights, where it survives conversion to "safe" formats
because it is just numbers.

S3  precision history: low bits that should be zero given how the model was stored
S4  bit-plane statistics: low mantissa bits that do not look like a natural model's
S5  differential: bit-level changes against a claimed base model

Every check returns Findings; none of them needs to run the model.
"""
from __future__ import annotations

import numpy as np

from .bits import FLOAT_LAYOUT, RawTensor, informative
from .model import Finding, ModelFile

MIN_ELEMENTS = 256          # below this, statistics are too noisy to judge
FP16_EXACT_LOW = 13         # fp32 bits that are zero when a value came from fp16
BF16_EXACT_LOW = 16         # fp32 bits that are zero when a value came from bf16


# --- helpers ----------------------------------------------------------------

def _load_floats(mf: ModelFile, f) -> dict[str, RawTensor]:
    out = {}
    for t in mf.tensors:
        if t.dtype in FLOAT_LAYOUT and t.offset is not None and t.nbytes:
            raw = mf.tensor_bytes(f, t)
            if len(raw) == t.nbytes:
                out[t.name] = RawTensor.from_bytes(t.dtype, t.shape, raw)
    return out


# --- S3: precision history --------------------------------------------------

def s3_precision_history(floats: dict[str, RawTensor]) -> list[Finding]:
    """Many published models are trained in fp16/bf16 and uploaded as fp32. Those
    fp32 values then have a block of guaranteed-zero low bits. Any model that is
    fp32-on-disk but whose low bits are zero has room that a payload would fill;
    if the zero block is broken in only some tensors, that is where to look."""
    findings = []
    f32 = {n: t for n, t in floats.items() if t.dtype == "F32"}
    if not f32:
        return findings

    profiles = {}
    for name, t in f32.items():
        u = t.data
        info = informative(u, "F32")
        if info.sum() < MIN_ELEMENTS:
            continue
        v = u[info]
        # Largest L such that the low L bits are zero for every informative element.
        any_bit = int(np.bitwise_or.reduce(v))
        trailing_zero = (any_bit & -any_bit).bit_length() - 1 if any_bit else 23
        profiles[name] = min(trailing_zero, 23)

    if not profiles:
        return findings
    counts = np.bincount(list(profiles.values()), minlength=24)
    model_floor = int(np.argmax(counts))  # the precision most tensors agree on

    if model_floor >= BF16_EXACT_LOW:
        origin, exact = "bfloat16", BF16_EXACT_LOW
    elif model_floor >= FP16_EXACT_LOW:
        origin, exact = "float16", FP16_EXACT_LOW
    else:
        return findings  # genuinely fp32-trained; S3 has nothing to say, S4 takes over

    findings.append(Finding("S3", "info", "precision-origin",
                            f"stored as float32 but the weights carry only {origin} precision "
                            f"(low {exact} bits are zero in most tensors)",
                            {"origin": origin, "exact_low_bits": exact, "tensors": len(profiles)}))
    for name, floor in sorted(profiles.items()):
        if floor < exact - 1:  # this tensor uses low bits the rest of the model does not
            dirty = exact - floor
            findings.append(Finding("S3", "high", "precision-anomaly",
                                    f"tensor {name!r} uses {dirty} low bit(s) that a {origin} model leaves zero",
                                    {"tensor": name, "unexpected_low_bits": dirty, "origin": origin}))
    return findings


# --- S4: bit-plane statistics -----------------------------------------------

SIGNATURE_MIN_ELEMENTS = 4096  # tensors large enough to estimate the model's signature
Z_FLAG = 6.0                   # |z| above this flags a tensor (strict: many tensors x bits)


def _bit_means(v: np.ndarray, nbits: int) -> np.ndarray:
    """Fraction of ones in each of the low `nbits` bit planes."""
    return np.array([float(((v >> np.uint64(b)) & np.uint64(1)).mean()) for b in range(nbits)])


def _signature(stats: list[tuple[str, int, np.ndarray]], nbits: int):
    """The model's own low-bit signature and how much clean tensors vary around it.

    Clean models are not all fair coins in their low bits: torchvision's
    MobileNetV3 has bit 0 set in ~67% of weights, consistently across every
    tensor. So each tensor is tested against the median of the model's large
    tensors (sig), with tau = the spread between tensors beyond sampling noise
    (a random-effects null). With fewer than three tensors there is no
    population, and the fair coin (0.5) is the fallback."""
    big = [s for s in stats if s[1] >= SIGNATURE_MIN_ELEMENTS]
    pool = big if len(big) >= 3 else stats if len(stats) >= 3 else []
    if not pool:
        return np.full(nbits, 0.5), np.zeros(nbits), False
    m = np.array([s[2] for s in pool])
    n = np.array([s[1] for s in pool], dtype=np.float64)
    sig = np.median(m, axis=0)
    mad = 1.4826 * np.median(np.abs(m - sig), axis=0)
    sampling = np.median(sig * (1 - sig) / n[:, None], axis=0)
    tau = np.sqrt(np.maximum(0.0, mad ** 2 - sampling))
    return sig, tau, True


def s4_bit_planes(floats: dict[str, RawTensor], low_bits: int = 8) -> list[Finding]:
    """Per-tensor low-bit statistics, read against the model's own signature.

    A tensor is flagged when one of its low bit planes departs from what the rest
    of the model shows: a payload pulls a structured model's bits towards 0.5,
    and structured payloads (text, zeros) bias bits in any model. An encrypted
    payload in a model whose low bits are already fair coins is, by design,
    invisible here; that limit is what experiment E1 measures."""
    findings = []
    by_dtype: dict[str, list] = {}
    for name, t in sorted(floats.items()):
        v = t.data[informative(t.data, t.dtype)]
        if v.size < MIN_ELEMENTS:
            continue
        kb = min(low_bits, FLOAT_LAYOUT[t.dtype][1])
        by_dtype.setdefault(t.dtype, []).append((name, int(v.size), _bit_means(v, kb)))

    for dtype, stats in by_dtype.items():
        kb = len(stats[0][2])
        sig, tau, has_pop = _signature(stats, kb)
        if has_pop and np.max(np.abs(sig - 0.5)) > 0.02:
            findings.append(Finding("S4", "info", "lowbit-signature",
                                    f"{dtype} weights carry a low-bit signature (bit means "
                                    + ", ".join(f"{x:.2f}" for x in sig[:4]) + "); tensors are tested against it",
                                    {"dtype": dtype, "signature": [round(float(x), 4) for x in sig]}))
        # Where the model has a signature, only movement towards (or past) the fair
        # coin counts: that is what a random payload does, while clean layers of a
        # different type can carry a stronger version of the same signature
        # (MobileNetV3's squeeze-excitation layers). Where it has none, any
        # departure counts, which keeps text or constant payloads visible.
        # This rule was chosen after seeing MobileNetV3 in E0; it is validated on
        # models that were not used to choose it.
        structured = np.abs(sig - 0.5) > 0.02
        toward_coin = -np.sign(sig - 0.5)
        for name, n, m in stats:
            se = np.sqrt(np.maximum(sig * (1 - sig), 1e-6) / n)
            z = (m - sig) / np.sqrt(se ** 2 + tau ** 2)
            score = np.where(structured, z * toward_coin, np.abs(z))
            b = int(np.argmax(score))
            if score[b] > Z_FLAG:
                findings.append(Finding("S4", "medium", "lowbit-anomaly",
                                        f"tensor {name!r}: low bit {b} is set in {m[b]:.1%} of weights, "
                                        f"against {sig[b]:.1%} across the model (z {z[b]:+.1f})",
                                        {"tensor": name, "bit": b, "z": round(float(z[b]), 2),
                                         "mean": round(float(m[b]), 4), "expected": round(float(sig[b]), 4),
                                         "elements": n}))
    return findings


# --- S5: differential against a base model ----------------------------------

def s5_differential(floats: dict[str, RawTensor], base: dict[str, RawTensor]) -> list[Finding]:
    """Compare against the claimed base model, tensor by tensor. An honest fine-tune
    changes whole weights in the layers it trained; a payload changes only the low
    bits, and often in layers a fine-tune would leave frozen. Any shared tensor that
    differs *only* in its low bits is reported, at any size — this is the strongest
    check when a base model is available."""
    findings = []
    shared = sorted(set(floats) & set(base))
    if not shared:
        findings.append(Finding("S5", "info", "no-shared-tensors",
                                "the base model shares no tensor names with this model"))
        return findings

    for name in shared:
        a, b = floats[name], base[name]
        if a.dtype != b.dtype or a.data.shape != b.data.shape:
            continue
        diff = a.data != b.data
        if not diff.any():
            continue
        changed = int(diff.sum())
        mant_bits = FLOAT_LAYOUT[a.dtype][1]
        # Of the changed elements, how many differ only within the low bits?
        xor = (a.data[diff].astype(np.uint64) ^ b.data[diff].astype(np.uint64))
        high_mask = np.uint64(~((1 << (mant_bits // 2)) - 1) & ((1 << 64) - 1))
        low_only = int((xor & high_mask == 0).sum())
        frac_low = low_only / changed
        if frac_low > 0.98 and changed >= MIN_ELEMENTS:
            findings.append(Finding("S5", "high", "low-bit-only-change",
                                    f"tensor {name!r} differs from the base model in {changed} elements, "
                                    f"{frac_low:.0%} of them only in the low bits - not how a fine-tune changes weights",
                                    {"tensor": name, "changed": changed, "low_only_fraction": round(frac_low, 4)}))
    return findings


# --- entry point used by report.py ------------------------------------------

def analyse(mf: ModelFile, f, base: dict[str, RawTensor] | None = None) -> list[Finding]:
    floats = _load_floats(mf, f)
    if not floats:
        return []
    findings = s3_precision_history(floats) + s4_bit_planes(floats)
    if base is not None:
        findings += s5_differential(floats, base)
    return findings


def load_base(path, open_model) -> dict[str, RawTensor]:
    """Load a base model's float tensors for S5."""
    from pathlib import Path
    path = Path(path)
    with open(path, "rb") as fh:
        mf = open_model(path, fh)
        return _load_floats(mf, fh)
