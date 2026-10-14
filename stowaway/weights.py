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

import math

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

def _bitplane_bias(v: np.ndarray, bit: int) -> tuple[float, float]:
    """Mean of one bit plane and a two-sided z-score against the fair-coin null.

    In a natural float model the low mantissa bits are close to a fair coin, so a
    plane whose mean is far from 0.5 is unusual. An all-zero plane (a payload that
    never touched this bit, or cleared precision) also shows here."""
    b = ((v >> np.uint64(bit)) & np.uint64(1)).astype(np.float64)
    n = b.size
    mean = float(b.mean())
    z = (mean - 0.5) / (0.5 / math.sqrt(n)) if n else 0.0
    return mean, z


def _byte_chi2(v: np.ndarray, nbits: int) -> tuple[float, int]:
    """Chi-square of the low-nbits value distribution against uniform."""
    vals = (v & np.uint64((1 << nbits) - 1)).astype(np.int64)
    k = 1 << nbits
    obs = np.bincount(vals, minlength=k).astype(np.float64)
    exp = v.size / k
    return float(((obs - exp) ** 2 / exp).sum()), k - 1


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution (regularised upper gamma),
    computed without SciPy so the package stays dependency-light."""
    if x <= 0:
        return 1.0
    # Wilson–Hilferty normal approximation; adequate for the large df used here.
    t = (x / df) ** (1 / 3)
    mu, sigma = 1 - 2 / (9 * df), math.sqrt(2 / (9 * df))
    z = (t - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


def s4_bit_planes(floats: dict[str, RawTensor], low_bits: int = 8) -> list[Finding]:
    """Per-tensor statistics on the low mantissa bits. Flags tensors whose low bits
    depart from a natural model's: a biased bit plane, or a low-byte distribution
    that is not uniform (both signatures of structured or partial payloads). An
    encrypted payload that fills the low bits uniformly is, by design, invisible
    here — that limit is the point of experiment E1."""
    findings = []
    for name, t in sorted(floats.items()):
        u = t.data
        info = informative(u, t.dtype)
        v = u[info]
        if v.size < MIN_ELEMENTS:
            continue
        mant_bits = FLOAT_LAYOUT[t.dtype][1]
        kb = min(low_bits, mant_bits)

        worst_z, worst_bit = 0.0, -1
        for bit in range(kb):
            _, z = _bitplane_bias(v, bit)
            if abs(z) > abs(worst_z):
                worst_z, worst_bit = z, bit
        # Bonferroni across the bits inspected; flag only strong departures.
        if abs(worst_z) > 6.0:
            findings.append(Finding("S4", "medium", "biased-bitplane",
                                    f"tensor {name!r}: low bit {worst_bit} is biased "
                                    f"(mean z-score {worst_z:+.1f} vs a fair coin)",
                                    {"tensor": name, "bit": worst_bit, "z": round(worst_z, 2),
                                     "elements": int(v.size)}))

        chi2, df = _byte_chi2(v, kb)
        p = _chi2_sf(chi2, df)
        if p < 1e-6:
            findings.append(Finding("S4", "medium", "nonuniform-low-bits",
                                    f"tensor {name!r}: low {kb}-bit values are not uniform "
                                    f"(chi-square p < 1e-6)",
                                    {"tensor": name, "low_bits": kb, "chi2": round(chi2, 1),
                                     "df": df, "elements": int(v.size)}))
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
