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

import re

import numpy as np

from .bits import FLOAT_LAYOUT, RawTensor, informative
from .model import Finding, ModelFile

MIN_ELEMENTS = 256          # below this, statistics are too noisy to judge
FP16_EXACT_LOW = 13         # fp32 bits that are zero when a value came from fp16
BF16_EXACT_LOW = 16         # fp32 bits that are zero when a value came from bf16


# --- helpers ----------------------------------------------------------------

def _float_infos(mf: ModelFile) -> dict:
    """Float tensors the checks read; a repeated name keeps the last, as a dict does."""
    return {t.name: t for t in mf.tensors if t.dtype in FLOAT_LAYOUT and t.offset is not None and t.nbytes}


def _read(mf: ModelFile, f, t) -> RawTensor | None:
    raw = mf.tensor_bytes(f, t)
    return RawTensor.from_bytes(t.dtype, t.shape, raw) if len(raw) == t.nbytes else None


def _load_floats(mf: ModelFile, f) -> dict[str, RawTensor]:
    out = {}
    for name, t in _float_infos(mf).items():
        rt = _read(mf, f, t)
        if rt is not None:
            out[name] = rt
    return out


class _Summary:
    """What S3 and S4 read from one tensor: its informative elements' count, the
    OR of their bits, and the fraction set in each low bit plane. Computed once,
    so the tensor itself can be dropped (the scan holds one tensor at a time)."""
    __slots__ = ("dtype", "ndim", "n", "or_bits", "means")

    def __init__(self, t: RawTensor, nbits: int, ndim: int | None = None):
        v = t.data[informative(t.data, t.dtype)]
        self.dtype, self.ndim, self.n = t.dtype, len(t.shape) if ndim is None else ndim, int(v.size)
        self.or_bits = int(np.bitwise_or.reduce(v)) if v.size else 0
        self.means = _bit_means(v, nbits) if v.size else np.zeros(nbits)


def _summaries(floats: dict[str, RawTensor], low_bits=None, shapes=None) -> dict[str, _Summary]:
    bits = S4_BITS if low_bits is None else low_bits
    shapes = shapes or {}
    return {n: _Summary(t, bits[t.dtype], len(shapes[n]) if n in shapes else None) for n, t in floats.items()}


# --- S3: precision history --------------------------------------------------

def s3_precision_history(floats: dict[str, RawTensor]) -> list[Finding]:
    """Many published models are trained in fp16/bf16 and uploaded as fp32. Those
    fp32 values then have a block of guaranteed-zero low bits. Any model that is
    fp32-on-disk but whose low bits are zero has room that a payload would fill;
    if the zero block is broken in only some tensors, that is where to look."""
    return _s3(_summaries(floats))


def _s3(sums: dict[str, _Summary]) -> list[Finding]:
    findings = []
    f32 = {n: t for n, t in sums.items() if t.dtype == "F32"}
    if not f32:
        return findings

    profiles = {}
    for name, t in f32.items():
        if t.n < MIN_ELEMENTS:
            continue
        # Largest L such that the low L bits are zero for every informative element.
        any_bit = t.or_bits
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

# Bit planes S4 examines: the same low fraction of the mantissa in every format
# (8 of float32's 23 bits, ~35%). Higher mantissa bits encode the shape of the
# value distribution (layer-norm weights near 1.0, quantization scales), which
# differs between clean tensors; the first S4 used 8 bits everywhere, i.e. the
# whole bf16 mantissa, and flagged those (Hugging Face survey, week 4).
S4_BITS = {dt: round(m * 8 / 23) for dt, (_, m, _) in FLOAT_LAYOUT.items()}  # F32 8, F16 3, BF16 2


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


# S4b (PLAN.md, E0d): a reference of fewer than SMALL_ROLE_PEERS role peers
# cannot show less between-tensor spread than the whole model does.
SMALL_ROLE_PEERS = 6
S4_SMALL_FLOOR = False


def s4_bit_planes(floats: dict[str, RawTensor], low_bits: dict[str, int] | None = None,
                  shapes: dict[str, tuple] | None = None, small_floor: bool | None = None) -> list[Finding]:
    """Per-tensor low-bit statistics, read against the model's own signature.

    A tensor is flagged when one of its low bit planes departs from what the rest
    of the model shows: a payload pulls a structured model's bits towards 0.5,
    and structured payloads (text, zeros) bias bits in any model. An encrypted
    payload in a model whose low bits are already fair coins is, by design,
    invisible here; that limit is what experiment E1 measures.

    `shapes` gives the tensors' full shapes when `floats` holds samples."""
    return _s4(_summaries(floats, low_bits, shapes), small_floor)


def _s4(sums: dict[str, _Summary], small_floor: bool | None = None) -> list[Finding]:
    small_floor = S4_SMALL_FLOOR if small_floor is None else small_floor
    findings = []
    by_dtype: dict[str, list] = {}
    for name, t in sorted(sums.items()):
        # Quantization metadata (AWQ `.scales`, FP8 `weight_scale_inv`) is not
        # weights and its low bits follow the quantization grid, not a model
        # signature; S4 would read it as anomalous. It is also a poor place to
        # hide a payload (few values, read at load time).
        if "scale" in name.lower():
            continue
        if t.n < MIN_ELEMENTS:
            continue
        by_dtype.setdefault(t.dtype, []).append((name, t.n, t.means))

    for dtype, stats in by_dtype.items():
        kb = len(stats[0][2])
        model_ref = _signature(stats, kb)
        sig, _, has_pop = model_ref
        if has_pop and np.max(np.abs(sig - 0.5)) > 0.02:
            means = ", ".join(str(round(float(x), 2)) for x in sig[:4])
            findings.append(Finding("S4", "info", "lowbit-signature",
                                    f"{dtype} weights carry a low-bit signature (bit means {means}); tensors are tested against it",
                                    {"dtype": dtype, "signature": [round(float(x), 4) for x in sig]}))
        # Peers share the role and the number of dimensions: numbering alone
        # merges module types (ResNet's downsample.0 is a conv, downsample.1 a
        # BatchNorm), and those carry different low-bit statistics.
        def key(name):
            return role(name), sums[name].ndim

        roles: dict[tuple, list] = {}
        for s in stats:
            roles.setdefault(key(s[0]), []).append(s)

        for name, n, m in stats:
            # Reference: the other tensors of the same role (same name, other
            # layers), when there are at least three; otherwise the whole model.
            # Different kinds of tensor carry different low-bit signatures (bf16
            # routers and embeddings at 50% where the model sits at 39-44%), and
            # read against the whole model those look exactly like a payload.
            # Leaving the tensor itself out keeps a payload out of its own reference.
            peers = [s for s in roles[key(name)] if s[0] != name]
            if len(peers) >= 3:
                sig, tau, _ = _signature(peers, kb)
                if small_floor and len(peers) < SMALL_ROLE_PEERS:
                    tau = np.maximum(tau, model_ref[1])
                against = f"its {len(peers)} role peers"
            else:
                sig, tau, _ = model_ref
                against = "the model"
            # A random payload adds entropy, so it moves a bit plane *towards* the
            # fair coin (0.5) and never away from it. So the only suspicious
            # direction is closer to 0.5 than the reference. Flagging any departure
            # (the earlier rule) also flagged tensors that are *further* from 0.5 --
            # clean low-precision tensors whose low bit is stuck at 0, common in
            # 16-bit and quantized files -- which a payload could never produce
            # (Hugging Face fp16 survey, week 5). This loses constant/zero payloads,
            # which bias a bit away from 0.5 and are indistinguishable from
            # precision from one file; the base diff (S5) catches those.
            se = np.sqrt(np.maximum(sig * (1 - sig), 1e-6) / n)
            denom = np.sqrt(se ** 2 + tau ** 2)
            closer = (np.abs(sig - 0.5) - np.abs(m - 0.5)) / denom  # >0: m nearer 0.5 than ref
            b = int(np.argmax(closer))
            if closer[b] > Z_FLAG:
                mean, expected, z = round(float(m[b]), 4), round(float(sig[b]), 4), round(float(closer[b]), 2)
                # Message uses the same rounded values as the detail, so the Python
                # and TypeScript cores format the numbers identically (see canon).
                findings.append(Finding("S4", "medium", "lowbit-anomaly",
                                        f"tensor {name!r}: low bit {b} is nearer a fair coin ({mean}) than {against} "
                                        f"({expected}); z {z}",
                                        {"tensor": name, "bit": b, "z": z,
                                         "mean": mean, "expected": expected,
                                         "reference": "role" if len(peers) >= 3 else "model",
                                         "elements": n}))
    return findings


def role(name: str) -> str:
    """A tensor's role: its name with layer and block numbers removed,
    e.g. model.layers.#.mlp.up_proj.weight."""
    return re.sub(r"\d+", "#", name)


# --- S5: differential against a base model ----------------------------------

# Honest fine-tuning changes weights by whole magnitudes: in every one of 189
# fine-tuned float32 tensors measured (E4, dev), the changes reached the top
# mantissa bit (22) at the 99th percentile. A payload in the low k bits reaches
# only bit k-1. So a tensor whose changes stay below (mantissa - MARGIN) bits did
# not change the way fine-tuning does. MARGIN = 3 leaves a gap below the honest
# floor; the cutoff is 20 bits for float32, 7 for fp16, 4 for bf16.
S5_MARGIN = 3
S5_PERCENTILE = 99.0  # of changed elements; robust to a few coincidental high flips


def _highest_changed_bit(xor: np.ndarray, mant_bits: int) -> np.ndarray:
    """Per element, the highest mantissa bit set in `xor` (-1 if none)."""
    mant = xor & np.uint64((1 << mant_bits) - 1)
    hi = np.full(mant.size, -1, dtype=np.int64)
    for b in range(mant_bits):
        hi = np.where((mant >> np.uint64(b)) & np.uint64(1), b, hi)
    return hi


def s5_differential(floats: dict[str, RawTensor], base: dict[str, RawTensor]) -> list[Finding]:
    """Compare against the claimed base model, tensor by tensor. Honest fine-tuning
    changes whole weights, reaching the top of the mantissa; a payload changes only
    the low bits. A shared tensor whose changes stay in the low bits did not change
    the way fine-tuning does, and is flagged at any payload size — the strongest
    check when a base model is available. The cutoff is calibrated in E4 from real
    fine-tune/base pairs, not guessed."""
    findings = []
    shared = sorted(set(floats) & set(base))
    if not shared:
        findings.append(Finding("S5", "info", "no-shared-tensors",
                                "the base model shares no tensor names with this model"))
        return findings

    for name in shared:
        f = _s5_tensor(name, floats[name], base[name])
        if f is not None:
            findings.append(f)
    return findings


def _s5_tensor(name: str, a: RawTensor, b: RawTensor) -> Finding | None:
    if a.dtype != b.dtype or a.data.shape != b.data.shape:
        return None
    diff = a.data != b.data
    changed = int(diff.sum())
    if changed < MIN_ELEMENTS:
        return None
    mant_bits = FLOAT_LAYOUT[a.dtype][1]
    cutoff = mant_bits - S5_MARGIN
    xor = a.data[diff].astype(np.uint64) ^ b.data[diff].astype(np.uint64)
    hi = _highest_changed_bit(xor, mant_bits)
    reach = float(np.percentile(hi, S5_PERCENTILE))
    if reach > cutoff:
        return None
    return Finding("S5", "high", "low-bit-only-change",
                   f"tensor {name!r} differs from the base model in {changed} elements, "
                   f"but the changes reach only bit {int(reach)} of {mant_bits} "
                   f"(fine-tuning reaches the top bit) - a low-bit payload, not training",
                   {"tensor": name, "changed": changed,
                    "highest_changed_bit_p99": round(reach, 1),
                    "cutoff": cutoff, "mantissa_bits": mant_bits})


# --- entry point used by report.py ------------------------------------------

# --- aligning tensor names with the base ------------------------------------

_LEGACY_NAMES = {"gamma": "weight", "beta": "bias"}  # old LayerNorm parameter names
MAX_DROP = 2


def _canon(name: str, drop: int) -> str:
    parts = name.split(".")[drop:]
    if parts and parts[-1] in _LEGACY_NAMES:
        parts[-1] = _LEGACY_NAMES[parts[-1]]
    return ".".join(parts)


def align(model: dict[str, tuple], base: dict[str, tuple]) -> tuple[dict[str, str], int, int]:
    """Pair each tensor of `model` with the base tensor it fine-tunes.

    Fine-tunes often wrap the base (`bert.x` saved as `encoder.x`) or use the
    old LayerNorm names (`gamma`/`beta` for `weight`/`bias`). Each argument maps
    a tensor name to (dtype, shape). Tries dropping 0..MAX_DROP leading name
    components on each side, with the old names mapped, and keeps the choice
    that pairs the most tensors of identical dtype and shape (the first, i.e.
    fewest drops, on a tie). A name that is ambiguous after dropping is left out.
    Returns (model name -> base name, dropped from model, dropped from base)."""
    best = ({}, 0, 0)
    for dm in range(MAX_DROP + 1):
        for db in range(MAX_DROP + 1):
            bkeys: dict[str, list] = {}
            for b in base:
                bkeys.setdefault(_canon(b, db), []).append(b)
            mkeys: dict[str, list] = {}
            for m in model:
                mkeys.setdefault(_canon(m, dm), []).append(m)
            pairs = {}
            for key, ms in mkeys.items():
                bs = bkeys.get(key)
                if key and len(ms) == 1 and bs and len(bs) == 1 and model[ms[0]] == base[bs[0]]:
                    pairs[ms[0]] = bs[0]
            if len(pairs) > len(best[0]):
                best = (pairs, dm, db)
    return best


class BaseFile:
    """A claimed base model for S5, read one tensor at a time while it is compared.
    It may span several files (a sharded checkpoint): pass each (ModelFile, file)."""

    def __init__(self, mf: ModelFile, f, *more):
        parts = [(mf, f), *zip(more[::2], more[1::2])]
        self.infos = {}
        for pmf, pf in parts:
            for name, t in _float_infos(pmf).items():
                self.infos[name] = (pmf, pf, t)

    def get(self, name: str) -> RawTensor | None:
        hit = self.infos.get(name)
        return _read(*hit) if hit is not None else None

    def shapes(self) -> dict[str, tuple]:
        return {n: (t.dtype, tuple(t.shape)) for n, (_, _, t) in self.infos.items()}


def analyse(mf: ModelFile, f, base: "dict[str, RawTensor] | BaseFile | None" = None) -> list[Finding]:
    """S3, S4 and (with a base) S5, holding one tensor - and its base counterpart -
    in memory at a time. Results are identical to the dictionary-based checks."""
    sums: dict[str, _Summary] = {}
    s5: list[tuple[str, Finding]] = []
    shared = 0
    infos = _float_infos(mf)
    pairing, dm, db = {}, 0, 0
    if isinstance(base, BaseFile):
        pairing, dm, db = align({n: (t.dtype, tuple(t.shape)) for n, t in infos.items()}, base.shapes())
    for name, t in infos.items():
        a = _read(mf, f, t)
        if a is None:
            continue
        sums[name] = _Summary(a, S4_BITS[a.dtype])
        if base is not None:
            b = base.get(pairing.get(name, name)) if isinstance(base, BaseFile) else base.get(name)
            if b is not None:
                shared += 1
                found = _s5_tensor(name, a, b)
                if found is not None:
                    s5.append((name, found))
        del a
    if not sums:
        return []
    findings = _s3(sums) + _s4(sums)
    renamed = sum(1 for m, b in pairing.items() if m != b)
    if renamed:
        findings.append(Finding("S5", "info", "aligned-names",
                                f"paired {len(pairing)} tensors with the base by name after renaming {renamed} "
                                f"(dropped {dm} leading name part(s) here and {db} in the base, "
                                f"gamma/beta read as weight/bias)",
                                {"paired": len(pairing), "renamed": renamed, "dropped_model": dm, "dropped_base": db}))
    if base is not None:
        if not shared:
            findings.append(Finding("S5", "info", "no-shared-tensors",
                                    "the base model shares no tensor names with this model"))
        else:
            findings += [x for _, x in sorted(s5, key=lambda p: p[0])]
    return findings


def load_base(path, open_model) -> dict[str, RawTensor]:
    """Load a base model's float tensors for S5 (all in memory; for small models
    and experiments - report.scan streams the base with BaseFile instead)."""
    from pathlib import Path
    path = Path(path)
    with open(path, "rb") as fh:
        mf = open_model(path, fh)
        return _load_floats(mf, fh)
