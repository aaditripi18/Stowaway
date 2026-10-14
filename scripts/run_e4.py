"""E4: how honest fine-tunes change weights, to calibrate S5.

S5 flags a tensor whose changes against the claimed base stay in the low bits,
because an honest fine-tune is believed to change weights by more. Week 3 set
the cutoff at half the mantissa (11 of 23 float32 bits) by guess; at k = 16 a
payload changes bits above it and is missed. E4 measures the real boundary.

For each pair, for every shared tensor the fine-tune actually changed, we take
base XOR fine-tune and record the highest set bit in the mantissa (0..22). A
tensor whose highest changed bit is low is one a payload could hide behind; we
measure how often honest fine-tuning produces such tensors, by tensor role.

--split dev writes the distribution used to choose the cutoff; --split heldout
is the one-time validation and additionally runs the chosen S5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import hf
from experiments.ledger import Run
from stowaway.bits import FLOAT_LAYOUT
from stowaway.weights import role

RESULTS = Path(__file__).resolve().parent.parent / "results"


def highest_changed_bit(ft, base):
    """For a float tensor, the highest mantissa bit that differs, per element,
    and the fraction of elements that changed at all."""
    mant = FLOAT_LAYOUT[ft.dtype][1]
    x = ft.data.astype(np.uint64) ^ base.data.astype(np.uint64)
    changed = x != 0
    mant_bits = x & np.uint64((1 << mant) - 1)
    # highest set bit of the mantissa difference (0 if none)
    hi = np.zeros(x.size, dtype=np.int64)
    for b in range(mant):
        hi = np.where((mant_bits >> np.uint64(b)) & np.uint64(1), b, hi)
    return hi, changed, mant


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout"], default="dev")
    args = ap.parse_args(argv)
    pairs = json.loads((RESULTS / f"ft_pairs_{args.split}.json").read_text(encoding="utf-8"))["pairs"]
    run = Run("E4", split=args.split, n_pairs=len(pairs))

    for p in pairs:
        try:
            ft, base, info = hf.sample_pair(p["finetune"], p["file"], p["base"], hf.weights_file_for(p["base"]))
        except Exception as e:
            run.record(finetune=p["finetune"], base=p["base"], error=f"{type(e).__name__}: {e}"[:300])
            print(f"{p['finetune']}: ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        changed_tensors = 0
        for n in info["shared_tensors"]:
            hi, changed, mant = highest_changed_bit(ft[n], base[n])
            frac_changed = float(changed.mean())
            if frac_changed < 0.5:  # a tensor the fine-tune left essentially frozen
                continue
            changed_tensors += 1
            # among changed elements, the 99th percentile of the highest changed bit
            top = np.percentile(hi[changed], 99) if changed.any() else 0
            run.record(finetune=p["finetune"], base=p["base"], tensor=n, role=role(n),
                       dtype=ft[n].dtype, mantissa_bits=mant, elements=int(ft[n].data.size),
                       frac_changed=round(frac_changed, 4),
                       hi_bit_p99=int(top), hi_bit_max=int(hi[changed].max()) if changed.any() else 0)
        print(f"{p['finetune']}: {changed_tensors}/{len(info['shared_tensors'])} shared tensors changed", flush=True)
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
