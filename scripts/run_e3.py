"""E3: robustness of the single-file checks, measured on their known weak spots.

A scanner is only as good as its worst case, so this stress-tests S3/S4 the way
any defence is evaluated: assume whoever hid the data knew how the checks work,
place random bytes where the checks are weakest, and measure what still gets
through -- so the gaps can be reported and, where principled, closed.

Every placement here is an ordinary call to stowaway.embed (which tensors, how
thinly spread, how deep); no new embedding capability is added. The payloads are
random bytes, nothing is extracted or executed, and S5 (the base-model diff) is
reported alongside, since it is unaffected by where in the file the bytes go.

Placements, from the weaknesses found in E0/E1/E4:
  spread_all   low k bits of every float tensor, one weight in every s -- thins
               the per-tensor bit change so no bit-plane mean trips S4's z-test
  fair_only    low k bits, but only in tensors whose low bits are already ~fair
               coins (S4 has no signal there); skips tensors with a signature
  role_fill    low k bits of every tensor of one role -- the peers S4 reads a
               tensor against are changed too, so none stands out

For each placement and model, find the largest payload S3/S4 miss, and report
whether S5 would still catch it given the clean model as base.
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments import corpus
from experiments.ledger import Run
from stowaway.bits import informative
from stowaway.embed import capacity, embed
from stowaway.weights import role

FLAG = ("medium", "high", "critical")


def single_file_sees(rep, written, clean_flagged):
    return any(x["check"] in ("S3", "S4") and x["severity"] in FLAG
              and x["detail"].get("tensor") in written - clean_flagged
              for x in rep["findings"])


def base_sees(rep, written, clean_flagged):
    return any(x["check"] == "S5" and x["severity"] in FLAG
              and x["detail"].get("tensor") in written - clean_flagged
              for x in rep["findings"])


def fair_coin_tensors(raw):
    """Tensors whose low bit is already close to a fair coin: S4 has no signal."""
    out = []
    for n, t in raw.items():
        v = t.data[informative(t.data, t.dtype)]
        if v.size >= 4096 and abs(float((v & np.uint64(1)).mean()) - 0.5) < 0.01:
            out.append(n)
    return out


def role_groups(raw):
    groups = {}
    for n, t in raw.items():
        if t.data[informative(t.data, t.dtype)].size >= 4096:
            groups.setdefault(role(n), []).append(n)
    return {r: ns for r, ns in groups.items() if len(ns) >= 4}


def placements(raw):
    yield "spread_all", {"spread": 64, "names": None}
    fair = fair_coin_tensors(raw)
    if fair:
        yield "fair_only", {"spread": 1, "names": fair}
    groups = role_groups(raw)
    if groups:
        biggest = max(groups.values(), key=lambda ns: sum(raw[n].data.size for n in ns))
        yield "role_fill", {"spread": 1, "names": biggest}


def changed_tensors(poisoned, clean):
    return {n for n in poisoned if not np.array_equal(poisoned[n].data, clean[n].data)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args(argv)
    model, _ = corpus.load_vision(args.model)
    clean = corpus.state_to_raw(model.state_dict())
    clean_rep = corpus.scan_raw(clean)
    clean_flagged = {x["detail"].get("tensor") for x in clean_rep["findings"] if x["severity"] in FLAG}
    run = Run("E3", model=args.model, k=args.k)
    rng = np.random.default_rng(0)

    for name, opt in placements(clean):
        cap = capacity(clean, args.k, opt["spread"], opt["names"])
        for size in (1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20):
            if size > cap:
                continue
            poisoned = embed(clean, rng.bytes(size), k=args.k, spread=opt["spread"], names=opt["names"])
            written = changed_tensors(poisoned, clean)
            rep = corpus.scan_raw(poisoned, base_raw=clean)
            sf = single_file_sees(rep, written, clean_flagged)
            bs = base_sees(rep, written, clean_flagged)
            run.record(model=args.model, placement=name, k=args.k, spread=opt["spread"],
                       payload_bytes=size, tensors_written=len(written),
                       single_file_detected=sf, base_detected=bs)
            print(f"{name:11} size={size:>8} tensors={len(written):>3} single={sf!s:5} base={bs}", flush=True)
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
