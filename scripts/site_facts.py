"""Every number on the project page, computed from the ledger, and the gate
that keeps the page honest.

    python scripts/site_facts.py            check web/index.html, exit 1 on any problem
    python scripts/site_facts.py --write    rewrite the page's numbers from the ledger

On the page, a number is written as <span data-fact="key">text</span>. The check
fails if:

  - a data-fact's text differs from what the ledger says now,
  - a data-fact names a key this script does not compute,
  - any other number appears in the page's visible text, unless it is on the
    short list of names and method constants in ALLOWED below, or inside an
    element marked data-nofacts (the decorative hex dump, the shell commands),
  - an assertion behind a sentence on the page ("missed at every size") is false.

Test counts come from collecting the suites (pytest --collect-only, vitest
list); --no-counts skips that, for the unit test that runs this check.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import load  # noqa: E402
from scripts import report as R  # noqa: E402

PAGE = ROOT / "web" / "index.html"

# Numbers that are names or fixed method parameters, not results.
ALLOWED = [
    r"S[1-5]", r"H[1-4]", r"E[0-5][a-c]?", r"0[1-9]",          # check, hypothesis, experiment, section ids
    r"ResNet-18", r"MobileNetV3(-Small)?", r"V3", r"float(16|32)", r"fp(16|32)", r"bf16", r"F(16|32)",
    r"\.safetensors", r"S[1-5]/S[1-5]", r"S3/S4",
    r"k = 1, 4(,)? (or|and) 8", r"k = 1, 4 or 8", r"bit 0",
    r"1K", r"4K", r"16K", r"64K", r"256K", r"1M",              # E1 strip axis labels
    r"64 KB samples", r"2026", r"\(pre-1\.6\)", r"1\.6",
    r"0 clean, 1 suspicious, 2 dangerous", r"5 checks", r"1 KB or more", r"low 8 bits", r"[Tt]op-1", r"1:(?=\s|$)",
]
_ALLOWED_RE = re.compile("|".join(f"(?:{a})" for a in ALLOWED))


# ---------------------------------------------------------------- facts

def _pct(x):
    return f"{100 * x:.1f}%"


def _cell(rs, key):
    return f"{sum(bool(r[key]) for r in rs)} / {len(rs)}"


def _e2(rows, model):
    _, recs = R._latest_run([r for r in rows if r.get("model") == model], "E2")
    clean = [r for r in recs if r["payload_bytes"] == 0][0]["accuracy"]
    low = [r for r in recs if r["payload_bytes"] and r["k"] <= 8]
    zero = [r for r in low if r["accuracy_drop"] == 0]
    return clean, max(abs(r["accuracy_drop"]) for r in low), max(r["payload_bytes"] for r in zero), recs


def _hf(rows):
    mine = [r for r in rows if r.get("meta", {}).get("list", "hf_models.json") == "hf_models.json"]
    _, survey = R._latest_run(mine, "E0b")
    _, attack = R._latest_run(mine, "E1b")
    survey = [r for r in survey if r["dtype"] == "F32"]  # the page's survey is the float32 files
    ok = [r for r in survey if "error" not in r]
    valid = [r for r in attack if not r["target_flagged_clean"] and r["dtype"] == "F32"]
    half = ("F32 file, bfloat16 precision", "F32 file, float16 precision")
    groups = {"half": lambda g: g in half, "sig": lambda g: g == "F32, low-bit signature",
              "fair": lambda g: g == "F32, fair-coin low bits"}
    out, asserts = {}, {}
    for name, pick in groups.items():
        cells = {k: [r for r in valid if pick(r["group"]) and r["k"] == k] for k in (1, 4, 8)}
        per_k = {k: _cell(rs, "detected") for k, rs in cells.items()}
        asserts[f"hf {name}: same result at k = 1, 4 and 8"] = len(set(per_k.values())) == 1
        out[f"hf.{name}"] = str(sum(1 for r in ok if pick(r["group"])))
        out[f"hf.{name}_cell"] = per_k[1]
    import json
    frozen = json.loads((ROOT / "results" / "hf_models.json").read_text(encoding="utf-8"))["models"]
    out["hf.listed"] = str(sum(1 for m in frozen if m["dtype"] == "F32"))
    out["hf.sampled"] = str(len(ok))
    exposed = int(out["hf.half_cell"].split(" / ")[0]) + int(out["hf.sig_cell"].split(" / ")[0])
    out["hf.exposed"] = str(exposed)
    return out, asserts


def _counts():
    py = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                        capture_output=True, text=True).stdout
    n_py = int(re.search(r"(\d+) tests? collected", py).group(1))
    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    ts = subprocess.run([npx, "vitest", "list"], cwd=ROOT / "web", capture_output=True, text=True).stdout
    n_ts = sum(1 for line in ts.splitlines() if ".test.ts >" in line)
    return {"tests.py": str(n_py), "tests.ts": str(n_ts), "tests.total": str(n_py + n_ts)}


def _mobilenet_bit0():
    import numpy as np

    from experiments import corpus
    from stowaway.bits import informative
    model, _ = corpus.load_vision("mobilenet_v3_small")
    raw = corpus.state_to_raw(model.state_dict())
    v = np.concatenate([t.data[informative(t.data, t.dtype)] for t in raw.values()])
    return f"~{round(100 * float((v & 1).mean()))}%"


def facts(counts=True, weights=True):
    rows = load()
    f, asserts = {}, {}

    for model, key in (("resnet18", "resnet"), ("mobilenet_v3_small", "mobilenet")):
        clean, worst, biggest, recs = _e2(rows, model)
        f[f"e2.{key}.clean"] = _pct(clean)
        f[f"e2.{key}.max_zero"] = R._fmt_size(biggest)
        f[f"e2.{key}.delta"] = f"{100 * worst:.1f}"
        f["e2.n_images"] = str(recs[0]["n_images"])
    asserts["E2 ResNet-18: no accuracy change at k <= 8"] = f["e2.resnet.delta"] == "0.0"

    _, e1 = R._latest_run([r for r in rows if r.get("model") == "resnet18"], "E1")
    asserts["E1 ResNet-18: single-file checks miss every size"] = not any(r["single_file_detected"] for r in e1)
    asserts["E1 ResNet-18: base diff catches every size at k <= 8"] = all(
        r["base_detected"] for r in e1 if r["k"] <= 8)
    f["e1.min"] = R._fmt_size(min(r["payload_bytes"] for r in e1))
    f["e1.max"] = R._fmt_size(max(r["payload_bytes"] for r in e1))
    _, e1m = R._latest_run([r for r in rows if r.get("model") == "mobilenet_v3_small"], "E1")
    asserts["E1 MobileNetV3: naive attack seen from the file at k <= 8"] = all(
        r["single_file_detected"] for r in e1m if r["k"] <= 8 and r["spread"] == 1)

    hf, a = _hf(rows)
    f.update(hf); asserts.update(a)

    _, e3 = R._latest_run([r for r in rows if r.get("model") == "mobilenet_v3_small"], "E3")
    spread = sorted((r for r in e3 if r["placement"] == "spread_all"), key=lambda r: r["payload_bytes"])
    asserts["E3 MobileNetV3 spread_all: missed by the file alone at every size"] = not any(
        r["single_file_detected"] for r in spread)
    f["e3.spread"] = str(spread[0]["spread"])
    f["e3.spread_max"] = R._fmt_size(spread[-1]["payload_bytes"])
    f["e3.spread_base_miss"] = R._fmt_size(max(r["payload_bytes"] for r in spread if not r["base_detected"]))
    f["e3.spread_base_from"] = R._fmt_size(min(r["payload_bytes"] for r in spread if r["base_detected"]))

    _, e1c = R._latest_run(rows, "E1c")
    models = {r["repo"]: r for r in e1c}
    f["e1c.models"] = str(len(models))
    f["e1c.origin"] = f"{sum(r['s3_origin'] == 'bfloat16' for r in models.values())} / {len(models)}"
    f["e1c.fp"] = str(sum(1 for r in models.values() if r["false_positives"]))
    kb = {k: _cell([r for r in e1c if r["payload"] == "1KB" and r["k"] == k], "detected") for k in (1, 4, 8)}
    asserts["E1c: 1 KB result equal at k = 1, 4, 8"] = len(set(kb.values())) == 1
    f["e1c.kb"] = kb[1]
    f["e1c.kb_size"] = R._fmt_size([r for r in e1c if r["payload"] == "1KB"][0]["payload_bytes"])

    _, val = R._latest_run(rows, "E4val")
    honest = [r for r in val if r["kind"] == "honest"]
    pay = [r for r in val if r["kind"] == "payload"]
    ks = sorted({r["k"] for r in pay})
    caught = [k for k in ks if all(r["detected_on_base"] for r in pay if r["k"] == k)]
    f["e4.pairs"] = str(len(honest))
    f["e4.fp"] = str(sum(r["s5_false_positives"] for r in honest))
    f["e4.caught_ks"] = ", ".join(map(str, caught[:-1])) + f" and {caught[-1]}"
    f["e4.caught"] = _cell([r for r in pay if r["k"] == caught[0]], "detected_on_base")
    f["e4.trained"] = _cell([r for r in pay if r["k"] == caught[0]], "detected_in_trained")
    asserts["E4: trained-tensor result equal at every caught k"] = len(
        {_cell([r for r in pay if r["k"] == k], "detected_in_trained") for k in caught}) == 1

    _, c3 = R._latest_run([r for r in rows if r.get("meta", {}).get("list") == "hf_models_batch3.json"], "E0c")
    for dt in ("BF16", "F16"):
        rs = [r for r in c3 if r["dtype"] == dt]
        f[f"e0c.{dt.lower()}"] = (f"{sum(r['old_fp'] > 0 for r in rs)} → {sum(r['new_fp'] > 0 for r in rs)}"
                                  f" of {len(rs)}")

    s = R.e5_summary(rows)
    if s:
        g = s["groups"]
        f["e5.code"] = str(g["code"]["files"])
        f["e5.picklescan_code"] = f"{g['code']['picklescan']['flagged']} / {g['code']['files']}"
        f["e5.stowaway_code"] = f"{g['code']['stowaway']['flagged']} / {g['code']['files']}"
        f["e5.weight_pass_picklescan"] = (f"{g['weight']['files'] - g['weight']['picklescan']['flagged']}"
                                          f" / {g['weight']['files']}")
        c = s["conversion"]
        f["e5.weight_survived"] = f"{c['weight_survived']} / {c['weight_files']}"
        asserts["E5: H1 supported"] = (g["code"]["picklescan"]["flagged"] == g["code"]["stowaway"]["flagged"]
                                       == g["code"]["files"] and g["weight"]["picklescan"]["flagged"] == 0
                                       and c["weight_survived"] == c["weight_files"])

    if weights:
        f["e0.mobilenet_bit0"] = _mobilenet_bit0()
    if counts:
        f.update(_counts())
    return f, asserts


# ---------------------------------------------------------------- page

VOID = {"br", "img", "input", "meta", "link", "hr", "source", "wbr", "area", "col", "path", "use", "rect",
        "circle", "line", "polyline", "polygon", "stop"}


class _Page(HTMLParser):
    """Collects data-fact spans and the visible text outside them."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []        # (tag, kind) with kind in {None, "fact", "skip"}
        self.facts = []        # (key, text)
        self.loose = []        # visible text not inside a fact or a skipped element

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        a = dict(attrs)
        kind = ("skip" if tag in ("script", "style", "title", "head") or "data-nofacts" in a
                else "fact" if "data-fact" in a else None)
        if kind == "fact":
            self.facts.append([a["data-fact"], ""])
        self.stack.append((tag, kind))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        while self.stack:
            t, _ = self.stack.pop()
            if t == tag:
                break

    def handle_data(self, data):
        kinds = [k for _, k in self.stack]
        if "skip" in kinds:
            return
        if "fact" in kinds:
            self.facts[-1][1] += data
        else:
            self.loose.append(data)


def stray_numbers(loose_text):
    text = re.sub(r"\s+", " ", " ".join(loose_text))
    text = _ALLOWED_RE.sub(" ", text)
    return sorted(set(re.findall(r"[^\s]*\d[^\s]*", text)))


def e1_strips(rows, k=8):
    """{model: (single-file cells, base cells)} as hit/miss, smallest size first."""
    out = {}
    for model in ("resnet18", "mobilenet_v3_small"):
        _, recs = R._latest_run([r for r in rows if r.get("model") == model], "E1")
        rs = sorted((r for r in recs if r["k"] == k and r["spread"] == 1), key=lambda r: r["payload_bytes"])
        out[model] = (["hit" if r["single_file_detected"] else "miss" for r in rs],
                      ["hit" if r["base_detected"] else "miss" for r in rs])
    return out


def check_strips(html, strips):
    problems = []
    for model, (single, base) in strips.items():
        m = re.search(rf'data-e1="{model}">(.*?)</div>', html, flags=re.S)
        if not m:
            problems.append(f"E1 strip for {model} missing from the page")
            continue
        lanes = re.findall(r'<span class="lane">[^<]*</span>((?:<i class="(?:hit|miss)"></i>)+)', m.group(1))
        cells = [re.findall(r'class="(hit|miss)"', lane) for lane in lanes]
        if cells != [single, base]:
            problems.append(f"E1 strip for {model}: page {cells}, ledger {[single, base]}")
    return problems


def check(html, f, asserts):
    p = _Page()
    p.feed(html)
    problems = []
    for key, text in p.facts:
        if key not in f:
            problems.append(f"unknown fact key {key!r} (page says {text!r})")
        elif text.strip() != f[key]:
            problems.append(f"{key}: page says {text.strip()!r}, ledger says {f[key]!r}")
    for n in stray_numbers(p.loose):
        problems.append(f"number not backed by a fact: {n!r}")
    for claim, ok in asserts.items():
        if not ok:
            problems.append(f"claim no longer true: {claim}")
    return problems


def write(html, f):
    def sub(m):
        key = m.group(2)
        return m.group(1) + (f[key] if key in f else m.group(3)) + m.group(4)
    return re.sub(r'(<span[^>]*data-fact="([^"]+)"[^>]*>)(.*?)(</span>)', sub, html, flags=re.S)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-counts", action="store_true", help="skip collecting the test suites")
    args = ap.parse_args(argv)
    f, asserts = facts(counts=not args.no_counts)
    html = PAGE.read_text(encoding="utf-8")
    if args.write:
        html = write(html, f)
        PAGE.write_text(html, encoding="utf-8")
    problems = check(html, f, asserts) + check_strips(html, e1_strips(load()))
    if args.no_counts:
        problems = [x for x in problems if not x.startswith("unknown fact key 'tests.")]
    for x in problems:
        print(x)
    print(f"{len(f)} facts, {len(asserts)} claims, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
