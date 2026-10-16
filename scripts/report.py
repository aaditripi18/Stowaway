"""Generate docs/RESULTS.md from the ledger. Never edit RESULTS.md by hand.

Only the most recent run of each experiment is tabled, so reruns supersede old
numbers. Every table says which commit produced it.
"""
from __future__ import annotations

from pathlib import Path

from experiments.ledger import load

OUT = Path(__file__).resolve().parent.parent / "docs" / "RESULTS.md"


def _latest_run(rows, experiment):
    runs = [r for r in rows if r["experiment"] == experiment]
    if not runs:
        return None, []
    last = max(r["run_id"] for r in runs)
    return last, [r for r in runs if r["run_id"] == last]


def _latest_runs_by_model(rows, experiment):
    """{model: (run_id, records)} using each model's most recent run."""
    out = {}
    for model in sorted({r["model"] for r in rows if r["experiment"] == experiment}):
        _, recs = _latest_run([r for r in rows if r.get("model") == model], experiment)
        out[model] = (recs[0]["run_id"], recs)
    return out


def _fmt_size(n):
    for unit, div in (("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:g} {unit}"
    return f"{n} B"


def section_e0(rows):
    run_id, recs = _latest_run(rows, "E0")
    if not recs:
        return "## E0 — false positives on clean models\n\n_No run yet._\n"
    lines = ["## E0 — false positives on clean models", "",
             f"Run `{run_id}`. Single-file checks (S3+S4) on clean ImageNet models; "
             "0 means no false alarm on an unmodified model. It does not mean a payload would "
             "be seen: that is E1.", "",
             "`dev` models were looked at while the checks were designed; `held out` models were not.", "",
             "| Model | Split | Float tensors | Verdict | False positives |", "|---|---|--:|---|--:|"]
    for r in sorted(recs, key=lambda r: (r.get("split", "dev") != "dev", r["model"])):
        split = "held out" if r.get("split") == "held_out" else "dev"
        lines.append(f"| {r['model']} | {split} | {r['n_float_tensors']} | {r['verdict']} | {r['false_positives']} |")
    return "\n".join(lines) + "\n"


def section_e1(rows):
    runs = _latest_runs_by_model(rows, "E1")
    if not runs:
        return "## E1 — detection vs payload size\n\n_No run yet._\n"
    out = ["## E1 — detection vs payload size", "",
           "Random (encrypted-style) payloads in the low k bits. `single` = seen by S3/S4 from "
           "the file alone; `base` = seen by S5 given the clean model as the claimed base.", ""]
    for model, (run_id, recs) in runs.items():
        ks = sorted({r["k"] for r in recs})
        sizes = sorted({r["payload_bytes"] for r in recs})
        out += [f"### {model}", "", f"Run `{run_id}`.", "",
                "| Payload | " + " | ".join(f"k={k} single | k={k} base" for k in ks) + " |",
                "|---|" + "---|---|" * len(ks)]
        for size in sizes:
            cells = []
            for k in ks:
                m = [r for r in recs if r["k"] == k and r["payload_bytes"] == size]
                if not m:
                    cells += ["–", "–"]
                    continue
                cells.append("detected" if m[0]["single_file_detected"] else "**missed**")
                cells.append("detected" if m[0]["base_detected"] else "**missed**")
            out.append(f"| {_fmt_size(size)} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def section_e2(rows):
    runs = _latest_runs_by_model(rows, "E2")
    if not runs:
        return "## E2 — accuracy cost of the payload\n\n_No run yet._\n"
    out = ["## E2 — accuracy cost of the payload", ""]
    for model, (run_id, recs) in runs.items():
        base = [r for r in recs if r["payload_bytes"] == 0]
        n = recs[0].get("n_images")
        out += [f"### {model}", "", f"Run `{run_id}`, top-1 on {n} Imagenette images."
                + (f" Clean accuracy **{base[0]['accuracy']:.4f}**." if base else ""), "",
                "| low bits k | Payload | Accuracy | Drop vs clean |", "|--:|---|--:|--:|"]
        for r in sorted((r for r in recs if r["payload_bytes"]), key=lambda r: (r["k"], r["payload_bytes"])):
            out.append(f"| {r['k']} | {_fmt_size(r['payload_bytes'])} | {r['accuracy']:.4f} | {r['accuracy_drop']:+.4f} |")
        out.append("")
    return "\n".join(out)


GROUP_ORDER = ["F32 file, bfloat16 precision", "F32 file, float16 precision",
               "F32, low-bit signature", "F32, fair-coin low bits", "BF16 file", "F16 file"]


def _ordered(groups):
    return sorted(groups, key=lambda g: (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, g))


def section_hf(rows):
    lists = sorted({r.get("meta", {}).get("list", "hf_models.json") for r in rows if r["experiment"] == "E0b"})
    if not lists:
        return "## E0b / E1b — Hugging Face survey\n\n_No run yet._\n"
    return "\n".join(_section_hf_list(rows, name) for name in lists)


def _section_hf_list(rows, list_name):
    mine = [r for r in rows if r.get("meta", {}).get("list", "hf_models.json") == list_name]
    run0, survey = _latest_run(mine, "E0b")
    run1, attack = _latest_run(mine, "E1b")
    meta = survey[0].get("meta", {})
    how = (f"{meta['roles']} roles x 4 layers (role peers for S4), up to {meta['max_tensors']} tensors, "
           f"first {meta['chunk'] // 1024} KB of each" if meta.get("roles")
           else "up to 12 tensors spread through the file, first 64 KB of each")
    ok = [r for r in survey if "error" not in r]
    errors = [r for r in survey if "error" in r]
    out = [f"## E0b / E1b — single-file checks on public Hugging Face models (`{list_name}`)", "",
           f"Runs `{run0}` and `{run1}`. Sampling: {how}, fetched with HTTP range requests; "
           f"the list is frozen in `results/{list_name}`. "
           "Groups come from what S3/S4 measure on the clean samples.", "",
           "### E0b — what the weights look like, and false positives", "",
           "| Group | Models | Models with a false positive | Flagged tensors / sampled |",
           "|---|--:|--:|--:|"]
    groups = _ordered({r["group"] for r in ok})
    for g in groups:
        rs = [r for r in ok if r["group"] == g]
        fpm = sum(1 for r in rs if r["false_positives"])
        fpt = sum(len(r["fp_tensors"]) for r in rs)
        out.append(f"| {g} | {len(rs)} | {fpm} | {fpt} / {sum(r['n_sampled'] for r in rs)} |")
    out += ["", f"{len(ok)} models sampled, {len(errors)} could not be."]
    if errors:
        out += [""] + [f"- `{r['repo']}`: {r['error']}" for r in sorted(errors, key=lambda r: r["repo"])]

    if attack:
        out += ["", "### E1b — would the file alone expose a random low-bit payload?", "",
                "Random bytes fill the low k bits of the sample of the largest tensor the clean "
                "scan did not flag. Cells: models where S3 or S4 flagged that tensor / models tested.", "",
                "| Group | k=1 | k=4 | k=8 |", "|---|--:|--:|--:|"]
        valid = [r for r in attack if not r["target_flagged_clean"]]
        for g in _ordered({r["group"] for r in valid}):
            cells = []
            for k in (1, 4, 8):
                rs = [r for r in valid if r["group"] == g and r["k"] == k]
                cells.append(f"{sum(r['detected'] for r in rs)} / {len(rs)}" if rs else "–")
            out.append(f"| {g} | " + " | ".join(cells) + " |")
        skipped = len({r["repo"] for r in attack if r["target_flagged_clean"]})
        if skipped:
            out += ["", f"{skipped} model(s) excluded: every sampled tensor was flagged on the clean scan."]

    out += ["", "### Per model", "",
            "| Model | Stored | Group | Signature (bits 0–3) | False positives | Payload seen at k=1 / 4 / 8 |",
            "|---|---|---|---|--:|---|"]
    by_repo = {}
    for r in attack:
        by_repo.setdefault(r["repo"], {})[r["k"]] = r
    for r in sorted(ok, key=lambda r: (_ordered([r["group"]]) and GROUP_ORDER.index(r["group"])
                                         if r["group"] in GROUP_ORDER else 99, r["repo"])):
        sig = ", ".join(f"{x:.2f}" for x in r["signature"][:4]) if r.get("signature") else "–"
        hits = by_repo.get(r["repo"], {})
        seen = " / ".join(("yes" if hits[k]["detected"] else "no") if k in hits else "–" for k in (1, 4, 8))
        out.append(f"| `{r['repo']}` | {r['dtype']} | {r['group']} | {sig} | {r['false_positives']} | {seen} |")
    return "\n".join(out) + "\n"


def section_e0c(rows):
    out = ["## E0c — week-4 S4 rule against the current one, same samples", ""]
    lists = sorted({r["meta"].get("list") for r in rows if r["experiment"] == "E0c"})
    if not lists:
        return "\n".join(out + ["_No run yet._", ""])
    for name in lists:
        run_id, recs = _latest_run([r for r in rows if r.get("meta", {}).get("list") == name], "E0c")
        out += [f"`{name}`, run `{run_id}`. Models with at least one S4 false positive on the clean samples:", "",
                "| Stored as | Models | Week-4 rule | Current rule |", "|---|--:|--:|--:|"]
        for dt in ("F32", "BF16", "F16"):
            rs = [r for r in recs if r["dtype"] == dt]
            if rs:
                out.append(f"| {dt} | {len(rs)} | {sum(r['old_fp'] > 0 for r in rs)} | {sum(r['new_fp'] > 0 for r in rs)} |")
        out.append("")
    return "\n".join(out)


def section_e1c(rows):
    run_id, recs = _latest_run(rows, "E1c")
    if not recs:
        return "## E1c — S3 on bf16 weights saved as float32\n\n_No run yet._\n"
    models = {r["repo"]: r for r in recs}
    origin_ok = sum(1 for r in models.values() if r["s3_origin"] == "bfloat16")
    fp = sum(1 for r in models.values() if r["false_positives"])
    out = ["## E1c — S3 on bf16 weights saved as float32", "",
           f"Run `{run_id}`. The survey's BF16 samples widened to float32 exactly as `model.float()` "
           "does, then a random payload in the low k bits of one tensor.", "",
           f"- S3 identified bfloat16 precision in **{origin_ok} / {len(models)}** models.",
           f"- Models with a false positive on the clean widened weights: **{fp} / {len(models)}**.", "",
           "| Payload | k=1 | k=4 | k=8 |", "|---|--:|--:|--:|"]
    for label in ("1KB", "full"):
        cells = []
        for k in (1, 4, 8):
            rs = [r for r in recs if r["payload"] == label and r["k"] == k]
            cells.append(f"{sum(r['detected'] for r in rs)} / {len(rs)}")
        name = "1 KB" if label == "1KB" else "whole sampled tensor"
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def section_e3(rows):
    runs = _latest_runs_by_model(rows, "E3")
    if not runs:
        return "## E3 - robustness of the single-file checks\n\n_No run yet._\n"
    out = ["## E3 - robustness of the single-file checks", "",
           "Random payloads placed where S3/S4 are weakest. `single` = S3 or S4 flagged a "
           "written tensor; `base` = S5 (given the clean base) did. Each cell is the smallest "
           "payload at which the placement is **caught**, or — (missed, up to the largest size tested) if "
           "it is missed at every size tested. A placement is tested only at sizes that fit it: at one "
           "weight in 64 a model holds much less."]
    for model, (run_id, recs) in runs.items():
        out += ["", f"### {model}", "", f"Run `{run_id}`, k = {recs[0]['k']}.", "",
                "| Placement | single-file caught from | with base caught from |", "|---|---|---|"]
        for pl in sorted({r["placement"] for r in recs}):
            rs = sorted((r for r in recs if r["placement"] == pl), key=lambda r: r["payload_bytes"])
            def first(key):
                hit = [r for r in rs if r[key]]
                return (_fmt_size(min(r["payload_bytes"] for r in hit)) if hit
                        else f"— (missed, up to {_fmt_size(max(r['payload_bytes'] for r in rs))})")
            out.append(f"| {pl} | {first('single_file_detected')} | {first('base_detected')} |")
    return "\n".join(out) + "\n"


def section_e4(rows):
    _, dev = _latest_run([r for r in rows if r.get("meta", {}).get("split") == "dev"], "E4")
    dev_t = [r for r in dev if "tensor" in r]
    run_id, val = _latest_run(rows, "E4val")
    out = ["## E4 — S5 calibrated on real fine-tune/base pairs", ""]
    if dev_t:
        import statistics
        p99 = [r["hi_bit_p99"] for r in dev_t]
        mant = dev_t[0]["mantissa_bits"]
        out += [f"On {len(dev_t)} changed tensors across {len({r['finetune'] for r in dev_t})} dev fine-tunes, "
                f"the 99th-percentile highest changed mantissa bit is "
                f"{min(p99)}–{max(p99)} (median {int(statistics.median(p99))}) of {mant}: honest "
                f"fine-tuning reaches the top of the mantissa. S5 flags a tensor whose changes reach "
                f"only below bit {mant - 3}.", ""]
    if val:
        honest = [r for r in val if r["kind"] == "honest"]
        pay = [r for r in val if r["kind"] == "payload"]
        out += [f"Held-out validation (`{run_id}`): {len(honest)} honest pairs, "
                f"**{sum(r['s5_false_positives'] for r in honest)}** S5 false positives.", ""]
        if pay:
            ks = sorted({r["k"] for r in pay})
            out += ["Payload detection (caught / tested):", "",
                    "| Placement | " + " | ".join(f"k={k}" for k in ks) + " |",
                    "|---|" + "--:|" * len(ks)]
            for kind, label in (("detected_on_base", "tensor matching base"),
                                ("detected_in_trained", "tensor the fine-tune trained")):
                cells = [f"{sum(r[kind] for r in pay if r['k'] == k)} / {sum(1 for r in pay if r['k'] == k)}"
                         for k in ks]
                out.append(f"| {label} | " + " | ".join(cells) + " |")
            out += ["", "A payload in a tensor that should match the base is caught up to the cutoff; "
                    "one hidden inside a genuinely trained tensor is camouflaged by the training changes."]
    return "\n".join(out) + "\n"


E4B_T = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)


def s5c_threshold(n: int) -> float:
    """The chosen S5c threshold (PLAN.md, E4b): 3 binomial SDs below 50%."""
    return 0.5 - 1.5 / n ** 0.5


def e4b_rules(T):
    """S5 as calibrated in E4, and the two refinements PLAN.md phase 2 compares."""
    return {
        "S5": lambda r: r["p99"] <= r["cutoff"],
        "S5c (T(n), chosen)": lambda r: r["low_share"] >= s5c_threshold(r["changed"]),
        f"S5b (T={T})": lambda r: r["p99"] <= r["cutoff"] and r["top_share"] >= T,
        f"S5c (T={T})": lambda r: r["low_share"] >= T,
    }


def e4b_summary(rows, split):
    run_id, recs = _latest_run([r for r in rows if r.get("meta", {}).get("split") == split], "E4b")
    if not recs:
        return None
    honest = [r for r in recs if r["kind"] == "honest"]
    pay = [r for r in recs if r["kind"] == "payload"]
    pairs = [r for r in recs if r["kind"] == "pair"]
    out = {"run_id": run_id, "pairs": len(pairs), "errors": [r["finetune"] for r in recs if r["kind"] == "error"],
           "honest_tensors": len(honest), "rules": {}}
    for T in E4B_T:
        for name, rule in e4b_rules(T).items():
            if name in out["rules"]:
                continue
            fp = [r for r in honest if rule(r)]
            cases = {}
            for case in ("a_base_matching", "b_low_gradient"):
                for k in sorted({r["k"] for r in pay}):
                    rs = [r for r in pay if r["case"] == case and r["k"] == k]
                    cases[f"{case}|{k}"] = (sum(1 for r in rs if rule(r)), len(rs))
            out["rules"][name] = {"fp_tensors": len(fp), "fp_models": len({r["finetune"] for r in fp}),
                                  "fp_examples": sorted({r["tensor"] for r in fp})[:6], "caught": cases}
    return out


def section_e4b(rows):
    out = ["## E4b — S5 on whole files (PLAN.md, phase 2)", ""]
    for split in ("dev", "fresh"):
        s = e4b_summary(rows, split)
        if not s:
            continue
        ks = sorted({int(c.split("|")[1]) for c in next(iter(s["rules"].values()))["caught"]})
        out += [f"### {split}", "", f"Run `{s['run_id']}`: {s['pairs']} pairs, {s['honest_tensors']} changed shared tensors"
                + (f"; could not fetch {', '.join(s['errors'])}" if s["errors"] else "") + ".", "",
                "| Rule | Honest tensors flagged | Honest models flagged | " +
                " | ".join(f"(a) k={k}" for k in ks) + " | " + " | ".join(f"(b) k={k}" for k in ks) + " |",
                "|---|--:|--:|" + "--:|" * (2 * len(ks))]
        for name, v in s["rules"].items():
            cells = [f"{a} / {n}" for a, n in (v["caught"][f"a_base_matching|{k}"] for k in ks)]
            cells += [f"{a} / {n}" for a, n in (v["caught"][f"b_low_gradient|{k}"] for k in ks)]
            out.append(f"| {name} | {v['fp_tensors']} | {v['fp_models']} / {s['pairs']} | " + " | ".join(cells) + " |")
        out += ["", "(a) payload in the largest tensor the fine-tune left identical to the base; "
                "(b) payload in the changed tensor with the smallest p99 bit (the low-gradient case).", ""]
    return "\n".join(out) + "\n"


# The S5 rule E6's base detection is scored with. E4b (PLAN.md, phase 2) decides
# whether this stays the E4 rule or becomes a refinement; change it here only by
# that decision.
E6_S5 = lambda r: r.get("changed", 0) >= 256 and r["p99"] <= r["cutoff"]  # noqa: E731


def e6_records(rows):
    """The latest E6 run of each model."""
    return {m: recs for m, (_, recs) in _latest_runs_by_model(rows, "E6").items()}


def e6_hypotheses(rows):
    by_model = e6_records(rows)
    if not by_model:
        return []
    out = []
    worst = {}
    for m, recs in by_model.items():
        cells = [r for r in recs if r.get("kind") == "cell" and r["payload"] == "random"
                 and r["payload_bytes"] <= 1 << 20 and r["k"] <= 8]
        if cells:
            worst[m] = min(r["agreement"] for r in cells)
    ok5 = all(v >= 0.99 for v in worst.values()) if worst else None
    out.append({"id": "H5", "statement": "A random payload of up to 1 MB at k <= 8 leaves >= 99% of top-1 predictions "
                "unchanged, on every model and seed.", "verdict": "pending" if ok5 is None else
                ("supported" if ok5 else "not supported"),
                "evidence": ", ".join(f"{m} {100 * v:.1f}%" for m, v in sorted(worst.items(), key=lambda x: x[1]))})

    def caught(kind, key):
        n = c = 0
        for recs in by_model.values():
            for r in recs:
                if r.get("kind") == "cell" and r["payload"] == kind and r["seed"] == 0:
                    n += 1
                    if key == "single":
                        c += r["single_file_detected"]
                    else:
                        c += any(E6_S5(x) for x in r["s5_stats"].values())
        return c, n
    single = {k: caught(k, "single") for k in ("random", "zeros", "text")}
    base = {k: caught(k, "base") for k in ("random", "zeros", "text")}
    ok6 = (single["zeros"][0] < single["random"][0] and single["text"][0] < single["random"][0]
           and base["zeros"][0] >= base["random"][0] and base["text"][0] >= base["random"][0]) if single["random"][1] else None
    out.append({"id": "H6", "statement": "S4's one-sided rule costs what it says: zero and text payloads are caught "
                "from the file alone less often than random ones, and the base diff catches them at least as often.",
                "verdict": "pending" if ok6 is None else ("supported" if ok6 else "not supported"),
                "evidence": "file alone " + ", ".join(f"{k} {c}/{n}" for k, (c, n) in single.items())
                + "; with base " + ", ".join(f"{k} {c}/{n}" for k, (c, n) in base.items())})
    return out


def section_e6(rows):
    by_model = e6_records(rows)
    if not by_model:
        return "## E6 — nine models, five payload types\n\n_No run yet._\n"
    out = ["## E6 — nine models, five payload types (PLAN.md, phase 2)", "",
           "Seed 0 for every payload type; random payloads also at seeds 1-2. `file` = S3/S4 from the planted "
           "file alone; `base` = S5 with the clean model as base; `agree` = share of 1,000 Imagenette images whose "
           "top-1 prediction is unchanged (minimum over the seeds).", ""]
    kinds = ("random", "zeros", "text", "compressed", "executable")
    for m in sorted(by_model, key=lambda m: (m not in ("resnet18", "mobilenet_v3_small"), m)):
        recs = by_model[m]
        clean = next((r for r in recs if r.get("kind") == "clean"), None)
        out += [f"### {m}" + (f" (clean top-1 {clean['accuracy']:.3f})" if clean else ""), "",
                "| Payload | k | " + " | ".join(_fmt_size(s) for s in sorted({r['payload_bytes'] for r in recs if r.get('kind') == 'cell'})) + " |",
                "|---|--:|" + "---|" * len({r['payload_bytes'] for r in recs if r.get('kind') == 'cell'})]
        sizes = sorted({r["payload_bytes"] for r in recs if r.get("kind") == "cell"})
        for kind in kinds:
            for k in sorted({r["k"] for r in recs if r.get("kind") == "cell"}):
                cells = []
                for size in sizes:
                    rs = [r for r in recs if r.get("kind") == "cell" and r["payload"] == kind and r["k"] == k
                          and r["payload_bytes"] == size]
                    if not rs:
                        cells.append("–")
                        continue
                    f = sum(r["single_file_detected"] for r in rs)
                    b = sum(any(E6_S5(x) for x in r["s5_stats"].values()) for r in rs)
                    agree = min(r["agreement"] for r in rs)
                    cells.append(f"file {f}/{len(rs)}, base {b}/{len(rs)}, agree {100 * agree:.1f}%")
                out.append(f"| {kind} | {k} | " + " | ".join(cells) + " |")
        out.append("")
    hs = e6_hypotheses(rows)
    out += ["### Verdicts", "", "| | Hypothesis | Verdict | Evidence |", "|---|---|---|---|"]
    out += [f"| {h['id']} | {h['statement']} | **{h['verdict']}** | {h['evidence']} |" for h in hs]
    return "\n".join(out) + "\n"


E5_TOOLS = ("picklescan", "modelscan", "fickling", "fickling-ml", "stowaway", "stowaway+base")


def e5_summary(rows):
    """Counts the E5 section and the site both read: per group and tool, how many
    files were flagged out of how many the tool scanned."""
    run_id, recs = _latest_run(rows, "E5")
    if not recs:
        return None
    verdicts = [r for r in recs if r["tool"] != "conversion"]
    out = {"run_id": run_id, "commit": recs[0]["commit"], "versions": recs[0]["meta"]["versions"],
           "groups": {}}
    for group in ("clean", "code", "weight", "structure"):
        files = sorted({r["file"] for r in verdicts if r["group"] == group})
        g = {"files": len(files)}
        for tool in E5_TOOLS:
            rs = [r for r in verdicts if r["group"] == group and r["tool"] == tool]
            if rs:
                g[tool] = {"flagged": sum(r["verdict"] == "flagged" for r in rs),
                           "scanned": sum(r["verdict"] not in ("not scanned",) for r in rs),
                           "errors": sum(r["verdict"] == "error" for r in rs)}
        out["groups"][group] = g
    conv = [r for r in recs if r["tool"] == "conversion"]
    out["conversion"] = {
        "code_refused": sum(r["outcome"] == "load refused" for r in conv if r["group"] == "code"),
        "code_files": sum(1 for r in conv if r["group"] == "code"),
        "weight_survived": sum(bool(r.get("weight_payload_survives")) for r in conv if r["group"] == "weight"),
        "weight_files": sum(1 for r in conv if r["group"] == "weight"),
    }
    return out


def section_e5(rows):
    run_id, recs = _latest_run(rows, "E5")
    if not recs:
        return "## E5 — the same files through existing scanners\n\n_No run yet._\n"
    v = recs[0]["meta"]["versions"]
    verdicts = [r for r in recs if r["tool"] != "conversion"]
    cell = {(r["file"], r["tool"]): r["verdict"] for r in verdicts}
    short = {"flagged": "**flag**", "clean": "clean", "not scanned": "—", "error": "error"}
    out = ["## E5 — the same files through existing scanners", "",
           f"Run `{run_id}`. picklescan {v['picklescan']}, ModelScan {v['modelscan']}, fickling "
           f"{v['fickling']} (`fickling` = its default check, as the CLI runs it; `fickling-ml` = its "
           f"ML allowlist analysis), torch {v['torch']}, safetensors {v['safetensors']}. "
           "`—` = the tool did not scan the file (unsupported format). `stowaway+base` = with the "
           "clean model as claimed base (S5).", "",
           "| File | Group | " + " | ".join(E5_TOOLS) + " |", "|---|---|" + "---|" * len(E5_TOOLS)]
    order = {"clean": 0, "code": 1, "weight": 2, "structure": 3}
    files = sorted({(r["group"], r["file"]) for r in verdicts}, key=lambda x: (order[x[0]], x[1]))
    for group, f in files:
        out.append(f"| `{f}` | {group} | " + " | ".join(short.get(cell.get((f, t)), "") for t in E5_TOOLS) + " |")
    s = e5_summary(rows)
    out += ["", "Flagged / scanned, per group:", "",
            "| Group | Files | " + " | ".join(E5_TOOLS) + " |", "|---|--:|" + "--:|" * len(E5_TOOLS)]
    for group, g in s["groups"].items():
        cells = [f"{g[t]['flagged']} / {g[t]['scanned']}" if t in g else "" for t in E5_TOOLS]
        out.append(f"| {group} | {g['files']} | " + " | ".join(cells) + " |")
    c = s["conversion"]
    out += ["", "Conversion to safetensors (`torch.load(weights_only=True)` + `safetensors.torch.save_file`) "
            "of each `.pt` file:", "",
            f"- code payloads: load refused for **{c['code_refused']} / {c['code_files']}** (nothing executed, "
            "nothing converted)",
            f"- weight payloads: carried across bit for bit and still extractable in "
            f"**{c['weight_survived']} / {c['weight_files']}**"]
    conv = {r["file"]: r for r in recs if r["tool"] == "conversion"}
    for f, r in sorted(conv.items()):
        if r["group"] in ("structure",):
            out.append(f"- `{f}`: {r['outcome']}; Stowaway after conversion: {r.get('stowaway_after')}")
    return "\n".join(out) + "\n"


def main():
    rows = load()
    body = "\n".join([
        "# Results", "",
        "> Generated by `scripts/report.py` from `results/ledger.jsonl`. Do not edit by hand.", "",
        section_e0(rows), "", section_e1(rows), "", section_e2(rows), "",
        section_hf(rows), "", section_e0c(rows), "", section_e1c(rows), "", section_e3(rows), "", section_e4(rows), "", section_e4b(rows), "", section_e5(rows), "", section_e6(rows),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} ledger rows)")


if __name__ == "__main__":
    main()
