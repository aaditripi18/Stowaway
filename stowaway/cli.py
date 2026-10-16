"""stowaway scan FILE|hf:org/name[@rev] ... [--base FILE|hf:org/name[@rev]|declared] [--json]

A target or base written hf:org/name is fetched from the Hugging Face Hub
(pinned to a commit, checked against the Hub's SHA-256; see hub.py). With
--base declared, each hf: target is compared with the base_model its own model
card declares.

Exit status: 0 clean, 1 suspicious, 2 dangerous, 3 could not run.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import hub
from .report import scan

_MARK = {"info": " ", "low": ".", "medium": "?", "high": "!", "critical": "X"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _fetch(spec: str):
    res = hub.resolve(spec)
    return res, hub.download(res, log=_log)


def _targets(specs: list[str]):
    """(display name, local path, source) for every file to scan."""
    for spec in specs:
        if hub.is_spec(spec):
            res, paths = _fetch(spec)
            for name, p in zip(res.files, paths):
                yield f"{res.repo}@{res.commit[:10]}/{name}", p, res
        else:
            yield spec, spec, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stowaway", description="Find what model scanners miss.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="scan model files")
    sc.add_argument("files", nargs="+", help="model files, or hf:org/name[@revision]")
    sc.add_argument("--base", metavar="MODEL",
                    help="run the S5 differential against this claimed base model: a file, "
                         "hf:org/name[@revision], or 'declared' (the base_model an hf: target's card names)")
    sc.add_argument("--json", action="store_true", help="print the full JSON report")
    args = ap.parse_args(argv)

    try:
        fixed_base, fixed_source = None, None
        if args.base and args.base != "declared":
            if hub.is_spec(args.base):
                fixed_source, fixed_base = _fetch(args.base)
            else:
                fixed_base = args.base
        declared_cache: dict[str, tuple] = {}
        status, reports = 0, []
        for display, path, source in _targets(args.files):
            base, base_source = fixed_base, fixed_source
            if args.base == "declared":
                if source is None:
                    raise hub.HubError(f"--base declared needs an hf: target; {display} is a local file")
                if not source.base_model:
                    raise hub.HubError(f"{source.repo} declares no base_model in its model card")
                if len(source.base_model) > 1:
                    _log(f"{source.repo} declares {len(source.base_model)} base models; using {source.base_model[0]}")
                spec = "hf:" + source.base_model[0]
                if spec not in declared_cache:
                    declared_cache[spec] = _fetch(spec)
                base_source, base = declared_cache[spec]
            rep = scan(path, base=base)
            if source is not None:
                rep["source"] = {"hub": source.repo, "revision": source.revision, "commit": source.commit,
                                 "file": display.split("/", 2)[-1], "declared_base_model": source.base_model}
            if base_source is not None:
                rep["base"] = {"hub": base_source.repo, "revision": base_source.revision,
                               "commit": base_source.commit, "files": base_source.files}
            elif base is not None:
                rep["base"] = {"file": str(base)}
            reports.append(rep)
            if rep["verdict"] == "dangerous":
                status = 2
            elif rep["verdict"] == "suspicious":
                status = max(status, 1)
            if args.json:
                continue
            t = rep["tensors"]
            print(f"{display}: {rep['verdict'].upper()}  [{rep['format']}, {t['count']} tensors, {rep['size']:,} bytes]")
            if "base" in rep:
                b = rep["base"]
                print(f"  base: {b['hub'] + '@' + b['commit'][:10] if 'hub' in b else b['file']}")
            for x in rep["findings"]:
                print(f"  {_MARK[x['severity']]} {x['severity']:<8} {x['check']:<5} {x['message']}")
    except (hub.HubError, OSError) as e:
        _log(f"stowaway: {e}")
        return 3
    if args.json:
        json.dump(reports if len(reports) > 1 else reports[0], sys.stdout, indent=2)
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
