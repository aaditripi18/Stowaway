"""stowaway scan FILE [FILE ...] [--json]"""
from __future__ import annotations

import argparse
import json
import sys

from .report import scan

_MARK = {"info": " ", "low": ".", "medium": "?", "high": "!", "critical": "X"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stowaway", description="Find what model scanners miss.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="scan model files")
    sc.add_argument("files", nargs="+")
    sc.add_argument("--base", metavar="MODEL",
                    help="run the S5 differential against this claimed base model")
    sc.add_argument("--json", action="store_true", help="print the full JSON report")
    args = ap.parse_args(argv)

    status = 0
    reports = []
    for path in args.files:
        rep = scan(path, base=args.base)
        reports.append(rep)
        if rep["verdict"] == "dangerous":
            status = 2
        elif rep["verdict"] == "suspicious":
            status = max(status, 1)
        if args.json:
            continue
        t = rep["tensors"]
        print(f"{rep['file']}: {rep['verdict'].upper()}  [{rep['format']}, {t['count']} tensors, {rep['size']:,} bytes]")
        for x in rep["findings"]:
            print(f"  {_MARK[x['severity']]} {x['severity']:<8} {x['check']:<5} {x['message']}")
    if args.json:
        json.dump(reports if len(reports) > 1 else reports[0], sys.stdout, indent=2)
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
