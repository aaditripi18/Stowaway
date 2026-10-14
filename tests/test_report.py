import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import ledger  # noqa: E402
from scripts import report  # noqa: E402


def test_report_tables_from_ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER", path)
    monkeypatch.setattr(report, "OUT", tmp_path / "RESULTS.md")
    rows = [
        {"experiment": "E0", "run_id": "E0-a", "model": "m", "n_float_tensors": 3,
         "verdict": "clean", "false_positives": 0},
        {"experiment": "E1", "run_id": "E1-a", "model": "m", "k": 8, "payload_bytes": 1024,
         "single_file_detected": False, "base_detected": True},
        {"experiment": "E2", "run_id": "E2-a", "model": "m", "k": None, "payload_bytes": 0,
         "accuracy": 0.9, "accuracy_drop": 0.0, "n_images": 10},
        {"experiment": "E2", "run_id": "E2-a", "model": "m", "k": 8, "payload_bytes": 1024,
         "accuracy": 0.8, "accuracy_drop": 0.1, "n_images": 10},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report.main()
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| m | dev | 3 | clean | 0 |" in text
    assert "| 1 KB | **missed** | detected |" in text
    assert "### m" in text
    assert "| 8 | 1 KB | 0.8000 | +0.1000 |" in text


def test_hf_section(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER", path)
    monkeypatch.setattr(report, "OUT", tmp_path / "RESULTS.md")
    g = "F32, fair-coin low bits"
    rows = [
        {"experiment": "E0b", "run_id": "s", "repo": "a/b", "dtype": "F32", "group": g, "n_sampled": 12,
         "signature": None, "false_positives": 1, "fp_tensors": ["t"]},
        {"experiment": "E0b", "run_id": "s", "repo": "c/d", "dtype": "F32", "error": "HTTPError 403"},
        {"experiment": "E1b", "run_id": "a", "repo": "a/b", "group": g, "k": 1,
         "target_flagged_clean": False, "detected": False},
        {"experiment": "E1b", "run_id": "a", "repo": "a/b", "group": g, "k": 8,
         "target_flagged_clean": False, "detected": True},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report.main()
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert f"| {g} | 1 | 1 | 1 / 12 |" in text
    assert f"| {g} | 0 / 1 | – | 1 / 1 |" in text
    assert "`c/d`: HTTPError 403" in text
    assert "| `a/b` | F32 | " in text and "| no / – / yes |" in text
