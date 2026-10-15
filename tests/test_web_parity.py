"""Guard: the committed golden reports in web/fixtures match a fresh Python scan.

The TypeScript parity test (web/test/parity.test.ts) checks the TS scanner against
these same golden files, so together they pin Python and TypeScript to the same
canonical report. If this fails, re-run `web/test/gen_fixtures.py`.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "web" / "fixtures"
sys.path.insert(0, str(ROOT / "web" / "test"))

pytest.importorskip("safetensors")
from gen_fixtures import canon  # noqa: E402

from stowaway.report import scan  # noqa: E402

fixtures = sorted(FIX.glob("*.safetensors")) if FIX.exists() else []


@pytest.mark.skipif(not fixtures, reason="run web/test/gen_fixtures.py first")
@pytest.mark.parametrize("path", fixtures, ids=lambda p: p.stem)
def test_golden_matches_fresh_scan(path):
    golden = (path.with_suffix(".json")).read_text(encoding="utf-8").strip()
    assert canon(scan(path)) == golden
