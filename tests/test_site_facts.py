"""The project page's numbers must be the ledger's numbers (scripts/site_facts.py)."""
from pathlib import Path

import pytest

from scripts import site_facts as SF

ROOT = Path(__file__).resolve().parent.parent
HAVE_MODELS = (ROOT / "models" / "torch").exists()


def test_gate_flags_a_wrong_fact_and_a_loose_number():
    html = ('<p>Accuracy <span data-fact="a">66.8%</span> on 500 images, '
            '<span data-fact="b">3 / 4</span>, <span data-fact="zz">1</span>.</p>'
            '<div data-nofacts>3f84 9b</div>')
    problems = SF.check(html, {"a": "66.8%", "b": "4 / 4"}, {"claim": False})
    assert "b: page says '3 / 4', ledger says '4 / 4'" in problems
    assert "unknown fact key 'zz' (page says '1')" in problems
    assert "number not backed by a fact: '500'" in problems
    assert "claim no longer true: claim" in problems
    assert not any("3f84" in p for p in problems)  # data-nofacts is skipped
    assert len(problems) == 4


def test_write_fills_spans_from_facts():
    html = '<b><span class="x" data-fact="a">old</span></b>'
    assert SF.write(html, {"a": "new"}) == '<b><span class="x" data-fact="a">new</span></b>'


def test_allowed_names_are_not_numbers():
    assert SF.stray_numbers(["ResNet-18 and MobileNetV3-Small, S1 to S5, H4, float32, k = 1, 4 or 8"]) == []


@pytest.mark.skipif(not (ROOT / "results" / "ledger.jsonl").exists(), reason="no ledger")
def test_project_page_matches_the_ledger():
    f, asserts = SF.facts(counts=False, weights=HAVE_MODELS)
    problems = []
    for page in SF.PAGES:
        html = page.read_text(encoding="utf-8")
        problems += SF.check(html, f, asserts if page == SF.PAGE else {})
    problems += SF.check_strips(SF.PAGE.read_text(encoding="utf-8"), SF.e1_strips(SF.load()))
    skipped = ("tests.",) + (() if HAVE_MODELS else ("e0.mobilenet_bit",))
    problems = [p for p in problems if not any(p.startswith(f"unknown fact key '{s}") for s in skipped)]
    assert problems == []
