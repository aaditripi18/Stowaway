from pathlib import Path

import pytest

from stowaway.formats import open_model


@pytest.fixture
def parse():
    """Parse a file and return (ModelFile, {tensor name: bytes read through our offsets})."""
    def _parse(path: Path):
        with open(path, "rb") as f:
            mf = open_model(Path(path), f)
            data = {t.name: mf.tensor_bytes(f, t) for t in mf.tensors if t.offset is not None}
        return mf, data
    return _parse


def codes(report, check=None):
    return [x["code"] for x in report["findings"] if check is None or x["check"] == check]


class Evil:
    """Pickles to a call that would create `marker` if anything ever executed it."""

    def __init__(self, marker: Path):
        self.marker = str(marker)

    def __reduce__(self):
        return exec, (f"open({self.marker!r}, 'w').write('pwned')",)
