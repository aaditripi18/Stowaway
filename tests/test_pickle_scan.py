import pickle

import pytest

from stowaway.pickle_scan import analyse, classify, opcode_imports
from conftest import Evil


@pytest.mark.parametrize("mod,name,want", [
    ("torch._utils", "_rebuild_tensor_v2", "safe"),
    ("collections", "OrderedDict", "safe"),
    ("torch", "FloatStorage", "safe"),
    ("os", "system", "dangerous"),
    ("posix", "system", "dangerous"),
    ("nt", "system", "dangerous"),
    ("builtins", "exec", "dangerous"),
    ("subprocess.foo", "Popen", "dangerous"),   # submodule of a denied module
    ("functools", "partial", "dangerous"),
    ("torch.nn.modules.linear", "Linear", "unknown"),
    ("osmosis", "thing", "unknown"),            # prefix match is per dotted segment
])
def test_classify(mod, name, want):
    assert classify(mod, name) == want


@pytest.mark.parametrize("protocol", range(0, pickle.HIGHEST_PROTOCOL + 1))
def test_opcode_walk_finds_import(tmp_path, protocol):
    data = pickle.dumps([Evil(tmp_path / "m")], protocol=protocol)
    found, err = opcode_imports(data)
    assert ("builtins", "exec") in found and err is None


def test_stack_global_through_memo(tmp_path):
    # Same callable twice: protocol 4 memoizes the strings and reuses them via BINGET.
    data = pickle.dumps([Evil(tmp_path / "1"), Evil(tmp_path / "2"), eval], protocol=4)
    found, _ = opcode_imports(data)
    assert ("builtins", "exec") in found and ("builtins", "eval") in found


def test_truncated_pickle_still_reports_import(tmp_path):
    data = pickle.dumps({"x": Evil(tmp_path / "m"), "pad": list(range(100))}, protocol=4)
    cut = data[: len(data) - 20]
    findings = analyse("cut", cut)
    assert any(f.severity == "critical" for f in findings)
    assert not (tmp_path / "m").exists()
