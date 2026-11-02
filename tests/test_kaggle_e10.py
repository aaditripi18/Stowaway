"""The Kaggle notebook of every experiment is valid Python, defines every name it
uses, and runs that experiment's script (an E8 session once failed with NameError)."""
import ast
import builtins

from scripts.kaggle_e10 import JOBS, notebook_source


def test_every_job_notebook_defines_what_it_uses():
    for job, (_, script) in JOBS.items():
        src = notebook_source(job, "abc1234")
        tree = ast.parse(src)
        assert f"SCRIPT = {script!r}" in src
        bound = {a.asname or a.name.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        bound |= {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                  if isinstance(t, ast.Name)}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert used <= bound | set(dir(builtins)), used - bound
