"""Static guard: every name the pipeline entry point calls must actually exist.

WHY THIS EXISTS: a refactor deleted resolve_target_week() and left its call site in
main(). All sixteen other tests passed, because they import the inner helper
(resolve_week_from_starts) directly and never touch the entry point. The result was a
NameError on every run that omits --week -- raised before setup_logging(), and silently
swallowed by weekly-update.yml's `|| true`, so the workflow goes GREEN having published
nothing while still firing the site's deploy hook.

Unit tests that import helpers can never catch that. This one resolves every name used
inside each module-level function against the module's own globals plus builtins.
"""
import ast
import builtins
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODULES = ["main.py", "utils.py", "artifacts/schedule.py"]


def _unresolved_names(path: Path):
    tree = ast.parse(path.read_text())
    module_level = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_level.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_level.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                module_level.add((a.asname or a.name).split(".")[0])

    problems = []
    for fn in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local = set(module_level)
        for sub in ast.walk(fn):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(sub.name)   # nested def/class is a local binding
            elif isinstance(sub, ast.arg):
                local.add(sub.arg)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store,)):
                local.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    local.add((a.asname or a.name).split(".")[0])
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                local.add(sub.name)
            elif isinstance(sub, (ast.comprehension,)):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        local.add(t.id)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id not in local:
                    problems.append((fn.name, sub.id, sub.lineno))
    return problems


def test_no_undefined_names_in_pipeline_modules():
    failures = []
    for rel in MODULES:
        for fn, name, line in _unresolved_names(REPO / rel):
            failures.append(f"{rel}:{line} {fn}() references undefined name {name!r}")
    assert not failures, "undefined names:\n  " + "\n  ".join(failures)


def test_entry_point_helpers_are_importable():
    """The specific regression: main() calls resolve_target_week, so it must exist."""
    import main
    assert callable(main.resolve_target_week)
    assert callable(main.resolve_week_from_starts)
    assert callable(main.main)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
