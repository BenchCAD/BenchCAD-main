"""Guard that the publishable environment packages under `environments/` resolve
from their own wheel metadata alone.

These are pushed to the Prime Intellect hub, where a consumer runs
`uv pip install benchcad-<task>` into a bare venv — none of this repo's uv config
is present. A `[tool.uv]` override that reconciles the CAD stack while developing
*in that directory* is therefore invisible at install time, and the package fails
to resolve for everyone but us. That is exactly what shipped in v0.1.0 of
benchcad-vision2code: `nlopt==2.10.0` (which declares numpy>=2,<3) alongside
`numpy==1.26.4`, held together only by an override that never left the repo
(issue #48).

Offline and metadata-only — no network, no resolution. These check the two
invariants that break silently rather than every possible conflict.
"""

import pathlib
import tomllib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENVS = sorted(p for p in (_ROOT / "environments").glob("*/pyproject.toml"))

# nlopt 2.8.0 was the first release to declare `numpy>=2,<3`; 2.7.1 declares
# `numpy>=1.14`. A package pinning numpy<2 can only pin nlopt <= 2.7.1.
_LAST_NLOPT_ALLOWING_NUMPY1 = (2, 7, 1)

# Last CPython minor with a numpy 1.26.4 wheel (cp39-cp312) — `requires-python`
# must not admit an interpreter numpy cannot be installed on.
_LAST_PY_MINOR_WITH_NUMPY_126_WHEEL = 12


def _version(spec: str) -> tuple[int, ...]:
    """`"nlopt==2.7.1"` -> `(2, 7, 1)`. Only used on `==` pins."""
    return tuple(int(p) for p in spec.split("==", 1)[1].split("."))


def _pins(deps: list[str]) -> dict[str, str]:
    """Map bare package name -> its requirement string."""
    out = {}
    for d in deps:
        name = d.split("==")[0].split(">")[0].split("<")[0].split("[")[0].strip()
        out[name.lower().replace("_", "-")] = d
    return out


@pytest.mark.parametrize("path", _ENVS, ids=lambda p: p.parent.name)
def test_no_uv_workspace_overrides(path):
    """uv overrides/constraints don't survive into the wheel, so a published
    package must not depend on them to resolve."""
    cfg = tomllib.loads(path.read_text())
    uv = cfg.get("tool", {}).get("uv", {})
    for key in ("override-dependencies", "constraint-dependencies"):
        assert key not in uv, (
            f"{path.parent.name}: `[tool.uv] {key}` is workspace-only config — it is "
            f"not written into the built wheel, so `uv pip install` from the hub will "
            f"not see it. Express the constraint in `[project] dependencies` instead."
        )


@pytest.mark.parametrize("path", _ENVS, ids=lambda p: p.parent.name)
def test_nlopt_pin_agrees_with_numpy_pin(path):
    """nlopt >= 2.8 declares numpy>=2, which contradicts the numpy<2 that
    cadquery 2.3.0 requires."""
    deps = _pins(tomllib.loads(path.read_text()).get("project", {}).get("dependencies", []))
    if "nlopt" not in deps or "numpy" not in deps:
        pytest.skip("no CAD stack in this environment")
    if not deps["numpy"].startswith("numpy==1."):
        pytest.skip("numpy is not pinned to a 1.x")
    assert deps["nlopt"].startswith("nlopt=="), f"{path.parent.name}: pin nlopt exactly"
    assert _version(deps["nlopt"]) <= _LAST_NLOPT_ALLOWING_NUMPY1, (
        f"{path.parent.name}: {deps['nlopt']} declares numpy>=2,<3, which cannot "
        f"resolve against {deps['numpy']}."
    )


@pytest.mark.parametrize("path", _ENVS, ids=lambda p: p.parent.name)
def test_requires_python_excludes_interpreters_without_wheels(path):
    """An upper bound wider than the pinned numpy's wheel coverage lets an
    installer pick a Python the package can never be installed on."""
    cfg = tomllib.loads(path.read_text()).get("project", {})
    deps = _pins(cfg.get("dependencies", []))
    if deps.get("numpy", "") != "numpy==1.26.4":
        pytest.skip("numpy is not pinned to 1.26.4")
    bound = next(
        (c.strip()[1:] for c in cfg["requires-python"].split(",") if c.strip().startswith("<")),
        None,
    )
    assert bound, f"{path.parent.name}: requires-python needs an upper bound"
    assert int(bound.split(".")[1]) <= _LAST_PY_MINOR_WITH_NUMPY_126_WHEEL + 1, (
        f"{path.parent.name}: requires-python allows Python "
        f"3.{_LAST_PY_MINOR_WITH_NUMPY_126_WHEEL + 1}+, which has no numpy 1.26.4 wheel."
    )
