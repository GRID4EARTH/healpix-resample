"""
tests/test_notebook_environment.py

Guards that ``notebooks/`` stays runnable. Three things have gone wrong here,
none caught by the test suite, CI, or review:

* **Imports no environment declared.** Twelve modules used by
  ``test-resample-paper.ipynb`` -- the notebook producing the paper's tables
  and figures -- were absent from every pixi environment, so it could not be
  run from a checkout at all.

* **A hardcoded CUDA device.** ``DEVICE = 'cuda:0'`` meant the notebook failed
  with ``AssertionError: Torch not compiled with CUDA enabled`` on any machine
  without an NVIDIA GPU, whatever the environment.

* **An ``engine=`` string naming a backend nothing provides.** The Sentinel-2
  read used ``engine="xarray-eopf"``, but that package registers its backend as
  ``eopf-zarr`` and always has -- in every release from 0.0.1 to 0.3.0. That
  path could never have run; the notebook only worked for whoever already had
  the Zarr cache on disk.

The first two are caught by **executing the real setup cells** rather than
parsing them, so this file cannot drift out of sync with the notebook the way a
hand-maintained dependency list would. Full execution is not possible in CI:
the data cells need the EOPF STAC endpoint, Esri basemap tiles, and pass
``force=True`` everywhere, so nothing is cached. The setup boundary is the cell
defining ``DEVICE``.

The engine check remains static, because verifying it for real needs a dataset
open over the network. It is deliberately the only parsed thing left.

Needs the ``notebooks`` environment, so it is marked and deselected by default::

    pixi run -e notebooks pytest -m notebooks
"""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebooks

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
REPO_ROOT = NOTEBOOK_DIR.parent
PAPER_NOTEBOOK = "test-resample-paper.ipynb"


def _code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return [
        "".join(c["source"]) for c in nb.get("cells", []) if c.get("cell_type") == "code"
    ]


def _strip_magics(src: str) -> str:
    return "\n".join(
        line for line in src.splitlines() if not line.strip().startswith(("%", "!"))
    )


def test_notebook_directory_is_discoverable():
    """Without this, the parametrised tests below would collect nothing and a
    silent no-op would look like a pass."""
    assert NOTEBOOK_DIR.is_dir()
    assert (NOTEBOOK_DIR / PAPER_NOTEBOOK).is_file()


def test_paper_notebook_setup_executes(tmp_path, monkeypatch):
    """Run the notebook's own setup cells, up to and including the one defining
    DEVICE. Executing rather than parsing means every import, and the GPU
    fallback, are exercised as written -- so this cannot silently drift.
    """
    # The setup cells mkdir data/, figures/ and tables/ relative to the CWD.
    # Run them somewhere disposable so the suite does not litter the repo.
    # Make the notebook-local helper importable after chdir(), then stub only
    # its data-presence check: this test validates imports and device setup,
    # while the frozen Zenodo archive is deliberately not downloaded in CI.
    monkeypatch.syspath_prepend(str(NOTEBOOK_DIR))
    paper_data_guard = importlib.import_module("paper_data_guard")
    monkeypatch.setattr(
        paper_data_guard,
        "require_paper_data",
        lambda _notebook_name: REPO_ROOT,
    )
    monkeypatch.chdir(tmp_path)

    cells = _code_cells(NOTEBOOK_DIR / PAPER_NOTEBOOK)
    boundary = next(
        (i for i, src in enumerate(cells) if re.search(r"^DEVICE\s*=", src, re.M)), None
    )
    assert boundary is not None, (
        "no cell defines DEVICE; the setup boundary this test relies on has "
        "moved -- update it deliberately rather than widening the range"
    )

    namespace: dict = {"__name__": "__main__"}
    for i, src in enumerate(cells[: boundary + 1]):
        try:
            exec(compile(_strip_magics(src), f"<{PAPER_NOTEBOOK} cell {i}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"{PAPER_NOTEBOOK} setup cell {i} raised {type(exc).__name__}: {exc}\n"
                f"The notebook cannot be run from a checkout. If this is a missing "
                f"module, declare it under [feature.notebooks.dependencies]."
            )

    assert "DEVICE" in namespace, "setup ran but never defined DEVICE"
    # The GPU fallback: whatever this box has, DEVICE must be usable here.
    import torch

    if not torch.cuda.is_available():
        assert namespace["DEVICE"] == "cpu", (
            f"DEVICE resolved to {namespace['DEVICE']!r} on a machine without CUDA; "
            f"the notebook will fail with 'Torch not compiled with CUDA enabled'"
        )


def _engine_cases():
    cases = []
    for nb in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        src = "\n".join(_code_cells(nb))
        for engine in sorted(set(re.findall(r"""engine\s*=\s*["']([^"']+)["']""", src))):
            cases.append(pytest.param(nb.name, engine, id=f"{nb.name}::{engine}"))
    return cases


@pytest.mark.parametrize("notebook,engine", _engine_cases())
def test_notebook_xarray_engines_are_registered(notebook, engine):
    """A wrong engine name raises only when that line is reached, needs network
    to reach, and is invisible to an import scan because the providing package
    is never imported directly. Hence checking the registry instead.

    """
    xarray = pytest.importorskip("xarray")
    available = sorted(xarray.backends.list_engines())
    assert engine in available, (
        f"{notebook} passes engine={engine!r} to xarray, but no installed backend "
        f"registers that name. Available: {available}."
    )
