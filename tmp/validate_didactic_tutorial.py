from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def compile_notebook(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    ids = [cell.get("id") for cell in notebook["cells"] if cell.get("id")]
    assert len(ids) == len(set(ids)), f"duplicate IDs in {path}"
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path.name}:cell-{index}", "exec")
    return notebook


tutorial_path = ROOT / "docs" / "tutorials" / "zenodo_resamplers.ipynb"
tutorial = compile_notebook(tutorial_path)
tutorial_text = "\n".join("".join(cell.get("source", [])) for cell in tutorial["cells"])
for name in (
    "NearestResampler", "BilinearResampler", "BicubicResampler",
    "CloughTocherResampler", "PSFResampler", "GroupByResampler",
    "ConservativeResampler", "CellPointResampler", "CategoricalResampler",
    "BitmaskResampler",
):
    assert name in tutorial_text, name
assert "healpix-resample-paper-core-data-v1.zip" in tutorial_text
assert "healpix-resample-paper-esri-multipatch-v1.zip" not in tutorial_text
assert tutorial["metadata"]["mystnb"]["execution_mode"] == "off"
print("OK didactic tutorial structure and syntax")

loader_path = ROOT / "notebooks" / "load_data_in_zenodo.ipynb"
loader = compile_notebook(loader_path)
loader_code = [
    "".join(cell.get("source", []))
    for cell in loader["cells"]
    if cell.get("cell_type") == "code"
]
namespace = {}
exec(loader_code[0], namespace)
exec(loader_code[1], namespace)
assert "tarfile.is_tarfile" in loader_code[1]

archive = ROOT / "tmp" / "zenodo-core-inspect.zip"
destination = ROOT / "tmp" / "loader-tar-extract-test"
if destination.exists():
    shutil.rmtree(destination)
written, skipped = namespace["safe_extract"](archive, destination)
assert written > 0 and skipped == 0
assert (destination / "notebooks" / "data" / "urban_data.zarr" / ".zgroup").is_file()
assert (destination / "notebooks" / "outputFLUX.grib").is_file()
print(f"OK real Zenodo TAR extraction: {written} files")
shutil.rmtree(destination)

index = (ROOT / "docs" / "tutorials" / "index.md").read_text(encoding="utf-8")
assert "zenodo_resamplers" in index
assert tutorial_path.is_file()
print("OK documentation reference")
