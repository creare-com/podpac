# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Install for development (runs `pre-commit install` via a custom `develop` cmdclass):
```
pip install -e .[devall]
```

Run the test suite (pytest config lives in `setup.cfg`; `testpaths = podpac`):
```
pytest -m "not integration"                       # full unit suite
pytest podpac/core/test/test_node.py              # single file
pytest podpac/core/test/test_node.py::TestNode    # single class
pytest podpac --ci                                # CI mode; skips tests marked `aws`
```

Custom pytest markers (registered in `podpac/conftest.py`): `aws`, `integration`. The `--ci` flag is defined in that conftest and MUST come after the `podpac` path.

Coverage (matches CI):
```
coverage run --branch -m pytest -m "not integration" --continue-on-collection-errors
coverage xml -o coverage.xml
```

Formatting / lint (CI gates both):
```
black --check --diff -l 120 .
flake8 --format=pylint --ignore=E,W,D,I,N806,N815,N818,Q000,Q001,Q002,S001,B008,B028 .
```
`pyproject.toml` pins black line length to 120. A pre-commit hook runs black on files under `podpac/` or `dist/`.

Docs (Sphinx; doctest target is what CI runs):
```
cd doc && ./test-docs.sh           # sphinx-build -b doctest source build
```

## Architecture

PODPAC's public API (see `podpac/__init__.py`) is a curated re-export layer over `podpac.core`. The top-level files `algorithm.py`, `data.py`, `compositor.py`, `coordinates.py`, `interpolators.py`, `managers.py`, `authentication.py`, `caches.py`, `style.py`, `utils.py` are thin wrappers that expose curated namespaces — edits to behavior almost always belong in `podpac/core/<subpackage>/`, not in these wrappers.

Four concepts tie the library together (see `doc/source/design.rst`):

- **Node** (`core/node.py`) — base class for every computation. Nodes are **designed to fail on eval, not on instantiation**; expensive work is deferred until `eval(coordinates)` is called. Nodes use `traitlets` for typed attributes. When adding or changing a node, preserve this lazy contract.
- **Coordinates** (`core/coordinates/`) — the geospatial index. PODPAC is restricted to four dimension names: `lat`, `lon`, `time`, `alt` (enforced via `VALID_DIMENSION_NAMES`). Stacked, uniform, affine, and polar variants live in sibling files; construction helpers `crange`/`clinspace` are exported at top level.
- **UnitsDataArray** (`core/units.py`) — the output format of every Node eval. Wraps `xarray.DataArray` with units (`pint`) and CRS metadata. Any function that returns array data from a Node should return (or be coerced into) a `UnitsDataArray`.
- **Pipelines** — not an explicit class. Any `Node` composition is a pipeline; serialize with `Node.definition` and rebuild with `Node.from_definition(...)`. This serialization is also what drives AWS Lambda execution via `core/managers/aws.py`.

Subpackage roles under `podpac/core/`:

- `data/` — `DataSource` is the abstract base; concrete sources (`rasterio_source`, `zarr_source`, `pydap_source`, `h5py_source`, `csv_source`, `ogc`, `ogr`, …) implement `get_data(coordinates, coordinates_index)` and declare their native `coordinates`. When wrapping a new dataset, subclass `DataSource` here — don't modify `datasource.py`.
- `interpolation/` — `InterpolationManager` dispatches among `Interpolator` subclasses (`nearest_neighbor`, `scipy`, `xarray`, `rasterio`, `none`). `selector.py` is the fast-path that lets a DataSource pre-trim data for a given interpolator before `get_data` runs.
- `compositor/` — combines multiple sources (`OrderedCompositor`, `TileCompositor`).
- `algorithm/` — node-level math, stats, reprojection, coordinate selection, and generic expression nodes (`numexpr`-backed when the `algorithms` extra is installed).
- `managers/` — execution backends: `multi_threading` (gated by `settings["MULTITHREADING"]`/`N_THREADS`), `multi_process`, `parallel`, and `aws` (Lambda deployment).
- `cache/` — `CacheCtrl` dispatches to `ram`, `disk`, `s3`, and `zarr` stores. Controlled by `settings["DEFAULT_CACHE"]` / `ENABLE_CACHE`. The test session sets `DEFAULT_CACHE=[]` in `conftest.py` — rely on that rather than disabling per-test.
- `coordinates/` — all Coordinates machinery; most bug fixes to selection/indexing behavior land here.

`podpac/alglib/` holds domain-specific algorithms (e.g. climatology) that depend on `core` but aren't part of it. A `datalib` package is referenced in docs but is not vendored in this repo.

## Settings and state

Runtime configuration is a dict-like singleton at `podpac.settings` (`core/settings.py`). Defaults live in `DEFAULT_SETTINGS`; users can override via `~/.config/podpac/settings.json` or `./settings.json` (cwd wins). Tests snapshot and restore settings around each session in `conftest.py`, so tests may mutate `settings` freely without cleanup.

## CI

`.github/workflows/gitlab-python-worflow.yml` runs four jobs on push to `main`/`develop` and on PRs: `Formatting` (black + flake8, both gating), `Unit Testing` (coverage), `Document Testing` (sphinx doctest against the sibling `creare-com/podpac-examples` repo), and `SonarQube Scan`. Unit and doctest jobs currently swallow failures (`|| true`), so a green CI run does not prove tests pass — read the job logs or the coverage artifact.
