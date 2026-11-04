import importlib
import sys
import pytest
import textwrap
from pathlib import Path
from typing import Iterator

from podpac.core.lazy_loading import lazy_module


class TestLazyModule:
    @pytest.fixture
    def package_name(self) -> str:
        """Name of the temporary package.

        Returns
        -------
        str
            Top-level package name.
        """
        return "podpac_lazy_loading_temp_package"

    @pytest.fixture
    def tracker(self, tmp_path: Path) -> Path:
        """File that each module of the temporary package appends its name to when it runs.

        Used to track which modules have loaded.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.

        Returns
        -------
        Path
            Tracker file path.
        """
        return tmp_path / "modules.txt"

    @pytest.fixture
    def package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package_name: str, tracker: Path
    ) -> Iterator[str]:
        """Temporary package for loading with submodules.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        monkeypatch : pytest.MonkeyPatch
            Used to put the package on ``sys.path``.
        package_name : str
            Top-level package name.
        tracker : Path
            File each module appends to when it loads.

        Yields
        ------
        str
            Top-level package name.
        """
        # Write module name to the tracker file
        source = textwrap.dedent(f"""\
            with open({str(tracker)!r}, "a") as file:
                print(__name__, file=file)
            VALUE = __name__
            """)

        root = tmp_path / package_name
        (root / "sub").mkdir(parents=True)
        for path in ["__init__.py", "sub/__init__.py", "sub/leaf.py"]:
            (root / path).write_text(source)

        # Add tmp_path to the system path so modules can be found
        monkeypatch.syspath_prepend(tmp_path)

        yield package_name

        # Cleanup the package addition
        for name in list(sys.modules):
            if name.split(".")[0] == package_name:
                del sys.modules[name]

    def assert_ran(self, tracker: Path, expected: list) -> None:
        """Assert which modules of the temporary package have run, in order.

        Parameters
        ----------
        tracker : Path
            Tracker file each module appends to when it runs.
        expected : list
            Module names expected to have run.
        """
        ran = tracker.read_text().splitlines() if tracker.exists() else []
        assert ran == expected

    def test_top_level_deferred(self, package: str, tracker: Path) -> None:
        """A top-level package does not run until an attribute is accessed."""
        lazy_module(package)
        self.assert_ran(tracker, [])

    def test_top_level_loads_on_access(self, package: str, tracker: Path) -> None:
        """A top-level package runs on first attribute access."""
        assert lazy_module(package).VALUE == package
        self.assert_ran(tracker, [package])

    def test_submodule_deferred(self, package: str, tracker: Path) -> None:
        """Neither a submodule nor its parent package runs until an attribute is accessed."""
        lazy_module(package + ".sub")
        self.assert_ran(tracker, [])

    def test_submodule_loads_parent_first(self, package: str, tracker: Path) -> None:
        """A submodule runs its parent package first on first attribute access."""
        assert lazy_module(package + ".sub").VALUE == package + ".sub"
        self.assert_ran(tracker, [package, package + ".sub"])

    def test_nested_submodule_loads_parents_first(self, package: str, tracker: Path) -> None:
        """A nested submodule runs each parent package in order on first attribute access."""
        assert lazy_module(package + ".sub.leaf").VALUE == package + ".sub.leaf"
        self.assert_ran(tracker, [package, package + ".sub", package + ".sub.leaf"])

    def test_already_imported_returns_module(self, package: str) -> None:
        """An already imported submodule is returned directly."""
        module = importlib.import_module(package + ".sub")
        assert lazy_module(package + ".sub") is module

    def test_missing_top_level_raises_on_access(self, package_name: str) -> None:
        """A missing top-level package raises on first attribute access, not on load."""
        module = lazy_module(package_name + "_missing")
        with pytest.raises(ModuleNotFoundError):
            module.VALUE

    def test_missing_parent_raises_on_access(self, package_name: str) -> None:
        """A submodule of a missing package raises on first attribute access, not on load."""
        module = lazy_module(package_name + "_missing.sub")
        with pytest.raises(ModuleNotFoundError):
            module.VALUE

    def test_missing_submodule_raises_on_access(self, package: str, tracker: Path) -> None:
        """A missing submodule of an installed package raises on first attribute access, not on load."""
        module = lazy_module(package + ".missing")
        self.assert_ran(tracker, [])
        with pytest.raises(ModuleNotFoundError):
            module.VALUE
