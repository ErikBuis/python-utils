import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class TestFindImportingPackage:
    def test_find_importing_package_propagates(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tests.custom.init.folder.file"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_PROJECT_ROOT,
        )
        assert result.stdout.strip() == "tests.custom.init.folder"

    def test_find_importing_package_parent_propagates(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.custom.init.folder.subfolder.subfile",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=_PROJECT_ROOT,
        )
        assert result.stdout.strip() == "tests.custom.init.folder"

    def test_find_importing_package_direct_import_raises(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import python_utils.custom.init"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        assert result.returncode != 0
        assert "ImportError" in result.stderr
