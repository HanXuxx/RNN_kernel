#!/usr/bin/env python3
"""从 src/rnn_kernel/a100/prod 导出独立 A100GRUH256 产品包。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_PACKAGE = SOURCE_ROOT / "a100_gru_h256"
CUBIN_RELATIVE_PATH = Path("a100_gru_h256/kernels/a100_gru_h256_sm80.cubin")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _copy_package(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if not SOURCE_PACKAGE.is_dir():
        raise FileNotFoundError(f"missing source package: {SOURCE_PACKAGE}")
    if not (SOURCE_PACKAGE / "kernels" / "a100_gru_h256_sm80.cubin").is_file():
        raise FileNotFoundError("missing prebuilt sm80 cubin in source package")

    stale_paths = [
        output_root / "pyproject.toml",
        output_root / "MANIFEST.in",
        output_root / "dist",
        output_root / "a100_gru_h256.tar.gz",
    ]
    for stale_path in stale_paths:
        _remove_path(stale_path)

    package_target = output_root / "a100_gru_h256"
    package_tmp = output_root / ".a100_gru_h256.tmp"
    _remove_path(package_tmp)
    shutil.copytree(
        SOURCE_PACKAGE,
        package_tmp,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    _remove_path(package_target)
    package_tmp.rename(package_target)


def _build_wheel(package_root: Path) -> Path:
    for path in (
        package_root / "build",
        package_root / "a100_gru_h256.egg-info",
        package_root / "dist",
    ):
        _remove_path(path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            "dist",
        ],
        cwd=package_root,
        check=True,
    )
    _remove_path(package_root / "build")
    _remove_path(package_root / "a100_gru_h256.egg-info")

    wheels = sorted((package_root / "dist").glob("a100_gru_h256-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel, got {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        if CUBIN_RELATIVE_PATH.as_posix() not in archive.namelist():
            raise RuntimeError(f"wheel missing {CUBIN_RELATIVE_PATH}")
    return wheel


def _build_archive(package_root: Path) -> Path:
    archive_path = package_root / "dist" / "a100_gru_h256-0.1.0.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(
            package_root,
            arcname="a100_gru_h256",
            filter=_archive_filter,
        )
    return archive_path


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts:
        return None
    if len(parts) >= 2 and parts[0] == "a100_gru_h256" and parts[1] == "dist":
        return None
    if info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "prod",
        help="产品包输出目录，默认是仓库根目录下的 prod。",
    )
    parser.add_argument(
        "--no-build-artifacts",
        action="store_true",
        help="只复制源码，不构建 wheel 和 tar.gz。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    _copy_package(output_root)
    package_root = output_root / "a100_gru_h256"
    print(f"copied {SOURCE_PACKAGE} -> {package_root}")
    if args.no_build_artifacts:
        return
    wheel = _build_wheel(package_root)
    archive = _build_archive(package_root)
    print(f"built {wheel}")
    print(f"built {archive}")


if __name__ == "__main__":
    main()
