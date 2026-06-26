#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUBIN="${PACKAGE_ROOT}/kernels/a100_gru_h256_sm80.cubin"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -f "${CUBIN}" ]]; then
  echo "missing ${CUBIN}; run ${PYTHON_BIN} scripts/build_cubin.py first" >&2
  exit 1
fi

cd "${PACKAGE_ROOT}"
"${PYTHON_BIN}" -m pip wheel . --no-deps --no-build-isolation -w dist
"${PYTHON_BIN}" - <<'PY'
import zipfile
from pathlib import Path

wheel = next(Path("dist").glob("a100_gru_h256-*.whl"))
with zipfile.ZipFile(wheel) as zf:
    required = "a100_gru_h256/kernels/a100_gru_h256_sm80.cubin"
    if required not in zf.namelist():
        raise SystemExit(f"wheel missing {required}")
print(f"built {wheel}")
PY
