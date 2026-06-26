#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_PARENT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
ARCHIVE_DIR="${ARCHIVE_DIR:-${PACKAGE_ROOT}/dist}"
ARCHIVE_PATH="${1:-${ARCHIVE_DIR}/a100_gru_h256-0.1.0.tar.gz}"

mkdir -p "$(dirname "${ARCHIVE_PATH}")"
tar \
  --exclude='a100_gru_h256/__pycache__' \
  --exclude='a100_gru_h256/*/__pycache__' \
  --exclude='a100_gru_h256/*/*/__pycache__' \
  --exclude='a100_gru_h256/dist' \
  -C "${PACKAGE_PARENT}" \
  -czf "${ARCHIVE_PATH}" \
  a100_gru_h256

echo "built ${ARCHIVE_PATH}"
