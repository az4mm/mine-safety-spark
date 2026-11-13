#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m mine_safety generate
.venv/bin/python -m mine_safety batch
.venv/bin/python -m mine_safety serve
