#!/bin/bash
set -e
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv
"$VENV/bin/python" bench.py
