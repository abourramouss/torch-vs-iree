#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv
time "$VENV/bin/python" bench.py
