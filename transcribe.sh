#!/bin/bash
# Обёртка: ./transcribe.sh "<файл>" [--model small|medium|large-v3] [--language ru] [--out <папка>] [--probe]
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" -X utf8 "$DIR/transcribe-local.py" "$@"
