#!/bin/bash
# Веб-интерфейс расшифровки: http://127.0.0.1:8765 (порт меняется через TRANSCRIBE_PORT). Остановить - Ctrl+C.
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${TRANSCRIBE_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
if curl -fs "$URL/api/config" >/dev/null 2>&1; then
  echo "Уже запущено: $URL"; open "$URL"; exit 0
fi
[ "$1" = "--no-browser" ] || (sleep 1.5; open "$URL") &
echo "Расшифровка: $URL  (Ctrl+C - остановить)"
exec "$DIR/.venv/bin/python" "$DIR/web/server.py"
