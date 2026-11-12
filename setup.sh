#!/bin/bash
# Установка окружения расшифровки (идемпотентно, без sudo и brew). Нужна сеть: PyPI, GitHub, Hugging Face.
#   ./setup.sh            - venv на Python 3.12 + пакеты + модели (large-v3-turbo для MLX, модели спикеров)
#   ./setup.sh --no-models - без предзагрузки моделей (скачаются при первом запуске)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "== uv"
python3 -m uv --version >/dev/null 2>&1 || python3 -m pip install --user --quiet "uv==0.13.1"

echo "== Python 3.12"
python3 -m uv python install 3.12

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
  echo "== venv"
  rm -rf .venv
  python3 -m uv venv --python 3.12 .venv
fi

echo "== пакеты"
python3 -m uv pip install --python .venv/bin/python -r requirements.txt

if [ "${1:-}" != "--no-models" ]; then
  echo "== модели"
  .venv/bin/python - <<'PY'
import sys, contextlib, io
sys.path.insert(0, '.')
from transcriber.config import setup_hf_home
setup_hf_home()
import static_ffmpeg
with contextlib.redirect_stdout(io.StringIO()):
    static_ffmpeg.add_paths()
from huggingface_hub import snapshot_download
print('whisper large-v3-turbo (MLX):', snapshot_download('mlx-community/whisper-large-v3-turbo'))
from transcriber import diarize
print('спикеры:', *diarize.ensure_models())
PY
fi
echo "Готово. CLI: ./transcribe.sh <файл>   UI: ./transcribe-ui.sh"
