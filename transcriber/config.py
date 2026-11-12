import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / 'transcribe.config.json'
TERMS_PATH = BASE_DIR / 'terms.txt'
MODELS_DIR = BASE_DIR / 'models'

MEDIA_EXT = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.wmv', '.flv', '.ts',
             '.mp3', '.m4a', '.wav', '.ogg', '.oga', '.opus', '.flac', '.aac', '.wma', '.amr', '.3gp'}
FORMATS = ('txt', 'md', 'srt', 'vtt', 'json')


def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
        except Exception:
            return {}
    return {}


def setup_hf_home(cfg=None):
    """HF_HOME должен быть выставлен до импорта mlx_whisper / huggingface_hub."""
    cfg = cfg if cfg is not None else load_config()
    hf = (cfg.get('paths') or {}).get('hfHome') or str(MODELS_DIR)
    os.environ['HF_HOME'] = os.path.expanduser(os.path.expandvars(hf))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    return os.environ['HF_HOME']


def read_terms(extra_path=None):
    """Словарь терминов: по одному на строку, # - комментарий."""
    terms = []
    for p in [TERMS_PATH] + ([Path(extra_path)] if extra_path else []):
        if p and p.exists():
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                # комментарий - строка с # в начале или « #» после термина; C#, F# не трогаем
                line = re.split(r'(?:^|\s)#', line, 1)[0].strip()
                if line and line not in terms:
                    terms.append(line)
    return terms


def terms_prompt(terms, max_words=100):
    """Подсказка для Whisper: термины через запятую, не длиннее ~100 слов (окно промпта 224 токена)."""
    out, words = [], 0
    for t in terms:
        n = len(t.split())
        if words + n > max_words:
            break
        out.append(t)
        words += n
    return ', '.join(out) + '.' if out else None
