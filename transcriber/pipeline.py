"""Конвейер: probe -> WAV 16 кГц -> распознавание -> (диаризация) -> запись форматов. Плюс пакетная обработка."""
import os
import tempfile
import time
from pathlib import Path

from . import audio, writers
from .config import FORMATS, MEDIA_EXT, load_config, read_terms, terms_prompt

# доля общего прогресса на каждый этап
STAGES = {'convert': (0.0, 0.03), 'model': (0.03, 0.05), 'transcribe': (0.05, 0.97), 'diarize': (0.92, 0.97), 'write': (0.97, 1.0)}
STAGE_NAMES = {'convert': 'извлечение звука', 'model': 'загрузка модели', 'transcribe': 'распознавание',
               'diarize': 'разделение по спикерам', 'write': 'запись результата'}


def resolve_options(**kw):
    """Слить аргументы с transcribe.config.json."""
    cfg = load_config()
    engine = kw.get('engine') or cfg.get('engine') or 'mlx'
    models = cfg.get('models') or {}
    model = kw.get('model') or models.get(engine) or ('large-v3-turbo')
    language = kw.get('language') or cfg.get('language') or 'ru'
    formats = kw.get('formats') or cfg.get('formats') or list(FORMATS)
    formats = [f for f in formats if f in FORMATS]
    if 'json' not in formats:
        formats.append('json')  # json - основной файл результата, по нему определяется «уже расшифровано»
    return {
        'engine': engine, 'model': model,
        'language': None if language == 'auto' else language,
        'formats': formats,
        'diarize': bool(kw.get('diarize')),
        'num_speakers': kw.get('num_speakers') or None,
        'threshold': kw['threshold'] if kw.get('threshold') is not None else (cfg.get('diarize') or {}).get('threshold'),
        'terms': kw.get('terms'),
        'no_terms': bool(kw.get('no_terms')),
        'out': kw.get('out'),
        'force': bool(kw.get('force')),
    }


def expand_inputs(paths):
    """Файлы как есть, папки - рекурсивно по медиа-расширениям."""
    out = []
    for p in paths:
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in MEDIA_EXT
                              and not f.name.startswith('.')))
        else:
            out.append(p)
    return out


def output_paths(src, opts, stem=None):
    out_dir = Path(opts['out']).expanduser() if opts.get('out') else src.parent
    return out_dir, stem or src.stem


def assign_stems(files, opts):
    """Имена результатов без коллизий: a.mp3 и a.mp4 в одной папке (или одноимённые файлы из разных подпапок
    при общем --out) получают имена «a.mp3.*», «a.mp4.*», при повторе - с именем родительской папки."""
    keys = {}
    for f in files:
        out_dir, _ = output_paths(f, opts)
        keys.setdefault((str(out_dir), f.stem), []).append(f)
    stems = {}
    for (_, stem), group in keys.items():
        if len(group) == 1:
            stems[group[0]] = stem
            continue
        names = [g.name for g in group]
        for g in group:
            stems[g] = g.name if names.count(g.name) == 1 else '%s_%s' % (g.parent.name, g.name)
    return stems


def run_file(src, opts, on_event=None, stem=None):
    """Расшифровать один файл. on_event(dict) получает события прогресса."""
    src = Path(src)
    t0 = time.time()

    def emit(stage, frac=None, **extra):
        if not on_event:
            return
        lo, hi = STAGES[stage]
        if stage == 'transcribe' and opts['diarize']:
            hi = STAGES['diarize'][0]
        total = lo + (hi - lo) * (frac or 0.0)
        on_event(dict(event='progress', file=str(src), stage=stage, stage_name=STAGE_NAMES[stage],
                      stage_progress=frac, progress=round(total, 4), elapsed=round(time.time() - t0, 1), **extra))

    if not src.exists():
        raise FileNotFoundError('файл не найден: %s' % src)
    out_dir, stem = output_paths(src, opts, stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = out_dir / (stem + '.json')
    if existing.exists() and not opts['force']:
        files = {f: str(out_dir / ('%s.%s' % (stem, f))) for f in FORMATS if (out_dir / ('%s.%s' % (stem, f))).exists()}
        return {'file': str(src), 'skipped': True, 'files': files, 'txt': files.get('txt'), 'json': files.get('json')}

    from .engines import get_engine, is_hallucination, split_segments

    duration = audio.probe_duration(src)
    with tempfile.TemporaryDirectory() as tmp:
        emit('convert', 0.0)
        wav = os.path.join(tmp, 'audio.wav')
        audio.to_wav(src, wav)
        samples = audio.load_wav(wav)
    duration = duration or len(samples) / 16000

    emit('model', 0.0)
    if opts['engine'] == 'mlx':
        from huggingface_hub import snapshot_download
        from .engines import mlx_repo
        repo = mlx_repo(opts['model'])
        if not os.path.isdir(repo):
            try:
                snapshot_download(repo_id=repo, local_files_only=True)  # уже в кэше - без сети
            except Exception:
                snapshot_download(repo_id=repo)
    engine = get_engine(opts['engine'], opts['model'])

    prompt = None if opts.get('no_terms') else terms_prompt(read_terms(opts.get('terms')))
    started = time.time()

    def on_asr(frac):
        eta = (time.time() - started) / frac * (1 - frac) if frac > 0.02 else None
        emit('transcribe', frac, eta=round(eta) if eta else None)

    emit('transcribe', 0.0)
    segments = engine.transcribe(samples, opts['language'], prompt=prompt, on_progress=on_asr,
                                 word_timestamps=True)
    segments = split_segments([s for s in segments if not is_hallucination(s['text'])])
    asr_seconds = time.time() - started

    utterances = None
    speakers = None
    if opts['diarize'] and segments:
        from . import diarize
        emit('diarize', 0.0)
        speakers = diarize.diarize(samples, segments, num_speakers=opts['num_speakers'], threshold=opts['threshold'],
                                   on_progress=lambda f: emit('diarize', f))
        utterances = diarize.utterances(segments)

    emit('write', 0.0)
    meta = {'name': src.name, 'duration': duration, 'model': opts['model'], 'engine': opts['engine'],
            'language': opts['language'] or getattr(engine, 'language', None)}
    files = writers.write_all(out_dir, stem, opts['formats'], segments, utterances, meta)
    emit('write', 1.0)
    return {'file': str(src), 'txt': files.get('txt'), 'json': files.get('json'), 'files': files,
            'duration': round(duration, 1), 'seconds': round(time.time() - t0, 1), 'asr_seconds': round(asr_seconds, 1),
            'engine': opts['engine'], 'model': opts['model'], 'language': meta['language'],
            'segments': len(segments), 'speakers': speakers}
