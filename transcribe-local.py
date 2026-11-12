# Локальная расшифровка аудио и видео. Ничего не уходит в облако.
# Движки: mlx (GPU Apple Silicon, по умолчанию) и cpu (faster-whisper).
# Использование:
#   ./transcribe.sh "<файл или папка>" [...] [--engine mlx|cpu] [--model large-v3-turbo] [--language ru|auto]
#                   [--out <папка>] [--formats txt,md,srt,vtt,json] [--diarize [--num-speakers N]]
#                   [--terms <файл>] [--force]
#   ./transcribe.sh "<файл>" --probe      (только длительность, JSON)
# Результат рядом с файлом (или в --out): <имя>.txt/.md/.srt/.vtt/.json. Итог по каждому файлу - строка JSON в stdout.
# Основа: https://github.com/J3MeTron/Assistant (vault-template/00_System/Расшифровка/transcribe-local.py)
import argparse
import json
import os
import sys
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from transcriber.config import FORMATS, setup_hf_home  # noqa: E402

setup_hf_home()  # до импорта mlx_whisper / huggingface_hub


def human_progress():
    state = {'last': None}
    tty = sys.stderr.isatty()

    def on_event(ev):
        pct = int(ev['progress'] * 100)
        eta = ev.get('eta')
        line = '%3d%%  %s%s' % (pct, ev['stage_name'], ('  ~%d:%02d осталось' % divmod(eta, 60)) if eta else '')
        key = (ev['stage'], pct // (1 if tty else 5))
        if key == state['last']:
            return
        state['last'] = key
        if tty:
            sys.stderr.write('\r\033[K' + line)
        else:
            sys.stderr.write(line + '\n')
        sys.stderr.flush()

    def done():
        if tty and state['last']:
            sys.stderr.write('\n')

    on_event.done = done
    return on_event


def json_progress(ev):
    sys.stderr.write(json.dumps(ev, ensure_ascii=False) + '\n')
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description='Локальная расшифровка аудио и видео')
    ap.add_argument('inputs', nargs='+', help='файлы и/или папки')
    ap.add_argument('--engine', choices=['mlx', 'cpu'])
    ap.add_argument('--model')
    ap.add_argument('--language', help='ru, en, ... или auto')
    ap.add_argument('--out')
    ap.add_argument('--formats', help='через запятую: ' + ','.join(FORMATS))
    ap.add_argument('--diarize', action='store_true', help='разделить по спикерам')
    ap.add_argument('--num-speakers', type=int)
    ap.add_argument('--threshold', type=float, help='порог кластеризации спикеров (меньше - больше спикеров)')
    ap.add_argument('--terms', help='доп. файл со словарём терминов')
    ap.add_argument('--no-terms', action='store_true', help='не использовать словарь терминов')
    ap.add_argument('--force', action='store_true', help='перерасшифровать, даже если результат уже есть')
    ap.add_argument('--stem', help='имя файлов результата (только для одного входного файла)')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--progress-json', action='store_true', help='события прогресса JSON-строками в stderr')
    args = ap.parse_args()

    from transcriber import audio
    from transcriber.pipeline import assign_stems, expand_inputs, resolve_options, run_file

    files = expand_inputs(args.inputs)
    if args.probe:
        for f in files:
            print(json.dumps({'file': str(f), 'duration': audio.probe_duration(f)}, ensure_ascii=False))
        return 0
    if not files:
        print('нет медиафайлов для расшифровки', file=sys.stderr)
        return 2

    opts = resolve_options(engine=args.engine, model=args.model, language=args.language, out=args.out,
                           formats=args.formats.split(',') if args.formats else None, diarize=args.diarize,
                           num_speakers=args.num_speakers, threshold=args.threshold, terms=args.terms,
                           no_terms=args.no_terms, force=args.force)
    stems = {files[0]: args.stem} if args.stem and len(files) == 1 else assign_stems(files, opts)
    on_event = json_progress if args.progress_json else human_progress()
    failed = 0
    for i, f in enumerate(files, 1):
        if len(files) > 1 and not args.progress_json:
            print('[%d/%d] %s' % (i, len(files), f.name), file=sys.stderr)
        try:
            res = run_file(f, opts, on_event, stem=stems.get(f))
        except Exception as e:
            failed += 1
            res = {'file': str(f), 'error': str(e) or e.__class__.__name__}
            if os.environ.get('TRANSCRIBE_DEBUG'):
                traceback.print_exc()
        finally:
            getattr(on_event, 'done', lambda: None)()
        print(json.dumps(res, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
