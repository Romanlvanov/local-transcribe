"""Запись результата: txt (таймкоды), md (чистый текст), srt, vtt, json.
Каждый файл пишется атомарно (tmp + replace); json - последним, по нему определяется «готово»."""
import json
import os


def _save(path, text):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def fmt_clock(t):
    t = int(t)
    return '%02d:%02d' % (t // 60, t % 60) if t < 3600 else '%d:%02d:%02d' % (t // 3600, (t % 3600) // 60, t % 60)


def _ts(t, sep):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return '%02d:%02d:%02d%s%03d' % (h, m, s, sep, ms)


def spk(n):
    return 'Спикер %d' % n


def write_txt(path, segments, utterances):
    if utterances:
        lines = ['[%s] %s: %s' % (fmt_clock(u['start']), spk(u['speaker']), u['text']) for u in utterances]
    else:
        lines = ['[%s] %s' % (fmt_clock(s['start']), s['text']) for s in segments]
    _save(path, '\n'.join(lines) + '\n')


def _paragraphs(segments, pause=2.0, soft_chars=350, max_chars=800):
    """Абзац заканчивается на паузе, либо на конце предложения после ~350 символов, либо жёстко после 800."""
    paras, cur, last_end = [], [], None
    for s in segments:
        size = sum(len(x) for x in cur)
        sentence_end = bool(cur) and cur[-1].rstrip().endswith(('.', '!', '?', '…'))
        if cur and (s['start'] - last_end > pause or (size > soft_chars and sentence_end) or size > max_chars):
            paras.append(' '.join(cur))
            cur = []
        cur.append(s['text'])
        last_end = s['end']
    if cur:
        paras.append(' '.join(cur))
    return paras


def write_md(path, segments, utterances, meta):
    head = ['# %s' % meta['name'], '',
            '_Длительность %s, модель %s (%s), язык %s_' % (fmt_clock(meta.get('duration') or 0), meta['model'],
                                                           meta['engine'], meta.get('language') or '?'), '']
    if utterances:
        body = ['**%s** [%s]: %s\n' % (spk(u['speaker']), fmt_clock(u['start']), u['text']) for u in utterances]
    else:
        body = [p + '\n' for p in _paragraphs(segments)]
    _save(path, '\n'.join(head + body))


def _cue_text(s):
    return ('%s: %s' % (spk(s['speaker']), s['text'])) if 'speaker' in s else s['text']


def write_srt(path, segments):
    out = []
    for i, s in enumerate(segments, 1):
        out.append('%d\n%s --> %s\n%s\n' % (i, _ts(s['start'], ','), _ts(s['end'], ','), _cue_text(s)))
    _save(path, '\n'.join(out))


def write_vtt(path, segments):
    out = ['WEBVTT\n']
    for s in segments:
        out.append('%s --> %s\n%s\n' % (_ts(s['start'], '.'), _ts(s['end'], '.'), _cue_text(s)))
    _save(path, '\n'.join(out))


def write_json(path, segments):
    # формат совместим с исходным transcribe-local.py: список {start, end, text} (+ speaker при диаризации)
    items = []
    for s in segments:
        item = {'start': round(s['start'], 2), 'end': round(s['end'], 2), 'text': s['text']}
        if 'speaker' in s:
            item['speaker'] = s['speaker']
        items.append(item)
    _save(path, json.dumps(items, ensure_ascii=False, indent=1))


def write_all(out_dir, stem, formats, segments, utterances, meta):
    files = {}
    for fmt in sorted(formats, key=lambda f: f == 'json'):
        p = out_dir / ('%s.%s' % (stem, fmt))
        if fmt == 'txt':
            write_txt(p, segments, utterances)
        elif fmt == 'md':
            write_md(p, segments, utterances, meta)
        elif fmt == 'srt':
            write_srt(p, segments)
        elif fmt == 'vtt':
            write_vtt(p, segments)
        elif fmt == 'json':
            write_json(p, segments)
        else:
            continue
        files[fmt] = str(p)
    return files
