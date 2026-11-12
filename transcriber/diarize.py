"""Разделение по говорящим: sherpa-onnx (pyannote segmentation 3.0 + 3D-Speaker eres2net), CPU, без токенов HF.
Сопоставление слов со спикерами перенесено из ~/.claude/skills/transcribe/scripts/transcribe_local.py."""
import os
import sys
import tarfile
import urllib.request

from .config import MODELS_DIR

DIAR_DIR = MODELS_DIR / 'diarization'
SEG_DIR = DIAR_DIR / 'sherpa-onnx-pyannote-segmentation-3-0'
SEG_MODEL = SEG_DIR / 'model.onnx'
EMB_MODEL = DIAR_DIR / '3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx'
BASE_URL = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
SEG_URL = BASE_URL + 'speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2'
EMB_URL = BASE_URL + 'speaker-recongition-models/' + EMB_MODEL.name


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + '.part')
    print('[diarize] загрузка %s' % url, file=sys.stderr, flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def ensure_models():
    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    if not SEG_MODEL.exists():
        arc = DIAR_DIR / 'segmentation.tar.bz2'
        _download(SEG_URL, arc)
        with tarfile.open(arc) as t:
            t.extractall(DIAR_DIR, filter='data')
        arc.unlink()
    if not EMB_MODEL.exists():
        _download(EMB_URL, EMB_MODEL)
    return SEG_MODEL, EMB_MODEL


def diarize(samples, num_speakers=None, threshold=0.5, on_progress=None):
    """-> [{start, end, speaker}] по времени."""
    import sherpa_onnx
    ensure_models()
    threads = max(1, min(4, (os.cpu_count() or 4) // 2))
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            provider='cpu', num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), provider='cpu', num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise RuntimeError('некорректная конфигурация диаризации')
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def cb(done, total):
        if on_progress and total:
            on_progress(done / total)
        return 0

    result = sd.process(samples, callback=cb).sort_by_start_time()
    return [{'start': float(r.start), 'end': float(r.end), 'speaker': int(r.speaker)} for r in result]


def _assign(mid, turns):
    for t in turns:
        if t['start'] <= mid <= t['end']:
            return t['speaker']
    if not turns:
        return 0
    return min(turns, key=lambda t: min(abs(t['start'] - mid), abs(t['end'] - mid)))['speaker']


def _smooth(words, min_run=1.2):
    """Короткая «пробивка» другого спикера между двумя кусками одного и того же - переатрибутировать."""
    if not words:
        return
    runs, i, n = [], 0, len(words)
    while i < n:
        j = i
        while j + 1 < n and words[j + 1]['speaker'] == words[i]['speaker']:
            j += 1
        runs.append([i, j, words[i]['speaker'], words[j]['end'] - words[i]['start']])
        i = j + 1
    changed = True
    while changed:
        changed = False
        for k in range(1, len(runs) - 1):
            ri, rj, rspk, rdur = runs[k]
            li, _, lspk, _ = runs[k - 1]
            nspk = runs[k + 1][2]
            if rdur < min_run and lspk == nspk and lspk != rspk:
                for x in range(ri, rj + 1):
                    words[x]['speaker'] = lspk
                nj = runs[k + 1][1]
                runs[k - 1] = [li, nj, lspk, words[nj]['end'] - words[li]['start']]
                del runs[k:k + 2]
                changed = True
                break


def merge(segments, turns, gap=1.5):
    """Сегменты с пословными таймкодами + реплики диаризации -> реплики [{start, end, speaker, text}].
    Каждому сегменту тоже проставляется seg['speaker'] (для субтитров).
    Номера спикеров перенумеровываются по порядку появления (1, 2, ...)."""
    words, loose = [], []
    for idx, seg in enumerate(segments):
        if seg.get('words'):
            for w in seg['words']:
                words.append(dict(w, speaker=_assign((w['start'] + w['end']) / 2, turns), _seg=idx))
        else:
            seg['speaker'] = _assign((seg['start'] + seg['end']) / 2, turns)
            loose.append({'start': seg['start'], 'end': seg['end'], 'text': seg['text'],
                          'speaker': _assign((seg['start'] + seg['end']) / 2, turns)})
    _smooth(words)
    votes = {}
    for w in words:
        v = votes.setdefault(w['_seg'], {})
        v[w['speaker']] = v.get(w['speaker'], 0) + (w['end'] - w['start']) + 1e-3
    for idx, v in votes.items():
        segments[idx]['speaker'] = max(v, key=v.get)

    utts = []
    for w in words:
        if utts and utts[-1]['speaker'] == w['speaker'] and not utts[-1].get('_loose'):
            utts[-1]['end'] = w['end']
            utts[-1]['text'] += w['text']
        else:
            utts.append({'start': w['start'], 'end': w['end'], 'speaker': w['speaker'], 'text': w['text']})
    utts.extend(dict(u, _loose=True) for u in loose)
    utts.sort(key=lambda u: u['start'])

    merged = []
    for u in utts:
        u.pop('_loose', None)
        u['text'] = u['text'].strip()
        if not u['text']:
            continue
        if merged and merged[-1]['speaker'] == u['speaker'] and u['start'] - merged[-1]['end'] < gap:
            merged[-1]['end'] = u['end']
            merged[-1]['text'] += ' ' + u['text']
        else:
            merged.append(u)

    order = {}
    for u in merged:
        order.setdefault(u['speaker'], len(order) + 1)
        u['speaker'] = order[u['speaker']]
    for seg in segments:
        if 'speaker' in seg:
            seg['speaker'] = order.setdefault(seg['speaker'], len(order) + 1)
    return merged
