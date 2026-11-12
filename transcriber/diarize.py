"""Разделение по говорящим.

Единица - сегмент распознавания (фраза до ~8 с, см. engines.split_segments). Для каждой фразы считается
голосовой эмбеддинг (3D-Speaker eres2net через sherpa-onnx), затем агломеративная кластеризация по косинусному
сходству (average linkage, центроиды взвешены длительностью). Мелкие кластеры (меньше 2% речи и меньше 20 с) -
это шум от коротких реплик, смеха, наложений: они приклеиваются к ближайшему крупному спикеру.

Почему не штатный OfflineSpeakerDiarization (pyannote-сегментация + FastClustering): на реальной встрече
4 человек он выдавал 47-81 «спикера» и работал в 10 раз дольше. Подход по фразам на той же записи даёт
4 спикера, и разметка совпадает у трёх разных моделей эмбеддингов (ARI 0.97-1.0) в диапазоне порогов 0.5-0.6.
"""
import os
import sys
import urllib.request

import numpy as np

from .config import MODELS_DIR

SR = 16000
DIAR_DIR = MODELS_DIR / 'diarization'
EMB_MODEL = DIAR_DIR / '3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx'
EMB_URL = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/' + EMB_MODEL.name

DEFAULT_THRESHOLD = 0.55   # косинусное сходство для слияния; больше -> больше спикеров
MIN_EMBED_SEC = 0.5        # более короткие фразы ненадёжны: спикер берётся у соседней фразы
MIN_SPEAKER_SHARE = 0.02   # спикер должен набрать хотя бы 2% речи...
MIN_SPEAKER_SEC = 20.0     # ...и при этом 20 секунд (или 10% речи - для коротких записей)


def ensure_models():
    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    if not EMB_MODEL.exists():
        tmp = EMB_MODEL.with_suffix('.part')
        print('[diarize] загрузка %s' % EMB_URL, file=sys.stderr, flush=True)
        with urllib.request.urlopen(EMB_URL, timeout=60) as r, open(tmp, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, EMB_MODEL)
    return EMB_MODEL


def _embeddings(samples, segments, on_progress):
    import sherpa_onnx
    threads = max(1, min(4, (os.cpu_count() or 4) // 2))
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ensure_models()), num_threads=threads))
    vecs, idx = [], []
    for i, s in enumerate(segments):
        a, b = int(s['start'] * SR), int(s['end'] * SR)
        if b - a >= MIN_EMBED_SEC * SR:
            st = ex.create_stream()
            st.accept_waveform(SR, samples[a:b])
            st.input_finished()
            vecs.append(np.asarray(ex.compute(st), dtype=np.float32))
            idx.append(i)
        if on_progress and i % 20 == 0:
            on_progress(0.9 * i / len(segments))
    if not vecs:
        return None, []
    e = np.stack(vecs)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9), idx


def _cluster(e, w, threshold, n_clusters=None):
    """Агломеративная кластеризация. Без n_clusters - сливаем, пока сходство центроидов >= threshold;
    с n_clusters - пока не останется ровно столько кластеров."""
    n = len(e)
    members = [[i] for i in range(n)]
    cent, wt = e.copy(), w.astype(np.float64).copy()
    alive = np.ones(n, bool)
    sim = cent @ cent.T
    np.fill_diagonal(sim, -2.0)
    count = n
    while count > 1:
        k = int(np.argmax(sim))
        a, b = divmod(k, n)
        if n_clusters is None and sim[a, b] < threshold:
            break
        if n_clusters is not None and count <= n_clusters:
            break
        members[a] += members[b]
        c = cent[a] * wt[a] + cent[b] * wt[b]
        wt[a] += wt[b]
        cent[a] = c / max(np.linalg.norm(c), 1e-9)
        alive[b] = False
        sim[b, :] = -2.0
        sim[:, b] = -2.0
        row = cent @ cent[a]
        row[~alive] = -2.0
        row[a] = -2.0
        sim[a, :] = row
        sim[:, a] = row
        count -= 1
    keep = np.flatnonzero(alive)
    return [members[i] for i in keep], cent[keep]


def _is_noise(sec, total):
    share = sec / total
    return share < MIN_SPEAKER_SHARE or (sec < MIN_SPEAKER_SEC and share < 0.10)


def _select_speakers(clusters, cent, e, w, num_speakers=None):
    """Выбор настоящих спикеров среди кластеров, остальные (шум) приклеиваются к ближайшему.
    Если число спикеров задано: лишних крупных сливаем по сходству, недостающих добираем из следующих по размеру.
    -> метка для каждой фразы."""
    total = w.sum()
    dur = [float(w[c].sum()) for c in clusters]
    order = sorted(range(len(clusters)), key=lambda i: -dur[i])
    big = [i for i in order if not _is_noise(dur[i], total)] or order[:1]
    groups = [[i] for i in big]
    if num_speakers:
        if len(groups) < num_speakers:
            groups += [[i] for i in order if i not in big][:num_speakers - len(groups)]
        while len(groups) > num_speakers:  # слить два самых похожих крупных
            gc = np.stack([_centroid(cent, dur, g) for g in groups])
            sim = gc @ gc.T
            np.fill_diagonal(sim, -2)
            a, b = divmod(int(np.argmax(sim)), len(groups))
            groups[a] += groups.pop(b)
    gc = np.stack([_centroid(cent, dur, g) for g in groups])
    owner = {ci: gi for gi, g in enumerate(groups) for ci in g}
    labels = np.empty(len(e), int)
    for ci, c in enumerate(clusters):
        for s in c:
            labels[s] = owner[ci] if ci in owner else int(np.argmax(gc @ e[s]))
    return labels


def _centroid(cent, dur, group):
    c = sum(cent[i] * dur[i] for i in group)
    return c / max(np.linalg.norm(c), 1e-9)


def diarize(samples, segments, num_speakers=None, threshold=None, on_progress=None):
    """Проставляет seg['speaker'] (1, 2, ... по порядку появления) каждому сегменту. Возвращает число спикеров."""
    if not segments:
        return 0
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    e, idx = _embeddings(samples, segments, on_progress)
    if e is None:
        for s in segments:
            s['speaker'] = 1
        return 1
    w = np.array([segments[i]['end'] - segments[i]['start'] for i in idx])
    clusters, cent = _cluster(e, w, threshold)
    labels = _select_speakers(clusters, cent, e, w, num_speakers)
    for j, i in enumerate(idx):
        segments[i]['speaker'] = int(labels[j])

    # короткие фразы без эмбеддинга - спикер ближайшей по времени фразы
    labeled = [i for i in range(len(segments)) if 'speaker' in segments[i]]
    for i, s in enumerate(segments):
        if 'speaker' not in s:
            near = min(labeled, key=lambda k: min(abs(segments[k]['start'] - s['end']), abs(s['start'] - segments[k]['end'])))
            s['speaker'] = segments[near]['speaker']

    order = {}
    for s in segments:
        s['speaker'] = order.setdefault(s['speaker'], len(order) + 1)
    if on_progress:
        on_progress(1.0)
    return len(order)


def utterances(segments, gap=5.0):
    """Соседние фразы одного спикера (пауза меньше gap) -> реплики [{start, end, speaker, text}]."""
    out = []
    for s in segments:
        if out and out[-1]['speaker'] == s['speaker'] and s['start'] - out[-1]['end'] < gap:
            out[-1]['end'] = s['end']
            out[-1]['text'] += ' ' + s['text']
        else:
            out.append({'start': s['start'], 'end': s['end'], 'speaker': s['speaker'], 'text': s['text']})
    return out
