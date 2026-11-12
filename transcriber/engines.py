"""Движки распознавания. Общий интерфейс:
    engine.transcribe(samples, language, prompt, on_progress, word_timestamps) -> [{start, end, text, words?}]
samples - float32 моно 16 кГц; on_progress(fraction) получает долю 0..1.
В одном процессе держим только один движок (unified memory на Mac общая с GPU).
"""
import contextlib
import os
import re
import sys
import types

SAMPLE_RATE = 16000

HALLUCINATIONS = ('спасибо за просмотр', 'продолжение следует', 'субтитры сделал', 'субтитры создавал',
                  'подписывайтесь на канал', 'редактор субтитров', 'субтитры подогнал',
                  'thank you for watching', 'thanks for watching')


def mlx_repo(name):
    """Короткое имя модели -> репозиторий mlx-community."""
    if '/' in name or os.path.isdir(name):
        return name
    if name in ('turbo', 'large-v3-turbo'):
        return 'mlx-community/whisper-large-v3-turbo'
    if name == 'tiny':
        return 'mlx-community/whisper-tiny'
    return 'mlx-community/whisper-%s-mlx' % name


def _norm(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def is_hallucination(text):
    """Отбрасываем только короткие фразы, целиком состоящие из типичной галлюцинации,
    а не любую фразу, где она встречается как подстрока."""
    t = _norm(text)
    if not t:
        return True
    return any(t.startswith(h) and len(t) <= len(h) + 25 for h in HALLUCINATIONS)


def speech_spans(samples):
    """Участки речи по Silero VAD, в секундах. Порог низкий (0.3), чтобы не терять тихую речь."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    speech = get_speech_timestamps(samples, VadOptions(threshold=0.3, min_silence_duration_ms=500, speech_pad_ms=300))
    return [(s['start'] / SAMPLE_RATE, s['end'] / SAMPLE_RATE) for s in speech]


def chunk_bounds(duration, spans, chunk=1200.0, window=90.0):
    """Границы кусков ~20 минут, разрезанных посередине ближайшей паузы (чтобы не резать слово).
    Нужно для очень длинных записей: mlx_whisper строит спектрограмму всего входа сразу."""
    if duration <= chunk * 1.5:
        return [(0.0, duration)]
    gaps = [((a_end + b_start) / 2, b_start - a_end) for (_, a_end), (b_start, _) in zip(spans, spans[1:])]
    cuts, t = [], chunk
    while t < duration - chunk * 0.5:
        near = [g for g in gaps if abs(g[0] - t) <= window]
        cut = max(near, key=lambda g: (g[1], -abs(g[0] - t)))[0] if near else t
        cuts.append(cut)
        t = cut + chunk
    edges = [0.0] + cuts + [duration]
    return list(zip(edges, edges[1:]))


def vad_filter(segments, spans, min_overlap=0.3):
    """Убирает сегменты, которые почти не пересекаются с речью (галлюцинации на тишине)."""
    kept = []
    for seg in segments:
        dur = max(seg['end'] - seg['start'], 0.01)
        overlap = sum(max(0.0, min(seg['end'], e) - max(seg['start'], s)) for s, e in spans)
        if overlap / dur >= min_overlap:
            kept.append(seg)
    return kept


class MlxEngine:
    name = 'mlx'

    def __init__(self, model):
        self.model = mlx_repo(model)
        import mlx_whisper  # noqa: F401  (регистрирует модуль mlx_whisper.transcribe)
        self._mod = sys.modules['mlx_whisper.transcribe']
        self._orig_tqdm = self._mod.tqdm
        self._orig_opts = self._mod.DecodingOptions
        self._orig_tok = self._mod.get_tokenizer

    def _patch(self, on_progress, prompt):
        mod = self._mod
        state = {'tokenizer': None, 'prompt_tokens': None}

        # 1. Прогресс: mlx_whisper рисует tqdm по кадрам мел-спектрограммы (только при verbose=False).
        class Bar(self._orig_tqdm.tqdm):
            def __init__(self, *a, **kw):
                kw['disable'] = True
                super().__init__(*a, **kw)
                self._done = 0

            def update(self, n=1):
                self._done += n
                if on_progress and self.total:
                    on_progress(min(self._done / self.total, 1.0))

        mod.tqdm = types.SimpleNamespace(tqdm=Bar)

        # 2. Словарь терминов в КАЖДОМ окне. Штатный initial_prompt при condition_on_previous_text=False
        # действует только на первые 30 секунд, поэтому подмешиваем термины в prompt каждого окна.
        if prompt:
            def get_tokenizer(*a, **kw):
                tok = self._orig_tok(*a, **kw)
                state['tokenizer'] = tok
                state['prompt_tokens'] = tok.encode(' ' + prompt.strip())[:200]
                return tok

            def decoding_options(**kw):
                pt = state['prompt_tokens']
                cur = kw.get('prompt')
                if pt and (cur is None or isinstance(cur, list)):
                    cur = list(cur or [])
                    if cur[:len(pt)] != pt:
                        kw['prompt'] = pt + cur
                return self._orig_opts(**kw)

            mod.get_tokenizer = get_tokenizer
            mod.DecodingOptions = decoding_options

    def _unpatch(self):
        self._mod.tqdm = self._orig_tqdm
        self._mod.DecodingOptions = self._orig_opts
        self._mod.get_tokenizer = self._orig_tok

    def _run(self, audio, kwargs):
        import mlx_whisper
        # mlx_whisper печатает "Detected language" в stdout - не даём ломать JSON-вывод CLI
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return mlx_whisper.transcribe(audio, **kwargs)
            except Exception as e:
                if not kwargs.get('word_timestamps'):
                    raise
                print('[mlx] пословные таймкоды не сработали (%s), повтор без них' % e, file=sys.stderr)
                kw = dict(kwargs, word_timestamps=False)
                kw.pop('hallucination_silence_threshold', None)
                return mlx_whisper.transcribe(audio, **kw)

    def transcribe(self, samples, language, prompt=None, on_progress=None, word_timestamps=True):
        kwargs = dict(path_or_hf_repo=self.model, verbose=False, language=language,
                      condition_on_previous_text=False, word_timestamps=word_timestamps)
        if word_timestamps:
            kwargs['hallucination_silence_threshold'] = 2.0
        duration = len(samples) / SAMPLE_RATE
        spans = speech_spans(samples)
        chunks = chunk_bounds(duration, spans)
        segments = []
        self.language = language
        for a, b in chunks:
            def chunk_progress(f, a=a, b=b):
                if on_progress and duration:
                    on_progress(min((a + (b - a) * f) / duration, 1.0))

            self._patch(chunk_progress, prompt)
            try:
                result = self._run(samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)], kwargs)
            finally:
                self._unpatch()
                self.clear()
            if kwargs['language'] is None and result.get('language'):
                kwargs['language'] = self.language = result['language']  # язык определяем один раз
            for s in result.get('segments', []):
                seg = {'start': a + float(s['start']), 'end': a + float(s['end']), 'text': s['text'].strip()}
                if s.get('words'):
                    seg['words'] = [{'start': a + float(w['start']), 'end': a + float(w['end']), 'text': w['word']}
                                    for w in s['words']]
                segments.append(seg)
        return vad_filter(segments, spans)

    @staticmethod
    def clear():
        try:
            import mlx.core as mx
            (getattr(mx, 'clear_cache', None) or mx.metal.clear_cache)()
        except Exception:
            pass


class FasterWhisperEngine:
    name = 'cpu'

    def __init__(self, model):
        from faster_whisper import WhisperModel
        self.model = model
        threads = min(8, os.cpu_count() or 4)
        self._model = WhisperModel(model, device='cpu', compute_type='int8', cpu_threads=threads)

    def transcribe(self, samples, language, prompt=None, on_progress=None, word_timestamps=True):
        duration = len(samples) / SAMPLE_RATE
        segs, info = self._model.transcribe(samples, language=language, vad_filter=True, beam_size=5,
                                            condition_on_previous_text=False, word_timestamps=word_timestamps,
                                            hotwords=prompt)
        out = []
        for s in segs:
            seg = {'start': float(s.start), 'end': float(s.end), 'text': s.text.strip()}
            if s.words:
                seg['words'] = [{'start': float(w.start), 'end': float(w.end), 'text': w.word} for w in s.words]
            out.append(seg)
            if on_progress and duration:
                on_progress(min(s.end / duration, 1.0))
        self.language = info.language
        return out


_engine = None


def get_engine(kind, model):
    """Один движок на процесс; при смене движка/модели старый освобождается."""
    global _engine
    if _engine is not None and (_engine.name != kind or getattr(_engine, '_req_model', None) != model):
        _engine = None
        import gc
        gc.collect()
        MlxEngine.clear()
    if _engine is None:
        _engine = MlxEngine(model) if kind == 'mlx' else FasterWhisperEngine(model)
        _engine._req_model = model
    return _engine


def split_segments(segments, max_dur=8.0, max_chars=90):
    """Режет длинные сегменты по пословным таймкодам: на концах предложений, а если фраза всё ещё длинная -
    на запятой или просто по словам. Нужно для субтитров и читаемых таймкодов."""
    out = []
    for seg in segments:
        words = seg.get('words')
        if not words or (seg['end'] - seg['start'] <= max_dur and len(seg['text']) <= max_chars):
            out.append(seg)
            continue
        cur = []

        def flush():
            if cur:
                text = ''.join(w['text'] for w in cur).strip()
                if text:
                    out.append({'start': cur[0]['start'], 'end': cur[-1]['end'], 'text': text, 'words': list(cur)})
                cur.clear()

        for w in words:
            cur.append(w)
            t = w['text'].strip()
            dur = cur[-1]['end'] - cur[0]['start']
            chars = sum(len(x['text']) for x in cur)
            if t.endswith(('.', '!', '?', '…')) and (dur > 1.5 or chars > 20):
                flush()
            elif (dur > max_dur * 0.6 or chars > max_chars * 0.6) and t.endswith((',', ';', ':')):
                flush()
            elif dur > max_dur or chars > max_chars:
                flush()
        flush()
    return out
