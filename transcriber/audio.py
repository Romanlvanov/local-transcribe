import contextlib
import io
import shutil
import subprocess


def find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    try:
        import static_ffmpeg
        # при первом вызове static_ffmpeg качает бинарники и пишет в stdout - не даем ему ломать JSON-вывод
        with contextlib.redirect_stdout(io.StringIO()):
            static_ffmpeg.add_paths()
    except Exception:
        return None
    return shutil.which(name)


def probe_duration(src):
    ffprobe = find_tool('ffprobe')
    if not ffprobe:
        return None
    out = subprocess.run([ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(src)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def to_wav(src, dst):
    """Любой медиафайл -> WAV 16 кГц моно."""
    ffmpeg = find_tool('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg не найден: установите пакет static-ffmpeg в .venv или ffmpeg в PATH')
    res = subprocess.run([ffmpeg, '-nostdin', '-y', '-v', 'error', '-i', str(src), '-vn', '-ac', '1', '-ar', '16000', str(dst)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError('ffmpeg: ' + (res.stderr.strip()[:300] or 'не удалось извлечь звук'))


def load_wav(path):
    import soundfile as sf
    samples, sr = sf.read(str(path), dtype='float32')
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        raise RuntimeError('ожидался WAV 16 кГц, получено %d' % sr)
    return samples
