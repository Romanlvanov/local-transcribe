"""Веб-интерфейс расшифровки: http://127.0.0.1:8765 (только локально).
Каждое задание выполняется отдельным процессом transcribe-local.py по одному за раз:
MLX/GPU не живёт в потоках веб-сервера, отмена = завершение процесса, память освобождается после файла."""
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from transcriber.config import FORMATS, TERMS_PATH, load_config  # noqa: E402
from transcriber.pipeline import assign_stems, expand_inputs  # noqa: E402

CLI = BASE / 'transcribe-local.py'
UPLOADS = BASE / 'uploads'
OUTPUT = BASE / 'output'
STATE = OUTPUT / 'jobs.json'
STATIC = Path(__file__).resolve().parent / 'static'
PORT = int(os.environ.get('TRANSCRIBE_PORT', '8765'))
ALLOWED_HOSTS = {'127.0.0.1:%d' % PORT, 'localhost:%d' % PORT}
MIME = {'txt': 'text/plain', 'md': 'text/markdown', 'srt': 'application/x-subrip', 'vtt': 'text/vtt',
        'json': 'application/json'}

app = FastAPI(title='Расшифровка')
lock = threading.Lock()
wake = threading.Event()
jobs = {}          # id -> job dict
procs = {}         # id -> Popen


# ---------- состояние ----------

def save_state():
    OUTPUT.mkdir(exist_ok=True)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(list(jobs.values()), ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, STATE)


def load_state():
    if not STATE.exists():
        return
    try:
        for j in json.loads(STATE.read_text(encoding='utf-8')):
            if j['status'] == 'running':
                kill_leftover(j.get('pid'))
                j.update(status='queued', progress=0, stage_name=None, eta=None, pid=None)
            jobs[j['id']] = j
    except Exception:
        pass


def kill_leftover(pid):
    """Процесс задания, переживший прошлый запуск сервера (Ctrl+C до него не доходит), - завершить."""
    if not pid:
        return
    try:
        cmd = subprocess.run(['ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True).stdout
        if 'transcribe-local.py' in cmd:
            os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def kill_all():
    for p in list(procs.values()):
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


atexit.register(kill_all)


def public(j):
    return {k: v for k, v in j.items() if not k.startswith('_')}


# ---------- воркер ----------

def build_cmd(j):
    o = j['options']
    cmd = [sys.executable, str(CLI), j['source'], '--out', j['out_dir'], '--progress-json',
           '--engine', o['engine'], '--model', o['model'], '--language', o['language'],
           '--formats', ','.join(o['formats'])]
    if o.get('diarize'):
        cmd.append('--diarize')
        if o.get('num_speakers'):
            cmd += ['--num-speakers', str(int(o['num_speakers']))]
    if o.get('force'):
        cmd.append('--force')
    if j.get('stem'):
        cmd += ['--stem', j['stem']]
    return cmd


def run_job(j, run_id):
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with lock:
        # задание могли отменить или перезапустить, пока воркер его выбирал
        if j['status'] != 'queued' or j.get('run') != run_id:
            return
        p = subprocess.Popen(build_cmd(j), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                             start_new_session=True)
        procs[j['id']] = p
        j.update(status='running', started=time.time(), progress=0, stage_name='запуск', eta=None, error=None,
                 pid=p.pid)
        save_state()
    errors = []
    for line in p.stderr:
        line = line.strip()
        if not line:
            continue
        if line.startswith('{'):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get('event') == 'progress':
                with lock:
                    if j['status'] == 'running':
                        j.update(progress=ev['progress'], stage_name=ev['stage_name'], eta=ev.get('eta'))
                continue
        errors.append(line)
    out = p.stdout.read()
    p.wait()
    result = None
    for line in out.splitlines():
        if line.startswith('{'):
            try:
                result = json.loads(line)
            except ValueError:
                pass
    with lock:
        procs.pop(j['id'], None)
        j['pid'] = None
        if j.get('run') != run_id:
            return  # задание уже перезапущено - этот результат устарел
        j['finished'] = time.time()
        if j['status'] == 'cancelled':
            pass
        elif result and not result.get('error'):
            j.update(status='done', progress=1, stage_name=None, eta=None, result=result)
        else:
            msg = (result or {}).get('error') or '\n'.join(errors[-5:]) or 'процесс завершился с кодом %s' % p.returncode
            j.update(status='error', error=msg, stage_name=None, eta=None)
        save_state()


def worker():
    while True:
        wake.wait(1.0)
        wake.clear()
        while True:
            with lock:
                nxt = next((j for j in sorted(jobs.values(), key=lambda x: x['created']) if j['status'] == 'queued'), None)
                run_id = nxt and nxt.get('run')
            if not nxt:
                break
            try:
                run_job(nxt, run_id)
            except Exception as e:  # воркер не должен умирать
                with lock:
                    nxt.update(status='error', error=str(e))
                    save_state()


# ---------- защита: только локальные запросы ----------

@app.middleware('http')
async def local_only(request: Request, call_next):
    if request.headers.get('host', '') not in ALLOWED_HOSTS:
        return PlainTextResponse('forbidden host', status_code=403)
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD') and origin and origin.split('://', 1)[-1] not in ALLOWED_HOSTS:
        return PlainTextResponse('forbidden origin', status_code=403)
    return await call_next(request)


# ---------- API ----------

def normalize_options(raw):
    cfg = load_config()
    engine = raw.get('engine') if raw.get('engine') in ('mlx', 'cpu') else (cfg.get('engine') or 'mlx')
    formats = [f for f in (raw.get('formats') or cfg.get('formats') or FORMATS) if f in FORMATS]
    return {
        'engine': engine,
        'model': str(raw.get('model') or (cfg.get('models') or {}).get(engine) or 'large-v3-turbo'),
        'language': str(raw.get('language') or cfg.get('language') or 'ru'),
        'formats': formats or list(FORMATS),
        'diarize': bool(raw.get('diarize')),
        'num_speakers': int(raw['num_speakers']) if str(raw.get('num_speakers') or '').isdigit() else None,
        'force': bool(raw.get('force')),
    }


def new_job(name, source, out_dir, origin, options):
    jid = uuid.uuid4().hex[:10]
    j = {'id': jid, 'run': uuid.uuid4().hex[:8], 'name': name, 'source': str(source), 'out_dir': str(out_dir),
         'origin': origin, 'options': dict(options), 'status': 'queued', 'progress': 0, 'stage_name': None, 'eta': None,
         'created': time.time(), 'started': None, 'finished': None, 'result': None, 'error': None}
    jobs[jid] = j
    return j


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/config')
def get_config():
    cfg = load_config()
    return {'engine': cfg.get('engine') or 'mlx', 'models': cfg.get('models') or {},
            'language': cfg.get('language') or 'ru', 'formats': cfg.get('formats') or list(FORMATS),
            'all_formats': list(FORMATS)}


@app.post('/api/jobs/upload')
def add_uploads(files: list[UploadFile] = File(...), options: str = Form('{}')):
    # обычный def: FastAPI выполняет его в пуле потоков, копирование больших файлов не блокирует сервер
    opts = normalize_options(json.loads(options or '{}'))
    created = []
    for f in files:
        jid_dir = UPLOADS / uuid.uuid4().hex[:10]
        jid_dir.mkdir(parents=True, exist_ok=True)
        name = Path(f.filename or '').name
        if name in ('', '.', '..'):
            name = 'file'
        dst = jid_dir / name
        with open(dst, 'wb') as out:
            shutil.copyfileobj(f.file, out, 1 << 20)
        with lock:
            j = new_job(name, dst, OUTPUT / jid_dir.name, 'upload', opts)
            created.append(public(j))
    with lock:
        save_state()
    wake.set()
    return created


@app.post('/api/jobs/paths')
async def add_paths(request: Request):
    body = await request.json()
    opts = normalize_options(body.get('options') or {})
    raw = [p for p in (body.get('paths') or []) if str(p).strip()]
    missing = [p for p in raw if not Path(os.path.expanduser(str(p).strip())).exists()]
    if missing:
        raise HTTPException(400, 'не найдено: ' + ', '.join(missing))
    files = expand_inputs([str(p).strip() for p in raw])
    if not files:
        raise HTTPException(400, 'медиафайлы не найдены')
    stems = assign_stems(files, {})  # a.mp3 и a.mp4 в одной папке не перезаписывают друг друга
    created = []
    with lock:
        for f in files:
            out = f.parent if body.get('save_next', True) else OUTPUT / ('p_' + uuid.uuid4().hex[:8])
            j = new_job(f.name, f, out, 'path', opts)
            j['stem'] = stems.get(f)
            created.append(public(j))
        save_state()
    wake.set()
    return created


@app.get('/api/jobs')
def list_jobs():
    with lock:
        return [public(j) for j in sorted(jobs.values(), key=lambda x: x['created'], reverse=True)]


def get_job(jid):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, 'задание не найдено')
    return j


@app.get('/api/jobs/{jid}/result')
def job_result(jid: str):
    j = get_job(jid)
    files = (j.get('result') or {}).get('files') or {}
    if not files.get('json') or not Path(files['json']).exists():
        raise HTTPException(404, 'результата нет')
    segs = json.loads(Path(files['json']).read_text(encoding='utf-8'))
    return {'segments': segs, 'files': {k: Path(v).name for k, v in files.items() if Path(v).exists()},
            'result': j['result']}


@app.get('/api/jobs/{jid}/download/{fmt}')
def download(jid: str, fmt: str):
    j = get_job(jid)
    path = ((j.get('result') or {}).get('files') or {}).get(fmt)
    if not path or not Path(path).exists():
        raise HTTPException(404, 'файла нет')
    return FileResponse(path, media_type=MIME.get(fmt, 'application/octet-stream'), filename=Path(path).name)


@app.post('/api/jobs/{jid}/reveal')
def reveal(jid: str):
    j = get_job(jid)
    files = (j.get('result') or {}).get('files') or {}
    target = files.get('txt') or files.get('json') or j['source']
    subprocess.run(['open', '-R', target], timeout=10)
    return {'ok': True}


@app.post('/api/jobs/{jid}/retry')
def retry(jid: str):
    with lock:
        j = get_job(jid)
        if j['status'] in ('queued', 'running') or jid in procs:
            raise HTTPException(409, 'задание ещё в работе или останавливается, попробуйте через пару секунд')
        j['options'] = dict(j['options'], force=True)
        j.update(status='queued', run=uuid.uuid4().hex[:8], progress=0, error=None, result=None, created=time.time())
        save_state()
    wake.set()
    return public(j)


@app.delete('/api/jobs/{jid}')
def delete_job(jid: str):
    with lock:
        j = get_job(jid)
        if j['status'] in ('queued', 'running'):
            j.update(status='cancelled', stage_name=None, eta=None)
            p = procs.get(jid)
            if p and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        else:
            jobs.pop(jid)
            # исходник, загруженный через браузер, больше не нужен (результаты в output/ остаются)
            src_dir = Path(j['source']).parent
            if j['origin'] == 'upload' and src_dir.parent == UPLOADS:
                shutil.rmtree(src_dir, ignore_errors=True)
        save_state()
    return {'ok': True}


@app.get('/api/terms')
def get_terms():
    return {'text': TERMS_PATH.read_text(encoding='utf-8') if TERMS_PATH.exists() else ''}


@app.put('/api/terms')
async def put_terms(request: Request):
    body = await request.json()
    TERMS_PATH.write_text(str(body.get('text', '')).replace('\r\n', '\n'), encoding='utf-8')
    return {'ok': True}


load_state()
threading.Thread(target=worker, daemon=True).start()
wake.set()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')
