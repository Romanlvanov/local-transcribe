"""Иконка расшифровки в строке меню macOS.

Значок показывает, запущен ли веб-интерфейс (заполненная волна - работает, контур - остановлен),
а во время расшифровки рядом виден процент. Из меню: открыть интерфейс, запустить/остановить сервер,
открыть папку результатов. Сервер, запущенный отдельно через transcribe-ui.sh, тоже распознаётся.
Запуск: ./transcribe-menubar.sh
"""
import json
import os
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

import AppKit
import rumps

BASE = Path(__file__).resolve().parent
PORT = int(os.environ.get('TRANSCRIBE_PORT', '8765'))
URL = 'http://127.0.0.1:%d' % PORT
SERVER = BASE / 'web' / 'server.py'
PYTHON = BASE / '.venv' / 'bin' / 'python'
LOG = BASE / 'output' / 'server.log'


def api(path, timeout=0.7):
    try:
        with urllib.request.urlopen(URL + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def server_pids():
    out = subprocess.run(['pgrep', '-f', str(SERVER)], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.strip() and int(p) != os.getpid()]


def symbol(name):
    img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, 'Расшифровка')
    img.setTemplate_(True)  # сам подстраивается под светлую/тёмную строку меню
    return img


class MenuBar(rumps.App):
    def __init__(self):
        super().__init__('Расшифровка', title=None, quit_button=None)
        self.status_item = rumps.MenuItem('Проверка...')
        self.job_item = rumps.MenuItem('')
        self.open_item = rumps.MenuItem('Открыть интерфейс', callback=self.open_ui)
        self.toggle_item = rumps.MenuItem('Запустить сервер', callback=self.toggle)
        self.menu = [self.status_item, self.job_item, None, self.open_item, self.toggle_item,
                     rumps.MenuItem('Папка с результатами', callback=self.open_output), None,
                     rumps.MenuItem('Выйти (сервер остановится)', callback=self.quit_all)]
        self.running = None
        self.starting = False
        self.icons = {True: symbol('waveform.circle.fill'), False: symbol('waveform.circle')}
        self.timer = rumps.Timer(self.refresh, 2)
        self.timer.start()

    # ---- состояние ----
    def set_icon(self, running, text):
        item = self._nsapp.nsstatusitem
        item.button().setImage_(self.icons[running])
        item.button().setImagePosition_(AppKit.NSImageLeft)
        item.setTitle_(text)

    def refresh(self, _=None):
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        jobs = api('/api/jobs')
        running = jobs is not None
        if running:
            self.starting = False
        active = [j for j in (jobs or []) if j['status'] == 'running']
        queued = [j for j in (jobs or []) if j['status'] == 'queued']
        text = ''
        if active:
            j = active[0]
            pct = int((j.get('progress') or 0) * 100)
            text = ' %d%%' % pct
            eta = j.get('eta')
            line = '%s: %d%%, %s' % (j['name'], pct, j.get('stage_name') or '')
            if eta:
                line += ', ~%d:%02d' % divmod(int(eta), 60)
            if queued:
                line += ' (ещё в очереди: %d)' % len(queued)
            self.job_item.title = line
        else:
            self.job_item.title = ('В очереди: %d' % len(queued)) if queued else ''
        self.job_item._menuitem.setHidden_(not self.job_item.title)  # строка задания видна только при работе
        self.set_icon(running, text)
        if self.starting and not running:
            self.status_item.title = 'Сервер запускается...'
        else:
            self.status_item.title = ('Сервер работает: ' + URL) if running else 'Сервер остановлен'
        self.toggle_item.title = 'Остановить сервер' if running else 'Запустить сервер'
        self.open_item.title = 'Открыть интерфейс' if running else 'Запустить и открыть интерфейс'
        self.running = running

    # ---- действия ----
    def start_server(self):
        if self.running or self.starting:
            return
        LOG.parent.mkdir(exist_ok=True)
        log = open(LOG, 'a')
        subprocess.Popen([str(PYTHON), str(SERVER)], cwd=str(BASE), stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        self.starting = True
        self.refresh()

    def stop_server(self):
        busy = [j for j in (api('/api/jobs') or []) if j['status'] == 'running']
        if busy and rumps.alert('Остановить сервер?', 'Сейчас идёт расшифровка «%s». Она прервётся и при '
                                'следующем запуске начнётся заново.' % busy[0]['name'],
                                ok='Остановить', cancel='Отмена') != 1:
            return False
        for pid in server_pids():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.refresh()
        return True

    def toggle(self, _):
        if self.running:
            self.stop_server()
        else:
            self.start_server()

    def open_ui(self, _):
        if not self.running:
            self.start_server()
            self._open_when_ready = 20  # ~10 секунд ждём, пока сервер поднимется
            rumps.Timer(self._open_later, 0.5).start()
        else:
            subprocess.run(['open', URL])

    def _open_later(self, timer):
        self._open_when_ready -= 1
        if api('/api/config') is not None:
            timer.stop()
            subprocess.run(['open', URL])
        elif self._open_when_ready <= 0:
            timer.stop()
            rumps.alert('Сервер не запустился', 'Подробности в журнале: %s' % LOG)

    def open_output(self, _):
        (BASE / 'output').mkdir(exist_ok=True)
        subprocess.run(['open', str(BASE / 'output')])

    def quit_all(self, _):
        if self.running and not self.stop_server():
            return
        rumps.quit_application()


if __name__ == '__main__':
    others = [p for p in subprocess.run(['pgrep', '-f', str(BASE / 'menubar.py')], capture_output=True,
                                        text=True).stdout.split() if int(p) != os.getpid()]
    if others:
        print('Иконка уже запущена', file=sys.stderr)
        sys.exit(0)
    MenuBar().run()
