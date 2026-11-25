"""Small dependency-free progress display + machine-readable heartbeat."""
import json
import sys
import time
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return '--:--'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class Progress:
    def __init__(self, output, variant='data', quiet=False):
        self.output = Path(output)
        self.variant = variant
        self.quiet = quiet
        self.started = time.monotonic()
        self.last_line = False

    def update(self, stage, done, total, **metrics):
        elapsed = time.monotonic() - self.started
        stage_elapsed = metrics.pop('stage_elapsed', elapsed)
        remaining = stage_elapsed / done * (total - done) if done and total else None
        ratio = min(1., done / total) if total else 0.
        value = {'variant': self.variant, 'stage': stage, 'done': done, 'total': total,
                 'percent': ratio * 100, 'elapsed_seconds': elapsed,
                 'eta_stage_seconds': remaining, 'heartbeat_unix': time.time(), **metrics}
        save_json(self.output / 'progress.json', value)
        if self.quiet:
            return
        bar = '#' * int(ratio * 24) + '-' * (24 - int(ratio * 24))
        detail = ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                          for k, v in metrics.items())
        line = f'[{self.variant}] {stage} [{bar}] {done}/{total} {ratio:.0%} ETA {duration(remaining)} {detail}'
        if sys.stdout.isatty():
            print('\r' + line + ' ' * 8, end='', flush=True)
            self.last_line = True
        else:
            print(line, flush=True)

    def message(self, text):
        if self.last_line:
            print(); self.last_line = False
        if not self.quiet:
            print(f'[{self.variant}] {text}', flush=True)
