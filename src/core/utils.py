import os, pathlib, time, pytz, yaml
from datetime import datetime, timezone

_TZ_CACHE = {}

def get_tz(name: str):
    if name not in _TZ_CACHE:
        _TZ_CACHE[name] = pytz.timezone(name)
    return _TZ_CACHE[name]

def now_ms() -> int:
    return int(time.time() * 1000)

def ensure_dir(path: str) -> str:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def epoch_ms_to_local(ms: int, tz_name: str) -> str:
    tz = get_tz(tz_name)
    dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
