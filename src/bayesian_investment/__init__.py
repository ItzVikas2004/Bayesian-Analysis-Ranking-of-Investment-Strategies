"""Bayesian investment strategy research package."""

import os
from pathlib import Path

import platformdirs

# Scientific libraries use platform cache directories during import. Keep those
# writes local so the package also works in containers and restricted CI agents.
_cache = Path.cwd() / ".cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(_cache)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=FAST_COMPILE")


def _local_cache_dir(appname=None, appauthor=None, **_kwargs):
    """Return a writable application cache directory for sandboxed execution."""
    parts = [str(_cache)]
    if appname:
        parts.append(str(appname))
    if appauthor:
        parts.append(str(appauthor))
    return str(Path(*parts))


platformdirs.user_cache_dir = _local_cache_dir

__version__ = "1.0.0"
