from __future__ import annotations

import importlib
from types import ModuleType


def import_module_fresh(name: str) -> ModuleType | None:
    """Best-effort module refresh for long-lived app runtimes.

    Streamlit Community Cloud may reload an entrypoint before every dependency
    file from the same deploy has replaced the module already held in sys.modules.
    Prefer the freshly reloaded module when possible. If reload fails, keep the
    previously imported module so callers can resolve only the exports that
    still exist and fail soft on newer optional features.
    """
    try:
        module = importlib.import_module(name)
    except Exception:
        return None

    try:
        return importlib.reload(module)
    except Exception:
        return module
