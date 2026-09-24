from __future__ import annotations
import os, subprocess, sys

os.environ.setdefault("SPORTS_EDGE_MODE", "on_demand")
raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"]))
