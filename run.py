"""
run.py  —  project-root entry point for the NER Studio Flask app.

Usage (from the project root):
    python run.py

Or via the virtualenv interpreter:
    .venv/bin/python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the templates/ directory is importable as a plain module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from templates.app import app  # noqa: E402

if __name__ == "__main__":
    # use_reloader=False avoids the signal-handler crash that occurs when
    # Flask's reloader spawns a background thread in Python 3.11+.
    app.run(host="127.0.0.1", port=5003, debug=True, use_reloader=False)
