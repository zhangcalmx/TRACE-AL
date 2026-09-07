"""Shared project-root resolution that survives non-editable installs.

Source checkouts and editable installs find the project root by walking up
from this file. A regular (wheel-style) install lands in site-packages where
no project data lives; in that case we fall back to the process working
directory, because the Streamlit app and pipeline scripts are always launched
from the project root. An explicit environment override wins over both.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV_VAR = "ANASTOMOTIC_LEAK_RISKS_ROOT"


def project_root() -> Path:
    override = os.environ.get(ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "seed").is_dir() and (parent / "configs").is_dir():
            return parent
    return Path.cwd()
