"""Standalone runner for the evidence Q&A page.

The canonical entry point is ``app/streamlit_app.py`` (multipage navigation).
This file exists so the page can also be launched directly during development
with the same chrome: page config, language switch, and app header.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.streamlit_app import (
    chat_tab,
    render_app_header,
    render_language_control,
    require_access_code,
    set_page_config,
)

set_page_config()
require_access_code()
render_language_control()
render_app_header()
chat_tab()
