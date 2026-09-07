"""Standalone runner for the structured risk-screening page.

The canonical entry point is ``app/streamlit_app.py`` (multipage navigation).
This file exists so the page can also be launched directly during development
with the same chrome: page config, language switch, app header, and sidebar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.streamlit_app import (
    render_app_header,
    render_language_control,
    require_access_code,
    screening_sidebar,
    screening_tab,
    set_page_config,
)

set_page_config()
require_access_code()
render_language_control()
render_app_header()
use_real_llm, use_lightrag, top_k = screening_sidebar()
screening_tab(use_real_llm, use_lightrag, top_k)
