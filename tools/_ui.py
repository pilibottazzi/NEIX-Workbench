"""
Shared UI constants and CSS for all workbench tools.

Usage in each tool:
    from tools._ui import inject_tool_css, NEIX_PRIMARY, TEXT, MUTED, BORDER, CARD_BG
"""
from __future__ import annotations

import streamlit as st

# =========================================================
# PALETTE — NEIX brand
# =========================================================
NEIX_PRIMARY = "#b91c1c"
NEIX_RED = NEIX_PRIMARY  # backward compat alias
TEXT = "#0f172a"
MUTED = "#64748b"
BORDER = "rgba(15,23,42,0.08)"
BORDER2 = "rgba(15,23,42,0.05)"
CARD_BG = "#ffffff"
SOFT_BG = "#f9fafb"
APP_BG = "#faf8f5"


def inject_tool_css(*, max_width: str = "1180px") -> None:
    """Inject the shared NEIX tool CSS. Call once at the top of each render()."""
    st.markdown(
        f"""
<style>
  /* ============================
     LAYOUT
     ============================ */
  .block-container {{
    max-width: {max_width};
    padding-top: .8rem;
    padding-bottom: 1.6rem;
  }}

  /* ============================
     TYPOGRAPHY
     ============================ */
  h1 {{
    color: {TEXT} !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    margin-bottom: 2px !important;
    margin-top: 0 !important;
  }}
  h2 {{
    color: {TEXT} !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    margin-bottom: 2px !important;
    margin-top: 0 !important;
  }}
  h3 {{
    color: {TEXT} !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    margin-bottom: 2px !important;
  }}
  .stCaption {{
    color: {MUTED} !important;
    font-size: .85rem !important;
  }}

  /* ============================
     CARDS
     ============================ */
  .neix-card {{
    border: 1px solid {BORDER};
    background: {CARD_BG};
    border-radius: 12px;
    padding: 14px 16px 12px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    margin-top: 8px;
    margin-bottom: 8px;
  }}

  /* ============================
     LABELS
     ============================ */
  label {{
    font-weight: 600 !important;
    color: {TEXT} !important;
    font-size: .85rem !important;
  }}

  /* ============================
     FILE UPLOADERS
     ============================ */
  [data-testid="stFileUploader"] {{
    border: 1px solid {BORDER};
    background: {SOFT_BG};
    border-radius: 10px;
    padding: 6px 10px;
  }}
  [data-testid="stFileUploader"] section {{
    padding: 0;
  }}
  [data-testid="stFileUploaderDropzone"] {{
    border: 0 !important;
    background: transparent !important;
    padding: 6px 8px !important;
  }}
  [data-testid="stFileUploader"] small {{
    font-size: .78rem !important;
  }}

  /* ============================
     BUTTONS — uniformes y compactos
     ============================ */
  div.stButton > button {{
    background: {NEIX_PRIMARY} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.45rem 0.9rem !important;
    min-height: 38px !important;
    height: 38px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    width: 100% !important;
    box-shadow: 0 2px 6px rgba(185,28,28,0.12) !important;
    transition: all .12s ease !important;
  }}
  div.stButton > button:hover {{
    filter: brightness(1.08) !important;
    box-shadow: 0 3px 10px rgba(185,28,28,0.18) !important;
  }}
  div.stButton > button:disabled {{
    opacity: 0.5 !important;
    box-shadow: none !important;
  }}

  /* ============================
     DOWNLOAD BUTTONS
     ============================ */
  div[data-testid="stDownloadButton"] > button {{
    width: 100% !important;
    background: {NEIX_PRIMARY} !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    padding: 0.45rem 0.9rem !important;
    min-height: 38px !important;
    height: 38px !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(185,28,28,0.12) !important;
  }}

  /* ============================
     DATA DISPLAY
     ============================ */
  div[data-testid="stAlert"] {{ border-radius: 10px !important; }}

  div[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid {BORDER};
  }}
  div[data-testid="stExpander"] {{
    border-radius: 10px;
    border: 1px solid {BORDER};
    overflow: hidden;
  }}

  /* ============================
     INPUTS
     ============================ */
  .stTextArea textarea, .stTextInput input {{
    border-radius: 10px !important;
    font-size: .88rem !important;
  }}
  .stSelectbox > div > div {{
    border-radius: 10px !important;
  }}

  /* ============================
     STATUS PILLS
     ============================ */
  .pill {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 11px; border-radius: 8px;
    border: 1px solid {BORDER}; font-weight: 600; font-size: .84rem;
    width: fit-content; background: {SOFT_BG}; margin: 4px 0;
  }}
  .pill.ok  {{ border-color: rgba(22,163,74,0.30);  background: rgba(22,163,74,0.08); color: #15803d; }}
  .pill.warn {{ border-color: rgba(217,119,6,0.30); background: rgba(217,119,6,0.08); color: #b45309; }}
  .pill.bad  {{ border-color: rgba(185,28,28,0.30);  background: rgba(185,28,28,0.08); color: #991b1b; }}

  /* ============================
     SPACING REDUCTIONS
     ============================ */
  hr {{ border: 0; border-top: 1px solid {BORDER2}; margin: 10px 0; }}

  div[data-testid="stVerticalBlock"] > div {{
    gap: 0.4rem !important;
  }}

  [data-testid="stFileUploader"] {{
    margin-top: 0 !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


def tag(msg: str, kind: str = "ok") -> None:
    """Render a colored status pill."""
    st.markdown(f'<div class="pill {kind}">{msg}</div>', unsafe_allow_html=True)
