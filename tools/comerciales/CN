# tools/comercial/cn.py
from __future__ import annotations

import io
import re
from typing import Dict, List

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
SHEETS = ["WSC A", "WSC B", "INSIGNEO"]

REQUIRED_COLS = [
    "Fecha",
    "Agente",
    "Producto",
    "Neto Agente",
    "Gross Agente",
    "Id_Off",
    "MANAGER",
    "OFICIAL",
]

NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "rgba(255,255,255,0.96)"


# =========================
# UI helpers
# =========================
def _inject_css() -> None:
    st.markdown(
        f"""
<style>
  .block-container {{
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
  }}

  .card {{
    background:{CARD_BG};
    border:1px solid {BORDER};
    border-radius: 16px;
    padding: 16px 16px 14px;
    box-shadow: 0 10px 30px rgba(17,24,39,0.05);
  }}

  .hint {{
    color:{MUTED};
    font-size:.92rem;
    margin-top:10px;
  }}

  /* Botón rojo NEIX */
  div.stDownloadButton > button,
  div.stButton > button {{
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    padding: 0.62rem 0.9rem !important;
    font-weight: 700 !important;
  }}
  div.stDownloadButton > button {{
    background: {NEIX_RED} !important;
    color: white !important;
  }}
  div.stDownloadButton > button:hover {{
    filter: brightness(0.97);
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# Data helpers
# =========================
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def _find_missing(required: List[str], cols: List[str]) -> List[str]:
    cols_set = set(cols)
    return [c for c in required if c not in cols_set]


def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df = _normalize_headers(df)

    missing = _find_missing(REQUIRED_COLS, list(df.columns))
    if missing:
        raise ValueError(
            f"Hoja '{sheet_name}': faltan columnas {missing}. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    df = df[REQUIRED_COLS].copy()
    df.insert(0, "Banco", sheet_name)

    return df


def _to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\*\?/\\:]", "-", name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe)
    bio.seek(0)
    return bio.read()


# =========================
# Tool entrypoint (Workbench)
# =========================
def render(back_to_home=None) -> None:
    """
    Tool CN (Comercial) - Consolidar hojas WSC A / WSC B / INSIGNEO.
    Integración Workbench: llamá cn.render(back_to_home=...)
    """
    _inject_css()

    st.markdown("## CN · Consolidar Bancos")
    st.caption("Comercial · WSC A / WSC B / INSIGNEO → 1 consolidado descargable")

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    up = st.file_uploader(
        "Subí el Excel (un solo archivo con 3 hojas)",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="cn_bancos_uploader",
    )

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        include_clean_sheets = st.checkbox(
            "Incluir también hojas individuales limpias en el Excel de salida",
            value=True,
        )
    with c2:
        st.caption("Columnas requeridas (en todas las hojas):")
        st.code(" | ".join(REQUIRED_COLS), language="text")

    st.markdown(
        '<div class="hint">Hojas esperadas: <b>WSC A</b>, <b>WSC B</b>, <b>INSIGNEO</b>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not up:
        return

    try:
        # más robusto en Streamlit Cloud
        data = up.getvalue()
        xls = pd.ExcelFile(io.BytesIO(data))
        available = list(xls.sheet_names)

        missing_sheets = [s for s in SHEETS if s not in available]
        if missing_sheets:
            st.error(f"Faltan hojas en el Excel: {missing_sheets}. Hojas detectadas: {available}")
            return

        dfs: Dict[str, pd.DataFrame] = {}
        for s in SHEETS:
            dfs[s] = _read_one_sheet(xls, s)

        df_all = pd.concat([dfs[s] for s in SHEETS], ignore_index=True)

        st.markdown("### Preview consolidado")
        st.dataframe(df_all, use_container_width=True, height=620)

        st.markdown("### Descargar")
        out_sheets: Dict[str, pd.DataFrame] = {"Consolidado": df_all}
        if include_clean_sheets:
            for s in SHEETS:
                out_sheets[s] = dfs[s]

        st.download_button(
            "⬇️ Descargar Excel consolidado",
            data=_to_excel_bytes(out_sheets),
            file_name="cn_bancos_consolidado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error("Se rompió el procesamiento.")
        st.exception(e)
