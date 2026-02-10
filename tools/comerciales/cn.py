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

# "Agente" eliminado
REQUIRED_COLS = [
    "Fecha",
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


def _parse_fecha_por_hoja(df: pd.DataFrame, col: str = "Fecha") -> pd.DataFrame:
    """
    Excel guarda fechas como seriales o strings, y el formato (DD/MM vs MM/DD)
    puede variar por hoja. Esto fuerza un parseo robusto por hoja.

    Estrategia:
    - Probar dayfirst=True (AR).
    - Si queda mucha NaT, probar dayfirst=False (US).
    """
    df = df.copy()

    # Si viene como datetime/serial, pandas normalmente lo convierte bien.
    # Si viene como string ambiguo, esto lo resuelve por hoja.
    fechas_ar = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    ratio_nat = float(fechas_ar.isna().mean())

    if ratio_nat > 0.4:
        fechas_us = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
        df[col] = fechas_us
    else:
        df[col] = fechas_ar

    return df


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

    # Parseo de fecha robusto por hoja (AR vs US)
    df = _parse_fecha_por_hoja(df, "Fecha")

    df.insert(0, "Banco", sheet_name)
    return df


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    # datetime_format define cómo se VE en el Excel final
    with pd.ExcelWriter(bio, engine="openpyxl", datetime_format="DD/MM/YYYY") as writer:
        df.to_excel(writer, index=False, sheet_name="Consolidado")
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

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    up = st.file_uploader(
        "CN: Subí el Excel para consolidar bancos",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="cn_bancos_uploader",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if not up:
        return

    try:
        data = up.getvalue()
        xls = pd.ExcelFile(io.BytesIO(data))
        available = list(xls.sheet_names)

        missing_sheets = [s for s in SHEETS if s not in available]
        if missing_sheets:
            st.error(f"Faltan hojas en el Excel: {missing_sheets}. Hojas detectadas: {available}")
            return

        dfs: Dict[str, pd.DataFrame] = {s: _read_one_sheet(xls, s) for s in SHEETS}
        df_all = pd.concat([dfs[s] for s in SHEETS], ignore_index=True)

        # Evita el "contador" del preview y asegura índice limpio
        df_all = df_all.reset_index(drop=True)

        # Descargar arriba (solo 1 hoja) y texto "Excel"
        st.download_button(
            "Excel",
            data=_to_excel_bytes(df_all),
            file_name="cn_bancos_consolidado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("### Consolidado")
        # hide_index depende de versión de streamlit; fallback incluido
        try:
            st.dataframe(df_all, use_container_width=True, height=620, hide_index=True)
        except TypeError:
            st.dataframe(df_all, use_container_width=True, height=620)

    except Exception as e:
        st.error("Se rompió el procesamiento.")
        st.exception(e)

