# tools/comercial/cn.py
from __future__ import annotations

import io
import re
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st


# =========================================================
# Config
# =========================================================
SHEETS = ["WSC A", "WSC B", "INSIGNEO"]

OUTPUT_COLS = [
    "Fecha",
    "Cuenta",
    "Producto",
    "Neto Agente",
    "Gross Agente",
    "Id_Off",
    "MANAGER",
    "OFICIAL",
]

NEIX_RED = "#ff3b30"


# =========================================================
# UI
# =========================================================
def _inject_css() -> None:
    st.markdown(
        f"""
<style>
  .block-container {{
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
  }}

  /* Botón Excel full ancho */
  div[data-testid="stDownloadButton"] {{
    width: 100% !important;
  }}
  div[data-testid="stDownloadButton"] > button {{
    width: 100% !important;
    background: {NEIX_RED} !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    padding: 0.95rem 1rem !important;
    border: 0 !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers
# =========================================================
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _key(s: str) -> str:
    return _norm_col(s).lower().replace("_", " ")


def _clean_text_series(x: pd.Series) -> pd.Series:
    """Limpieza mínima segura (no toca comas/puntos ni fechas)."""
    s = x.astype(str)
    s = s.str.replace("\u00a0", " ", regex=False)  # NBSP
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.replace({"nan": "", "None": "", "NaT": ""})


# =========================================================
# Column aliases (simple)
# =========================================================
ALIASES: Dict[str, List[str]] = {
    "Fecha": ["fecha", "fec", "date"],
    "Cuenta": ["cuenta", "cta", "account"],
    "Producto": ["producto", "product"],
    "Neto Agente": ["neto agente", "neto"],
    "Gross Agente": ["gross agente", "gross"],
    "Id_Off": ["id off", "id_off", "idoff", "id oficial", "id_oficial"],
    "MANAGER": ["manager", "nombre manager", "managernombre", "manager nombre"],
    "OFICIAL": ["oficial", "nombre oficial", "oficialnombre", "oficial nombre"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    key_map = {_key(c): c for c in cols}

    rename: Dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for cand in [canonical] + aliases:
            k = _key(cand)
            if k in key_map:
                rename[key_map[k]] = canonical
                break

    return df.rename(columns=rename) if rename else df


# =========================================================
# Read sheet
# =========================================================
def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        # dtype=str: NO tocamos decimales ni fechas
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    # Validación mínima
    if any(c not in df.columns for c in OUTPUT_COLS):
        return None

    df = df[OUTPUT_COLS].copy()

    # ✅ Fecha: DEJAR TAL CUAL VIENE (solo limpieza de espacios raros, sin parsear)
    df["Fecha"] = _clean_text_series(df["Fecha"])

    # Limpieza mínima en el resto (sin tocar coma/punto)
    for c in ["Cuenta", "Producto", "Id_Off", "MANAGER", "OFICIAL", "Neto Agente", "Gross Agente"]:
        df[c] = _clean_text_series(df[c])

    df.insert(0, "Banco", sheet_name)
    return df


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Consolidado")
    bio.seek(0)
    return bio.read()


# =========================================================
# Entrypoint
# =========================================================
def render(back_to_home=None) -> None:
    _inject_css()

    up = st.file_uploader(
        "CN: Subí el Excel para consolidar bancos",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="cn_bancos_uploader",
    )
    if not up:
        return

    try:
        xls = pd.ExcelFile(io.BytesIO(up.getvalue()))
    except Exception:
        st.error("No pude leer el archivo. Probá guardarlo como .xlsx y re-subirlo.")
        return

    dfs: List[pd.DataFrame] = []
    for s in SHEETS:
        one = _read_one_sheet(xls, s)
        if one is not None:
            dfs.append(one)

    if not dfs:
        st.warning("No encontré hojas válidas con las columnas esperadas.")
        return

    df_all = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    st.download_button(
        "Excel",
        data=_to_excel_bytes(df_all),
        file_name="cn_bancos_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("### Consolidado")
    try:
        st.dataframe(df_all, use_container_width=True, height=620, hide_index=True)
    except TypeError:
        st.dataframe(df_all, use_container_width=True, height=620)
