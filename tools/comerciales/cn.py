# tools/comercial/cn.py
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================================================
# Config
# =========================================================
SHEETS = ["WSC A", "WSC B", "INSIGNEO"]

# 👉 Se mantiene Id_Off y SOLO nombres (sin Id_manager / Id_oficial)
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
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "rgba(255,255,255,0.96)"


# =========================================================
# UI helpers
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

  div.stDownloadButton > button {{
    width: 100% !important;
    background: {NEIX_RED} !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 0.9rem 1rem !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# Data helpers
# =========================================================
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _key(s: str) -> str:
    s = _norm_col(s).lower().replace("_", " ")
    return re.sub(r"\s+", " ", s)


ALIASES = {
    "Fecha": ["fecha", "fec", "date"],
    "Cuenta": ["cuenta", "cta", "account"],
    "Producto": ["producto", "product"],
    "Neto Agente": ["neto agente", "neto"],
    "Gross Agente": ["gross agente", "gross"],
    "Id_Off": ["id off", "id_off", "id oficial"],
    "MANAGER": ["manager", "nombre manager"],
    "OFICIAL": ["oficial", "nombre oficial"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    key_map = {_key(c): c for c in df.columns}

    rename = {}
    for canonical, aliases in ALIASES.items():
        for a in [canonical] + aliases:
            k = _key(a)
            if k in key_map:
                rename[key_map[k]] = canonical
                break

    if rename:
        df = df.rename(columns=rename)

    return df


def _parse_fecha(df: pd.DataFrame) -> pd.DataFrame:
    fechas_ar = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    if fechas_ar.isna().mean() > 0.4:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=False)
    else:
        df["Fecha"] = fechas_ar
    return df


def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    # si no tiene las columnas mínimas, se ignora silenciosamente
    for c in OUTPUT_COLS:
        if c not in df.columns:
            return None

    df = df[OUTPUT_COLS].copy()
    df = _parse_fecha(df)

    df["Cuenta"] = df["Cuenta"].astype(str).str.strip()
    df["MANAGER"] = df["MANAGER"].astype(str).str.strip()
    df["OFICIAL"] = df["OFICIAL"].astype(str).str.strip()

    df.insert(0, "Banco", sheet_name)
    return df


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl", datetime_format="DD/MM/YYYY") as writer:
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
    )

    if not up:
        return

    xls = pd.ExcelFile(io.BytesIO(up.getvalue()))

    dfs: List[pd.DataFrame] = []
    for s in SHEETS:
        df = _read_one_sheet(xls, s)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return

    df_all = pd.concat(dfs, ignore_index=True)

    st.download_button(
        "Excel",
        data=_to_excel_bytes(df_all),
        file_name="cn_bancos_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("### Consolidado")
    st.dataframe(df_all, use_container_width=True, height=620)

