# tools/comercial/cn.py
from __future__ import annotations

import io
import re
from typing import List, Optional

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
# Helpers
# =========================================================
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _key(s: str) -> str:
    return _norm_col(s).lower().replace("_", " ")


ALIASES = {
    "Fecha": ["fecha", "fec", "date"],
    "Cuenta": ["cuenta", "cta", "account"],
    "Producto": ["producto", "product"],
    "Neto Agente": ["neto agente", "neto"],
    "Gross Agente": ["gross agente", "gross"],
    "Id_Off": ["id off", "id_off", "id of"],
    "MANAGER": ["manager", "nombre manager"],
    "OFICIAL": ["oficial", "nombre oficial"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    key_map = {_key(c): c for c in cols}

    rename = {}
    for canonical, aliases in ALIASES.items():
        # pruebo canonical + aliases
        for cand in [canonical] + aliases:
            k = _key(cand)
            if k in key_map:
                rename[key_map[k]] = canonical
                break

    if rename:
        df = df.rename(columns=rename)

    return df


def _parse_fecha(df: pd.DataFrame) -> pd.DataFrame:
    if "Fecha" not in df.columns:
        return df

    fechas_ar = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    if float(fechas_ar.isna().mean()) > 0.4:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=False)
    else:
        df["Fecha"] = fechas_ar
    return df


def _comma_to_dot_number(x: pd.Series) -> pd.Series:
    """
    Regla SIMPLE:
    - tomar como texto
    - reemplazar ',' por '.'
    - convertir a número
    """
    s = x.astype(str).str.strip()
    s = s.str.replace("\u00a0", "", regex=False)  # NBSP
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    # si no existe / no se lee -> ignorar
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    # si faltan columnas mínimas -> ignorar
    for c in OUTPUT_COLS:
        if c not in df.columns:
            return None

    df = df[OUTPUT_COLS].copy()

    # fecha robusta
    df = _parse_fecha(df)

    # texto limpio
    df["Cuenta"] = df["Cuenta"].astype(str).str.strip()
    df["MANAGER"] = df["MANAGER"].astype(str).str.strip()
    df["OFICIAL"] = df["OFICIAL"].astype(str).str.strip()

    # ✅ decimal simple
    df["Neto Agente"] = _comma_to_dot_number(df["Neto Agente"])
    df["Gross Agente"] = _comma_to_dot_number(df["Gross Agente"])

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
        return

    dfs: List[pd.DataFrame] = []
    for s in SHEETS:
        one = _read_one_sheet(xls, s)
        if one is not None:
            dfs.append(one)

    if not dfs:
        return

    df_all = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    st.download_button(
        "Excel",
        data=_to_excel_bytes(df_all),
        file_name="cn_bancos_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("### Consolidado")
    try:
        st.dataframe(df_all, use_container_width=True, height=620, hide_index=True)
    except TypeError:
        st.dataframe(df_all, use_container_width=True, height=620)
