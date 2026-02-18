# tools/comercial/cn.py
from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd
import streamlit as st


SHEETS = ["WSC A", "WSC B", "INSIGNEO"]

OUTPUT_COLS = [
    "Fecha",
    "Cuenta",
    "Producto",
    "Neto Agente",
    "Gross Agente",
    "Id_Off",
    "Id_manager",
    "MANAGER",
    "Id_oficial",
    "OFICIAL",
]

NEIX_RED = "#ff3b30"


def _inject_css() -> None:
    st.markdown(
        f"""
<style>
  .block-container {{
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
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


def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    # limpiar espacios en headers
    df.columns = df.columns.str.strip()

    # verificar que estén todas
    missing = [c for c in OUTPUT_COLS if c not in df.columns]
    if missing:
        st.warning(f"{sheet_name} falta columnas: {missing}")
        return None

    df = df[OUTPUT_COLS].copy()
    df.insert(0, "Banco", sheet_name)

    return df


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Consolidado")
    bio.seek(0)
    return bio.read()


def render(back_to_home=None) -> None:
    _inject_css()

    up = st.file_uploader(
        "CN: Subí el Excel para consolidar bancos",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
    )
    if not up:
        return

    try:
        xls = pd.ExcelFile(io.BytesIO(up.getvalue()))
    except Exception:
        st.error("No pude leer el archivo.")
        return

    dfs: List[pd.DataFrame] = []
    for s in SHEETS:
        one = _read_one_sheet(xls, s)
        if one is not None:
            dfs.append(one)

    if not dfs:
        st.warning("No encontré hojas válidas.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    st.download_button(
        "Excel",
        data=_to_excel_bytes(df_all),
        file_name="cn_bancos_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("### Consolidado")
    st.dataframe(df_all, use_container_width=True, height=620)
