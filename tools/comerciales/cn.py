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

# 👉 Mantener Id_Off + nombres (sin Id_manager / Id_oficial)
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
    s = _norm_col(s).lower().replace("_", " ")
    return re.sub(r"\s+", " ", s)


ALIASES = {
    "Fecha": ["fecha", "fec", "date"],
    "Cuenta": ["cuenta", "cta", "account"],
    "Producto": ["producto", "product"],
    "Neto Agente": ["neto agente", "neto", "net agent"],
    "Gross Agente": ["gross agente", "gross", "gross agent"],
    "Id_Off": ["id off", "id_off", "id oficial", "id of"],
    "MANAGER": ["manager", "nombre manager", "manager nombre"],
    "OFICIAL": ["oficial", "nombre oficial", "oficial nombre"],
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
    if "Fecha" not in df.columns:
        return df

    fechas_ar = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    if float(fechas_ar.isna().mean()) > 0.4:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=False)
    else:
        df["Fecha"] = fechas_ar
    return df


def _parse_decimal_coma_strict(series: pd.Series) -> pd.Series:
    """
    Regla:
    - Si viene como texto con coma: '82,3905' => 82.3905
    - Si viene sin coma y es todo dígitos y largo (ej '823905'): asumimos 4 decimales implícitos => /10000
    - Si ya es número razonable, queda
    """
    s = series.copy()

    # Trabajamos con representación textual "cruda"
    raw = s.astype(str).str.strip()

    # Normalizamos NBSP y espacios raros
    raw = raw.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)

    # Caso A: trae coma decimal -> parse AR
    has_comma = raw.str.contains(",", regex=False)

    # Convertimos AR: miles '.' y decimal ','
    raw_ar = (
        raw.where(has_comma, other=None)
        .dropna()
        .str.replace(".", "", regex=False)   # miles
        .str.replace(",", ".", regex=False)  # decimal
    )

    out = pd.to_numeric(raw_ar, errors="coerce")
    s_out = pd.Series(index=s.index, dtype="float64")
    s_out.loc[out.index] = out

    # Caso B: NO trae coma, NO trae punto, es todo dígitos y "largo" (ej 823905) -> 4 decimales implícitos
    mask_digits = (~has_comma) & (~raw.str.contains(r"\.", regex=True)) & raw.str.fullmatch(r"\d{5,}")
    if mask_digits.any():
        as_num = pd.to_numeric(raw.where(mask_digits), errors="coerce")
        # 4 decimales implícitos
        s_out.loc[mask_digits] = (as_num / 10000.0).astype("float64")

    # Caso C: resto -> intentar numérico directo (por si vino como float ya ok)
    mask_rest = s_out.isna()
    if mask_rest.any():
        s_out.loc[mask_rest] = pd.to_numeric(raw.where(mask_rest), errors="coerce")

    return s_out


def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    # si no existe o no se puede leer, se ignora SILENCIOSAMENTE
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=object)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    # si no tiene las columnas mínimas, se ignora SILENCIOSAMENTE
    for c in OUTPUT_COLS:
        if c not in df.columns:
            return None

    df = df[OUTPUT_COLS].copy()

    # Fecha robusta
    df = _parse_fecha(df)

    # Tipos texto
    df["Cuenta"] = df["Cuenta"].astype(str).str.strip()
    df["MANAGER"] = df["MANAGER"].astype(str).str.strip()
    df["OFICIAL"] = df["OFICIAL"].astype(str).str.strip()

    # ✅ Decimales: coma decimal SIEMPRE
    for col in ["Neto Agente", "Gross Agente"]:
        df[col] = _parse_decimal_coma_strict(df[col])

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
