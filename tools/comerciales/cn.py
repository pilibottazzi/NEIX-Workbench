# tools/comercial/cn.py
from __future__ import annotations

import io
from pathlib import Path
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

TEMPLATE_PATH = Path("data") / "Capital N - herramienta de datos.xlsx"


# =========================
# UI CSS
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


# =========================
# Helpers
# =========================
def _read_template_bytes() -> bytes | None:
    if not TEMPLATE_PATH.exists():
        return None
    return TEMPLATE_PATH.read_bytes()


def _normalize_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia espacios y transforma strings vacíos en NA.
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                    "NaN": pd.NA,
                    "NAN": pd.NA,
                }
            )

    return df


def _parse_fecha(series: pd.Series) -> pd.Series:
    """
    Convierte la fecha a datetime sin hora.
    Intenta primero formato día/mes/año.
    """
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt.dt.normalize()


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas vacías del consolidado.
    Regla:
    - si todos los campos de negocio están vacíos => se elimina
    - además, exige que al menos uno de los campos principales tenga dato
    """
    df = df.copy()

    business_cols = [c for c in OUTPUT_COLS if c in df.columns]

    # Eliminar filas donde TODO lo relevante está vacío
    df = df.dropna(subset=business_cols, how="all").copy()

    # Refuerzo: exigir que exista al menos alguno de estos campos clave
    key_cols = [c for c in ["Fecha", "Cuenta", "Producto", "Neto Agente", "Gross Agente"] if c in df.columns]
    if key_cols:
        mask_has_key_data = df[key_cols].notna().any(axis=1)
        df = df.loc[mask_has_key_data].copy()

    return df


def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    df.columns = df.columns.str.strip()

    missing = [c for c in OUTPUT_COLS if c not in df.columns]
    if missing:
        st.warning(f"{sheet_name} falta columnas: {missing}")
        return None

    df = df[OUTPUT_COLS].copy()
    df = _normalize_empty_strings(df)
    df = _drop_empty_rows(df)

    if df.empty:
        return None

    if "Fecha" in df.columns:
        df["Fecha"] = _parse_fecha(df["Fecha"])

    df.insert(0, "Banco", sheet_name)

    return df


def _coerce_ar_number_to_float(series: pd.Series) -> pd.Series:
    """
    Convierte números que pueden venir como:
      - "1.234,56" (AR)
      - "1234,56"
      - "1234.56" (EN)
      - "1,234.56" (US)
      - con espacios / $ etc
    a float. Lo que no pueda, queda NaN.
    """
    s = series.astype(str).str.strip()

    s = s.replace({"": None, "None": None, "nan": None, "NaN": None})

    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("USD", "", regex=False)
    s = s.str.replace("ARS", "", regex=False)

    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)

    out = s.copy()

    both = has_comma & has_dot
    if both.any():
        last_comma = out[both].str.rfind(",")
        last_dot = out[both].str.rfind(".")
        ar_mask = last_comma > last_dot
        us_mask = ~ar_mask

        idx_ar = out[both].index[ar_mask]
        out.loc[idx_ar] = (
            out.loc[idx_ar]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        idx_us = out[both].index[us_mask]
        out.loc[idx_us] = out.loc[idx_us].str.replace(",", "", regex=False)

    only_comma = has_comma & ~has_dot
    if only_comma.any():
        out.loc[only_comma] = out.loc[only_comma].str.replace(",", ".", regex=False)

    return pd.to_numeric(out, errors="coerce")


def _prepare_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _normalize_empty_strings(df)
    df = _drop_empty_rows(df)

    if "Fecha" in df.columns:
        df["Fecha"] = _parse_fecha(df["Fecha"])

    return df.reset_index(drop=True)


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Exporta el consolidado a Excel dejando:
    - Fecha como FECHA real sin hora
    - Neto/Gross como NÚMEROS
    """
    df = _prepare_final_dataframe(df)

    num_cols = ["Neto Agente", "Gross Agente"]

    for c in num_cols:
        if c in df.columns:
            df[c] = _coerce_ar_number_to_float(df[c])

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Consolidado")

        ws = writer.sheets["Consolidado"]

        # Formato fecha sin hora
        if "Fecha" in df.columns:
            fecha_idx = df.columns.get_loc("Fecha") + 1
            for col_cells in ws.iter_cols(min_col=fecha_idx, max_col=fecha_idx, min_row=2):
                for cell in col_cells:
                    cell.number_format = "dd/mm/yyyy"

        # Formato número
        for col_name in num_cols:
            if col_name in df.columns:
                col_idx = df.columns.get_loc(col_name) + 1
                for col_cells in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                    for cell in col_cells:
                        cell.number_format = "#,##0.00"

    bio.seek(0)
    return bio.read()


# =========================
# RENDER
# =========================
def render(back_to_home=None) -> None:
    _inject_css()

    template_bytes = _read_template_bytes()

    if template_bytes:
        st.download_button(
            "Descargar template para completar",
            data=template_bytes,
            file_name="Capital N - herramienta de datos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.warning("No encontré el template en /data")

    st.divider()

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
        if one is not None and not one.empty:
            dfs.append(one)

    if not dfs:
        st.warning("No encontré hojas válidas.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = _prepare_final_dataframe(df_all)

    st.download_button(
        "Excel consolidado",
        data=_to_excel_bytes(df_all),
        file_name="cn_bancos_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("### Consolidado")

    df_view = df_all.copy()
    if "Fecha" in df_view.columns:
        df_view["Fecha"] = pd.to_datetime(df_view["Fecha"], errors="coerce").dt.strftime("%d/%m/%Y")
        df_view["Fecha"] = df_view["Fecha"].fillna("")

    st.dataframe(df_view, use_container_width=True, height=620)
