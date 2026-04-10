# tools/mesa/filtro_especies_exterior.py
from __future__ import annotations

import logging
from io import BytesIO

import openpyxl
import pandas as pd
import streamlit as st

from tools._ui import inject_tool_css

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================
def _norm_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _find_required_columns(df: pd.DataFrame) -> dict[str, str]:
    cols_map = {str(c).strip().lower(): c for c in df.columns}

    required = {
        "fuente": "Fuente",
        "moneda": "Moneda",
        "csva": "CSVA",
        "precio": "Precio",
    }

    found: dict[str, str] = {}
    for key, original_name in required.items():
        if key not in cols_map:
            raise ValueError(f"No encontré la columna requerida: {original_name}")
        found[key] = cols_map[key]

    return found


def _build_output_excel(df_final: pd.DataFrame) -> BytesIO:
    buffer = BytesIO()
    df_final.to_excel(buffer, index=False)
    buffer.seek(0)

    wb = openpyxl.load_workbook(buffer)
    ws = wb.active
    ws.title = "Filtro Especies Exterior"

    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        for cell in row:
            cell.number_format = "0.00"

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 14

    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output


def process_excel_file(file) -> tuple[pd.DataFrame, BytesIO, dict]:

    df = pd.read_excel(file, header=1)

    if df.empty:
        raise ValueError("El archivo no tiene datos luego de leer la hoja con header=1.")

    cols = _find_required_columns(df)

    fuente_col = cols["fuente"]
    moneda_col = cols["moneda"]
    csva_col = cols["csva"]
    precio_col = cols["precio"]

    total_filas = len(df)

    fuente_norm = _norm_text(df[fuente_col]).str.lower()
    moneda_norm = _norm_text(df[moneda_col]).str.upper()

    df_filtrado = df[fuente_norm == "bbg"].copy()
    filas_bbg = len(df_filtrado)

    moneda_norm_filtrada = _norm_text(df_filtrado[moneda_col]).str.upper()
    df_filtrado = df_filtrado[moneda_norm_filtrada == "USD EXTERIOR"].copy()
    filas_usd_exterior = len(df_filtrado)

    df_final = df_filtrado[[csva_col, precio_col]].copy()
    df_final.columns = ["CSVA", "Precio"]

    df_final["CSVA"] = _norm_text(df_final["CSVA"])
    df_final["Precio"] = pd.to_numeric(df_final["Precio"], errors="coerce").round(2)

    df_final = df_final[df_final["CSVA"] != ""].copy()
    df_final = df_final.dropna(subset=["Precio"]).copy()
    df_final = df_final.drop_duplicates(subset=["CSVA"], keep="first").reset_index(drop=True)

    output = _build_output_excel(df_final)

    stats = {
        "total_filas": total_filas,
        "filas_bbg": filas_bbg,
        "filas_usd_exterior": filas_usd_exterior,
        "filas_finales": len(df_final),
    }

    return df_final, output, stats


# =========================================================
# RENDER
# =========================================================
def render() -> None:
    inject_tool_css()

    st.caption(
        "Filtra especies con **Fuente = BBG** y **Moneda = USD EXTERIOR**. "
        "Devuelve un Excel con columnas **CSVA** y **Precio**."
    )

    archivo = st.file_uploader(
        "Subí tu archivo Excel",
        type=["xlsx"],
        help="Se espera un archivo con encabezado real en la fila 2 del Excel.",
    )

    if not archivo:
        st.info("Esperando archivo para procesar.")
        return

    try:
        with st.spinner("Procesando archivo..."):
            df_final, output, stats = process_excel_file(archivo)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas leídas", f"{stats['total_filas']:,}")
        c2.metric("Fuente = BBG", f"{stats['filas_bbg']:,}")
        c3.metric("Moneda = USD EXTERIOR", f"{stats['filas_usd_exterior']:,}")
        c4.metric("Filas finales", f"{stats['filas_finales']:,}")

        st.markdown("### Vista previa")
        st.dataframe(df_final, use_container_width=True, hide_index=True)

        st.download_button(
            label="Descargar Excel filtrado",
            data=output,
            file_name="filtro_especies_exterior.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        logger.exception("Error procesando el archivo de filtro especies exterior")
        st.error(f"Error procesando el archivo: {e}")
