# tools/mesa/filtro_especies_exterior.py
from __future__ import annotations

from io import BytesIO

import openpyxl
import pandas as pd
import streamlit as st


# =========================================================
# UI / ESTILO NEIX
# =========================================================
NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "rgba(255,255,255,0.96)"


def _inject_ui_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 980px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }}

          .fes-wrap {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(17,24,39,0.05);
            margin-bottom: 1rem;
          }}

          .fes-title {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {TEXT};
            margin-bottom: 0.2rem;
          }}

          .fes-subtitle {{
            font-size: 0.95rem;
            color: {MUTED};
            margin-bottom: 0.8rem;
          }}

          .fes-line {{
            height: 3px;
            width: 100%;
            background: linear-gradient(90deg, {NEIX_RED}, rgba(255,59,48,0.18));
            border-radius: 999px;
            margin: 0.35rem 0 1rem 0;
          }}

          .fes-kpi-row {{
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
          }}

          .fes-kpi {{
            flex: 1 1 180px;
            background: white;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 0.9rem 1rem;
          }}

          .fes-kpi-label {{
            font-size: 0.80rem;
            color: {MUTED};
            margin-bottom: 0.2rem;
          }}

          .fes-kpi-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {TEXT};
          }}

          .stDownloadButton > button,
          .stButton > button {{
            width: 100%;
            border-radius: 12px;
            min-height: 2.8rem;
            font-weight: 600;
            border: 1px solid {NEIX_RED};
          }}

          .stDownloadButton > button {{
            background: {NEIX_RED};
            color: white;
          }}

          .stDownloadButton > button:hover {{
            background: #e1342a;
            color: white;
            border-color: #e1342a;
          }}

          .stAlert {{
            border-radius: 14px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    # Formato precio
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        for cell in row:
            cell.number_format = "0.00"

    # Anchos de columna
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
    _inject_ui_css()

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

        st.markdown(
            f"""
            <div class="fes-kpi-row">
                <div class="fes-kpi">
                    <div class="fes-kpi-label">Filas leídas</div>
                    <div class="fes-kpi-value">{stats['total_filas']:,}</div>
                </div>
                <div class="fes-kpi">
                    <div class="fes-kpi-label">Fuente = BBG</div>
                    <div class="fes-kpi-value">{stats['filas_bbg']:,}</div>
                </div>
                <div class="fes-kpi">
                    <div class="fes-kpi-label">Moneda = USD EXTERIOR</div>
                    <div class="fes-kpi-value">{stats['filas_usd_exterior']:,}</div>
                </div>
                <div class="fes-kpi">
                    <div class="fes-kpi-label">Filas finales</div>
                    <div class="fes-kpi-value">{stats['filas_finales']:,}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Vista previa")
        st.dataframe(df_final, use_container_width=True, hide_index=True)

        st.download_button(
            label="Descargar Excel filtrado",
            data=output,
            file_name="filtro_especies_exterior.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
