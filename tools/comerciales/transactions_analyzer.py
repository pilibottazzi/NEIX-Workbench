# tools/tenencias_to_db.py
from __future__ import annotations

import re
import calendar
import datetime as dt
from io import BytesIO
from typing import Optional, Tuple, Dict, List

import pandas as pd
import streamlit as st


# ============
# Parse meta
# ============
MONTHS_ES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12
}

def parse_sheet_meta(sheet_name: str) -> Optional[Tuple[str, dt.date]]:
    """
    Espera nombres tipo: '904 Ene-26', '904 dic-25', etc.
    Devuelve (comitente, fecha_fin_mes)
    """
    s = sheet_name.strip()
    m_com = re.match(r"^\s*(\d+)\s+(.+?)\s*$", s)
    if not m_com:
        return None
    comitente = m_com.group(1)
    rest = m_com.group(2).strip()

    m = re.search(r"(?i)\b(ene|feb|mar|abr|may|jun|jul|ago|sep|set|oct|nov|dic)\b\W*(\d{2}|\d{4})\b", rest)
    if not m:
        return None
    mon = MONTHS_ES[m.group(1).lower()]
    y_raw = m.group(2)
    year = int(y_raw)
    if year < 100:
        year = 2000 + year

    last_day = calendar.monthrange(year, mon)[1]
    return comitente, dt.date(year, mon, last_day)


# =================
# Normalizadores
# =================
def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def _to_float(x) -> float:
    s = _safe_str(x).strip()
    if s == "":
        return 0.0
    s = s.replace("%", "").strip()
    # 1.234,56 -> 1234.56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

def _find_col(df: pd.DataFrame, wanted: List[str]) -> Optional[str]:
    cols = {c: _safe_str(c).strip().lower() for c in df.columns}
    for w in wanted:
        w2 = w.strip().lower()
        for c, cl in cols.items():
            if cl == w2:
                return c
    return None


# =========================
# Clasificación por bloque
# =========================
TOTAL_TO_CLASS = {
    "TOTAL ACCIONES": "Acciones",
    "TOTAL TITULOS PUBLICOS": "Titulos Publicos",
    # si el día de mañana aparece esto, ya queda listo:
    "TOTAL TITULOS PÚBLICOS": "Titulos Publicos",
    "TOTAL FCI": "FCI",
    "TOTAL OBLIGACIONES NEGOCIABLES": "ON",
}

STOP_ROWS = {
    "TOTAL POSICION",
}

EXCLUDE_PREFIX = ("TOTAL ",)  # excluimos todas las filas TOTAL* (pero las usamos como “marcadores”)

def tenencias_sheet_to_rows(xls: pd.ExcelFile, sheet: str, comitente: str, fecha_cierre: dt.date) -> pd.DataFrame:
    raw = pd.read_excel(xls, sheet_name=sheet)

    col_especie = _find_col(raw, ["Especie"]) or raw.columns[0]
    col_cant = _find_col(raw, ["Cantidad"])
    col_precio = _find_col(raw, ["Precio"])
    col_importe = _find_col(raw, ["Importe"])
    col_part = _find_col(raw, ["Part.", "Part"])

    df = pd.DataFrame({
        "especie": raw[col_especie].map(_safe_str).str.strip(),
        "cantidad": raw[col_cant].map(_to_float) if col_cant else 0.0,
        "precio": raw[col_precio].map(_to_float) if col_precio else 0.0,
        "importe": raw[col_importe].map(_to_float) if col_importe else 0.0,
        "part": raw[col_part].map(_to_float) if col_part else 0.0,
    })

    df = df[df["especie"].ne("")]

//    # Si hay filas totalmente vacías las sacamos
    df = df[~((df["cantidad"] == 0) & (df["precio"] == 0) & (df["importe"] == 0) & (df["part"] == 0) & (df["especie"] == ""))]

    # Clasificación por bloque: vamos recorriendo en orden
    current_class = "SinClasificar"
    clases = []
    for esp in df["especie"].tolist():
        esp_u = esp.upper().strip()

        # cortar si llega a total posición (si existiese)
        if esp_u in STOP_ROWS:
            break

        # si es una fila TOTAL..., cambia el bloque (marcador)
        if esp_u in TOTAL_TO_CLASS:
            current_class = TOTAL_TO_CLASS[esp_u]
            clases.append("__MARKER__")
            continue

        # si es otra fila total (TOTAL CUENTA CORRIENTE, TOTAL..., etc.) -> marcador genérico
        if esp_u.startswith("TOTAL "):
            clases.append("__MARKER__")
            continue

        # instrumentos “especiales” (si querés tratarlos aparte)
        if esp_u in ("DOLAR MEP", "PESOS"):
            clases.append("Moneda")
            continue

        clases.append(current_class)

    df = df.iloc[:len(clases)].copy()
    df["clase"] = clases

    # nos quedamos solo con filas de instrumentos reales (no marcadores)
    df = df[df["clase"].ne("__MARKER__")].copy()

    # agregar columnas base-datos
    df.insert(0, "comitente", comitente)
    df.insert(1, "fecha_cierre", fecha_cierre)

    # orden final
    df = df[["comitente", "fecha_cierre", "clase", "especie", "cantidad", "precio", "importe", "part"]]
    return df


def to_excel_bytes(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="tenencias_db", index=False)
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name[:31], index=False)
    return bio.getvalue()


# =========================
# Streamlit
# =========================
st.set_page_config(page_title="Tenencias → Base de datos", layout="wide")
st.title("Tenencias valorizadas → Base de datos (comitente/especie)")
st.caption("Convierte el Excel (una hoja por comitente+mes) en una tabla tipo base de datos.")

uploaded = st.file_uploader("Subí el Excel de tenencias", type=["xlsx", "xls"])
if not uploaded:
    st.stop()

xls = pd.ExcelFile(uploaded)

meta = []
for sh in xls.sheet_names:
    m = parse_sheet_meta(sh)
    if m:
        com, fecha = m
        meta.append({"sheet": sh, "comitente": com, "fecha_cierre": fecha})

meta_df = pd.DataFrame(meta)
if meta_df.empty:
    st.error("No pude interpretar nombres de hojas. Ej: '904 Ene-26', '904 Dic-25'.")
    st.stop()

meta_df = meta_df.sort_values(["comitente", "fecha_cierre"])
st.subheader("Hojas detectadas")
st.dataframe(meta_df, use_container_width=True, hide_index=True)

comitentes = sorted(meta_df["comitente"].unique())
sel_com = st.multiselect("Comitentes a procesar", comitentes, default=comitentes)

only_latest = st.checkbox("Solo último mes por comitente", value=False)

if st.button("Procesar", type="primary"):
    work = meta_df[meta_df["comitente"].isin(sel_com)].copy()
    if only_latest:
        work = work.sort_values(["comitente", "fecha_cierre"]).groupby("comitente", as_index=False).tail(1)

    out_rows = []
    for r in work.itertuples(index=False):
        try:
            df_rows = tenencias_sheet_to_rows(xls, r.sheet, r.comitente, r.fecha_cierre)
            out_rows.append(df_rows)
        except Exception as e:
            st.warning(f"Error en hoja {r.sheet}: {e}")

    if not out_rows:
        st.error("No se generó ninguna fila.")
        st.stop()

    db = pd.concat(out_rows, ignore_index=True)

    st.success(f"Listo: {len(db):,} filas (instrumentos)")

    # Resumen por comitente/clase
    resumen = (
        db.groupby(["comitente", "fecha_cierre", "clase"], as_index=False)
          .agg(importe=("importe", "sum"), part=("part", "sum"), instrumentos=("especie", "count"))
          .sort_values(["comitente", "fecha_cierre", "importe"], ascending=[True, True, False])
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Base (tenencias_db)")
        st.dataframe(db, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Resumen por clase")
        st.dataframe(resumen, use_container_width=True, hide_index=True)

    excel_bytes = to_excel_bytes(db, {"resumen_clase": resumen})
    st.download_button(
        "Descargar Excel (base + resumen)",
        data=excel_bytes,
        file_name="tenencias_base_datos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
