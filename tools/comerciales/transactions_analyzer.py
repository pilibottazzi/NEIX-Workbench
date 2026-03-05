# tools/transactions_analyze.py
from __future__ import annotations

import re
import calendar
import datetime as dt
from io import BytesIO
from typing import Optional, Tuple, Dict, List

import pandas as pd
import streamlit as st


# =========================
# 1) Meta desde nombre hoja
# =========================
MONTHS_ES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12
}

def parse_sheet_meta(sheet_name: str) -> Optional[Tuple[str, dt.date]]:
    """
    Espera nombres tipo: '904 Ene-26', '904 dic-25', '904 Ene 26', etc.
    Devuelve (comitente, fecha_fin_mes).
    """
    s = sheet_name.strip()

    # comitente al inicio
    m_com = re.match(r"^\s*(\d{3,4})\s+(.+?)\s*$", s)
    if not m_com:
        return None

    comitente = m_com.group(1)
    rest = m_com.group(2).strip()

    m = re.search(r"(?i)\b(ene|feb|mar|abr|may|jun|jul|ago|sep|set|oct|nov|dic)\b\W*(\d{2}|\d{4})\b", rest)
    if not m:
        return None

    mon = MONTHS_ES[m.group(1).lower()]
    year = int(m.group(2))
    if year < 100:
        year = 2000 + year

    last_day = calendar.monthrange(year, mon)[1]
    return comitente, dt.date(year, mon, last_day)


# =========================
# 2) Normalización
# =========================
def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def _norm(s: str) -> str:
    s = _safe_str(s).strip().lower()
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
    s = re.sub(r"\s+", " ", s)
    return s

def _to_float(x) -> float:
    s = _safe_str(x).strip()
    if s in ("", "-", "–"):
        return float("nan")
    s = s.replace("%", "").replace("$", "").replace(" ", "")
    # 1.234,56 -> 1234.56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _find_col(df: pd.DataFrame, wanted: List[str]) -> Optional[str]:
    cols = {c: _norm(c) for c in df.columns}
    wanted_norm = [_norm(w) for w in wanted]
    for w in wanted_norm:
        for c, cn in cols.items():
            if cn == w:
                return c
    return None


# =========================
# 3) Lectura robusta de hoja
# =========================
def detect_header_row_df(tmp: pd.DataFrame, max_scan_rows: int = 30) -> int:
    """
    tmp: DataFrame leído con header=None.
    Busca fila donde aparezcan Especie/Cantidad/Precio/Importe/Part.
    """
    wanted = ["especie", "cantidad", "precio", "importe", "part", "part."]
    best_i, best_score = 0, -1

    n = min(max_scan_rows, len(tmp))
    for i in range(n):
        row = [_norm(v) for v in tmp.iloc[i].tolist()]
        score = 0
        for w in wanted:
            if any(w == rv for rv in row):
                score += 3
            elif any(w in rv for rv in row):
                score += 1
        if score > best_score:
            best_score, best_i = score, i

    return best_i

def read_sheet_smart(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Lee hoja de tenencias aunque tenga filas arriba / header corrido.
    """
    try:
        tmp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=40)
        h = detect_header_row_df(tmp)
        df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
        # si quedó todo Unnamed, fallback a header=0
        if len(df.columns) and all(_norm(c).startswith("unnamed") for c in df.columns):
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
        return df
    except Exception:
        return pd.read_excel(xls, sheet_name=sheet_name, header=0)


# =========================
# 4) Clasificación por bloque (TOTAL...)
# =========================
TOTAL_TO_CLASS = {
    "TOTAL ACCIONES": "Acciones",
    "TOTAL TITULOS PUBLICOS": "Titulos Publicos",
    "TOTAL TITULOS PÚBLICOS": "Titulos Publicos",
    "TOTAL OBLIGACIONES NEGOCIABLES": "ON",
    "TOTAL FCI": "FCI",
}

STOP_ROWS = {"TOTAL POSICION"}  # si aparece

def tenencias_sheet_to_rows(xls: pd.ExcelFile, sheet: str, comitente: str, fecha_cierre: dt.date) -> pd.DataFrame:
    raw = read_sheet_smart(xls, sheet)

    col_especie = _find_col(raw, ["Especie"]) or raw.columns[0]
    col_cant = _find_col(raw, ["Cantidad"])
    col_precio = _find_col(raw, ["Precio"])
    col_importe = _find_col(raw, ["Importe"])
    col_part = _find_col(raw, ["Part.", "Part"])

    df = pd.DataFrame({
        "especie": raw[col_especie].map(_safe_str).str.strip(),
        "cantidad": raw[col_cant].map(_to_float) if col_cant else float("nan"),
        "precio": raw[col_precio].map(_to_float) if col_precio else float("nan"),
        "importe": raw[col_importe].map(_to_float) if col_importe else float("nan"),
        "part": raw[col_part].map(_to_float) if col_part else float("nan"),
    })

    # quedarnos con filas con especie (aunque sea TOTAL...)
    df = df[df["especie"].ne("")].copy()

    current_class = "SinClasificar"
    out_rows = []

    for _, row in df.iterrows():
        esp = _safe_str(row["especie"]).strip()
        esp_u = esp.upper().strip()

        # cortar si llega a TOTAL POSICION
        if esp_u in STOP_ROWS:
            break

        # Marcadores de bloque
        if esp_u in TOTAL_TO_CLASS:
            current_class = TOTAL_TO_CLASS[esp_u]
            continue

        # cualquier TOTAL ... se ignora como data
        if esp_u.startswith("TOTAL "):
            continue

        # monedas/resúmenes
        if esp_u in ("DOLAR MEP", "PESOS"):
            clase = "Moneda"
        else:
            clase = current_class

        out_rows.append({
            "comitente": comitente,
            "fecha_cierre": fecha_cierre,
            "clase": clase,
            "especie": esp,
            "cantidad": row["cantidad"],
            "precio": row["precio"],
            "importe": row["importe"],
            "part": row["part"],
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    # limpieza final: si importe es NaN pero cantidad y precio están, dejamos NaN (no inventamos)
    return out


def to_excel_bytes(base: pd.DataFrame, sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        base.to_excel(writer, sheet_name="tenencias_db", index=False)
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name[:31], index=False)
    return bio.getvalue()


# =========================
# 5) Entry point Workbench
# =========================
def render() -> None:
    """
    Llamar desde el router del Workbench.
    No usar set_page_config acá.
    """
    st.header("Tenencias valorizadas → Base tipo DB")
    st.caption("Convierte un Excel (una hoja por comitente+mes) en una tabla: comitente / fecha_cierre / especie / importes.")

    uploaded = st.file_uploader("Subí el Excel de tenencias", type=["xlsx", "xls"], key="ten_up")
    if not uploaded:
        st.info("Subí el archivo para comenzar.")
        return

    xls = pd.ExcelFile(uploaded)

    # detectar hojas con meta
    meta = []
    for sh in xls.sheet_names:
        m = parse_sheet_meta(sh)
        if m:
            com, fecha = m
            meta.append({"sheet": sh, "comitente": com, "fecha_cierre": fecha})

    meta_df = pd.DataFrame(meta)
    if meta_df.empty:
        st.error("No pude interpretar nombres de hojas. Ejemplos válidos: '904 Ene-26', '904 Dic-25'.")
        return

    meta_df = meta_df.sort_values(["comitente", "fecha_cierre"])
    st.subheader("Hojas detectadas")
    st.dataframe(meta_df, use_container_width=True, hide_index=True)

    comitentes = sorted(meta_df["comitente"].unique())
    sel_com = st.multiselect("Comitentes a procesar", comitentes, default=comitentes, key="ten_coms")

    only_latest = st.checkbox("Solo último mes por comitente", value=False, key="ten_latest")

    colA, colB = st.columns([1, 1])
    with colA:
        run = st.button("Procesar", type="primary", key="ten_run")
    with colB:
        st.caption("Tip: si tenés muchas hojas, probá primero con 1 comitente para validar lectura.")

    if not run:
        return

    work = meta_df[meta_df["comitente"].isin(sel_com)].copy()
    if only_latest:
        work = work.sort_values(["comitente", "fecha_cierre"]).groupby("comitente", as_index=False).tail(1)

    out_rows = []
    errores = []

    for r in work.itertuples(index=False):
        try:
            df_rows = tenencias_sheet_to_rows(xls, r.sheet, r.comitente, r.fecha_cierre)
            if df_rows is None or df_rows.empty:
                errores.append(f"{r.sheet}: sin filas útiles (revisar formato)")
            else:
                out_rows.append(df_rows)
        except Exception as e:
            errores.append(f"{r.sheet}: {e}")

    if errores:
        st.warning("Algunas hojas no pudieron procesarse:")
        for e in errores[:15]:
            st.write("•", e)
        if len(errores) > 15:
            st.write(f"… y {len(errores)-15} más.")

    if not out_rows:
        st.error("No se generaron filas. Probá con un comitente específico para debug.")
        return

    db = pd.concat(out_rows, ignore_index=True)

    # Resumen por comitente/fecha/clase
    resumen_clase = (
        db.groupby(["comitente", "fecha_cierre", "clase"], as_index=False)
          .agg(
              importe=("importe", "sum"),
              part=("part", "sum"),
              instrumentos=("especie", "count"),
          )
          .sort_values(["comitente", "fecha_cierre", "importe"], ascending=[True, True, False])
    )

    st.success(f"Listo: {len(db):,} filas (instrumentos).")

    t1, t2, t3 = st.tabs(["Base", "Resumen por clase", "Control"])
    with t1:
        st.dataframe(db, use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(resumen_clase, use_container_width=True, hide_index=True)
    with t3:
        # chequeos útiles
        falt_importe = db[db["importe"].isna()].copy()
        st.write("Filas con Importe vacío:", len(falt_importe))
        st.dataframe(falt_importe.head(200), use_container_width=True, hide_index=True)

    excel_bytes = to_excel_bytes(db, {"resumen_clase": resumen_clase})
    st.download_button(
        "Descargar Excel (base + resumen)",
        data=excel_bytes,
        file_name="tenencias_base_datos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="ten_dl",
    )
