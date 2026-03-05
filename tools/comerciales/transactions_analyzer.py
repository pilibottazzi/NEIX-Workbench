# tools/db_operaciones_cleaner.py
from __future__ import annotations

import re
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================
# Normalización / parsing
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

def _norm_colname(s: str) -> str:
    s = _safe_str(s).strip().lower()
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "").replace("°", "").replace("º", "")
    return s

def _to_float(x) -> float:
    """
    Convierte a float:
    - '-' / '' -> NaN
    - '1.234,56' -> 1234.56
    - '1234,56' -> 1234.56
    - '1234.56' -> 1234.56
    """
    s = _safe_str(x).strip()
    if s == "" or s == "-":
        return float("nan")
    s = s.replace("$", "").replace(" ", "")
    # miles/decimales
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _to_date(x):
    # Excel puede venir como datetime, string dd/mm/yy, etc.
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

def pick_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Mapea columnas de tu base a nombres estándar.
    """
    cols = {c: _norm_colname(c) for c in df.columns}

    def find(*candidates: str) -> Optional[str]:
        for cand in candidates:
            cand = _norm_colname(cand)
            for orig, norm in cols.items():
                if norm == cand:
                    return orig
        return None

    return {
        "especie": find("Especie"),
        "referencia": find("Referencia"),
        "tipo_operacion": find("Tipo Operación", "Tipo Operacion"),
        "fecha_operacion": find("Fecha Operacion", "Fecha Operación"),
        "fecha_liquidacion": find("Fecha Liquidacion", "Fecha Liquidación"),
        "nro_operacion": find("Nro de operación", "Nro de operacion", "Nro operación", "Nro operacion"),
        "cantidad": find("Cantidad"),
        "moneda": find("Moneda"),
        "precio": find("Precio"),
        "importe": find("Importe"),
    }

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    m = pick_columns(df)

    out = pd.DataFrame()
    out["Especie"] = df[m["especie"]] if m["especie"] else ""
    out["Referencia"] = df[m["referencia"]] if m["referencia"] else ""
    out["TipoOperacion"] = df[m["tipo_operacion"]] if m["tipo_operacion"] else ""
    out["FechaOperacion"] = df[m["fecha_operacion"]].map(_to_date) if m["fecha_operacion"] else pd.NaT
    out["FechaLiquidacion"] = df[m["fecha_liquidacion"]].map(_to_date) if m["fecha_liquidacion"] else pd.NaT
    out["NroOperacion"] = df[m["nro_operacion"]] if m["nro_operacion"] else ""
    out["Cantidad"] = df[m["cantidad"]].map(_to_float) if m["cantidad"] else float("nan")
    out["Moneda"] = df[m["moneda"]] if m["moneda"] else ""
    out["Precio"] = df[m["precio"]].map(_to_float) if m["precio"] else float("nan")
    out["Importe"] = df[m["importe"]].map(_to_float) if m["importe"] else float("nan")

    # limpieza de strings
    for c in ["Especie", "Referencia", "TipoOperacion", "Moneda", "NroOperacion"]:
        out[c] = out[c].map(_safe_str).str.strip()

    return out


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Importe_faltante"] = out["Importe"].isna()
    out["Precio_faltante"] = out["Precio"].isna()
    out["Cantidad_faltante"] = out["Cantidad"].isna()

    # Importe calculable si tengo cantidad y precio, pero importe falta
    out["Importe_calculable"] = out["Importe_faltante"] & (~out["Cantidad"].isna()) & (~out["Precio"].isna())
    out["Importe_sugerido"] = out["Cantidad"] * out["Precio"]

    # “calidad” simple
    out["Quality"] = "OK"
    out.loc[out["Importe_faltante"], "Quality"] = "FALTA_IMPORTE"
    out.loc[out["Importe_calculable"], "Quality"] = "FALTA_IMPORTE_PERO_CALCULABLE"

    return out


def to_excel_bytes(clean: pd.DataFrame, missing: pd.DataFrame, resumen: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        clean.to_excel(writer, sheet_name="Limpia", index=False)
        missing.to_excel(writer, sheet_name="Faltantes", index=False)
        for name, df in resumen.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return bio.getvalue()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="DB Operaciones – Limpieza & Explorer", layout="wide")
st.title("DB Operaciones – Limpieza y análisis")
st.caption("Objetivo: detectar importes faltantes, limpiar y entender totales por categoría.")

up = st.file_uploader("Subí el Excel (operaciones/débitos)", type=["xlsx", "xls"])
if not up:
    st.stop()

xls = pd.ExcelFile(up)
sheet = st.selectbox("Hoja", options=["(todas)"] + xls.sheet_names, index=0)

dfs = []
if sheet == "(todas)":
    for sh in xls.sheet_names:
        df0 = pd.read_excel(xls, sheet_name=sh)
        df0["__sheet__"] = sh
        dfs.append(df0)
else:
    df0 = pd.read_excel(xls, sheet_name=sheet)
    df0["__sheet__"] = sheet
    dfs.append(df0)

raw = pd.concat(dfs, ignore_index=True)

std = standardize_df(raw)
std.insert(0, "OrigenHoja", raw["__sheet__"].values)
std = add_quality_flags(std)

# Opciones de limpieza
st.subheader("Controles de limpieza")

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    incluir_moneda = st.checkbox("Incluir filas Moneda vacía", value=True)
with c2:
    solo_con_especie = st.checkbox("Excluir filas sin Especie", value=True)
with c3:
    imputar = st.checkbox("Imputar Importe cuando sea calculable (Cantidad×Precio)", value=False)
with c4:
    st.write("")

work = std.copy()
if not incluir_moneda:
    work = work[work["Moneda"].ne("")]
if solo_con_especie:
    work = work[work["Especie"].ne("")]

# imputación
if imputar:
    mask = work["Importe_calculable"]
    work.loc[mask, "Importe"] = work.loc[mask, "Importe_sugerido"]
    work = add_quality_flags(work)  # recomputar flags

# Separar faltantes
faltantes = work[work["Importe_faltante"]].copy()
limpia = work[~work["Importe_faltante"]].copy()

# KPIs
total_importe = limpia["Importe"].sum() if len(limpia) else 0.0
cant_total = len(work)
cant_falt = len(faltantes)
pct_falt = (cant_falt / cant_total * 100) if cant_total else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Filas totales", f"{cant_total:,}")
k2.metric("Filas sin Importe", f"{cant_falt:,}")
k3.metric("% sin Importe", f"{pct_falt:,.2f}%")
k4.metric("Suma Importe (solo limpias)", f"{total_importe:,.2f}")

if cant_falt:
    st.warning("Hay filas SIN Importe. Abajo tenés el listado y podés exportar para corregir.")
else:
    st.success("No hay filas sin Importe 🎉")

# Resúmenes
def group_sum(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=by + ["Importe"])
    g = df.groupby(by, as_index=False)["Importe"].sum()
    return g.sort_values("Importe", ascending=False)

res_tipo = group_sum(limpia, ["TipoOperacion"])
res_mon = group_sum(limpia, ["Moneda"])
res_especie = group_sum(limpia, ["Especie"])
res_ref = group_sum(limpia, ["Referencia"])

# Layout
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Limpia", "Faltantes", "Diagnóstico"])

with tab1:
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Importe por Tipo Operación")
        st.dataframe(res_tipo, use_container_width=True, hide_index=True)
    with cB:
        st.subheader("Importe por Moneda")
        st.dataframe(res_mon, use_container_width=True, hide_index=True)

    cC, cD = st.columns(2)
    with cC:
        st.subheader("Top Especies por Importe")
        st.dataframe(res_especie.head(30), use_container_width=True, hide_index=True)
    with cD:
        st.subheader("Top Referencias por Importe")
        st.dataframe(res_ref.head(30), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Base limpia (con Importe)")
    st.dataframe(limpia.sort_values("FechaOperacion", ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Filas con Importe faltante")
    show_cols = [
        "OrigenHoja","Especie","Referencia","TipoOperacion","FechaOperacion","FechaLiquidacion",
        "NroOperacion","Cantidad","Moneda","Precio","Importe","Quality","Importe_calculable","Importe_sugerido"
    ]
    st.dataframe(faltantes[show_cols], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Chequeos útiles")
    # 1) faltantes calculables
    calc = work[work["Importe_calculable"]].copy()
    st.write("**Faltantes que sí se pueden calcular (Cantidad×Precio):**", len(calc))
    st.dataframe(calc[show_cols].head(200), use_container_width=True, hide_index=True)

    # 2) filas con moneda vacía
    mv = work[work["Moneda"].eq("")].copy()
    st.write("**Filas con Moneda vacía:**", len(mv))
    st.dataframe(mv[show_cols].head(200), use_container_width=True, hide_index=True)

    # 3) fechas nulas
    fn = work[work["FechaOperacion"].isna() | work["FechaLiquidacion"].isna()].copy()
    st.write("**Filas con Fecha nula (operación o liquidación):**", len(fn))
    st.dataframe(fn[show_cols].head(200), use_container_width=True, hide_index=True)

# Export
st.subheader("Exportar")

excel_bytes = to_excel_bytes(
    clean=limpia,
    missing=faltantes,
    resumen={
        "Resumen_TipoOp": res_tipo,
        "Resumen_Moneda": res_mon,
        "Resumen_Especie": res_especie,
        "Resumen_Referencia": res_ref,
    }
)

st.download_button(
    "Descargar Excel (Limpia + Faltantes + Resúmenes)",
    data=excel_bytes,
    file_name="db_operaciones_limpieza.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
