# tools/mesa/cartera_propia.py
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# UI / ESTILO
# =========================================================
NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "rgba(255,255,255,0.96)"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1240px;
            padding-top: 1.15rem;
            padding-bottom: 2rem;
          }}

          .cp-wrap {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(17,24,39,0.05);
            margin-bottom: 1rem;
          }}

          .cp-subtle {{
            color: {MUTED};
            font-size: 0.96rem;
            margin-top: -0.2rem;
            margin-bottom: 1rem;
          }}

          .cp-kpi {{
            border: 1px solid {BORDER};
            border-radius: 18px;
            background: #fff;
            padding: 0.95rem 1rem;
            min-height: 108px;
            box-shadow: 0 4px 16px rgba(17,24,39,0.03);
          }}

          .cp-kpi-label {{
            color: {MUTED};
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
          }}

          .cp-kpi-value {{
            color: {TEXT};
            font-size: 1.52rem;
            font-weight: 700;
            line-height: 1.08;
          }}

          .cp-kpi-sub {{
            margin-top: 0.35rem;
            font-size: 0.9rem;
            font-weight: 600;
          }}

          .cp-section {{
            margin-top: 1.2rem;
            margin-bottom: 0.35rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: {TEXT};
          }}

          .cp-note {{
            color: {MUTED};
            font-size: 0.9rem;
            margin-bottom: 0.65rem;
          }}

          .stDownloadButton button {{
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS
# =========================================================
def _fmt_money(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"$ {v:,.2f}"


def _fmt_num(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}"


def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}%"


def _safe_pct_change(fin: float, ini: float) -> float:
    if pd.isna(ini) or ini == 0:
        return np.nan
    return (fin / ini - 1) * 100


def _safe_share(value: float, total: float) -> float:
    if pd.isna(total) or total == 0:
        return np.nan
    return value / total * 100


def _normalize_text(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def _detect_currency_from_text(text: str) -> str:
    t = _normalize_text(text).upper()

    if "DOLAR LOCAL" in t or "DÓLAR LOCAL" in t:
        return "USD Local"
    if "U$S EXTERIOR" in t or "USD EXTERIOR" in t:
        return "USD Exterior"
    if "PESOS" in t:
        return "ARS"

    return "N/D"


def _detect_currency(row: pd.Series) -> str:
    especie = _normalize_text(row.get("Nombre de la Especie"))
    categoria = _normalize_text(row.get("Categoria"))
    return _detect_currency_from_text(f"{especie} {categoria}")


# =========================================================
# PARSEO DEL ARCHIVO EXACTO DE PORTAFOLIO
# =========================================================
def _read_raw_excel(file) -> pd.DataFrame:
    return pd.read_excel(file, header=None)


def parse_header_info(df_raw: pd.DataFrame) -> Dict[str, object]:
    """
    Estructura observada:
    fila 0: título
    fila 2: headers usuario/comitente/fecha...
    fila 3: valores
    filas 7 a 11: resumen superior
    """
    out: Dict[str, object] = {}

    # Datos generales
    try:
        out["usuario"] = df_raw.iloc[3, 0]
        out["comitente"] = df_raw.iloc[3, 1]
        out["fecha_desde"] = df_raw.iloc[3, 2]
        out["fecha_hasta"] = df_raw.iloc[3, 3]
        out["tipo"] = df_raw.iloc[3, 4]
        out["especie_filtro"] = df_raw.iloc[3, 5]
        out["filtro"] = df_raw.iloc[3, 6]
    except Exception:
        pass

    # Resumen superior
    labels = {
        "Total Posición": "total_posicion",
        "Portafolio Disponible": "portafolio_disponible",
        "Cuenta Corriente $": "cc_ars",
        "Cuenta Corriente U$S Exterior": "cc_usd_ext",
        "Cuenta Corriente Dolar Local": "cc_usd_local",
    }

    for i in range(min(len(df_raw), 20)):
        label = _normalize_text(df_raw.iloc[i, 2]) if df_raw.shape[1] > 2 else ""
        value = df_raw.iloc[i, 5] if df_raw.shape[1] > 5 else None
        if label in labels:
            out[labels[label]] = pd.to_numeric(value, errors="coerce")

    return out


def parse_detail_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Desde la fila 16 (índice Excel 17) viene:
    Nombre de la Especie | Estado | Cantidad | Precio | Importe | % S/Total | Costo | % Var | Resultado

    Lógica:
    - filas con especie real y estado vacío => posiciones
    - filas con estado completo y especie 'asd'/vacío => subtotales de categoría
    - filas 17/18/19 del ejemplo => caja / cuenta corriente
    """
    detail = df_raw.iloc[16:, 1:10].copy()
    detail.columns = [
        "Nombre de la Especie",
        "Estado",
        "Cantidad",
        "Precio",
        "Importe",
        "% S/Total",
        "Costo",
        "% Var",
        "Resultado",
    ]

    detail = detail.reset_index(drop=True)

    # Limpieza básica
    detail["Nombre de la Especie"] = detail["Nombre de la Especie"].apply(_normalize_text)
    detail["Estado"] = detail["Estado"].apply(_normalize_text)

    num_cols = ["Cantidad", "Precio", "Importe", "% S/Total", "Costo", "% Var", "Resultado"]
    for c in num_cols:
        detail[c] = pd.to_numeric(detail[c], errors="coerce")

    # quitar filas totalmente vacías
    detail = detail[
        ~(detail["Nombre de la Especie"].eq("") &
          detail["Estado"].eq("") &
          detail["Cantidad"].isna() &
          detail["Precio"].isna() &
          detail["Importe"].isna())
    ].copy()

    # identificar subtotales / categorías
    detail["is_subtotal"] = (
        detail["Estado"].ne("") &
        (
            detail["Nombre de la Especie"].str.lower().isin(["asd", "subtotal", "total", ""])
            | detail["Cantidad"].isna()
        )
    )

    # recorrer y asignar categoría a las filas anteriores hasta el último subtotal
    rows: List[Dict[str, object]] = []
    buffer_positions: List[Dict[str, object]] = []

    for _, row in detail.iterrows():
        especie = row["Nombre de la Especie"]
        estado = row["Estado"]

        if row["is_subtotal"]:
            categoria = estado if estado else "Sin categoría"
            for pos in buffer_positions:
                pos["Categoria"] = categoria
                rows.append(pos)
            buffer_positions = []
            continue

        # fila de posición
        rec = row.to_dict()
        rec["Categoria"] = ""
        buffer_positions.append(rec)

    # si quedaran filas sin subtotal final
    for pos in buffer_positions:
        pos["Categoria"] = "Sin categoría"
        rows.append(pos)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["Moneda"] = out.apply(_detect_currency, axis=1)

    # clasificaciones adicionales
    out["Tipo_Apertura"] = out["Categoria"].replace(
        {
            "Cuenta Corriente": "Caja / Cuenta Corriente",
            "Fondos Comunes PESOS": "FCI",
            "Letras PESOS": "Letras",
            "Titulos PublicosDolar Local": "Títulos Públicos",
            "Titulos Publicos Dolar Local": "Títulos Públicos",
            "Obligaciones Negociables PESOS": "ON",
            "Obligaciones NegociablesU$S Exterior": "ON",
            "Obligaciones NegociablesDolar Local": "ON",
            "Obligaciones Negociables Dolar Local": "ON",
        }
    )

    out["Especie_Key"] = (
        out["Nombre de la Especie"].astype(str).str.strip().str.upper()
    )

    out["Categoria_Key"] = (
        out["Categoria"].astype(str).str.strip().str.upper()
    )

    out["Moneda_Key"] = (
        out["Moneda"].astype(str).str.strip().str.upper()
    )

    return out


def load_portfolio(file) -> Tuple[Dict[str, object], pd.DataFrame]:
    df_raw = _read_raw_excel(file)
    header = parse_header_info(df_raw)
    detail = parse_detail_table(df_raw)
    return header, detail


# =========================================================
# AGREGADOS
# =========================================================
def build_exec_summary(header: Dict[str, object], detail: pd.DataFrame) -> Dict[str, float]:
    if detail.empty:
        return {
            "total_posicion": 0.0,
            "portafolio_disponible": 0.0,
            "cc_ars": 0.0,
            "cc_usd_ext": 0.0,
            "cc_usd_local": 0.0,
            "resultado_total": 0.0,
        }

    return {
        "total_posicion": pd.to_numeric(header.get("total_posicion"), errors="coerce"),
        "portafolio_disponible": pd.to_numeric(header.get("portafolio_disponible"), errors="coerce"),
        "cc_ars": pd.to_numeric(header.get("cc_ars"), errors="coerce"),
        "cc_usd_ext": pd.to_numeric(header.get("cc_usd_ext"), errors="coerce"),
        "cc_usd_local": pd.to_numeric(header.get("cc_usd_local"), errors="coerce"),
        "resultado_total": pd.to_numeric(detail["Resultado"], errors="coerce").fillna(0).sum(),
    }


def aggregate_by_currency(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["Moneda", "Importe", "Resultado", "Cantidad_Posiciones"])

    out = (
        detail.groupby("Moneda", as_index=False)
        .agg(
            Importe=("Importe", "sum"),
            Resultado=("Resultado", "sum"),
            Cantidad_Posiciones=("Nombre de la Especie", "count"),
        )
        .sort_values("Importe", ascending=False)
    )
    total = out["Importe"].sum()
    out["% Participacion"] = out["Importe"].apply(lambda x: _safe_share(x, total))
    return out


def aggregate_by_category(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["Categoria", "Importe", "Resultado", "Cantidad_Posiciones"])

    out = (
        detail.groupby(["Categoria", "Tipo_Apertura", "Moneda"], as_index=False)
        .agg(
            Importe=("Importe", "sum"),
            Resultado=("Resultado", "sum"),
            Cantidad_Posiciones=("Nombre de la Especie", "count"),
        )
        .sort_values("Importe", ascending=False)
    )
    total = out["Importe"].sum()
    out["% Participacion"] = out["Importe"].apply(lambda x: _safe_share(x, total))
    return out


def aggregate_by_species(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["Especie_Key", "Nombre de la Especie", "Categoria", "Moneda", "Importe"])

    out = (
        detail.groupby(
            ["Especie_Key", "Nombre de la Especie", "Categoria", "Tipo_Apertura", "Moneda"],
            as_index=False
        )
        .agg(
            Cantidad=("Cantidad", "sum"),
            Precio=("Precio", "mean"),
            Importe=("Importe", "sum"),
            Costo=("Costo", "mean"),
            Pct_Var=("% Var", "mean"),
            Resultado=("Resultado", "sum"),
        )
        .sort_values("Importe", ascending=False)
    )
    total = out["Importe"].sum()
    out["% Participacion"] = out["Importe"].apply(lambda x: _safe_share(x, total))
    return out


# =========================================================
# COMPARATIVOS INICIO VS FIN
# =========================================================
def compare_headers(h_ini: Dict[str, object], h_fin: Dict[str, object]) -> pd.DataFrame:
    rows = [
        ("Total Posición", h_ini.get("total_posicion"), h_fin.get("total_posicion")),
        ("Portafolio Disponible", h_ini.get("portafolio_disponible"), h_fin.get("portafolio_disponible")),
        ("Cuenta Corriente ARS", h_ini.get("cc_ars"), h_fin.get("cc_ars")),
        ("Cuenta Corriente USD Exterior", h_ini.get("cc_usd_ext"), h_fin.get("cc_usd_ext")),
        ("Cuenta Corriente USD Local", h_ini.get("cc_usd_local"), h_fin.get("cc_usd_local")),
    ]

    out = pd.DataFrame(rows, columns=["Indicador", "Inicio", "Fin"])
    out["Variacion"] = out["Fin"] - out["Inicio"]
    out["Variacion %"] = out.apply(lambda r: _safe_pct_change(r["Fin"], r["Inicio"]), axis=1)
    return out


def compare_currency(d_ini: pd.DataFrame, d_fin: pd.DataFrame) -> pd.DataFrame:
    a = aggregate_by_currency(d_ini).rename(
        columns={
            "Importe": "Importe_Ini",
            "Resultado": "Resultado_Ini",
            "Cantidad_Posiciones": "Posiciones_Ini",
            "% Participacion": "Peso_Ini",
        }
    )
    b = aggregate_by_currency(d_fin).rename(
        columns={
            "Importe": "Importe_Fin",
            "Resultado": "Resultado_Fin",
            "Cantidad_Posiciones": "Posiciones_Fin",
            "% Participacion": "Peso_Fin",
        }
    )

    out = pd.merge(a, b, on="Moneda", how="outer").fillna(0)
    out["Variacion"] = out["Importe_Fin"] - out["Importe_Ini"]
    out["Variacion %"] = out.apply(lambda r: _safe_pct_change(r["Importe_Fin"], r["Importe_Ini"]), axis=1)
    return out.sort_values("Importe_Fin", ascending=False)


def compare_category(d_ini: pd.DataFrame, d_fin: pd.DataFrame) -> pd.DataFrame:
    a = aggregate_by_category(d_ini).rename(
        columns={
            "Importe": "Importe_Ini",
            "Resultado": "Resultado_Ini",
            "Cantidad_Posiciones": "Posiciones_Ini",
            "% Participacion": "Peso_Ini",
        }
    )
    b = aggregate_by_category(d_fin).rename(
        columns={
            "Importe": "Importe_Fin",
            "Resultado": "Resultado_Fin",
            "Cantidad_Posiciones": "Posiciones_Fin",
            "% Participacion": "Peso_Fin",
        }
    )

    keys = ["Categoria", "Tipo_Apertura", "Moneda"]
    out = pd.merge(a, b, on=keys, how="outer").fillna(0)
    out["Variacion"] = out["Importe_Fin"] - out["Importe_Ini"]
    out["Variacion %"] = out.apply(lambda r: _safe_pct_change(r["Importe_Fin"], r["Importe_Ini"]), axis=1)
    return out.sort_values("Importe_Fin", ascending=False)


def compare_species(d_ini: pd.DataFrame, d_fin: pd.DataFrame) -> pd.DataFrame:
    a = aggregate_by_species(d_ini).rename(
        columns={
            "Cantidad": "Cantidad_Ini",
            "Precio": "Precio_Ini",
            "Importe": "Importe_Ini",
            "Costo": "Costo_Ini",
            "Pct_Var": "PctVar_Ini",
            "Resultado": "Resultado_Ini",
            "% Participacion": "Peso_Ini",
        }
    )

    b = aggregate_by_species(d_fin).rename(
        columns={
            "Cantidad": "Cantidad_Fin",
            "Precio": "Precio_Fin",
            "Importe": "Importe_Fin",
            "Costo": "Costo_Fin",
            "Pct_Var": "PctVar_Fin",
            "Resultado": "Resultado_Fin",
            "% Participacion": "Peso_Fin",
        }
    )

    keys = ["Especie_Key", "Nombre de la Especie", "Categoria", "Tipo_Apertura", "Moneda"]

    out = pd.merge(a, b, on=keys, how="outer").fillna(0)

    out["Var_Cantidad"] = out["Cantidad_Fin"] - out["Cantidad_Ini"]
    out["Var_Importe"] = out["Importe_Fin"] - out["Importe_Ini"]
    out["Var_Importe %"] = out.apply(lambda r: _safe_pct_change(r["Importe_Fin"], r["Importe_Ini"]), axis=1)

    out["Estado_Mov"] = np.select(
        [
            (out["Importe_Ini"] == 0) & (out["Importe_Fin"] != 0),
            (out["Importe_Ini"] != 0) & (out["Importe_Fin"] == 0),
            (out["Importe_Ini"] != 0) & (out["Importe_Fin"] != 0),
        ],
        ["Alta", "Baja", "Continua"],
        default="Sin datos",
    )

    return out.sort_values("Importe_Fin", ascending=False)


# =========================================================
# EXPORT
# =========================================================
def build_export_excel(
    header_ini: Dict[str, object],
    header_fin: Dict[str, object],
    detail_ini: pd.DataFrame,
    detail_fin: pd.DataFrame,
    df_header_compare: pd.DataFrame,
    df_currency_compare: pd.DataFrame,
    df_category_compare: pd.DataFrame,
    df_species_compare: pd.DataFrame,
) -> bytes:
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([header_ini]).to_excel(writer, sheet_name="Header_Inicio", index=False)
        pd.DataFrame([header_fin]).to_excel(writer, sheet_name="Header_Fin", index=False)
        detail_ini.to_excel(writer, sheet_name="Detalle_Inicio", index=False)
        detail_fin.to_excel(writer, sheet_name="Detalle_Fin", index=False)
        df_header_compare.to_excel(writer, sheet_name="Resumen_Comparado", index=False)
        df_currency_compare.to_excel(writer, sheet_name="Moneda", index=False)
        df_category_compare.to_excel(writer, sheet_name="Categoria", index=False)
        df_species_compare.to_excel(writer, sheet_name="Especies", index=False)

    bio.seek(0)
    return bio.read()


# =========================================================
# RENDER
# =========================================================
def render() -> None:
    _inject_css()

    st.title("Cartera Propia")
    st.markdown(
        '<div class="cp-subtle">Comparación de portafolio propio entre inicio y fin de mes, basada en el archivo "Portafolio valorizado a una fecha".</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cp-wrap">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        file_ini = st.file_uploader("Excel inicio de mes", type=["xlsx", "xls"], key="cp_ini")
    with c2:
        file_fin = st.file_uploader("Excel fin de mes", type=["xlsx", "xls"], key="cp_fin")

    run = st.button("Procesar cartera propia", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    if file_ini is None or file_fin is None:
        st.warning("Subí ambos archivos: inicio y fin de mes.")
        return

    try:
        header_ini, detail_ini = load_portfolio(file_ini)
        header_fin, detail_fin = load_portfolio(file_fin)

        exec_ini = build_exec_summary(header_ini, detail_ini)
        exec_fin = build_exec_summary(header_fin, detail_fin)

        df_header_compare = compare_headers(header_ini, header_fin)
        df_currency_compare = compare_currency(detail_ini, detail_fin)
        df_category_compare = compare_category(detail_ini, detail_fin)
        df_species_compare = compare_species(detail_ini, detail_fin)

    except Exception as e:
        st.error(f"Error procesando archivos: {e}")
        return

    # =====================================================
    # RESUMEN EJECUTIVO
    # =====================================================
    st.markdown('<div class="cp-section">Resumen ejecutivo</div>', unsafe_allow_html=True)

    total_ini = exec_ini["total_posicion"]
    total_fin = exec_fin["total_posicion"]
    total_var = total_fin - total_ini
    total_var_pct = _safe_pct_change(total_fin, total_ini)

    disp_ini = exec_ini["portafolio_disponible"]
    disp_fin = exec_fin["portafolio_disponible"]
    disp_var = disp_fin - disp_ini
    disp_var_pct = _safe_pct_change(disp_fin, disp_ini)

    res_ini = exec_ini["resultado_total"]
    res_fin = exec_fin["resultado_total"]
    res_var = res_fin - res_ini

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        color = "#16a34a" if total_var >= 0 else "#dc2626"
        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Total posición</div>
                <div class="cp-kpi-value">{_fmt_money(total_fin)}</div>
                <div class="cp-kpi-sub" style="color:{color};">
                    vs inicio: {_fmt_money(total_var)} | {_fmt_pct(total_var_pct)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        color = "#16a34a" if disp_var >= 0 else "#dc2626"
        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Portafolio disponible</div>
                <div class="cp-kpi-value">{_fmt_money(disp_fin)}</div>
                <div class="cp-kpi-sub" style="color:{color};">
                    vs inicio: {_fmt_money(disp_var)} | {_fmt_pct(disp_var_pct)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        color = "#16a34a" if res_var >= 0 else "#dc2626"
        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Resultado agregado</div>
                <div class="cp-kpi-value">{_fmt_money(res_fin)}</div>
                <div class="cp-kpi-sub" style="color:{color};">
                    vs inicio: {_fmt_money(res_var)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        altas = int((df_species_compare["Estado_Mov"] == "Alta").sum())
        bajas = int((df_species_compare["Estado_Mov"] == "Baja").sum())
        continuan = int((df_species_compare["Estado_Mov"] == "Continua").sum())

        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Movimientos de especies</div>
                <div class="cp-kpi-value">{continuan}</div>
                <div class="cp-kpi-sub" style="color:{MUTED};">
                    Altas: {altas} | Bajas: {bajas}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # COMPARACIÓN GENERAL
    # =====================================================
    st.markdown('<div class="cp-section">Comparación general inicio vs fin</div>', unsafe_allow_html=True)
    st.dataframe(
        df_header_compare.style.format(
            {
                "Inicio": "$ {:,.2f}",
                "Fin": "$ {:,.2f}",
                "Variacion": "$ {:,.2f}",
                "Variacion %": "{:,.2f}%",
            }
        ),
        use_container_width=True,
        height=260,
    )

    # =====================================================
    # APERTURA POR MONEDA
    # =====================================================
    st.markdown('<div class="cp-section">Apertura por moneda</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cp-note">Muestra cómo cambia la exposición entre ARS, USD Exterior y USD Local.</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_currency_compare.style.format(
            {
                "Importe_Ini": "$ {:,.2f}",
                "Importe_Fin": "$ {:,.2f}",
                "Variacion": "$ {:,.2f}",
                "Variacion %": "{:,.2f}%",
                "Peso_Ini": "{:,.2f}%",
                "Peso_Fin": "{:,.2f}%",
                "Resultado_Ini": "$ {:,.2f}",
                "Resultado_Fin": "$ {:,.2f}",
            }
        ),
        use_container_width=True,
        height=260,
    )

    # =====================================================
    # APERTURA POR CATEGORÍA
    # =====================================================
    st.markdown('<div class="cp-section">Apertura por categoría</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cp-note">Agrupa por Cuenta Corriente, Letras, ON, FCI y Títulos Públicos.</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_category_compare.style.format(
            {
                "Importe_Ini": "$ {:,.2f}",
                "Importe_Fin": "$ {:,.2f}",
                "Variacion": "$ {:,.2f}",
                "Variacion %": "{:,.2f}%",
                "Peso_Ini": "{:,.2f}%",
                "Peso_Fin": "{:,.2f}%",
                "Resultado_Ini": "$ {:,.2f}",
                "Resultado_Fin": "$ {:,.2f}",
            }
        ),
        use_container_width=True,
        height=360,
    )

    # =====================================================
    # APERTURA POR ESPECIE
    # =====================================================
    st.markdown('<div class="cp-section">Apertura por especie</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cp-note">Comparación puntual por instrumento entre inicio y fin de mes.</div>',
        unsafe_allow_html=True,
    )

    show_species = df_species_compare[
        [
            "Nombre de la Especie",
            "Categoria",
            "Tipo_Apertura",
            "Moneda",
            "Cantidad_Ini",
            "Cantidad_Fin",
            "Var_Cantidad",
            "Importe_Ini",
            "Importe_Fin",
            "Var_Importe",
            "Var_Importe %",
            "Peso_Ini",
            "Peso_Fin",
            "Estado_Mov",
        ]
    ].copy()

    st.dataframe(
        show_species.style.format(
            {
                "Cantidad_Ini": "{:,.2f}",
                "Cantidad_Fin": "{:,.2f}",
                "Var_Cantidad": "{:,.2f}",
                "Importe_Ini": "$ {:,.2f}",
                "Importe_Fin": "$ {:,.2f}",
                "Var_Importe": "$ {:,.2f}",
                "Var_Importe %": "{:,.2f}%",
                "Peso_Ini": "{:,.2f}%",
                "Peso_Fin": "{:,.2f}%",
            }
        ),
        use_container_width=True,
        height=520,
    )

    # =====================================================
    # TOP MOVIMIENTOS
    # =====================================================
    st.markdown('<div class="cp-section">Top movimientos</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("**Top subas por variación absoluta**")
        top_up = (
            df_species_compare.sort_values("Var_Importe", ascending=False)
            .head(10)[
                ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Ini", "Importe_Fin", "Var_Importe", "Var_Importe %"]
            ]
            .copy()
        )
        st.dataframe(
            top_up.style.format(
                {
                    "Importe_Ini": "$ {:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                    "Var_Importe %": "{:,.2f}%",
                }
            ),
            use_container_width=True,
            height=320,
        )

    with right:
        st.markdown("**Top bajas por variación absoluta**")
        top_down = (
            df_species_compare.sort_values("Var_Importe", ascending=True)
            .head(10)[
                ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Ini", "Importe_Fin", "Var_Importe", "Var_Importe %"]
            ]
            .copy()
        )
        st.dataframe(
            top_down.style.format(
                {
                    "Importe_Ini": "$ {:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                    "Var_Importe %": "{:,.2f}%",
                }
            ),
            use_container_width=True,
            height=320,
        )

    # =====================================================
    # ALTAS / BAJAS / CONTINUIDAD
    # =====================================================
    st.markdown('<div class="cp-section">Altas, bajas y continuidad</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Altas**")
        altas_df = df_species_compare[df_species_compare["Estado_Mov"] == "Alta"][
            ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Fin", "Peso_Fin"]
        ].copy()
        st.dataframe(
            altas_df.style.format(
                {
                    "Importe_Fin": "$ {:,.2f}",
                    "Peso_Fin": "{:,.2f}%",
                }
            ),
            use_container_width=True,
            height=260,
        )

    with c2:
        st.markdown("**Bajas**")
        bajas_df = df_species_compare[df_species_compare["Estado_Mov"] == "Baja"][
            ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Ini", "Peso_Ini"]
        ].copy()
        st.dataframe(
            bajas_df.style.format(
                {
                    "Importe_Ini": "$ {:,.2f}",
                    "Peso_Ini": "{:,.2f}%",
                }
            ),
            use_container_width=True,
            height=260,
        )

    with c3:
        st.markdown("**Continúan**")
        cont_df = df_species_compare[df_species_compare["Estado_Mov"] == "Continua"][
            ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Ini", "Importe_Fin", "Var_Importe"]
        ].copy()
        st.dataframe(
            cont_df.style.format(
                {
                    "Importe_Ini": "$ {:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                }
            ),
            use_container_width=True,
            height=260,
        )

    # =====================================================
    # EXPORT
    # =====================================================
    export_bytes = build_export_excel(
        header_ini=header_ini,
        header_fin=header_fin,
        detail_ini=detail_ini,
        detail_fin=detail_fin,
        df_header_compare=df_header_compare,
        df_currency_compare=df_currency_compare,
        df_category_compare=df_category_compare,
        df_species_compare=df_species_compare,
    )

    st.markdown('<div class="cp-section">Descarga</div>', unsafe_allow_html=True)
    st.download_button(
        "Descargar Excel comparativo de cartera propia",
        data=export_bytes,
        file_name="cartera_propia_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
