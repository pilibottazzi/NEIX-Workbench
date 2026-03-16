# tools/comerciales/cartera_propia.py
from __future__ import annotations

from io import BytesIO
from datetime import datetime
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
CARD_BG = "rgba(255,255,255,0.98)"
SUCCESS = "#16a34a"
DANGER = "#dc2626"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1260px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
          }}

          .cp-wrap {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
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
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 118px;
            box-shadow: 0 4px 16px rgba(17,24,39,0.03);
          }}

          .cp-kpi-main {{
            border: 1px solid rgba(239,59,48,0.12);
            border-radius: 20px;
            background: linear-gradient(180deg, #ffffff 0%, #fff7f7 100%);
            padding: 1.05rem 1rem 1rem 1rem;
            min-height: 126px;
            box-shadow: 0 10px 24px rgba(239,59,48,0.06);
          }}

          .cp-kpi-label {{
            color: {MUTED};
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
          }}

          .cp-kpi-value {{
            color: {TEXT};
            font-size: 1.55rem;
            font-weight: 750;
            line-height: 1.08;
          }}

          .cp-kpi-value-main {{
            color: {TEXT};
            font-size: 1.72rem;
            font-weight: 800;
            line-height: 1.05;
          }}

          .cp-kpi-sub {{
            margin-top: 0.36rem;
            font-size: 0.92rem;
            font-weight: 650;
          }}

          .cp-section {{
            margin-top: 1.25rem;
            margin-bottom: 0.4rem;
            font-size: 1.08rem;
            font-weight: 760;
            color: {TEXT};
            letter-spacing: -0.01em;
          }}

          .cp-section-main {{
            margin-top: 1.2rem;
            margin-bottom: 0.45rem;
            font-size: 1.18rem;
            font-weight: 800;
            color: {TEXT};
            letter-spacing: -0.01em;
          }}

          .cp-note {{
            color: {MUTED};
            font-size: 0.91rem;
            margin-bottom: 0.7rem;
          }}

          .cp-card {{
            background: #fff;
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 4px 14px rgba(17,24,39,0.03);
          }}

          .stDownloadButton button {{
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
          }}

          div[data-testid="stDataFrame"] {{
            border-radius: 16px;
            overflow: hidden;
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


def _to_datetime_safe(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value

    if isinstance(value, datetime):
        return pd.Timestamp(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return None

    dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        return None

    return dt


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


def _hide_index(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    try:
        return styler.hide(axis="index")
    except Exception:
        return styler


def get_reference_date(header: Dict[str, object]) -> Optional[pd.Timestamp]:
    fecha_hasta = _to_datetime_safe(header.get("fecha_hasta"))
    if fecha_hasta is not None:
        return fecha_hasta

    fecha_desde = _to_datetime_safe(header.get("fecha_desde"))
    if fecha_desde is not None:
        return fecha_desde

    return None


# =========================================================
# PARSEO DEL ARCHIVO
# =========================================================
def _read_raw_excel(file) -> pd.DataFrame:
    return pd.read_excel(file, header=None)


def parse_header_info(df_raw: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}

    try:
        out["usuario"] = df_raw.iloc[3, 0]
        out["comitente"] = df_raw.iloc[3, 1]
        out["fecha_desde"] = _to_datetime_safe(df_raw.iloc[3, 2])
        out["fecha_hasta"] = _to_datetime_safe(df_raw.iloc[3, 3])
        out["tipo"] = df_raw.iloc[3, 4]
        out["especie_filtro"] = df_raw.iloc[3, 5]
        out["filtro"] = df_raw.iloc[3, 6]
    except Exception:
        pass

    labels = {
        "Total Posición": "total_posicion",
        "Portafolio Disponible": "portafolio_disponible",
        "Cuenta Corriente $": "cc_ars",
        "Cuenta Corriente U$S Exterior": "cc_usd_ext",
        "Cuenta Corriente Dolar Local": "cc_usd_local",
        "Cuenta Corriente Dólar Local": "cc_usd_local",
    }

    for i in range(min(len(df_raw), 20)):
        label = _normalize_text(df_raw.iloc[i, 2]) if df_raw.shape[1] > 2 else ""
        value = df_raw.iloc[i, 5] if df_raw.shape[1] > 5 else None

        if label in labels:
            out[labels[label]] = pd.to_numeric(value, errors="coerce")

    return out


def parse_detail_table(df_raw: pd.DataFrame) -> pd.DataFrame:
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
    detail["Nombre de la Especie"] = detail["Nombre de la Especie"].apply(_normalize_text)
    detail["Estado"] = detail["Estado"].apply(_normalize_text)

    num_cols = ["Cantidad", "Precio", "Importe", "% S/Total", "Costo", "% Var", "Resultado"]
    for c in num_cols:
        detail[c] = pd.to_numeric(detail[c], errors="coerce")

    detail = detail[
        ~(detail["Nombre de la Especie"].eq("") &
          detail["Estado"].eq("") &
          detail["Cantidad"].isna() &
          detail["Precio"].isna() &
          detail["Importe"].isna())
    ].copy()

    detail["is_subtotal"] = (
        detail["Estado"].ne("") &
        (
            detail["Nombre de la Especie"].str.lower().isin(["asd", "subtotal", "total", ""])
            | detail["Cantidad"].isna()
        )
    )

    rows: List[Dict[str, object]] = []
    buffer_positions: List[Dict[str, object]] = []

    for _, row in detail.iterrows():
        if row["is_subtotal"]:
            categoria = row["Estado"] if row["Estado"] else "Sin categoría"
            for pos in buffer_positions:
                pos["Categoria"] = categoria
                rows.append(pos)
            buffer_positions = []
            continue

        rec = row.to_dict()
        rec["Categoria"] = ""
        buffer_positions.append(rec)

    for pos in buffer_positions:
        pos["Categoria"] = "Sin categoría"
        rows.append(pos)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["Moneda"] = out.apply(_detect_currency, axis=1)

    out["Tipo_Apertura"] = out["Categoria"].replace(
        {
            "Cuenta Corriente": "Caja / Cuenta Corriente",
            "Fondos Comunes PESOS": "FCI",
            "Letras PESOS": "Letras",
            "Titulos PublicosDolar Local": "Títulos Públicos",
            "Titulos Publicos Dolar Local": "Títulos Públicos",
            "Títulos PublicosDolar Local": "Títulos Públicos",
            "Títulos Publicos Dolar Local": "Títulos Públicos",
            "Obligaciones Negociables PESOS": "ON",
            "Obligaciones NegociablesU$S Exterior": "ON",
            "Obligaciones Negociables U$S Exterior": "ON",
            "Obligaciones NegociablesDolar Local": "ON",
            "Obligaciones Negociables Dolar Local": "ON",
            "Obligaciones NegociablesDólar Local": "ON",
            "Obligaciones Negociables Dólar Local": "ON",
        }
    )

    out["Especie_Key"] = out["Nombre de la Especie"].astype(str).str.strip().str.upper()
    out["Categoria_Key"] = out["Categoria"].astype(str).str.strip().str.upper()
    out["Moneda_Key"] = out["Moneda"].astype(str).str.strip().str.upper()

    return out


def load_portfolio(file) -> Tuple[Dict[str, object], pd.DataFrame]:
    df_raw = _read_raw_excel(file)
    header = parse_header_info(df_raw)
    detail = parse_detail_table(df_raw)
    return header, detail


def assign_initial_and_final(
    file_a,
    file_b,
) -> Tuple[
    Dict[str, object], pd.DataFrame, str,
    Dict[str, object], pd.DataFrame, str
]:
    header_a, detail_a = load_portfolio(file_a)
    header_b, detail_b = load_portfolio(file_b)

    ref_a = get_reference_date(header_a)
    ref_b = get_reference_date(header_b)

    if ref_a is None and ref_b is None:
        raise ValueError("No pude identificar fechas en ninguno de los dos archivos.")

    if ref_a is None:
        raise ValueError(
            f"No pude identificar la fecha del archivo: {getattr(file_a, 'name', 'archivo 1')}"
        )

    if ref_b is None:
        raise ValueError(
            f"No pude identificar la fecha del archivo: {getattr(file_b, 'name', 'archivo 2')}"
        )

    name_a = getattr(file_a, "name", "archivo_a")
    name_b = getattr(file_b, "name", "archivo_b")

    if ref_a <= ref_b:
        return header_a, detail_a, name_a, header_b, detail_b, name_b

    return header_b, detail_b, name_b, header_a, detail_a, name_a


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
            "cantidad_especies": 0,
        }

    return {
        "total_posicion": pd.to_numeric(header.get("total_posicion"), errors="coerce"),
        "portafolio_disponible": pd.to_numeric(header.get("portafolio_disponible"), errors="coerce"),
        "cc_ars": pd.to_numeric(header.get("cc_ars"), errors="coerce"),
        "cc_usd_ext": pd.to_numeric(header.get("cc_usd_ext"), errors="coerce"),
        "cc_usd_local": pd.to_numeric(header.get("cc_usd_local"), errors="coerce"),
        "resultado_total": pd.to_numeric(detail["Resultado"], errors="coerce").fillna(0).sum(),
        "cantidad_especies": int(detail["Especie_Key"].nunique()),
    }


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
# COMPARATIVOS
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
    out["Variación"] = out["Fin"] - out["Inicio"]
    out["Variación %"] = out.apply(lambda r: _safe_pct_change(r["Fin"], r["Inicio"]), axis=1)
    return out


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

    out["Movimiento"] = np.select(
        [
            out["Var_Cantidad"] > 0,
            out["Var_Cantidad"] < 0,
            out["Var_Cantidad"] == 0,
        ],
        ["Compra", "Venta", "Misma posición"],
        default="Misma posición",
    )

    out["Estado_Existencia"] = np.select(
        [
            (out["Importe_Ini"] == 0) & (out["Importe_Fin"] != 0),
            (out["Importe_Ini"] != 0) & (out["Importe_Fin"] == 0),
            (out["Importe_Ini"] != 0) & (out["Importe_Fin"] != 0),
        ],
        ["Alta", "Baja", "Continua"],
        default="Sin datos",
    )

    return out.sort_values(["Movimiento", "Var_Importe"], ascending=[True, False])


# =========================================================
# EXPORT
# =========================================================
def build_export_excel(
    header_ini: Dict[str, object],
    header_fin: Dict[str, object],
    detail_ini: pd.DataFrame,
    detail_fin: pd.DataFrame,
    df_header_compare: pd.DataFrame,
    df_species_compare: pd.DataFrame,
) -> bytes:
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([header_ini]).to_excel(writer, sheet_name="Header_Inicio", index=False)
        pd.DataFrame([header_fin]).to_excel(writer, sheet_name="Header_Fin", index=False)
        detail_ini.to_excel(writer, sheet_name="Detalle_Inicio", index=False)
        detail_fin.to_excel(writer, sheet_name="Detalle_Fin", index=False)
        df_header_compare.to_excel(writer, sheet_name="Resumen_Comparado", index=False)
        df_species_compare.to_excel(writer, sheet_name="Especies", index=False)

    bio.seek(0)
    return bio.read()


# =========================================================
# RENDER TABLAS
# =========================================================
def _styled_df(df: pd.DataFrame, formats: Dict[str, str]) -> pd.io.formats.style.Styler:
    styler = df.style.format(formats, na_rep="-")
    styler = _hide_index(styler)
    return styler


# =========================================================
# RENDER
# =========================================================
def render() -> None:
    _inject_css()

    st.title("Cartera Propia")
    st.markdown(
        '<div class="cp-subtle">Comparación ejecutiva del portafolio propio entre dos fechas. El sistema identifica automáticamente cuál es el archivo inicial y cuál es el final según la fecha del reporte.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cp-wrap">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        file_a = st.file_uploader("Excel 1", type=["xlsx", "xls"], key="cp_a")
    with c2:
        file_b = st.file_uploader("Excel 2", type=["xlsx", "xls"], key="cp_b")

    run = st.button("Procesar cartera propia", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    if file_a is None or file_b is None:
        st.warning("Subí ambos archivos. El sistema identifica automáticamente cuál es el inicial y cuál es el final.")
        return

    try:
        (
            header_ini,
            detail_ini,
            name_ini,
            header_fin,
            detail_fin,
            name_fin,
        ) = assign_initial_and_final(file_a, file_b)

        fecha_ini_ref = get_reference_date(header_ini)
        fecha_fin_ref = get_reference_date(header_fin)

        st.info(
            f"Archivo inicial detectado: {name_ini} "
            f"({fecha_ini_ref.strftime('%d/%m/%Y') if fecha_ini_ref is not None else 'sin fecha'})\n\n"
            f"Archivo final detectado: {name_fin} "
            f"({fecha_fin_ref.strftime('%d/%m/%Y') if fecha_fin_ref is not None else 'sin fecha'})"
        )

        exec_ini = build_exec_summary(header_ini, detail_ini)
        exec_fin = build_exec_summary(header_fin, detail_fin)

        df_header_compare = compare_headers(header_ini, header_fin)
        df_species_compare = compare_species(detail_ini, detail_fin)

    except Exception as e:
        st.error(f"Error procesando archivos: {e}")
        return

    # =====================================================
    # KPIS PRINCIPALES
    # =====================================================
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

    compras = int((df_species_compare["Movimiento"] == "Compra").sum())
    ventas = int((df_species_compare["Movimiento"] == "Venta").sum())
    mismas = int((df_species_compare["Movimiento"] == "Misma posición").sum())

    st.markdown('<div class="cp-section-main">KPIs principales</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        color = SUCCESS if total_var >= 0 else DANGER
        st.markdown(
            f"""
            <div class="cp-kpi-main">
                <div class="cp-kpi-label">Total posición final</div>
                <div class="cp-kpi-value-main">{_fmt_money(total_fin)}</div>
                <div class="cp-kpi-sub" style="color:{color};">
                    variación: {_fmt_money(total_var)} · {_fmt_pct(total_var_pct)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        color = SUCCESS if disp_var >= 0 else DANGER
        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Portafolio disponible final</div>
                <div class="cp-kpi-value">{_fmt_money(disp_fin)}</div>
                <div class="cp-kpi-sub" style="color:{color};">
                    variación: {_fmt_money(disp_var)} · {_fmt_pct(disp_var_pct)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        color = SUCCESS if res_var >= 0 else DANGER
        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Resultado agregado final</div>
                <div class="cp-kpi-value">{_fmt_money(res_fin)}</div>
                <div class="cp-kpi-sub" style="color:{color};">
                    variación: {_fmt_money(res_var)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="cp-kpi">
                <div class="cp-kpi-label">Movimientos de posición</div>
                <div class="cp-kpi-value">{compras + ventas + mismas}</div>
                <div class="cp-kpi-sub" style="color:{MUTED};">
                    Compras: {compras} · Ventas: {ventas} · Igual: {mismas}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # COMPARACIÓN GENERAL
    # =====================================================
    st.markdown('<div class="cp-section-main">Comparación general inicio vs fin</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cp-note">Bloque principal del análisis. Resume el cambio total del portafolio y de la liquidez entre ambos cortes.</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        _styled_df(
            df_header_compare,
            {
                "Inicio": "$ {:,.2f}",
                "Fin": "$ {:,.2f}",
                "Variación": "$ {:,.2f}",
                "Variación %": "{:,.2f}%",
            },
        ),
        use_container_width=True,
        height=245,
    )

    # =====================================================
    # DETALLE GENERAL POR ESPECIE
    # =====================================================
    st.markdown('<div class="cp-section">Detalle general por especie</div>', unsafe_allow_html=True)

    df_general = (
        df_species_compare[
            [
                "Nombre de la Especie",
                "Categoria",
                "Moneda",
                "Cantidad_Ini",
                "Cantidad_Fin",
                "Var_Cantidad",
                "Importe_Ini",
                "Importe_Fin",
                "Var_Importe",
                "Var_Importe %",
                "Movimiento",
            ]
        ]
        .sort_values(["Var_Importe"], ascending=False)
        .copy()
    )

    st.dataframe(
        _styled_df(
            df_general,
            {
                "Cantidad_Ini": "{:,.2f}",
                "Cantidad_Fin": "{:,.2f}",
                "Var_Cantidad": "{:,.2f}",
                "Importe_Ini": "$ {:,.2f}",
                "Importe_Fin": "$ {:,.2f}",
                "Var_Importe": "$ {:,.2f}",
                "Var_Importe %": "{:,.2f}%",
            },
        ),
        use_container_width=True,
        height=500,
    )

    # =====================================================
    # COMPRAS / VENTAS / MISMA POSICIÓN
    # =====================================================
    st.markdown('<div class="cp-section-main">Ordenado por compras, ventas y misma posición</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Compras**")
        compras_df = (
            df_species_compare[df_species_compare["Movimiento"] == "Compra"][
                [
                    "Nombre de la Especie",
                    "Categoria",
                    "Moneda",
                    "Cantidad_Ini",
                    "Cantidad_Fin",
                    "Var_Cantidad",
                    "Importe_Fin",
                    "Var_Importe",
                ]
            ]
            .sort_values(["Var_Cantidad", "Var_Importe"], ascending=[False, False])
            .copy()
        )

        st.dataframe(
            _styled_df(
                compras_df,
                {
                    "Cantidad_Ini": "{:,.2f}",
                    "Cantidad_Fin": "{:,.2f}",
                    "Var_Cantidad": "{:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                },
            ),
            use_container_width=True,
            height=300,
        )

    with col2:
        st.markdown("**Ventas**")
        ventas_df = (
            df_species_compare[df_species_compare["Movimiento"] == "Venta"][
                [
                    "Nombre de la Especie",
                    "Categoria",
                    "Moneda",
                    "Cantidad_Ini",
                    "Cantidad_Fin",
                    "Var_Cantidad",
                    "Importe_Fin",
                    "Var_Importe",
                ]
            ]
            .sort_values(["Var_Cantidad", "Var_Importe"], ascending=[True, True])
            .copy()
        )

        st.dataframe(
            _styled_df(
                ventas_df,
                {
                    "Cantidad_Ini": "{:,.2f}",
                    "Cantidad_Fin": "{:,.2f}",
                    "Var_Cantidad": "{:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                },
            ),
            use_container_width=True,
            height=300,
        )

    with col3:
        st.markdown("**Misma posición**")
        igual_df = (
            df_species_compare[df_species_compare["Movimiento"] == "Misma posición"][
                [
                    "Nombre de la Especie",
                    "Categoria",
                    "Moneda",
                    "Cantidad_Ini",
                    "Cantidad_Fin",
                    "Importe_Ini",
                    "Importe_Fin",
                    "Var_Importe",
                ]
            ]
            .sort_values(["Var_Importe"], ascending=False)
            .copy()
        )

        st.dataframe(
            _styled_df(
                igual_df,
                {
                    "Cantidad_Ini": "{:,.2f}",
                    "Cantidad_Fin": "{:,.2f}",
                    "Importe_Ini": "$ {:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                },
            ),
            use_container_width=True,
            height=300,
        )

    # =====================================================
    # TOP MOVIMIENTOS
    # =====================================================
    st.markdown('<div class="cp-section">Principales movimientos</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("**Mayores subas de valuación**")
        top_up = (
            df_species_compare.sort_values("Var_Importe", ascending=False)
            .head(10)[
                ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Ini", "Importe_Fin", "Var_Importe", "Var_Importe %"]
            ]
            .copy()
        )

        st.dataframe(
            _styled_df(
                top_up,
                {
                    "Importe_Ini": "$ {:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                    "Var_Importe %": "{:,.2f}%",
                },
            ),
            use_container_width=True,
            height=320,
        )

    with right:
        st.markdown("**Mayores bajas de valuación**")
        top_down = (
            df_species_compare.sort_values("Var_Importe", ascending=True)
            .head(10)[
                ["Nombre de la Especie", "Categoria", "Moneda", "Importe_Ini", "Importe_Fin", "Var_Importe", "Var_Importe %"]
            ]
            .copy()
        )

        st.dataframe(
            _styled_df(
                top_down,
                {
                    "Importe_Ini": "$ {:,.2f}",
                    "Importe_Fin": "$ {:,.2f}",
                    "Var_Importe": "$ {:,.2f}",
                    "Var_Importe %": "{:,.2f}%",
                },
            ),
            use_container_width=True,
            height=320,
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
