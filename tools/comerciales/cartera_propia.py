from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONSTANTES DE ESTILO
# =========================================================
PRIMARY = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.08)"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
INFO = "#2563eb"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }}

          /* Tarjetas generales */
          .cp-card {{
            background: #fff;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 12px rgba(17,24,39,0.04);
            margin-bottom: 1rem;
          }}

          /* KPIs */
          .cp-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 1.25rem;
          }}
          .cp-kpi {{
            background: #f9fafb;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1rem 1.1rem;
          }}
          .cp-kpi-label {{
            color: {MUTED};
            font-size: 0.82rem;
            margin-bottom: 0.3rem;
            letter-spacing: 0.02em;
          }}
          .cp-kpi-value {{
            color: {PRIMARY};
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.1;
          }}
          .cp-kpi-delta {{
            margin-top: 0.3rem;
            font-size: 0.83rem;
            font-weight: 600;
          }}

          /* Secciones */
          .cp-section {{
            font-size: 0.92rem;
            font-weight: 700;
            color: {MUTED};
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 1.4rem 0 0.6rem;
          }}

          /* Badges de movimiento */
          .badge {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 9px;
            border-radius: 20px;
          }}
          .badge-alta    {{ background: #dcfce7; color: #15803d; }}
          .badge-baja    {{ background: #fee2e2; color: #b91c1c; }}
          .badge-compra  {{ background: #dbeafe; color: #1d4ed8; }}
          .badge-venta   {{ background: #fef9c3; color: #a16207; }}
          .badge-igual   {{ background: #f3f4f6; color: #6b7280; }}

          div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
          .stDownloadButton button {{ border-radius: 10px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS DE FORMATO
# =========================================================
def _fmt_money(v: float, decimals: int = 2) -> str:
    if pd.isna(v):
        return "—"
    fmt = f"$ {{v:,.{decimals}f}}"
    return fmt.format(v=v)


def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:+,.2f}%"


def _pct_change(fin: float, ini: float) -> float:
    if pd.isna(ini) or ini == 0:
        return np.nan
    return (fin / ini - 1) * 100


def _share(value: float, total: float) -> float:
    if pd.isna(total) or total == 0:
        return np.nan
    return value / total * 100


def _norm(x) -> str:
    return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x).strip()


def _to_ts(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    ts = pd.to_datetime(str(value).strip(), dayfirst=True, errors="coerce")
    return None if pd.isna(ts) else ts


# =========================================================
# PARSEO
# =========================================================
def _detect_moneda(categoria: str) -> str:
    c = categoria.upper()
    if "U$S EXTERIOR" in c or "EXT" in c:
        return "USD Exterior"
    if "DOLAR LOCAL" in c or "DÓLAR LOCAL" in c:
        return "USD Local"
    return "ARS"


def _detect_tipo(categoria: str) -> str:
    c = categoria.upper()
    mapping = {
        "CUENTA CORRIENTE": "Cta. Corriente",
        "FONDOS COMUNES": "FCI",
        "LETRAS": "Letras",
        "TITULOS PUBLICOS": "T. Públicos",
        "TÍTULOS PUBLICOS": "T. Públicos",
        "OBLIGACIONES NEGOCIABLES": "ON",
        "ACCIONES": "Acciones",
    }
    for key, val in mapping.items():
        if key in c:
            return val
    return "Otros"


def _parse_header(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {
        "usuario": _norm(df.iloc[3, 0]),
        "comitente": _norm(df.iloc[3, 1]),
        "fecha": _to_ts(df.iloc[3, 2]),
    }
    label_map = {
        "Total Posición": "total_posicion",
        "Portafolio Disponible": "portafolio_disponible",
        "Cuenta Corriente $": "cc_ars",
        "Cuenta Corriente U$S Exterior": "cc_usd_ext",
        "Cuenta Corriente Dolar Local": "cc_usd_local",
        "Cuenta Corriente Dólar Local": "cc_usd_local",
    }
    for i in range(min(len(df), 20)):
        label = _norm(df.iloc[i, 2]) if df.shape[1] > 2 else ""
        val = df.iloc[i, 5] if df.shape[1] > 5 else None
        if label in label_map:
            out[label_map[label]] = pd.to_numeric(val, errors="coerce")
    return out


def _parse_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae el detalle de posiciones desde la fila 16 en adelante.
    Los subtotales se identifican cuando 'Cantidad' y 'Precio' son NaN
    pero 'Estado' tiene texto (incluye el caso 'asd' del sistema origen).
    """
    raw = df.iloc[16:, 1:10].copy()
    raw.columns = ["Especie", "Estado", "Cantidad", "Precio", "Importe",
                   "PctTotal", "Costo", "PctVar", "Resultado"]
    raw = raw.reset_index(drop=True)
    raw["Especie"] = raw["Especie"].apply(_norm)
    raw["Estado"]  = raw["Estado"].apply(_norm)

    for col in ["Cantidad", "Precio", "Importe", "PctTotal", "Costo", "PctVar", "Resultado"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Un subtotal es una fila sin Cantidad ni Precio pero con Estado no vacío
    def _is_subtotal(row) -> bool:
        return pd.isna(row["Cantidad"]) and pd.isna(row["Precio"]) and row["Estado"] != ""

    rows: List[dict] = []
    current_cat = ""

    for _, row in raw.iterrows():
        especie = row["Especie"]
        # Descarta filas completamente vacías y notas al pie
        if especie == "" and pd.isna(row["Cantidad"]) and pd.isna(row["Importe"]):
            continue
        if especie.startswith("* "):
            continue
        if _is_subtotal(row):
            current_cat = row["Estado"]
            continue

        rec = row.to_dict()
        rec["Categoria"] = current_cat
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Moneda"] = out["Categoria"].apply(_detect_moneda)
    out["Tipo"]   = out["Categoria"].apply(_detect_tipo)
    # Excluir filas de CC global (sin categoría asignada = primeras filas de FX)
    out = out[out["Categoria"] != ""].copy()
    return out


def load_portfolio(file) -> Tuple[Dict[str, object], pd.DataFrame]:
    df_raw = pd.read_excel(file, header=None)
    return _parse_header(df_raw), _parse_detail(df_raw)


def assign_dates(file_a, file_b) -> Tuple[
    Dict, pd.DataFrame, str,
    Dict, pd.DataFrame, str,
]:
    """Determina cuál archivo es el inicial y cuál el final según la fecha."""
    h_a, d_a = load_portfolio(file_a)
    h_b, d_b = load_portfolio(file_b)
    ref_a, ref_b = h_a.get("fecha"), h_b.get("fecha")

    name_a = getattr(file_a, "name", "archivo_a")
    name_b = getattr(file_b, "name", "archivo_b")

    for ref, name in [(ref_a, name_a), (ref_b, name_b)]:
        if ref is None:
            raise ValueError(f"No se pudo identificar la fecha en: {name}")

    if ref_a <= ref_b:
        return h_a, d_a, name_a, h_b, d_b, name_b
    return h_b, d_b, name_b, h_a, d_a, name_a


# =========================================================
# AGREGADOS
# =========================================================
def _agg_by_especie(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    return (
        detail.groupby(["Especie", "Categoria", "Tipo", "Moneda"], as_index=False)
        .agg(
            Cantidad=("Cantidad", "sum"),
            Precio=("Precio", "mean"),
            Importe=("Importe", "sum"),
            Costo=("Costo", "mean"),
            Resultado=("Resultado", "sum"),
        )
        .sort_values("Importe", ascending=False)
    )


def compare_headers(h_ini: Dict, h_fin: Dict) -> pd.DataFrame:
    rows = [
        ("Total Posición",          h_ini.get("total_posicion"),         h_fin.get("total_posicion")),
        ("Portafolio Disponible",    h_ini.get("portafolio_disponible"),   h_fin.get("portafolio_disponible")),
        ("Cuenta Corriente ARS",    h_ini.get("cc_ars"),                  h_fin.get("cc_ars")),
        ("CC USD Exterior",          h_ini.get("cc_usd_ext"),              h_fin.get("cc_usd_ext")),
        ("CC USD Local",            h_ini.get("cc_usd_local"),            h_fin.get("cc_usd_local")),
    ]
    df = pd.DataFrame(rows, columns=["Indicador", "Inicio", "Fin"])
    df["Variación $"] = df["Fin"] - df["Inicio"]
    df["Variación %"] = df.apply(lambda r: _pct_change(r["Fin"], r["Inicio"]), axis=1)
    return df


def compare_species(d_ini: pd.DataFrame, d_fin: pd.DataFrame) -> pd.DataFrame:
    keys = ["Especie", "Categoria", "Tipo", "Moneda"]

    a = _agg_by_especie(d_ini).rename(columns={
        "Cantidad": "Cant_Ini", "Precio": "Precio_Ini",
        "Importe": "Imp_Ini",   "Resultado": "Res_Ini",
    })
    b = _agg_by_especie(d_fin).rename(columns={
        "Cantidad": "Cant_Fin", "Precio": "Precio_Fin",
        "Importe": "Imp_Fin",   "Resultado": "Res_Fin",
    })

    comp = pd.merge(a, b, on=keys, how="outer").fillna(0)
    comp["Var_Cant"]    = comp["Cant_Fin"] - comp["Cant_Ini"]
    comp["Var_Importe"] = comp["Imp_Fin"]  - comp["Imp_Ini"]
    comp["Var_Pct"]     = comp.apply(lambda r: _pct_change(r["Imp_Fin"], r["Imp_Ini"]), axis=1)

    comp["Movimiento"] = np.select(
        [
            (comp["Imp_Ini"] == 0) & (comp["Imp_Fin"] != 0),
            (comp["Imp_Ini"] != 0) & (comp["Imp_Fin"] == 0),
            comp["Var_Cant"] > 0,
            comp["Var_Cant"] < 0,
        ],
        ["Alta", "Baja", "Compra", "Venta"],
        default="Sin cambio",
    )
    return comp.sort_values("Var_Importe", ascending=False)


# =========================================================
# EXPORT
# =========================================================
def build_export(
    h_ini: Dict, h_fin: Dict,
    d_ini: pd.DataFrame, d_fin: pd.DataFrame,
    df_header: pd.DataFrame, df_species: pd.DataFrame,
) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([h_ini]).to_excel(writer, sheet_name="Header_Inicio", index=False)
        pd.DataFrame([h_fin]).to_excel(writer, sheet_name="Header_Fin",    index=False)
        d_ini.to_excel(writer, sheet_name="Detalle_Inicio", index=False)
        d_fin.to_excel(writer, sheet_name="Detalle_Fin",    index=False)
        df_header.to_excel(writer,  sheet_name="Resumen",  index=False)
        df_species.to_excel(writer, sheet_name="Especies", index=False)
    bio.seek(0)
    return bio.read()


# =========================================================
# RENDER AUXILIARES
# =========================================================
def _badge_html(mov: str) -> str:
    cls_map = {
        "Alta": "badge-alta", "Baja": "badge-baja",
        "Compra": "badge-compra", "Venta": "badge-venta",
    }
    cls = cls_map.get(mov, "badge-igual")
    return f'<span class="badge {cls}">{mov}</span>'


def _kpi_html(label: str, value: str, delta: str, delta_color: str) -> str:
    return f"""
    <div class="cp-kpi">
        <div class="cp-kpi-label">{label}</div>
        <div class="cp-kpi-value">{value}</div>
        <div class="cp-kpi-delta" style="color:{delta_color};">{delta}</div>
    </div>
    """


def _styled(df: pd.DataFrame, fmts: Dict[str, str]):
    return df.style.format(fmts, na_rep="—").hide(axis="index")


# =========================================================
# RENDER PRINCIPAL
# =========================================================
def render() -> None:
    _inject_css()

    st.title("Cartera Propia")
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.95rem;margin-top:-0.3rem;margin-bottom:1.4rem;">'
        "Comparación ejecutiva del portafolio entre dos fechas. "
        "El sistema identifica automáticamente el archivo inicial y el final.</p>",
        unsafe_allow_html=True,
    )

    # ── Carga de archivos ─────────────────────────────────
    with st.container():
        c1, c2 = st.columns(2)
        file_a = c1.file_uploader("Excel 1", type=["xlsx", "xls"], key="cp_a")
        file_b = c2.file_uploader("Excel 2", type=["xlsx", "xls"], key="cp_b")
        run = st.button("Analizar cartera", use_container_width=True)

    if not run:
        return
    if file_a is None or file_b is None:
        st.warning("Subí ambos archivos para continuar.")
        return

    # ── Procesamiento ─────────────────────────────────────
    try:
        h_ini, d_ini, name_ini, h_fin, d_fin, name_fin = assign_dates(file_a, file_b)
    except ValueError as e:
        st.error(str(e))
        return

    fecha_ini = h_ini["fecha"]
    fecha_fin = h_fin["fecha"]
    fmt_fecha = lambda ts: ts.strftime("%d/%m/%Y") if ts else "—"

    st.info(
        f"**Inicial:** {name_ini} ({fmt_fecha(fecha_ini)})  \n"
        f"**Final:** {name_fin} ({fmt_fecha(fecha_fin)})"
    )

    df_header  = compare_headers(h_ini, h_fin)
    df_species = compare_species(d_ini, d_fin)

    total_ini  = pd.to_numeric(h_ini.get("total_posicion"), errors="coerce")
    total_fin  = pd.to_numeric(h_fin.get("total_posicion"), errors="coerce")
    disp_fin   = pd.to_numeric(h_fin.get("portafolio_disponible"), errors="coerce")
    disp_var   = disp_fin - pd.to_numeric(h_ini.get("portafolio_disponible"), errors="coerce")
    total_var  = total_fin - total_ini
    total_pct  = _pct_change(total_fin, total_ini)

    res_fin    = pd.to_numeric(d_fin["Resultado"], errors="coerce").fillna(0).sum()
    mov_counts = df_species["Movimiento"].value_counts().to_dict()

    # ── KPIs ──────────────────────────────────────────────
    st.markdown('<div class="cp-section">Resumen ejecutivo</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def _delta_color(v): return SUCCESS if v >= 0 else DANGER

    with k1:
        st.markdown(_kpi_html(
            "Total posición final",
            _fmt_money(total_fin),
            f"{_fmt_money(total_var)} · {_fmt_pct(total_pct)}",
            _delta_color(total_var),
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(_kpi_html(
            "Portafolio disponible",
            _fmt_money(disp_fin),
            f"var: {_fmt_money(disp_var)}",
            _delta_color(disp_var),
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(_kpi_html(
            "Resultado acumulado (fin)",
            _fmt_money(res_fin),
            "suma de resultados individuales",
            MUTED,
        ), unsafe_allow_html=True)

    with k4:
        altas   = mov_counts.get("Alta", 0)
        bajas   = mov_counts.get("Baja", 0)
        compras = mov_counts.get("Compra", 0)
        st.markdown(_kpi_html(
            "Movimientos",
            f"{len(df_species)} especies",
            f"Altas {altas} · Bajas {bajas} · Compras {compras}",
            MUTED,
        ), unsafe_allow_html=True)

    # ── Comparación general ────────────────────────────────
    st.markdown('<div class="cp-section">Comparación inicio vs. fin</div>', unsafe_allow_html=True)
    st.dataframe(
        _styled(df_header, {
            "Inicio":       "$ {:,.2f}",
            "Fin":          "$ {:,.2f}",
            "Variación $":  "$ {:,.2f}",
            "Variación %":  "{:+,.2f}%",
        }),
        use_container_width=True,
        height=240,
    )

    # ── Detalle por especie ────────────────────────────────
    st.markdown('<div class="cp-section">Detalle general por especie</div>', unsafe_allow_html=True)

    cols_general = [
        "Especie", "Categoria", "Tipo", "Moneda",
        "Cant_Ini", "Cant_Fin", "Var_Cant",
        "Imp_Ini",  "Imp_Fin",  "Var_Importe", "Var_Pct",
        "Movimiento",
    ]
    df_general = df_species[cols_general].copy()

    st.dataframe(
        _styled(df_general, {
            "Cant_Ini":    "{:,.2f}",
            "Cant_Fin":    "{:,.2f}",
            "Var_Cant":    "{:+,.2f}",
            "Imp_Ini":     "$ {:,.2f}",
            "Imp_Fin":     "$ {:,.2f}",
            "Var_Importe": "$ {:+,.2f}",
            "Var_Pct":     "{:+,.2f}%",
        }),
        use_container_width=True,
        height=500,
    )

    # ── Compras / Ventas / Sin cambio ─────────────────────
    st.markdown('<div class="cp-section">Por tipo de movimiento</div>', unsafe_allow_html=True)

    tab_compras, tab_ventas, tab_altas, tab_bajas, tab_igual = st.tabs(
        ["Compras", "Ventas", "Altas", "Bajas", "Sin cambio"]
    )

    cols_mov = [
        "Especie", "Categoria", "Moneda",
        "Cant_Ini", "Cant_Fin", "Var_Cant",
        "Imp_Ini", "Imp_Fin", "Var_Importe",
    ]
    fmt_mov = {
        "Cant_Ini": "{:,.2f}", "Cant_Fin": "{:,.2f}", "Var_Cant": "{:+,.2f}",
        "Imp_Ini":  "$ {:,.2f}", "Imp_Fin": "$ {:,.2f}", "Var_Importe": "$ {:+,.2f}",
    }

    def _render_mov_tab(mov: str, ascending=False):
        sub = df_species[df_species["Movimiento"] == mov][cols_mov].sort_values(
            "Var_Importe", ascending=ascending
        )
        if sub.empty:
            st.info(f"No hay operaciones de tipo '{mov}'.")
        else:
            st.dataframe(_styled(sub, fmt_mov), use_container_width=True, height=300)

    with tab_compras:  _render_mov_tab("Compra")
    with tab_ventas:   _render_mov_tab("Venta", ascending=True)
    with tab_altas:    _render_mov_tab("Alta")
    with tab_bajas:    _render_mov_tab("Baja", ascending=True)
    with tab_igual:    _render_mov_tab("Sin cambio")

    # ── Descarga ──────────────────────────────────────────
    st.markdown('<div class="cp-section">Exportar</div>', unsafe_allow_html=True)
    export_bytes = build_export(h_ini, h_fin, d_ini, d_fin, df_header, df_species)
    st.download_button(
        label="Descargar Excel comparativo",
        data=export_bytes,
        file_name="cartera_propia_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
