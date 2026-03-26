#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cartera_propia.py
# Módulo Streamlit — se integra al NEIX Workbench vía render()

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from tools.comerciales.cartera_propia_core import (
    CUENTAS_DEFAULT,
    DEFAULT_PARES_COMP,
    ConciliadorMensual,
    PeriodoConciliacion,
    RuleEngine,
    auditoria_intercuenta,
    build_excel_bytes,
    build_reglas_aplicadas,
    build_resumen_cuentas,
    build_top_pendientes,
    preparar_activity,
    preparar_portafolio,
    preparar_posiciones,
    read_any_file,
    _safe_concat,
)

# =============================================================================
# HELPERS UI
# =============================================================================

_SEMAFORO = {"OK": "🟢", "WARN": "🟡", "BAD": "🔴"}
_COLORES_CUENTA = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b",
    "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6",
    "#f97316", "#6366f1",
]


def _fmt_num(x, dec=0) -> str:
    """Número con separadores estilo AR: punto para miles, coma para decimal."""
    try:
        v = float(x)
        s = f"{v:,.{dec}f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(x)


def _fmt_compacto(x) -> str:
    """
    Formato compacto para métricas grandes: M para millones, K para miles.
    Estilo AR con punto de miles en la parte entera.
    Ej: 14.809.962.000 → 14.810 M  |  78.528.435 → 78.528 K
    """
    try:
        v = abs(float(x))
        signo = "-" if float(x) < 0 else ""
        if v >= 1_000_000_000:
            # miles de millones → mostrar en millones con separador
            m = v / 1_000_000
            return f"{signo}{_fmt_num(m, 0)} M"
        if v >= 1_000_000:
            m = v / 1_000_000
            return f"{signo}{_fmt_num(m, 1)} M"
        if v >= 1_000:
            return f"{signo}{_fmt_num(v / 1_000, 1)} K"
        return f"{signo}{_fmt_num(v, 0)}"
    except Exception:
        return str(x)


def _fmt_ars(x, dec=0) -> str:
    """Importe en ARS: $ 1.234.567"""
    try:
        return f"$ {_fmt_num(float(x), dec)}"
    except Exception:
        return str(x)


def _fmt_ars_compacto(x) -> str:
    """Importe en ARS compacto para métricas: $ 14.810 M"""
    try:
        return f"$ {_fmt_compacto(x)}"
    except Exception:
        return str(x)


def _fmt_usd(x, dec=0) -> str:
    """Importe en USD: U$S 1.234.567"""
    try:
        return f"U$S {_fmt_num(float(x), dec)}"
    except Exception:
        return str(x)


def _fmt_usd_compacto(x) -> str:
    """Importe en USD compacto para métricas: U$S 14.810 M"""
    try:
        return f"U$S {_fmt_compacto(x)}"
    except Exception:
        return str(x)


def _fmt_moneda(x, moneda: str, dec=0) -> str:
    """Formatea con signo según moneda (ARS o USD)."""
    moneda_up = str(moneda).upper()
    if "USD" in moneda_up or "U$S" in moneda_up:
        return _fmt_usd(x, dec)
    return _fmt_ars(x, dec)


def _fmt(x, dec=2):
    """Legado — número genérico sin signo."""
    return _fmt_num(x, dec)


def _state() -> dict:
    """Namespace aislado en session_state para este módulo."""
    if "cp_state" not in st.session_state:
        st.session_state["cp_state"] = {}
    return st.session_state["cp_state"]


def _file_cache() -> dict:
    """
    FIX #4 — caché de archivos separado del estado de resultados.
    Los uploaders guardan sus bytes acá al momento de carga, por lo que
    cambiar parámetros en Configuración no los borra.
    """
    if "cp_files" not in st.session_state:
        st.session_state["cp_files"] = {}
    return st.session_state["cp_files"]

def _grafico_pct_cierre(df_all: pd.DataFrame) -> None:
    """Barra de cierre ejecutiva."""
    df_res = build_resumen_cuentas(df_all)
    if df_res.empty:
        return
    total_esp  = int(df_res["especies"].sum())
    total_cerr = int(df_res["cerradas"].sum())
    pct = (total_cerr / total_esp * 100) if total_esp else 0.0
    color = "#166534" if pct >= 99 else ("#92400e" if pct >= 90 else "#b91c1c")
    st.markdown(f"""
        <div class="cp-progress-wrap">
            <div class="cp-progress-label">Cierre del período</div>
            <span class="cp-progress-pct" style="color:{color}">{pct:.1f}%</span>
            <span class="cp-progress-detail">{total_cerr} de {total_esp} especies conciliadas</span>
            <div class="cp-bar-outer">
                <div class="cp-bar-inner" style="width:{pct:.2f}%;background:{color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def _show_kpis(resumenes: list[dict], df_all: pd.DataFrame | None = None) -> None:
    total_esp  = sum(r["especies"]   for r in resumenes)
    total_cerr = sum(r["cerradas"]   for r in resumenes)
    total_pend = sum(r["pendientes"] for r in resumenes)
    pct_global = (total_cerr / total_esp * 100) if total_esp else 0.0

    # Valor en juego: suma del importe de las diferencias pendientes
    # (dif_importe = dif_final * precio_ref → valor económico real de la brecha)
    valor_en_juego_ars = 0.0
    valor_en_juego_usd = 0.0
    if df_all is not None and not df_all.empty and "dif_importe" in df_all.columns:
        df_pend = df_all[df_all["status"] == "PENDIENTE"]
        if not df_pend.empty:
            mask_usd = df_pend["moneda"].str.upper().str.contains("USD", na=False)
            valor_en_juego_ars = float(df_pend.loc[~mask_usd, "dif_importe"].abs().sum())
            valor_en_juego_usd = float(df_pend.loc[ mask_usd, "dif_importe"].abs().sum())

    _help = "Valor económico estimado (nominal × precio de referencia) — ARS y USD nunca se suman"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Especies", _fmt_num(total_esp, 0))
    c2.metric("Conciliadas ✅", f"{_fmt_num(total_cerr, 0)}  ({pct_global:.1f}%)")
    c3.metric(
        "Sin conciliar 🔴", _fmt_num(total_pend, 0),
        delta=None if total_pend == 0 else f"-{total_pend}",
        delta_color="inverse",
    )
    c4.metric(
        "Brecha ARS",
        _fmt_ars_compacto(valor_en_juego_ars) if valor_en_juego_ars > 0 else "$ 0",
        help=_help,
    )
    c5.metric(
        "Brecha USD",
        _fmt_usd_compacto(valor_en_juego_usd) if valor_en_juego_usd > 0 else "U$S 0",
        help=_help,
    )


# =============================================================================
# render()
# =============================================================================

# CSS ejecutivo inyectado una sola vez
_CSS = """
<style>
/* ── Importar tipografías ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Variables ───────────────────────────────────────────────────────────── */
:root {
    --cp-ink:      #0d1117;
    --cp-ink-2:    #374151;
    --cp-ink-3:    #6b7280;
    --cp-line:     #e5e7eb;
    --cp-line-2:   #f3f4f6;
    --cp-red:      #b91c1c;
    --cp-red-light:#fef2f2;
    --cp-green:    #166534;
    --cp-green-light:#f0fdf4;
    --cp-gold:     #92400e;
    --cp-gold-light:#fffbeb;
}

/* ── Encabezado del módulo ───────────────────────────────────────────────── */
.cp-header {
    padding: 28px 0 20px 0;
    border-bottom: 1px solid var(--cp-line);
    margin-bottom: 8px;
}
.cp-header-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--cp-red);
    margin-bottom: 6px;
}
.cp-header-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 26px;
    font-weight: 600;
    color: var(--cp-ink);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 4px;
}
.cp-header-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 300;
    color: var(--cp-ink-3);
    letter-spacing: 0.01em;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--cp-line) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--cp-ink-3) !important;
    padding: 10px 20px 12px !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cp-ink) !important;
    border-bottom: 2px solid var(--cp-ink) !important;
    font-weight: 500 !important;
}

/* ── Métricas (KPIs) ─────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    border-top: 2px solid var(--cp-ink) !important;
    border-radius: 0 !important;
    padding: 16px 0 8px 0 !important;
    box-shadow: none !important;
}
div[data-testid="metric-container"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--cp-ink-3) !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    color: var(--cp-ink) !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
}

/* ── Tablas ───────────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--cp-line) !important;
    border-radius: 2px !important;
    overflow: hidden !important;
}
div[data-testid="stDataFrame"] thead tr th {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--cp-ink-3) !important;
    background: var(--cp-line-2) !important;
    border-bottom: 1px solid var(--cp-line) !important;
    padding: 10px 12px !important;
}
div[data-testid="stDataFrame"] tbody tr td {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    color: var(--cp-ink) !important;
    border-bottom: 1px solid var(--cp-line-2) !important;
    padding: 9px 12px !important;
}
div[data-testid="stDataFrame"] tbody tr:last-child td {
    border-bottom: none !important;
}

/* ── Secciones internas ───────────────────────────────────────────────────── */
.cp-section {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 16px;
    font-weight: 400;
    color: var(--cp-ink);
    letter-spacing: -0.01em;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--cp-line);
}
.cp-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--cp-ink-3);
    margin: 20px 0 8px 0;
}
.cp-label-ars {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: var(--cp-line-2);
    color: var(--cp-ink-2);
    border: 1px solid var(--cp-line);
    border-radius: 2px;
    padding: 2px 8px;
    margin-right: 6px;
}
.cp-label-usd {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 2px;
    padding: 2px 8px;
    margin-right: 6px;
}
.cp-stat {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: var(--cp-ink-3);
    display: inline;
}
.cp-stat-val {
    font-weight: 500;
    color: var(--cp-ink);
}

/* ── Divider ───────────────────────────────────────────────────────────────── */
hr[data-testid="stDivider"] {
    border-color: var(--cp-line) !important;
    margin: 20px 0 !important;
}

/* ── Barra de progreso ────────────────────────────────────────────────────── */
.cp-progress-wrap {
    margin: 16px 0 24px 0;
}
.cp-progress-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cp-ink-3);
    margin-bottom: 10px;
}
.cp-progress-pct {
    font-family: 'DM Sans', sans-serif;
    font-size: 32px;
    font-weight: 300;
    letter-spacing: -0.03em;
    color: var(--cp-ink);
    display: inline;
}
.cp-progress-detail {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: var(--cp-ink-3);
    margin-left: 10px;
    display: inline;
}
.cp-bar-outer {
    background: var(--cp-line);
    border-radius: 0;
    height: 3px;
    width: 100%;
    overflow: hidden;
    margin-top: 10px;
}
.cp-bar-inner {
    height: 100%;
    border-radius: 0;
    transition: width .6s cubic-bezier(.4,0,.2,1);
}
</style>
"""


def render(_=None):
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
        <div class="cp-header">
            <div class="cp-header-eyebrow">NEIX · Comercial</div>
            <div class="cp-header-title">Conciliación de Cartera Propia</div>
            <div class="cp-header-sub">Ini + Activity = Fin &nbsp;·&nbsp; Por cuenta, especie y moneda</div>
        </div>
    """, unsafe_allow_html=True)

    tab_config, tab_archivos, tab_resultado, tab_detalle, tab_export = st.tabs([
        "Configuración",
        "Archivos",
        "Resultado",
        "Detalle",
        "Exportar",
    ])

    s  = _state()
    fc = _file_cache()

    # =========================================================================
    # TAB 1 — Configuración
    # =========================================================================
    with tab_config:
        st.markdown("<div class='cp-section'>Período y parámetros</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fecha_ini  = st.date_input("Fecha inicio", value=date(2026, 2, 1),  key="cp_fecha_ini")
            tolerancia = st.number_input(
                "Tolerancia de cierre", value=0.01, step=0.001, format="%.4f",
                help="Diferencia ≤ tolerancia → especie CERRADA",
                key="cp_tolerancia",
            )
        with col2:
            fecha_fin = st.date_input("Fecha fin", value=date(2026, 2, 28), key="cp_fecha_fin")

        st.divider()
        st.markdown("<div class='cp-sub'>Cuentas a conciliar</div>", unsafe_allow_html=True)
        cuentas_str = st.text_input(
            "Cuentas (separadas por coma)",
            value=", ".join(str(c) for c in CUENTAS_DEFAULT),
            key="cp_cuentas",
        )

        st.markdown("<div class='cp-sub'>Pares inter-cuenta</div>", unsafe_allow_html=True)
        pares_str = st.text_input(
            "Pares (formato: 904-992, 997-999)",
            value=", ".join(f"{a}-{b}" for a, b in DEFAULT_PARES_COMP),
            key="cp_pares",
        )
        st.caption("Se usarán para la auditoría de movimientos entre cuentas.")

        try:
            cuentas = [int(x.strip()) for x in cuentas_str.split(",") if x.strip()]
        except ValueError:
            st.error("Formato de cuentas inválido.")
            cuentas = list(CUENTAS_DEFAULT)

        try:
            pares = []
            for p in pares_str.split(","):
                p = p.strip()
                if "-" in p:
                    a, b = p.split("-", 1)
                    pares.append((int(a.strip()), int(b.strip())))
        except ValueError:
            st.error("Formato de pares inválido.")
            pares = list(DEFAULT_PARES_COMP)

        try:
            s["periodo"] = PeriodoConciliacion(
                fecha_ini=str(fecha_ini), fecha_fin=str(fecha_fin),
                cuentas=cuentas, pares_comp=pares,
            )
        except ValueError as e:
            st.error(f"⚠️ Período inválido: {e}")
            s.pop("periodo", None)   # no dejar un período roto guardado
        s["tolerancia"] = tolerancia

    # =========================================================================
    # TAB 2 — Archivos
    # FIX #4: los uploaders persisten en fc (file_cache) independientemente
    # de lo que cambie en Configuración. Solo se borran con "Limpiar archivos".
    # =========================================================================
    with tab_archivos:
        # Fallback seguro: si todavía no hay período configurado usamos uno
        # dummy con fechas distintas para no disparar la validación fecha_ini == fin.
        _periodo_fallback = s.get("periodo")
        if _periodo_fallback is None:
            from datetime import timedelta
            _hoy = date.today()
            try:
                _periodo_fallback = PeriodoConciliacion(
                    fecha_ini=str(_hoy.replace(day=1)),
                    fecha_fin=str(_hoy),
                )
            except ValueError:
                # Si hoy es día 1 (ini == fin), desplazamos fin +1
                _periodo_fallback = PeriodoConciliacion(
                    fecha_ini=str(_hoy),
                    fecha_fin=str(_hoy + timedelta(days=1)),
                )
        periodo: PeriodoConciliacion = _periodo_fallback

        st.markdown(
            f"Archivos para **{periodo.fecha_ini} → {periodo.fecha_fin}**. "
            "Formatos: CSV, XLSX, XLS."
        )

        # Callback que guarda bytes en fc cuando se sube un archivo,
        # evitando que el widget pierda el archivo al re-render.
        def _on_upload(key: str):
            uploaded = st.session_state.get(key)
            if uploaded is not None:
                uploaded.seek(0)
                fc[key] = {"name": uploaded.name, "data": uploaded.read()}

        col_ini, col_fin = st.columns(2)

        with col_ini:
            st.markdown("<div class='cp-sub'>Portafolios iniciales</div>", unsafe_allow_html=True)
            for cta in periodo.cuentas:
                key = f"cp_port_ini_{cta}"
                st.file_uploader(
                    f"Cta {cta} — Inicio", type=["csv", "xlsx", "xls"],
                    key=key, on_change=_on_upload, args=(key,),
                )
                if key in fc:
                    st.caption(f"✅ {fc[key]['name']}")

        with col_fin:
            st.markdown("<div class='cp-sub'>Portafolios finales</div>", unsafe_allow_html=True)
            for cta in periodo.cuentas:
                key = f"cp_port_fin_{cta}"
                st.file_uploader(
                    f"Cta {cta} — Fin", type=["csv", "xlsx", "xls"],
                    key=key, on_change=_on_upload, args=(key,),
                )
                if key in fc:
                    st.caption(f"✅ {fc[key]['name']}")

        st.markdown("<div class='cp-sub'>Activity</div>", unsafe_allow_html=True)
        cols_act = st.columns(min(len(periodo.cuentas), 4))
        for i, cta in enumerate(periodo.cuentas):
            with cols_act[i % len(cols_act)]:
                key = f"cp_activity_{cta}"
                st.file_uploader(
                    f"Activity Cta {cta}", type=["csv", "xlsx", "xls"],
                    key=key, on_change=_on_upload, args=(key,),
                )
                if key in fc:
                    st.caption(f"✅ {fc[key]['name']}")

        st.markdown("<div class='cp-sub'>Posiciones (auxiliar, opcional)</div>", unsafe_allow_html=True)
        cols_pos = st.columns(min(len(periodo.cuentas), 4))
        for i, cta in enumerate(periodo.cuentas):
            with cols_pos[i % len(cols_pos)]:
                key = f"cp_pos_{cta}"
                st.file_uploader(
                    f"Posiciones Cta {cta}", type=["csv", "xlsx", "xls"],
                    key=key, on_change=_on_upload, args=(key,),
                )
                if key in fc:
                    st.caption(f"✅ {fc[key]['name']}")

        n_cargados = len(fc)
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            run = st.button(
                f"🚀 Ejecutar conciliación ({n_cargados} archivo(s) cargado(s))",
                type="primary", use_container_width=True, key="cp_run",
                disabled=n_cargados == 0,
            )
        with col_btn2:
            if st.button("🗑 Limpiar archivos", use_container_width=True, key="cp_clear_files"):
                st.session_state["cp_files"] = {}
                st.rerun()

        if n_cargados == 0:
            st.info("Cargá al menos un archivo para continuar.")

        if run:
            _ejecutar(s, fc)

    # =========================================================================
    # TAB 3 — Resultado
    # =========================================================================
    with tab_resultado:
        if "resumenes" not in s:
            st.info("Ejecutá la conciliación en la pestaña **Archivos**.")
        else:
            resumenes: list[dict] = s["resumenes"]
            df_all: pd.DataFrame  = s["df_all"]

            # ── KPIs ─────────────────────────────────────────────────────────
            _show_kpis(resumenes, df_all)
            st.divider()

            # ── Barra de cierre global ────────────────────────────────────────
            _grafico_pct_cierre(df_all)
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            st.divider()

            # ── Resumen por cuenta ────────────────────────────────────────────
            st.markdown("<div class='cp-section'>Resumen por cuenta</div>", unsafe_allow_html=True)
            df_res = build_resumen_cuentas(df_all)
            if not df_res.empty:
                df_display = pd.DataFrame({
                    "Estado":      df_res["semaforo"].map(_SEMAFORO),
                    "Cuenta":      df_res["cuenta"],
                    "Especies":    df_res["especies"],
                    "Conciliadas": df_res["cerradas"],
                    "Pendientes":  df_res["pendientes"],
                    "% Cierre":    df_res["pct_cierre"].map(lambda x: f"{x:.1f}%"),
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)

            # ── Brecha por moneda (resumen rápido, sin mezclar) ───────────────
            if not df_all.empty and "dif_importe" in df_all.columns:
                df_pend_all = df_all[df_all["status"] == "PENDIENTE"]
                if not df_pend_all.empty:
                    resumen_mon = (
                        df_pend_all.groupby("moneda", as_index=False)
                        .agg(
                            pendientes=("especie_h", "count"),
                            brecha_importe=("dif_importe", lambda s: s.abs().sum()),
                        )
                        .sort_values("moneda")
                    )
                    resumen_mon["Brecha"] = [
                        _fmt_moneda(v, m)
                        for v, m in zip(resumen_mon["brecha_importe"], resumen_mon["moneda"])
                    ]
                    st.markdown("<div class='cp-sub'>Brecha por moneda</div>", unsafe_allow_html=True)
                    df_mon_display = pd.DataFrame({
                        "Moneda":     resumen_mon["moneda"],
                        "Pendientes": resumen_mon["pendientes"],
                        "Brecha":     resumen_mon["Brecha"],
                    })
                    st.dataframe(df_mon_display, use_container_width=True, hide_index=True)

            # ── Pendientes separados por moneda ───────────────────────────────
            df_pend = df_all[df_all["status"] == "PENDIENTE"].copy() if not df_all.empty else pd.DataFrame()
            if df_pend.empty:
                st.success("🎉 Sin diferencias pendientes.")
            else:
                st.markdown(f"<div class='cp-section'>Pendientes &nbsp;<span style='font-family:DM Sans,sans-serif;font-size:13px;font-weight:300;color:#6b7280;'>{len(df_pend)} especie(s)</span></div>", unsafe_allow_html=True)

                def _tabla_pendientes(df_bloque: pd.DataFrame) -> None:
                    """Renderiza una tabla de pendientes limpia y ordenada."""
                    if df_bloque.empty:
                        return
                    df_d = pd.DataFrame({
                        "Cuenta":       df_bloque["cuenta"],
                        "Especie":      df_bloque["especie_h"],
                        "Nominal ini":  df_bloque["ini"].map(lambda x: _fmt_num(x, 0)),
                        "Activity":     df_bloque["act"].map(lambda x: _fmt_num(x, 0)),
                        "Nominal fin":  df_bloque["fin"].map(lambda x: _fmt_num(x, 0)),
                        "Dif. nominal": df_bloque["dif_final"].map(lambda x: _fmt_num(x, 0)),
                        "Dif. importe": [_fmt_moneda(v, m) for v, m in
                                         zip(df_bloque["dif_importe"], df_bloque["moneda"])],
                    })
                    df_d["_abs"] = df_bloque["dif_importe"].abs().values
                    df_d = df_d.sort_values("_abs", ascending=False).drop(columns=["_abs"])
                    st.dataframe(df_d, use_container_width=True, hide_index=True)

                # ── ARS ───────────────────────────────────────────────────────
                pend_ars = df_pend[df_pend["moneda"] == "ARS"]
                if not pend_ars.empty:
                    total_imp_ars = pend_ars["dif_importe"].abs().sum()
                    st.markdown(
                        f"<span class='cp-label-ars'>ARS</span>"
                        f"<span class='cp-stat'>{len(pend_ars)} especie(s) &nbsp;·&nbsp; "
                        f"<span class='cp-stat-val'>{_fmt_ars_compacto(total_imp_ars)}</span> en juego</span>",
                        unsafe_allow_html=True,
                    )
                    _tabla_pendientes(pend_ars)

                # ── USD ───────────────────────────────────────────────────────
                pend_usd = df_pend[df_pend["moneda"] != "ARS"]
                if not pend_usd.empty:
                    total_imp_usd = pend_usd["dif_importe"].abs().sum()
                    st.markdown(
                        f"<span class='cp-label-usd'>USD</span>"
                        f"<span class='cp-stat'>{len(pend_usd)} especie(s) &nbsp;·&nbsp; "
                        f"<span class='cp-stat-val'>{_fmt_usd_compacto(total_imp_usd)}</span> en juego</span>",
                        unsafe_allow_html=True,
                    )
                    _tabla_pendientes(pend_usd)

    # =========================================================================
    # TAB 4 — Detalle
    # =========================================================================
    with tab_detalle:
        if "df_all" not in s:
            st.info("Ejecutá la conciliación primero.")
        else:
            df_all = s["df_all"]

            # ── Filtros ───────────────────────────────────────────────────────
            col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 3])
            cuentas_disp = sorted(df_all["cuenta"].unique().tolist()) if not df_all.empty else []
            monedas_disp = sorted(df_all["moneda"].unique().tolist()) if not df_all.empty else []

            cta_sel    = col_f1.multiselect("Cuenta",  cuentas_disp, default=cuentas_disp, key="cp_det_cta")
            status_sel = col_f2.multiselect("Estado",  ["CERRADA", "PENDIENTE"], default=["CERRADA", "PENDIENTE"], key="cp_det_status")
            mon_sel    = col_f3.multiselect("Moneda",  monedas_disp, default=monedas_disp, key="cp_det_moneda")
            buscar     = col_f4.text_input("🔍 Buscar especie", key="cp_det_esp", placeholder="Ej: AL30, GGAL...")

            df_view = df_all.copy()
            if cta_sel:
                df_view = df_view[df_view["cuenta"].isin(cta_sel)]
            if status_sel:
                df_view = df_view[df_view["status"].isin(status_sel)]
            if mon_sel:
                df_view = df_view[df_view["moneda"].isin(mon_sel)]
            if buscar:
                df_view = df_view[df_view["especie_h"].str.contains(buscar.upper(), na=False)]

            # ── Tabla limpia: solo columnas útiles ────────────────────────────
            st.caption(f"{len(df_view)} especie(s) mostrada(s)")

            if not df_view.empty:
                df_clean = pd.DataFrame({
                    "Estado":       df_view["status"].map({"CERRADA": "✅ Cerrada", "PENDIENTE": "🔴 Pendiente"}),
                    "Cuenta":       df_view["cuenta"],
                    "Especie":      df_view["especie_h"],
                    "Moneda":       df_view["moneda"],
                    "Nominal ini":  df_view["ini"].map(lambda x: _fmt_num(x, 0)),
                    "Activity":     df_view["act"].map(lambda x: _fmt_num(x, 0)),
                    "Nominal fin":  df_view["fin"].map(lambda x: _fmt_num(x, 0)),
                    "Dif. nominal": df_view["dif_final"].map(lambda x: _fmt_num(x, 0)),
                    "Dif. importe": [_fmt_moneda(v, m) for v, m in zip(df_view["dif_importe"], df_view["moneda"])],
                })
                st.dataframe(df_clean, use_container_width=True, hide_index=True)

            # ── Auditoría inter-cuenta ────────────────────────────────────────
            df_ic = s.get("df_intercuenta")
            if df_ic is not None and not df_ic.empty:
                st.divider()
                st.markdown("<div class='cp-section'>Movimientos entre cuentas</div>", unsafe_allow_html=True)
                df_ic_clean = pd.DataFrame({
                    "Cuenta origen":  df_ic["cuenta_origen"],
                    "Cuenta destino": df_ic["cuenta_destino"],
                    "Especie":        df_ic["especie_h"],
                    "Moneda":         df_ic["moneda"],
                    "Dif. origen":    [_fmt_moneda(v, m) for v, m in zip(df_ic["dif_origen"],  df_ic["moneda"])],
                    "Dif. destino":   [_fmt_moneda(v, m) for v, m in zip(df_ic["dif_destino"], df_ic["moneda"])],
                    "Neto":           [_fmt_moneda(v, m) for v, m in zip(df_ic["suma_neta"],   df_ic["moneda"])],
                })
                st.dataframe(df_ic_clean, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 5 — Exportar
    # =========================================================================
    with tab_export:
        if "df_all" not in s:
            st.info("Ejecutá la conciliación primero.")
        else:
            df_all   = s["df_all"]
            periodo  = s.get("periodo")

            df_res         = build_resumen_cuentas(df_all)
            df_top         = build_top_pendientes(df_all)
            df_pend        = df_all[df_all["status"] == "PENDIENTE"].copy() if not df_all.empty else pd.DataFrame()
            df_reglas      = build_reglas_aplicadas(df_all)
            df_intercuenta = s.get("df_intercuenta", pd.DataFrame())
            df_match       = s.get("df_match", pd.DataFrame())
            df_aud         = s.get("df_aud", pd.DataFrame())

            excel_bytes = build_excel_bytes(
                df_resumen=df_res, df_top_pend=df_top, df_pend=df_pend,
                df_all=df_all, df_auditoria=df_aud, df_reglas=df_reglas,
                df_match=df_match, df_intercuenta=df_intercuenta,
                periodo=periodo,
                tolerancia=s.get("tolerancia", 0.01),
            )

            nombre = "conciliacion"
            if periodo:
                nombre += f"_{periodo.fecha_ini}_{periodo.fecha_fin}"
            nombre += ".xlsx"

            st.download_button(
                label="⬇️ Descargar Excel completo",
                data=excel_bytes, file_name=nombre,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary",
            )
            st.markdown("**Hojas:** Resumen · Top Pendientes · Pendientes · Detalle · Auditoria · Reglas · Match · Intercuenta")


# =============================================================================
# LÓGICA DE EJECUCIÓN
# =============================================================================

def _ejecutar(s: dict, fc: dict) -> None:
    periodo: PeriodoConciliacion = s.get("periodo")
    tolerancia: float = s.get("tolerancia", 0.01)

    if not periodo:
        st.error("Configurá el período primero.")
        return

    import io as _io
    conciliador = ConciliadorMensual(
        periodo=periodo,
        tolerance=tolerancia,
        rule_engine=RuleEngine(),
    )

    todas_las_filas: list[pd.DataFrame] = []
    resumenes:       list[dict]         = []
    auditorias:      list[pd.DataFrame] = []
    matches:         list[pd.DataFrame] = []
    alerts_globales: list[str]          = []

    barra     = st.progress(0, text="Iniciando…")
    n_cuentas = len(periodo.cuentas)

    for idx, cta in enumerate(periodo.cuentas):
        barra.progress(idx / n_cuentas, text=f"Conciliando cuenta {cta}…")

        def _read(key: str) -> pd.DataFrame:
            cached = fc.get(key)
            if cached is None:
                return pd.DataFrame()
            try:
                # Reconstituir un file-like desde los bytes guardados
                buf = _io.BytesIO(cached["data"])
                buf.name = cached["name"]
                return read_any_file(buf)
            except Exception as exc:
                alerts_globales.append(f"Error leyendo {key}: {exc}")
                return pd.DataFrame()

        raw_ini = _read(f"cp_port_ini_{cta}")
        raw_fin = _read(f"cp_port_fin_{cta}")
        raw_act = _read(f"cp_activity_{cta}")
        raw_pos = _read(f"cp_pos_{cta}")

        df_ini, a1 = preparar_portafolio(raw_ini)
        df_fin, a2 = preparar_portafolio(raw_fin)
        df_act, a3 = preparar_activity(raw_act)
        df_pos, a4 = preparar_posiciones(raw_pos)
        alerts_globales += [x for x in a1 + a2 + a3 + a4 if "vacío" not in x]

        def _filtrar(df: pd.DataFrame, nombre_archivo: str) -> pd.DataFrame:
            # FIX #1: si el CSV no tiene ninguna fila para esta cuenta
            # devolvemos DataFrame vacío (no todo el archivo).
            # El fallback anterior contaminaba la cuenta con datos de otras.
            if df.empty or "cuenta" not in df.columns:
                return pd.DataFrame()
            sub = df[df["cuenta"] == cta]
            if sub.empty:
                alerts_globales.append(
                    f"Cta {cta} — {nombre_archivo}: sin filas para cuenta={cta} "
                    f"(cuentas encontradas: {sorted(df['cuenta'].unique().tolist())})"
                )
            return sub

        df_ini = _filtrar(df_ini, f"port_ini_{cta}")
        df_fin = _filtrar(df_fin, f"port_fin_{cta}")
        df_act = _filtrar(df_act, f"activity_{cta}")
        df_pos = _filtrar(df_pos, f"pos_{cta}")

        df_cta, resumen, df_aud, df_match = conciliador.conciliar_cuenta(
            cuenta=cta,
            df_posiciones=df_pos, df_activity=df_act,
            df_portafolio_ini=df_ini, df_portafolio_fin=df_fin,
        )

        todas_las_filas.append(df_cta)
        resumenes.append(resumen)
        auditorias.append(df_aud)
        matches.append(df_match)

    barra.progress(1.0, text="✅ Listo")

    df_all         = conciliador.generar_reporte(todas_las_filas)
    df_aud_all     = _safe_concat(auditorias)
    df_match_all   = _safe_concat(matches)
    df_intercuenta = auditoria_intercuenta(df_all, periodo.pares_comp)

    s["df_all"]         = df_all
    s["resumenes"]      = resumenes
    s["df_aud"]         = df_aud_all
    s["df_match"]       = df_match_all
    s["df_intercuenta"] = df_intercuenta

    if alerts_globales:
        with st.expander(f"⚠️ {len(alerts_globales)} advertencia(s)"):
            for a in alerts_globales:
                st.caption(a)

    total_pend = sum(r["pendientes"] for r in resumenes)
    total_esp  = sum(r["especies"]   for r in resumenes)
    if total_pend == 0:
        st.success(f"🎉 Conciliación completada — {total_esp} especies, todas cerradas.")
    else:
        st.warning(f"⚠️ {total_pend} especie(s) pendiente(s) sobre {total_esp}. Revisá la pestaña **Resultado**.")
