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


def _fmt(x, dec=2):
    try:
        return f"{float(x):,.{dec}f}"
    except Exception:
        return str(x)


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


# =============================================================================
# GRÁFICOS  (FIX #5)
# =============================================================================

def _grafico_pendientes_por_cuenta(df_all: pd.DataFrame) -> None:
    """Bar chart: pendientes y cerradas por cuenta."""
    try:
        import altair as alt
    except ImportError:
        st.info("Instalá `altair` para ver los gráficos: `pip install altair`")
        return

    df_res = build_resumen_cuentas(df_all)
    if df_res.empty:
        return

    df_melt = df_res[["cuenta", "cerradas", "pendientes"]].melt(
        id_vars="cuenta", var_name="tipo", value_name="cantidad"
    )
    df_melt["cuenta"] = df_melt["cuenta"].astype(str)

    color_scale = alt.Scale(
        domain=["cerradas", "pendientes"],
        range=["#22c55e", "#ef4444"],
    )
    chart = (
        alt.Chart(df_melt)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("cuenta:N", title="Cuenta", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("cantidad:Q", title="Especies", stack=True),
            color=alt.Color("tipo:N", scale=color_scale, legend=alt.Legend(title="")),
            tooltip=["cuenta:N", "tipo:N", "cantidad:Q"],
        )
        .properties(height=280, title="Especies por cuenta")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _grafico_dif_por_moneda(df_all: pd.DataFrame) -> None:
    """Bar chart horizontal: diferencia absoluta acumulada por moneda."""
    try:
        import altair as alt
    except ImportError:
        return

    if df_all.empty or "moneda" not in df_all.columns:
        return

    df_pend = df_all[df_all["status"] == "PENDIENTE"].copy()
    if df_pend.empty:
        st.success("Sin pendientes — nada que mostrar en el breakdown por moneda.")
        return

    df_mon = (
        df_pend.groupby("moneda", as_index=False)
        .agg(dif_abs=("dif_final", lambda s: float(s.abs().sum())),
             n_especies=("especie_h", "count"))
        .sort_values("dif_abs", ascending=False)
    )

    chart = (
        alt.Chart(df_mon)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#ef4444")
        .encode(
            y=alt.Y("moneda:N", sort="-x", title="Moneda"),
            x=alt.X("dif_abs:Q", title="Diferencia absoluta acumulada"),
            tooltip=["moneda:N", "dif_abs:Q", "n_especies:Q"],
        )
        .properties(height=max(120, len(df_mon) * 40), title="Diferencia acumulada por moneda (pendientes)")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _grafico_pct_cierre(df_all: pd.DataFrame) -> None:
    """Gauge-like: % de cierre global con una barra de progreso estilizada."""
    df_res = build_resumen_cuentas(df_all)
    if df_res.empty:
        return

    total_esp  = int(df_res["especies"].sum())
    total_cerr = int(df_res["cerradas"].sum())
    pct = (total_cerr / total_esp * 100) if total_esp else 0.0

    color = "#22c55e" if pct >= 99 else ("#f59e0b" if pct >= 90 else "#ef4444")
    st.markdown(
        f"""
        <div style="margin-bottom:8px;">
            <span style="font-size:.9rem;color:#64748b;">Cierre global</span>
            <span style="font-size:1.5rem;font-weight:800;color:{color};margin-left:10px;">{pct:.1f}%</span>
            <span style="font-size:.85rem;color:#94a3b8;margin-left:6px;">({total_cerr}/{total_esp} especies)</span>
        </div>
        <div style="background:#f1f5f9;border-radius:999px;height:10px;width:100%;overflow:hidden;">
            <div style="background:{color};width:{pct:.1f}%;height:100%;border-radius:999px;transition:width .4s;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_kpis(resumenes: list[dict]) -> None:
    total_esp  = sum(r["especies"]       for r in resumenes)
    total_cerr = sum(r["cerradas"]       for r in resumenes)
    total_pend = sum(r["pendientes"]     for r in resumenes)
    total_dif  = sum(r["dif_total_abs"]  for r in resumenes)
    pct_global = (total_cerr / total_esp * 100) if total_esp else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Especies totales", total_esp)
    c2.metric("Cerradas", f"{total_cerr}  ({pct_global:.1f}%)")
    c3.metric(
        "Pendientes", total_pend,
        delta=None if total_pend == 0 else f"-{total_pend}",
        delta_color="inverse",
    )
    c4.metric("Dif. total abs.", _fmt(total_dif))


# =============================================================================
# render()
# =============================================================================

def render(_=None):
    st.markdown("<div class='section-title'>Cartera Propia</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Conciliación mensual: Ini + Activity = Fin</div>",
        unsafe_allow_html=True,
    )

    tab_config, tab_archivos, tab_resultado, tab_detalle, tab_export = st.tabs([
        "⚙️ Configuración",
        "📁 Archivos",
        "📊 Resultado",
        "🔍 Detalle",
        "⬇️ Exportar",
    ])

    s  = _state()
    fc = _file_cache()

    # =========================================================================
    # TAB 1 — Configuración
    # =========================================================================
    with tab_config:
        st.markdown("#### Período y parámetros")

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
        st.markdown("#### Cuentas a conciliar")
        cuentas_str = st.text_input(
            "Cuentas (separadas por coma)",
            value=", ".join(str(c) for c in CUENTAS_DEFAULT),
            key="cp_cuentas",
        )

        st.markdown("#### Pares inter-cuenta")
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

        s["periodo"]    = PeriodoConciliacion(
            fecha_ini=str(fecha_ini), fecha_fin=str(fecha_fin),
            cuentas=cuentas, pares_comp=pares,
        )
        s["tolerancia"] = tolerancia

    # =========================================================================
    # TAB 2 — Archivos
    # FIX #4: los uploaders persisten en fc (file_cache) independientemente
    # de lo que cambie en Configuración. Solo se borran con "Limpiar archivos".
    # =========================================================================
    with tab_archivos:
        periodo: PeriodoConciliacion = s.get("periodo", PeriodoConciliacion(
            fecha_ini=str(date.today()), fecha_fin=str(date.today()),
        ))

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
            st.markdown("**Portafolios iniciales**")
            for cta in periodo.cuentas:
                key = f"cp_port_ini_{cta}"
                st.file_uploader(
                    f"Cta {cta} — Inicio", type=["csv", "xlsx", "xls"],
                    key=key, on_change=_on_upload, args=(key,),
                )
                if key in fc:
                    st.caption(f"✅ {fc[key]['name']}")

        with col_fin:
            st.markdown("**Portafolios finales**")
            for cta in periodo.cuentas:
                key = f"cp_port_fin_{cta}"
                st.file_uploader(
                    f"Cta {cta} — Fin", type=["csv", "xlsx", "xls"],
                    key=key, on_change=_on_upload, args=(key,),
                )
                if key in fc:
                    st.caption(f"✅ {fc[key]['name']}")

        st.markdown("**Activity**")
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

        st.markdown("**Posiciones (auxiliar, opcional)**")
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
    # TAB 3 — Resultado  (FIX #5: gráficos)
    # =========================================================================
    with tab_resultado:
        if "resumenes" not in s:
            st.info("Ejecutá la conciliación en la pestaña **Archivos**.")
        else:
            resumenes: list[dict] = s["resumenes"]
            df_all: pd.DataFrame  = s["df_all"]

            # KPIs
            _show_kpis(resumenes)
            st.divider()

            # Barra de progreso de cierre
            _grafico_pct_cierre(df_all)
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            # Gráficos
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                _grafico_pendientes_por_cuenta(df_all)
            with col_g2:
                _grafico_dif_por_moneda(df_all)

            st.divider()

            # Tabla resumen por cuenta
            st.markdown("#### Resumen por cuenta")
            df_res = build_resumen_cuentas(df_all)
            if not df_res.empty:
                df_res["🚦"] = df_res["semaforo"].map(_SEMAFORO)
                st.dataframe(
                    df_res[["🚦", "cuenta", "especies", "cerradas", "pendientes",
                             "dif_total_abs", "dif_total_neto", "pct_cierre"]],
                    use_container_width=True, hide_index=True,
                )

            # Top pendientes
            df_top = build_top_pendientes(df_all, top_n=25)
            if not df_top.empty:
                st.markdown("#### Top 25 pendientes")
                st.dataframe(df_top, use_container_width=True, hide_index=True)
            else:
                st.success("🎉 Sin diferencias pendientes.")

    # =========================================================================
    # TAB 4 — Detalle
    # =========================================================================
    with tab_detalle:
        if "df_all" not in s:
            st.info("Ejecutá la conciliación primero.")
        else:
            df_all = s["df_all"]

            col_f1, col_f2, col_f3 = st.columns(3)
            cuentas_disp = sorted(df_all["cuenta"].unique().tolist()) if not df_all.empty else []
            cta_sel    = col_f1.multiselect("Cuenta",  cuentas_disp, default=cuentas_disp, key="cp_det_cta")
            status_sel = col_f2.multiselect("Status",  ["CERRADA", "PENDIENTE"], default=["CERRADA", "PENDIENTE"], key="cp_det_status")
            buscar     = col_f3.text_input("Buscar especie", key="cp_det_esp")

            df_view = df_all.copy()
            if cta_sel:
                df_view = df_view[df_view["cuenta"].isin(cta_sel)]
            if status_sel:
                df_view = df_view[df_view["status"].isin(status_sel)]
            if buscar:
                df_view = df_view[df_view["especie_h"].str.contains(buscar.upper(), na=False)]

            st.caption(f"{len(df_view)} filas mostradas")
            st.dataframe(df_view, use_container_width=True, hide_index=True)

            if s.get("df_intercuenta") is not None and not s["df_intercuenta"].empty:
                st.divider()
                st.markdown("#### Auditoría inter-cuenta")
                st.dataframe(s["df_intercuenta"], use_container_width=True, hide_index=True)

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

        def _filtrar(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "cuenta" not in df.columns:
                return df
            sub = df[df["cuenta"] == cta]
            return sub if not sub.empty else df

        df_ini = _filtrar(df_ini)
        df_fin = _filtrar(df_fin)
        df_act = _filtrar(df_act)
        df_pos = _filtrar(df_pos)

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
