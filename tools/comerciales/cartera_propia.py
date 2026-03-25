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
    auditoria_intercuenta,
    build_excel_bytes,
    build_reglas_aplicadas,
    build_resumen_cuentas,
    build_top_pendientes,
    preparar_activity,
    preparar_portafolio,
    preparar_posiciones,
    read_any_file,
)

# =============================================================================
# HELPERS UI
# =============================================================================

_SEMAFORO = {"OK": "🟢", "WARN": "🟡", "BAD": "🔴"}


def _badge(semaforo: str) -> str:
    return _SEMAFORO.get(semaforo, "⚪")


def _fmt(x, dec=2):
    try:
        return f"{float(x):,.{dec}f}"
    except Exception:
        return str(x)


def _show_resumen_cards(resumenes: list[dict]):
    """Muestra KPIs globales en métricas de Streamlit."""
    total_esp  = sum(r["especies"]   for r in resumenes)
    total_cerr = sum(r["cerradas"]   for r in resumenes)
    total_pend = sum(r["pendientes"] for r in resumenes)
    total_dif  = sum(r["dif_total_abs"] for r in resumenes)
    pct_global = (total_cerr / total_esp * 100) if total_esp else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Especies totales",   total_esp)
    c2.metric("Cerradas",           f"{total_cerr}  ({pct_global:.1f}%)")
    c3.metric("Pendientes",         total_pend,  delta=None if total_pend == 0 else f"-{total_pend}", delta_color="inverse")
    c4.metric("Dif. total abs.",    _fmt(total_dif))


# =============================================================================
# ESTADO DE SESIÓN
# =============================================================================

_KEY = "cp_state"   # clave única en session_state para este módulo


def _state() -> dict:
    if _KEY not in st.session_state:
        st.session_state[_KEY] = {}
    return st.session_state[_KEY]


# =============================================================================
# render()
# =============================================================================

def render(_=None):
    st.markdown("<div class='section-title'>Cartera Propia</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Conciliación mensual: Ini + Activity = Fin</div>",
        unsafe_allow_html=True,
    )

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab_config, tab_archivos, tab_resultado, tab_detalle, tab_export = st.tabs([
        "⚙️ Configuración",
        "📁 Archivos",
        "📊 Resultado",
        "🔍 Detalle",
        "⬇️ Exportar",
    ])

    s = _state()

    # =========================================================================
    # TAB 1 — Configuración
    # =========================================================================
    with tab_config:
        st.markdown("#### Período y parámetros")

        col1, col2 = st.columns(2)
        with col1:
            fecha_ini = st.date_input("Fecha inicio", value=date(2026, 2, 1), key="cp_fecha_ini")
            tolerancia = st.number_input(
                "Tolerancia de cierre",
                value=0.01, step=0.001, format="%.4f",
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

        # Parsear cuentas y pares
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

        s["periodo"] = PeriodoConciliacion(
            fecha_ini=str(fecha_ini),
            fecha_fin=str(fecha_fin),
            cuentas=cuentas,
            pares_comp=pares,
        )
        s["tolerancia"] = tolerancia

    # =========================================================================
    # TAB 2 — Archivos
    # =========================================================================
    with tab_archivos:
        periodo: PeriodoConciliacion = s.get("periodo", PeriodoConciliacion(
            fecha_ini=str(date.today()),
            fecha_fin=str(date.today()),
        ))

        st.markdown(
            f"Cargá los archivos para el período **{periodo.fecha_ini} → {periodo.fecha_fin}**. "
            "Formatos aceptados: CSV, XLSX, XLS."
        )
        st.caption("Si no tenés un archivo opcional, dejalo vacío — el motor lo maneja.")

        archivos: dict = {}

        col_ini, col_fin = st.columns(2)

        with col_ini:
            st.markdown("**Portafolios iniciales**")
            for cta in periodo.cuentas:
                f = st.file_uploader(
                    f"Cta {cta} — Inicio",
                    type=["csv", "xlsx", "xls"],
                    key=f"cp_port_ini_{cta}",
                )
                if f:
                    archivos[f"port_ini_{cta}"] = f

        with col_fin:
            st.markdown("**Portafolios finales**")
            for cta in periodo.cuentas:
                f = st.file_uploader(
                    f"Cta {cta} — Fin",
                    type=["csv", "xlsx", "xls"],
                    key=f"cp_port_fin_{cta}",
                )
                if f:
                    archivos[f"port_fin_{cta}"] = f

        st.markdown("**Activity**")
        cols_act = st.columns(min(len(periodo.cuentas), 4))
        for i, cta in enumerate(periodo.cuentas):
            with cols_act[i % len(cols_act)]:
                f = st.file_uploader(
                    f"Activity Cta {cta}",
                    type=["csv", "xlsx", "xls"],
                    key=f"cp_activity_{cta}",
                )
                if f:
                    archivos[f"activity_{cta}"] = f

        st.markdown("**Posiciones (auxiliar, opcional)**")
        cols_pos = st.columns(min(len(periodo.cuentas), 4))
        for i, cta in enumerate(periodo.cuentas):
            with cols_pos[i % len(cols_pos)]:
                f = st.file_uploader(
                    f"Posiciones Cta {cta}",
                    type=["csv", "xlsx", "xls"],
                    key=f"cp_pos_{cta}",
                )
                if f:
                    archivos[f"pos_{cta}"] = f

        s["archivos"] = archivos
        n = len(archivos)
        if n:
            st.success(f"✅ {n} archivo(s) cargado(s).")
        else:
            st.info("Cargá al menos un archivo para continuar.")

        st.divider()

        # ── Botón ejecutar ───────────────────────────────────────────────────
        if st.button("🚀 Ejecutar conciliación", type="primary", use_container_width=True, key="cp_run"):
            _ejecutar(s)

    # =========================================================================
    # TAB 3 — Resultado
    # =========================================================================
    with tab_resultado:
        if "resumenes" not in s:
            st.info("Ejecutá la conciliación en la pestaña **Archivos**.")
        else:
            resumenes: list[dict] = s["resumenes"]
            df_all: pd.DataFrame  = s["df_all"]

            _show_resumen_cards(resumenes)
            st.divider()

            # Tabla resumen por cuenta
            st.markdown("#### Resumen por cuenta")
            df_res = build_resumen_cuentas(df_all)
            if not df_res.empty:
                df_res["🚦"] = df_res["semaforo"].map(_SEMAFORO)
                st.dataframe(
                    df_res[["🚦", "cuenta", "especies", "cerradas", "pendientes",
                             "dif_total_abs", "dif_total_neto", "pct_cierre"]],
                    use_container_width=True,
                    hide_index=True,
                )

            # Top pendientes
            df_top = build_top_pendientes(df_all, top_n=25)
            if not df_top.empty:
                st.markdown("#### Top 25 pendientes por diferencia")
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

            # Filtros
            col_f1, col_f2, col_f3 = st.columns(3)
            cuentas_disp = sorted(df_all["cuenta"].unique().tolist()) if not df_all.empty else []
            cta_sel = col_f1.multiselect("Cuenta", cuentas_disp, default=cuentas_disp, key="cp_det_cta")
            status_sel = col_f2.multiselect("Status", ["CERRADA", "PENDIENTE"], default=["CERRADA", "PENDIENTE"], key="cp_det_status")
            buscar = col_f3.text_input("Buscar especie", key="cp_det_esp")

            df_view = df_all.copy()
            if cta_sel:
                df_view = df_view[df_view["cuenta"].isin(cta_sel)]
            if status_sel:
                df_view = df_view[df_view["status"].isin(status_sel)]
            if buscar:
                df_view = df_view[df_view["especie_h"].str.contains(buscar.upper(), na=False)]

            st.caption(f"{len(df_view)} filas mostradas")
            st.dataframe(df_view, use_container_width=True, hide_index=True)

            # Auditoría inter-cuenta
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
            df_all        = s["df_all"]
            periodo       = s.get("periodo")

            df_res        = build_resumen_cuentas(df_all)
            df_top        = build_top_pendientes(df_all)
            df_pend       = df_all[df_all["status"] == "PENDIENTE"].copy() if not df_all.empty else pd.DataFrame()
            df_reglas     = build_reglas_aplicadas(df_all)
            df_intercuenta = s.get("df_intercuenta", pd.DataFrame())
            df_match      = s.get("df_match", pd.DataFrame())
            df_aud        = s.get("df_aud", pd.DataFrame())

            excel_bytes = build_excel_bytes(
                df_resumen    = df_res,
                df_top_pend   = df_top,
                df_pend       = df_pend,
                df_all        = df_all,
                df_auditoria  = df_aud,
                df_reglas     = df_reglas,
                df_match      = df_match,
                df_intercuenta= df_intercuenta,
            )

            nombre = "conciliacion"
            if periodo:
                nombre += f"_{periodo.fecha_ini}_{periodo.fecha_fin}"
            nombre += ".xlsx"

            st.download_button(
                label="⬇️ Descargar Excel completo",
                data=excel_bytes,
                file_name=nombre,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
            )

            st.markdown("**Hojas incluidas:** Resumen · Top Pendientes · Pendientes · Detalle · Auditoria · Reglas · Match · Intercuenta")


# =============================================================================
# LÓGICA DE EJECUCIÓN
# =============================================================================

def _ejecutar(s: dict):
    periodo: PeriodoConciliacion = s.get("periodo")
    archivos: dict = s.get("archivos", {})
    tolerancia: float = s.get("tolerancia", 0.01)

    if not periodo:
        st.error("Configurá el período primero.")
        return

    conciliador = ConciliadorMensual(periodo=periodo, tolerance=tolerancia)

    todas_las_filas: list[pd.DataFrame] = []
    resumenes:       list[dict]         = []
    auditorias:      list[pd.DataFrame] = []
    matches:         list[pd.DataFrame] = []
    alerts_globales: list[str]          = []

    barra = st.progress(0, text="Iniciando…")
    n_cuentas = len(periodo.cuentas)

    for idx, cta in enumerate(periodo.cuentas):
        barra.progress(idx / n_cuentas, text=f"Conciliando cuenta {cta}…")

        # Leer archivos de esta cuenta
        def _read(key):
            f = archivos.get(key)
            if f is None:
                return pd.DataFrame()
            try:
                raw = read_any_file(f)
                return raw
            except Exception as exc:
                alerts_globales.append(f"Error leyendo {key}: {exc}")
                return pd.DataFrame()

        raw_ini = _read(f"port_ini_{cta}")
        raw_fin = _read(f"port_fin_{cta}")
        raw_act = _read(f"activity_{cta}")
        raw_pos = _read(f"pos_{cta}")

        # Preparar
        df_ini, a1 = preparar_portafolio(raw_ini)
        df_fin, a2 = preparar_portafolio(raw_fin)
        df_act, a3 = preparar_activity(raw_act)
        df_pos, a4 = preparar_posiciones(raw_pos)
        alerts_globales += [x for x in a1 + a2 + a3 + a4 if "vacío" not in x]

        # Filtrar por cuenta
        def _filtrar(df):
            if df.empty or "cuenta" not in df.columns:
                return df
            sub = df[df["cuenta"] == cta]
            return sub if not sub.empty else df  # si no hay match devuelve todo

        df_ini = _filtrar(df_ini)
        df_fin = _filtrar(df_fin)
        df_act = _filtrar(df_act)
        df_pos = _filtrar(df_pos)

        # Conciliar
        df_cta, resumen, df_aud, df_match = conciliador.conciliar_cuenta(
            cuenta=cta,
            df_posiciones=df_pos,
            df_activity=df_act,
            df_portafolio_ini=df_ini,
            df_portafolio_fin=df_fin,
        )

        todas_las_filas.append(df_cta)
        resumenes.append(resumen)
        auditorias.append(df_aud)
        matches.append(df_match)

    barra.progress(1.0, text="✅ Listo")

    from tools.comerciales.cartera_propia_core import _safe_concat, auditoria_intercuenta

    df_all        = conciliador.generar_reporte(todas_las_filas)
    df_aud_all    = _safe_concat(auditorias)
    df_match_all  = _safe_concat(matches)
    df_intercuenta = auditoria_intercuenta(df_all, periodo.pares_comp)

    s["df_all"]        = df_all
    s["resumenes"]     = resumenes
    s["df_aud"]        = df_aud_all
    s["df_match"]      = df_match_all
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
