from __future__ import annotations

import pandas as pd
import streamlit as st

from tools.comerciales.cartera_propia_core import (
    CUENTAS_DEFAULT,
    PeriodoConciliacion,
    auditoria_intercuenta,
    build_excel_bytes,
    build_reglas_aplicadas,
    build_resumen_cuentas,
    build_top_pendientes,
    format_num,
    guess_file_role,
    preparar_activity,
    preparar_portafolio,
    preparar_posiciones,
    read_any_file,
    ConciliadorMensual,
)


# =============================================================================
# ESTILO
# =============================================================================

NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.08)"
CARD_BG = "rgba(255,255,255,0.98)"
BG = "#f5f7fa"
OK = "#15803d"
WARN = "#b45309"
BAD = "#b91c1c"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1520px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
          }}

          .main {{
            background: {BG};
          }}

          [data-testid="stSidebar"] {{
            display: none;
          }}

          .hero {{
            background: linear-gradient(135deg, #ffffff 0%, #fbfbfc 100%);
            border: 1px solid {BORDER};
            border-radius: 26px;
            padding: 1.35rem 1.35rem 1.1rem 1.35rem;
            box-shadow: 0 16px 36px rgba(17,24,39,0.05);
            margin-bottom: 1rem;
          }}

          .hero-title {{
            font-size: 1.85rem;
            font-weight: 760;
            letter-spacing: -0.03em;
            color: {TEXT};
            margin-bottom: 0.15rem;
          }}

          .hero-sub {{
            font-size: 0.97rem;
            color: {MUTED};
            margin: 0;
          }}

          .toolbar {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 10px 28px rgba(17,24,39,0.04);
            margin-bottom: 1rem;
          }}

          .section-title {{
            font-size: 1.02rem;
            font-weight: 730;
            color: {TEXT};
            margin: 0.1rem 0 0.8rem 0;
          }}

          .card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 8px 24px rgba(17,24,39,0.04);
            height: 100%;
          }}

          .label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {MUTED};
            margin-bottom: 0.35rem;
          }}

          .value {{
            font-size: 1.65rem;
            font-weight: 760;
            color: {TEXT};
            line-height: 1.1;
          }}

          .sub {{
            font-size: 0.82rem;
            color: {MUTED};
            margin-top: 0.35rem;
          }}

          .mini {{
            background: #fff;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 6px 18px rgba(17,24,39,0.03);
            margin-bottom: 0.7rem;
          }}

          .pill {{
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
          }}

          .ok {{
            background: rgba(34,197,94,0.10);
            color: {OK};
          }}

          .warn {{
            background: rgba(245,158,11,0.12);
            color: {WARN};
          }}

          .bad {{
            background: rgba(255,59,48,0.10);
            color: {BAD};
          }}

          .box {{
            background: #fff;
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(17,24,39,0.03);
          }}

          .stTabs [data-baseweb="tab-list"] {{
            gap: 0.4rem;
          }}

          .stTabs [data-baseweb="tab"] {{
            height: 42px;
            border-radius: 12px;
            padding-left: 1rem;
            padding-right: 1rem;
          }}

          div[data-testid="stDownloadButton"] > button,
          div[data-testid="stButton"] > button {{
            width: 100%;
            border-radius: 12px;
            min-height: 44px;
          }}

          div[data-testid="stDownloadButton"] > button {{
            background: {NEIX_RED} !important;
            color: #fff !important;
            border: 1px solid {NEIX_RED} !important;
          }}

          div[data-testid="stButton"] > button {{
            background: white !important;
            color: {TEXT} !important;
            border: 1px solid {BORDER} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="sub">{sub}</div>
    </div>
    """


def pill_html(text: str, level: str) -> str:
    cls = {"OK": "ok", "WARN": "warn", "BAD": "bad"}.get(level, "warn")
    return f"<span class='pill {cls}'>{text}</span>"


# =============================================================================
# RENDER HELPERS
# =============================================================================

def render_toolbar():
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Conciliación mensual de cartera propia</div>', unsafe_allow_html=True)    
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Parámetros de corrida</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        fecha_ini = st.date_input("Fecha inicio")
    with c2:
        fecha_fin = st.date_input("Fecha fin")
    with c3:
        cuentas_sel = st.multiselect("Cuentas", options=CUENTAS_DEFAULT, default=CUENTAS_DEFAULT)

    c4, c5 = st.columns([3, 1])
    with c4:
        bulk_files = st.file_uploader(
            "Carga masiva de archivos",
            type=["xlsx", "xls", "xlsm", "csv"],
            accept_multiple_files=True,
        )
    with c5:
        auto_naming = st.toggle("Auto naming", value=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return fecha_ini, fecha_fin, cuentas_sel, bulk_files, auto_naming


def render_kpis(df_all: pd.DataFrame) -> None:
    total = len(df_all)
    cerradas = int(df_all["status"].astype(str).str.contains("CERRADA").sum()) if total else 0
    pendientes = int((df_all["status"] == "PENDIENTE").sum()) if total else 0
    pct = (cerradas / total * 100) if total else 0
    dif_abs = float(df_all["dif_final"].abs().sum()) if total else 0
    reglas = int(df_all["reglas_aplicadas"].sum()) if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card("Especies", f"{total:,.0f}".replace(",", ".")), unsafe_allow_html=True)
    with c2:
        st.markdown(card("Cerradas", f"{cerradas:,.0f}".replace(",", "."), f"{pct:.1f}%"), unsafe_allow_html=True)
    with c3:
        st.markdown(card("Pendientes", f"{pendientes:,.0f}".replace(",", ".")), unsafe_allow_html=True)
    with c4:
        st.markdown(card("Dif. abs acumulada", format_num(dif_abs, 0)), unsafe_allow_html=True)
    with c5:
        st.markdown(card("Reglas aplicadas", f"{reglas:,.0f}".replace(",", ".")), unsafe_allow_html=True)


def render_semaforo(df_resumen: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Semáforo por cuenta</div>', unsafe_allow_html=True)
    cols = st.columns(min(4, max(len(df_resumen), 1)))

    for i, (_, row) in enumerate(df_resumen.iterrows()):
        with cols[i % len(cols)]:
            st.markdown("<div class='mini'>", unsafe_allow_html=True)
            st.markdown(f"**Cta {int(row['cuenta'])}**", unsafe_allow_html=True)
            st.markdown(pill_html(row["semaforo"], row["semaforo"]), unsafe_allow_html=True)
            st.caption(
                f"{int(row['cerradas'])}/{int(row['especies'])} cerradas · "
                f"{int(row['pendientes'])} pendientes · "
                f"Dif abs {format_num(row['dif_total_abs'], 0)}"
            )
            st.markdown("</div>", unsafe_allow_html=True)


def render_management_summary(df_resumen: pd.DataFrame, df_top: pd.DataFrame) -> None:
    total = len(df_resumen)
    ok = int((df_resumen["semaforo"] == "OK").sum()) if total else 0
    warn = int((df_resumen["semaforo"] == "WARN").sum()) if total else 0
    bad = int((df_resumen["semaforo"] == "BAD").sum()) if total else 0

    texto = f"Se procesaron {total} cuentas. {ok} quedaron en OK, {warn} en observación y {bad} con desvíos relevantes."
    if not df_top.empty:
        r = df_top.iloc[0]
        texto += f" El principal pendiente corresponde a la cuenta {int(r['cuenta'])}, especie {r['especie_h']}, con diferencia {format_num(r['dif_final'], 0)}."

    st.markdown(
        f"""
        <div class="box">
          <strong>Lectura ejecutiva</strong><br>
          <span style="color:#6b7280;">{texto}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    inject_css()

    fecha_ini, fecha_fin, cuentas_sel, bulk_files, auto_naming = render_toolbar()

    periodo = PeriodoConciliacion(
        fecha_ini=str(fecha_ini),
        fecha_fin=str(fecha_fin),
        cuentas=cuentas_sel,
    )
    conciliador = ConciliadorMensual(periodo)

    cuentas_data = {
        c: {
            "posiciones": None,
            "activity": None,
            "portafolio_ini": None,
            "portafolio_fin": None,
            "errores": [],
            "alertas": [],
            "fuentes": [],
        }
        for c in cuentas_sel
    }

    if bulk_files and auto_naming:
        for f in bulk_files:
            cuenta, role = guess_file_role(f.name)
            if cuenta not in cuentas_data or role is None:
                continue

            try:
                df_raw = read_any_file(f)
                if role == "posiciones":
                    df_ready, alerts = preparar_posiciones(df_raw)
                elif role == "activity":
                    df_ready, alerts = preparar_activity(df_raw)
                elif role == "portafolio_ini":
                    df_ready, alerts = preparar_portafolio(df_raw)
                elif role == "portafolio_fin":
                    df_ready, alerts = preparar_portafolio(df_raw)
                else:
                    continue

                cuentas_data[cuenta][role] = df_ready
                cuentas_data[cuenta]["alertas"].extend(alerts)
                cuentas_data[cuenta]["fuentes"].append(f"{role}: {f.name}")

            except Exception as e:
                cuentas_data[cuenta]["errores"].append(f"{role}: {f.name} → {e}")

    st.markdown('<div class="section-title">Carga y control por cuenta</div>', unsafe_allow_html=True)
    tabs_in = st.tabs([f"Cta {c}" for c in cuentas_sel]) if cuentas_sel else []

    for cuenta, tab in zip(cuentas_sel, tabs_in):
        with tab:
            a, b = st.columns(2)

            with a:
                pos_file = st.file_uploader(
                    f"Posiciones - {cuenta}",
                    type=["xlsx", "xls", "xlsm", "csv"],
                    key=f"pos_{cuenta}",
                )
                act_file = st.file_uploader(
                    f"Activity - {cuenta}",
                    type=["xlsx", "xls", "xlsm", "csv"],
                    key=f"act_{cuenta}",
                )

            with b:
                ini_file = st.file_uploader(
                    f"Portafolio inicial - {cuenta}",
                    type=["xlsx", "xls", "xlsm", "csv"],
                    key=f"ini_{cuenta}",
                )
                fin_file = st.file_uploader(
                    f"Portafolio final - {cuenta}",
                    type=["xlsx", "xls", "xlsm", "csv"],
                    key=f"fin_{cuenta}",
                )

            manual_map = [
                ("posiciones", pos_file, preparar_posiciones),
                ("activity", act_file, preparar_activity),
                ("portafolio_ini", ini_file, preparar_portafolio),
                ("portafolio_fin", fin_file, preparar_portafolio),
            ]

            for role, file_obj, prep_fn in manual_map:
                if file_obj is not None:
                    try:
                        df_ready, alerts = prep_fn(read_any_file(file_obj))
                        cuentas_data[cuenta][role] = df_ready
                        cuentas_data[cuenta]["alertas"].extend(alerts)
                        cuentas_data[cuenta]["fuentes"].append(f"{role}: {file_obj.name}")
                    except Exception as e:
                        cuentas_data[cuenta]["errores"].append(f"{role}: {e}")

            p = cuentas_data[cuenta]
            status_row = []
            status_row.append(pill_html("Posiciones OK" if p["posiciones"] is not None else "Falta posiciones", "OK" if p["posiciones"] is not None else "BAD"))
            status_row.append(pill_html("Activity OK" if p["activity"] is not None else "Falta activity", "OK" if p["activity"] is not None else "BAD"))
            status_row.append(pill_html("Portafolio ini" if p["portafolio_ini"] is not None else "Ini opcional", "OK" if p["portafolio_ini"] is not None else "WARN"))
            status_row.append(pill_html("Portafolio fin" if p["portafolio_fin"] is not None else "Fin opcional", "OK" if p["portafolio_fin"] is not None else "WARN"))
            st.markdown(" ".join(status_row), unsafe_allow_html=True)

            with st.expander("Fuentes / alertas / preview"):
                if p["fuentes"]:
                    st.markdown("**Fuentes**")
                    for x in p["fuentes"]:
                        st.write(f"- {x}")
                if p["alertas"]:
                    st.markdown("**Alertas**")
                    for x in p["alertas"]:
                        st.write(f"- {x}")
                if p["errores"]:
                    st.markdown("**Errores**")
                    for x in p["errores"]:
                        st.write(f"- {x}")

                if p["posiciones"] is not None:
                    st.markdown("**Preview posiciones**")
                    st.dataframe(p["posiciones"].head(8), use_container_width=True, hide_index=True)
                if p["activity"] is not None:
                    st.markdown("**Preview activity**")
                    st.dataframe(p["activity"].head(8), use_container_width=True, hide_index=True)

    st.markdown("---")
    if not st.button("Ejecutar conciliación"):
        return

    resultados = []
    resumenes = []
    auditorias = []
    matches = []
    errores_globales = []

    for cuenta in cuentas_sel:
        payload = cuentas_data[cuenta]
        df_pos = payload["posiciones"]
        df_act = payload["activity"]
        df_ini = payload["portafolio_ini"]
        df_fin = payload["portafolio_fin"]

        if df_pos is None or df_act is None:
            errores_globales.append(f"Cta {cuenta}: faltan posiciones y/o activity.")
            continue

        try:
            df_res, resumen, df_aud, df_match = conciliador.conciliar_cuenta(
                cuenta=cuenta,
                df_posiciones=df_pos,
                df_activity=df_act,
                df_portafolio_ini=df_ini,
                df_portafolio_fin=df_fin,
            )
            resultados.append(df_res)
            resumenes.append(resumen)
            auditorias.append(df_aud)
            matches.append(df_match)
        except Exception as e:
            errores_globales.append(f"Cta {cuenta}: {e}")

    for err in errores_globales:
        st.error(err)

    df_all = conciliador.generar_reporte(resultados)
    if df_all.empty:
        st.warning("No se generaron resultados.")
        return

    df_resumen = build_resumen_cuentas(df_all)
    df_top = build_top_pendientes(df_all)
    df_pend = df_all[df_all["status"] == "PENDIENTE"].copy().reset_index(drop=True)
    df_auditoria = pd.concat(auditorias, ignore_index=True) if auditorias else pd.DataFrame()
    df_match = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    df_reglas = build_reglas_aplicadas(df_all)
    df_intercuenta = auditoria_intercuenta(df_all, periodo.pares_comp)

    render_kpis(df_all)
    render_management_summary(df_resumen, df_top)
    render_semaforo(df_resumen)

    tabs_out = st.tabs([
        "Resumen",
        "Top pendientes",
        "Pendientes",
        "Match / sin match",
        "Intercuenta",
        "Auditoría",
        "Detalle",
        "Reglas",
    ])

    with tabs_out[0]:
        st.dataframe(
            df_resumen.style.format({
                "especies": "{:,.0f}",
                "cerradas": "{:,.0f}",
                "pendientes": "{:,.0f}",
                "reglas_aplicadas": "{:,.0f}",
                "dif_total_abs": "{:,.2f}",
                "dif_total_neto": "{:,.2f}",
                "pct_cierre": "{:,.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tabs_out[1]:
        if df_top.empty:
            st.success("No hay pendientes.")
        else:
            st.dataframe(
                df_top.style.format({
                    "ini": "{:,.2f}",
                    "act": "{:,.2f}",
                    "fin": "{:,.2f}",
                    "comp": "{:,.2f}",
                    "ini_ajust": "{:,.2f}",
                    "act_ajust": "{:,.2f}",
                    "fin_ajust": "{:,.2f}",
                    "comp_ajust": "{:,.2f}",
                    "dif_final": "{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    with tabs_out[2]:
        if df_pend.empty:
            st.success("No hay pendientes.")
        else:
            st.dataframe(
                df_pend.style.format({
                    "ini": "{:,.2f}",
                    "act": "{:,.2f}",
                    "fin": "{:,.2f}",
                    "comp": "{:,.2f}",
                    "ini_ajust": "{:,.2f}",
                    "act_ajust": "{:,.2f}",
                    "fin_ajust": "{:,.2f}",
                    "comp_ajust": "{:,.2f}",
                    "dif_final": "{:,.2f}",
                    "dif_cant": "{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    with tabs_out[3]:
        if df_match.empty:
            st.info("No hay tabla de match.")
        else:
            st.dataframe(df_match, use_container_width=True, hide_index=True)

    with tabs_out[4]:
        if df_intercuenta.empty:
            st.info("No hay auditoría intercuenta.")
        else:
            st.dataframe(
                df_intercuenta.style.format({
                    "dif_origen": "{:,.2f}",
                    "dif_destino": "{:,.2f}",
                    "suma_neta": "{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    with tabs_out[5]:
        if df_auditoria.empty:
            st.info("No hay eventos de auditoría.")
        else:
            st.dataframe(df_auditoria, use_container_width=True, hide_index=True)

    with tabs_out[6]:
        st.dataframe(
            df_all.style.format({
                "ini": "{:,.2f}",
                "act": "{:,.2f}",
                "fin": "{:,.2f}",
                "comp": "{:,.2f}",
                "ini_ajust": "{:,.2f}",
                "act_ajust": "{:,.2f}",
                "fin_ajust": "{:,.2f}",
                "comp_ajust": "{:,.2f}",
                "dif_final_original": "{:,.2f}",
                "dif_final": "{:,.2f}",
                "dif_importe": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tabs_out[7]:
        if df_reglas.empty:
            st.info("No hay reglas registradas.")
        else:
            st.dataframe(df_reglas, use_container_width=True, hide_index=True)

    excel_bytes = build_excel_bytes(
        df_resumen=df_resumen,
        df_top_pend=df_top,
        df_pend=df_pend,
        df_all=df_all,
        df_auditoria=df_auditoria,
        df_reglas=df_reglas,
        df_match=df_match,
        df_intercuenta=df_intercuenta,
    )

    st.markdown("---")
    st.download_button(
        "Descargar Excel premium",
        data=excel_bytes,
        file_name=f"conciliacion_cartera_v4_{str(fecha_fin).replace('-', '')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render():
    main()
