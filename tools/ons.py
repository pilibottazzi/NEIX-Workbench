import io
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------
# Helpers UI / formato
# -----------------------
def _fmt_pct(x):
    if pd.isna(x): return ""
    return f"{x:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_num(x, dec=2):
    if pd.isna(x): return ""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_int(x):
    if pd.isna(x): return ""
    try:
        return f"{int(x):,}".replace(",", ".")
    except:
        return str(x)

def _fmt_date(x):
    if pd.isna(x): return ""
    try:
        return pd.to_datetime(x).strftime("%d/%m/%Y")
    except:
        return str(x)

def _soft_diverging(val, vmin, vmax):
    """
    Color suave tipo NEIX: rojo pálido -> blanco -> verde pálido
    """
    if pd.isna(val):
        return ""
    # Normalizar 0..1
    t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    t = max(0.0, min(1.0, t))

    # Interpolar: rojo suave -> blanco -> verde suave
    # rojo: #FDE2E2, blanco: #FFFFFF, verde: #DCFCE7
    def lerp(a, b, t_):
        return int(a + (b - a) * t_)

    if t < 0.5:
        tt = t / 0.5
        r = lerp(0xFD, 0xFF, tt)
        g = lerp(0xE2, 0xFF, tt)
        b = lerp(0xE2, 0xFF, tt)
    else:
        tt = (t - 0.5) / 0.5
        r = lerp(0xFF, 0xDC, tt)
        g = lerp(0xFF, 0xFC, tt)
        b = lerp(0xFF, 0xE7, tt)

    return f"background-color: rgb({r},{g},{b});"

def _style_comercial(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # Rangos para color (robusto)
    tir = df["TIR"].dropna()
    md  = df["MD"].dropna()

    tir_min, tir_max = (tir.min(), tir.max()) if len(tir) else (0, 1)
    md_min, md_max   = (md.min(), md.max()) if len(md) else (0, 1)

    def color_tir(s):
        return [ _soft_diverging(v, tir_min, tir_max) for v in s ]

    def color_md(s):
        return [ _soft_diverging(v, md_min, md_max) for v in s ]

    sty = (
        df.style
        .apply(color_tir, subset=["TIR"])
        .apply(color_md, subset=["MD"])
        .set_properties(**{
            "border": "1px solid rgba(17,24,39,0.08)",
            "padding": "10px 12px",
            "font-size": "13px",
        })
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "rgba(17,24,39,0.04)"),
                ("color", "#111827"),
                ("font-weight", "700"),
                ("border", "1px solid rgba(17,24,39,0.08)"),
                ("padding", "10px 12px"),
                ("font-size", "12px"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "0.04em"),
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
                ("border-radius", "14px"),
                ("overflow", "hidden"),
                ("width", "100%"),
            ]},
        ])
    )

    # Formatos numéricos visibles
    if "TIR" in df.columns:
        sty = sty.format({"TIR": lambda x: _fmt_pct(x)})
    if "MD" in df.columns:
        sty = sty.format({"MD": lambda x: _fmt_num(x, 2)})
    if "Precio" in df.columns:
        sty = sty.format({"Precio": lambda x: _fmt_num(x, 4)})
    if "Volumen" in df.columns:
        sty = sty.format({"Volumen": lambda x: _fmt_num(x, 0)})

    return sty

def _to_excel_bytes(df: pd.DataFrame, sheet_name="ONs") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        wb = writer.book
        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)
        ws.set_default_row(18)
        # ancho col auto básico
        for i, col in enumerate(df.columns):
            width = max(12, min(36, int(df[col].astype(str).map(len).max() if len(df) else 12) + 2))
            ws.set_column(i, i, width)
    bio.seek(0)
    return bio.read()


# -----------------------
# RENDER (Front)
# -----------------------
def render(back_to_home=None):
    # CSS NEIX minimal (si ya usás general.css, esto suma sin romper)
    st.markdown("""
    <style>
      .neix-title { font-size: 28px; font-weight: 800; letter-spacing: 0.06em; margin-bottom: 2px; }
      .neix-sub { color: rgba(17,24,39,.65); margin-top: 0; }
      .cardline {
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 16px;
        padding: 14px 16px;
        background: white;
        box-shadow: 0 6px 20px rgba(17,24,39,0.06);
      }
      .kpi-label { color: rgba(17,24,39,.60); font-size: 12px; margin-bottom: 2px; text-transform: uppercase; letter-spacing: .06em; }
      .kpi-value { font-size: 22px; font-weight: 800; color: #111827; }
      .pill {
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        border:1px solid rgba(17,24,39,0.10);
        background: rgba(17,24,39,0.03);
        font-size:12px;
        color:#111827;
      }
      .divider { height: 1px; background: rgba(17,24,39,0.08); margin: 14px 0; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([0.82, 0.18])
    with c1:
        st.markdown('<div class="neix-title">NEIX · Obligaciones Negociables</div>', unsafe_allow_html=True)
        st.markdown('<div class="neix-sub">Rendimientos y métricas en USD (TIR, Duration, Modified Duration). Listo para compartir con el equipo comercial.</div>', unsafe_allow_html=True)
    with c2:
        if back_to_home is not None:
            st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Inputs (arriba, simple y comercial)
    topA, topB, topC, topD = st.columns([0.34, 0.22, 0.22, 0.22])

    with topA:
        cf_file = st.file_uploader("📎 Cashflows (xlsx)", type=["xlsx"], key="ons_cf")
    with topB:
        key_col = st.selectbox("Agrupar por", ["ticker_original", "root_key"], index=0)
    with topC:
        plazo = st.selectbox("Plazo (días)", [0,1,2], index=2)
    with topD:
        fuente_precios = st.selectbox("Precios", ["IOL (web)", "Excel"], index=0)

    if cf_file is None:
        st.info("Subí el Excel de cashflows para ver el panel.")
        return

    # ---- Leer cashflows (tu función existente) ----
    # Asumo que ya las tenés definidas en el archivo:
    # load_cashflows_long_from_file, build_cashflow_dict, _future_cashflows, _settlement, tir, duration, modified_duration, fetch_iol_on_prices, load_prices_from_excel
    try:
        df_cf = load_cashflows_long_from_file(cf_file)
        cashflows = build_cashflow_dict(df_cf, key_col=key_col)
    except Exception as e:
        st.error(f"No pude leer cashflows: {e}")
        return

    tickers_all = sorted(cashflows.keys())
    tickers_sel = st.multiselect("Tickers", options=tickers_all, default=tickers_all)

    st.markdown("")

    # Precios
    precios_df = None
    iol_dividir_100 = False
    precio_pct_excel = True

    if fuente_precios == "IOL (web)":
        cc1, cc2 = st.columns([0.25, 0.75])
        with cc1:
            if st.button("🔄 Traer precios", key="ons_btn_iol"):
                with st.spinner("Leyendo IOL..."):
                    try:
                        precios_df = fetch_iol_on_prices()
                        st.session_state["ons_precios_iol"] = precios_df
                        st.success(f"Precios cargados: {len(precios_df)}")
                    except Exception as e:
                        st.error(f"No pude leer IOL: {e}")
                        precios_df = None
        with cc2:
            iol_dividir_100 = st.checkbox(
                "Precio IOL en % → dividir por 100",
                value=False,
                help="Si tus cupones están por VN100, normalmente NO se divide (precio ~80-100). Si te queda 0.8, está mal la escala."
            )

        if precios_df is None:
            precios_df = st.session_state.get("ons_precios_iol")

        if precios_df is None:
            st.warning("Traé precios con el botón o cambiá a Excel.")
    else:
        px_file = st.file_uploader("📎 Excel de precios (Ticker, Precio, Volumen opc.)", type=["xlsx"], key="ons_px")
        precio_pct_excel = st.checkbox("Precio en % (par) → dividir por 100", value=True)
        if px_file is not None:
            try:
                precios_df = load_prices_from_excel(px_file)
                st.success(f"Precios Excel cargados: {len(precios_df)}")
            except Exception as e:
                st.error(str(e))
                precios_df = None
        else:
            st.warning("Subí un Excel de precios o elegí IOL.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Calcular
    if st.button("▶️ Calcular y armar tabla comercial", type="primary"):
        if not tickers_sel:
            st.warning("Elegí al menos 1 ticker.")
            return

        settlement = _settlement(plazo)
        rows = []
        diag = []

        for t in tickers_sel:
            cf = cashflows[t].copy()
            df_fut = _future_cashflows(cf, settlement)

            diag.append({
                "Ticker": t,
                "Flujos futuros": len(df_fut),
                "Primer flujo": df_fut["Fecha"].min() if not df_fut.empty else pd.NaT,
                "Último flujo": df_fut["Fecha"].max() if not df_fut.empty else pd.NaT,
            })

            # Precio
            precio = np.nan
            vol = np.nan

            if precios_df is not None and t in precios_df.index:
                if fuente_precios == "IOL (web)":
                    precio = float(precios_df.loc[t, "UltimoOperado"])
                    if iol_dividir_100:
                        precio = precio / 100.0
                    vol = float(precios_df.loc[t, "MontoOperado"])
                else:
                    precio = float(precios_df.loc[t, "Precio"])
                    if precio_pct_excel:
                        precio = precio / 100.0
                    vol = float(precios_df.loc[t, "Volumen"]) if "Volumen" in precios_df.columns else np.nan

            obs = ""
            if df_fut.empty:
                obs = "Sin flujos futuros (vencido o cashflow incompleto)."
            elif np.isfinite(precio) and precio > 0:
                cupon_med = float(df_fut["Cupon"].median())
                if precio < 5 and cupon_med > 0.05:
                    obs = "⚠️ Escala: el precio parece estar /100 vs cupón (revisar toggle)."

            if (not np.isfinite(precio)) or precio <= 0 or df_fut.empty:
                rows.append({
                    "Vencimiento": df_fut["Fecha"].max() if not df_fut.empty else pd.NaT,
                    "TIR": np.nan,
                    "MD": np.nan,
                    "Ticker": t,
                    "Instrumento": "",
                    "Pago": "",
                    "Mínimo": np.nan,
                    "Calificación": "",
                    "Precio": precio,
                    "Volumen": vol,
                    "Obs": obs,
                })
                continue

            tir_v = tir(cf, precio, plazo_dias=plazo)
            md_v  = modified_duration(cf, precio, plazo_dias=plazo)

            rows.append({
                "Vencimiento": df_fut["Fecha"].max(),
                "TIR": tir_v,
                "MD": md_v,
                "Ticker": t,
                "Instrumento": "",   # si lo tenés en otra tabla, después lo cruzamos
                "Pago": "",          # idem
                "Mínimo": np.nan,    # idem
                "Calificación": "",  # idem
                "Precio": precio,
                "Volumen": vol,
                "Obs": obs,
            })

        df = pd.DataFrame(rows)

        # Orden
        ord1, ord2 = st.columns([0.25, 0.75])
        with ord1:
            order_by = st.selectbox("Ordenar por", ["Vencimiento", "TIR", "MD", "Ticker"], index=0)
        ascending = (order_by in ["Vencimiento", "Ticker"])
        df = df.sort_values(order_by, ascending=ascending, na_position="last").reset_index(drop=True)

        # KPIs
        ok = df["TIR"].dropna()
        ok_md = df["MD"].dropna()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown('<div class="cardline"><div class="kpi-label">ONs</div><div class="kpi-value">%s</div></div>' % _fmt_int(len(df)), unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="cardline"><div class="kpi-label">TIR promedio</div><div class="kpi-value">%s</div></div>' % (_fmt_pct(ok.mean()) if len(ok) else "—"), unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="cardline"><div class="kpi-label">TIR mediana</div><div class="kpi-value">%s</div></div>' % (_fmt_pct(ok.median()) if len(ok) else "—"), unsafe_allow_html=True)
        with k4:
            st.markdown('<div class="cardline"><div class="kpi-label">MD promedio</div><div class="kpi-value">%s</div></div>' % (_fmt_num(ok_md.mean(),2) if len(ok_md) else "—"), unsafe_allow_html=True)

        st.markdown("")

        # Tabla comercial (vista)
        view = df[["Vencimiento","TIR","MD","Ticker","Precio","Volumen","Obs"]].copy()
        view["Vencimiento"] = view["Vencimiento"].apply(_fmt_date)

        # Para el styler, necesitamos TIR/MD numéricos (df original)
        sty_df = df[["Vencimiento","TIR","MD","Ticker","Precio","Volumen","Obs"]].copy()
        sty_df["Vencimiento"] = sty_df["Vencimiento"].apply(_fmt_date)
        sty_df["Ticker"] = sty_df["Ticker"].astype(str).str.strip()

        st.markdown("### Tabla comercial")
        st.caption("Colores suaves en TIR y MD para lectura rápida. 'Obs' marca problemas de escala o cashflow incompleto.")

        st.dataframe(sty_df, use_container_width=True)  # fallback si no querés styler

        # Si querés el look “pro” tipo grilla:
        st.markdown(_style_comercial(sty_df).to_html(), unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Export
        st.markdown("### Exportar")
        exp1, exp2 = st.columns([0.5, 0.5])

        export_df = df.copy()
        export_df["Vencimiento"] = export_df["Vencimiento"].apply(_fmt_date)
        export_df["TIR"] = export_df["TIR"].apply(_fmt_pct)
        export_df["MD"] = export_df["MD"].apply(lambda x: _fmt_num(x,2))
        export_df["Precio"] = export_df["Precio"].apply(lambda x: _fmt_num(x,4))
        export_df["Volumen"] = export_df["Volumen"].apply(lambda x: _fmt_num(x,0))

        with exp1:
            st.download_button(
                "⬇️ Descargar Excel (Comercial)",
                data=_to_excel_bytes(export_df, sheet_name="ONs"),
                file_name=f"ONs_comercial_{dt.datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with exp2:
            csv = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv,
                file_name=f"ONs_comercial_{dt.datetime.now():%Y%m%d}.csv",
                mime="text/csv",
            )

        st.markdown("### Texto listo para enviar")
        top_line = f"📌 ONs Ley NY – Rendimientos USD ({dt.datetime.now():%d/%m/%Y})"
        bullets = []
        for _, r in df.head(10).iterrows():
            if pd.isna(r["TIR"]) or pd.isna(r["MD"]):
                continue
            bullets.append(f"• {r['Ticker']}: TIR {_fmt_pct(r['TIR'])} | MD {_fmt_num(r['MD'],2)} | Vto {_fmt_date(r['Vencimiento'])}")
        msg = top_line + "\n" + "\n".join(bullets) if bullets else top_line + "\n(Sin datos suficientes para armar ranking)"
        st.code(msg, language="text")

        # Diagnóstico (oculto / para vos)
        with st.expander("🔎 Diagnóstico de cashflows (para control interno)"):
            diag_df = pd.DataFrame(diag)
            diag_df["Primer flujo"] = diag_df["Primer flujo"].apply(_fmt_date)
            diag_df["Último flujo"] = diag_df["Último flujo"].apply(_fmt_date)
            st.dataframe(diag_df, use_container_width=True)
