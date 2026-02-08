# =========================================================
# UI (NEIX — limpio y "normal", sin tocar la lógica)
# =========================================================
NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "#ffffff"


def _inject_ui_css() -> None:
    st.markdown(
        f"""
        <style>
          /* ancho y aire */
          .block-container {{
            max-width: 1120px;
            padding-top: 1.25rem;
            padding-bottom: 2.25rem;
          }}

          /* suaviza tipografía general */
          html, body, [class*="css"] {{
            color: {TEXT};
          }}

          /* Header simple tipo Backoffice */
          .sliq-head {{
            margin: 0 0 10px 0;
          }}
          .sliq-kicker {{
            display:flex;
            align-items:center;
            gap:10px;
            margin-bottom: 6px;
          }}
          .sliq-badge {{
            width:38px;
            height:38px;
            border-radius: 12px;
            border: 1px solid {BORDER};
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight: 800;
            letter-spacing: .05em;
            background:#fff;
          }}
          .sliq-title {{
            margin:0;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.15;
          }}
          .sliq-sub {{
            margin: 0 0 0 48px;
            color:{MUTED};
            font-size: .95rem;
          }}

          /* Cards simples */
          .sliq-card {{
            border:1px solid {BORDER};
            border-radius:16px;
            padding:14px 14px 12px 14px;
            background:{CARD_BG};
            box-shadow: 0 2px 12px rgba(0,0,0,0.03);
          }}
          .sliq-card-title {{
            margin:0 0 2px 0;
            font-weight: 700;
            font-size: 1.02rem;
          }}
          .sliq-card-hint {{
            margin:0 0 10px 0;
            color:{MUTED};
            font-size: .88rem;
          }}

          /* uploader */
          [data-testid="stFileUploaderDropzone"] {{
            border-radius: 14px !important;
            border: 1px dashed rgba(17,24,39,0.22) !important;
            background: rgba(249,250,251,0.75) !important;
          }}

          /* botón primary NEIX */
          div.stButton > button[kind="primary"] {{
            width: 100%;
            background: {NEIX_RED};
            border: 1px solid rgba(0,0,0,0.06);
            color: #fff;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 800;
            box-shadow: 0 10px 20px rgba(255,59,48,0.16);
            transition: transform .06s ease, box-shadow .12s ease, filter .12s ease;
          }}
          div.stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            filter: brightness(0.98);
            box-shadow: 0 14px 26px rgba(255,59,48,0.20);
          }}

          /* separadores */
          hr {{
            margin: 1.1rem 0;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_open() -> None:
    st.markdown('<div class="sliq-card">', unsafe_allow_html=True)


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Render (Streamlit tool) — misma lógica, mejor UI
# =========================================================
def render(back_to_home=None):
    _inject_ui_css()

    # Header "normal" (como MOC TARDE)
    st.markdown(
        f"""
        <div class="sliq-head">
          <div class="sliq-kicker">
            <div class="sliq-badge">N</div>
            <div class="sliq-title">Control SLIQ</div>
          </div>
          <div class="sliq-sub">Cargá NASDAQ (,) y SLIQ (;). Genera <b>Control SLIQ tarde.xlsx</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        _card_open()
        st.markdown('<div class="sliq-card-title">Instr. de Liquidación NASDAQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="sliq-card-hint">CSV separado por coma (<b>,</b>)</div>', unsafe_allow_html=True)
        f_nasdaq = st.file_uploader("", type=["csv"], key="sliq_nasdaq", label_visibility="collapsed")
        _card_close()

    with c2:
        _card_open()
        st.markdown('<div class="sliq-card-title">Especies para un Participante</div>', unsafe_allow_html=True)
        st.markdown('<div class="sliq-card-hint">CSV separado por punto y coma (<b>;</b>)</div>', unsafe_allow_html=True)
        f_sliq = st.file_uploader("", type=["csv"], key="sliq_sliq", label_visibility="collapsed")
        _card_close()

    st.write("")
    run = st.button('Generar "Control SLIQ"', type="primary", key="sliq_run")

    if not run:
        return

    if not f_nasdaq or not f_sliq:
        st.error("Faltan archivos: cargá NASDAQ y SLIQ (CSV).")
        return

    logs: List[str] = []

    def log(m: str):
        logs.append(m)

    try:
        log("Leyendo NASDAQ (,) y SLIQ (;)…")

        # ---- NASDAQ
        nas_txt = _read_text_with_fallback(f_nasdaq)
        df_n0 = _read_csv_as_table(nas_txt, sep=",")
        if df_n0.empty:
            st.error("NASDAQ: archivo vacío.")
            return

        log("Procesando NASDAQ…")
        df_nas_out, sum_by_inst = _build_nasdaq_detalle_from_table(df_n0)

        # ---- SLIQ
        sliq_txt = _read_text_with_fallback(f_sliq)
        bad_quotes = sum(1 for ln in sliq_txt.splitlines() if ln.count('"') % 2 == 1)
        if bad_quotes:
            st.warning(f"SLIQ: detecté {bad_quotes} línea(s) con comillas desbalanceadas. Se corrigieron automáticamente.")

        df_s0 = _read_csv_as_table(sliq_txt, sep=";")
        if df_s0.empty:
            st.error("SLIQ: archivo vacío.")
            return

        log("Procesando SLIQ…")
        df_sliq_out, sliq_by_code = _build_sliq_from_table(df_s0)

        # ---- CONTROL
        log("Armando Control SLIQ tarde…")
        df_control = _build_control(sum_by_inst, sliq_by_code)

        # ---- EXCEL
        log("Generando Excel…")
        xlsx_bytes = _export_excel(df_nas_out, df_control, df_sliq_out)

        st.success("Listo ✅ Se generó el archivo.")
        st.download_button(
            "Descargar Control SLIQ tarde.xlsx",
            data=xlsx_bytes,
            file_name="Control SLIQ tarde.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sliq_download",
        )

        st.divider()
        st.subheader("Preview — Control SLIQ tarde")
        st.dataframe(df_control.head(80), use_container_width=True, hide_index=True)

        with st.expander("Ver logs", expanded=False):
            st.write("\n".join(f"• {m}" for m in logs))

    except Exception as e:
        st.error("Error generando el Control SLIQ.")
        st.exception(e)
