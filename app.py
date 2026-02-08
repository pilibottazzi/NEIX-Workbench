import streamlit as st

# =========================
# IMPORTS NUEVA ESTRUCTURA
# =========================
from tools.mesa import cartera, ons, vencimientos, bonos
from tools.comerciales import cheques, cauciones_mae, cauciones_byma, alquileres
from tools.backoffice import cauciones, control_sliq, moc_tarde, ppt_manana, acreditacion_mav


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NEIX Workbench", page_icon="🧰", layout="wide")


# =========================================================
# ESTÉTICA (minimal + sidebar tabs)
# =========================================================
st.markdown(
    """
    <style>
      /* ===== Layout general ===== */
      .block-container{
        padding-top: 1.2rem;
        max-width: 1220px;
      }

      /* Quitar “aire raro” arriba */
      header[data-testid="stHeader"]{ height: 0rem; }

      /* ===== Header ===== */
      .neix-title{
        text-align:left;
        font-weight:900;
        letter-spacing: .10em;
        margin: 0;
        padding: 0;
        font-size: 1.55rem;
      }
      .neix-caption{
        text-align:left;
        color:#6b7280;
        margin-top: 6px;
        margin-bottom: 14px;
        font-size: .95rem;
      }

      /* ===== Sección ===== */
      .section-title{
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0 0 6px 0;
      }
      .section-sub{
        color:#6b7280;
        margin: 0 0 14px 0;
        font-size: .92rem;
      }

      /* ===== Grid botones (cards) ===== */
      .tool-grid{
        display:flex;
        gap:14px;
        flex-wrap:wrap;
        margin-top:10px;
      }

      .tool-btn{
        display:flex;
        align-items:center;
        justify-content:center;

        padding:14px 18px;
        border-radius:14px;

        border:1px solid rgba(17,24,39,0.08);
        background: rgba(255,255,255,0.92);

        text-decoration:none !important;
        font-weight:700;
        color:#0f172a;

        min-width:240px;
        box-shadow:0 2px 10px rgba(0,0,0,0.04);

        transition: transform .08s ease, box-shadow .08s ease, border-color .08s ease;
      }

      .tool-btn:hover{
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.08);
        border-color: rgba(255, 59, 59, 0.35);
      }

      /* ===== Sidebar estilo “tabs” ===== */
      section[data-testid="stSidebar"]{
        background: #ffffff;
        border-right: 1px solid rgba(0,0,0,0.06);
      }
      section[data-testid="stSidebar"] .block-container{
        padding-top: 1.0rem;
      }

      .sb-title{
        font-weight: 900;
        letter-spacing: .10em;
        font-size: .95rem;
        margin-bottom: .4rem;
      }
      .sb-sub{
        color:#6b7280;
        font-size: .86rem;
        margin-bottom: 1rem;
      }

      /* Radio “pills” */
      div[role="radiogroup"] label{
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 14px;
        padding: 10px 12px;
        margin: 6px 0;
        background: rgba(255,255,255,0.92);
      }
      div[role="radiogroup"] label:hover{
        border-color: rgba(255, 59, 59, 0.35);
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }

      /* Botón Volver */
      .back-wrap{
        margin: 10px 0 14px 0;
      }
      .back-note{
        color:#6b7280;
        font-size:.88rem;
        margin-top: .15rem;
      }

      /* Divider más suave */
      hr{
        border: none;
        border-top: 1px solid rgba(0,0,0,0.08);
        margin: 14px 0 18px 0;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPERS NAV
# =========================================================
def go_home():
    st.query_params.clear()
    st.rerun()

def back_to_home_factory(tool_key: str):
    def _back():
        with st.container():
            st.markdown('<div class="back-wrap"></div>', unsafe_allow_html=True)
            if st.button("← Volver al Home", key=f"btn_back_{tool_key}"):
                go_home()
            st.markdown('<div class="back-note">Volvés a la navegación por áreas</div>', unsafe_allow_html=True)
    return _back


# =========================================================
# ROUTER
# =========================================================
q = st.query_params
tool = (q.get("tool") or "").strip().lower()

if tool:
    st.markdown("<div class='neix-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
    st.markdown("<div class='neix-caption'>Navegación por áreas y proyectos</div>", unsafe_allow_html=True)
    st.divider()

    back_to_home = back_to_home_factory(tool_key=tool)
    back_to_home()

    st.divider()

    try:
        # Mesa
        if tool == "cartera":
            cartera.render(back_to_home)

        elif tool == "ons":
            ons.render(back_to_home)

        elif tool == "bonos":
            bonos.render(back_to_home)

        elif tool in ("tenencia", "tenencias", "vencimientos"):
            vencimientos.render(back_to_home)

        # Comercial
        elif tool == "cheques":
            cheques.render(back_to_home)

        elif tool in ("cauciones_mae", "cauciones-mae"):
            cauciones_mae.render(back_to_home)

        elif tool in ("cauciones_byma", "cauciones-byma"):
            cauciones_byma.render(back_to_home)

        elif tool == "alquileres":
            alquileres.render(back_to_home)

        # Operaciones (antes Backoffice)
        elif tool in ("op_cauciones", "bo_cauciones", "cauciones"):
            cauciones.render(back_to_home)

        elif tool in ("op_control_sliq", "bo_control_sliq", "control_sliq"):
            control_sliq.render(back_to_home)

        elif tool in ("op_moc_tarde", "bo_moc_tarde", "moc_tarde"):
            moc_tarde.render(back_to_home)

        elif tool in ("op_ppt_manana", "bo_ppt_manana", "ppt_manana"):
            ppt_manana.render(back_to_home)

        elif tool in ("op_acreditacion_mav", "bo_acreditacion_mav", "acreditacion_mav"):
            acreditacion_mav.render(back_to_home)

        # Placeholder del tool que falta armar
        elif tool in ("op_pendiente", "pendiente"):
            st.info("🧩 Esta herramienta todavía no está armada. Cuando la tengas, la conectamos acá.")
            st.caption("Tip: creá tools/backoffice/pendiente.py con un render(back_to_home).")

        else:
            st.error("Herramienta no encontrada.")
            st.caption("Volvé a Home y verificá el link.")

    except Exception as e:
        st.error("Error cargando la herramienta.")
        st.exception(e)

    st.stop()


# =========================================================
# HOME (Sidebar izquierda + cards)
# =========================================================
st.sidebar.markdown("<div class='sb-title'>N E I X &nbsp;Workbench</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sb-sub'>Áreas y herramientas</div>", unsafe_allow_html=True)

area = st.sidebar.radio(
    "Área",
    ["Mesa", "Comercial", "Operaciones"],
    label_visibility="collapsed"
)

st.markdown("<div class='neix-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
st.markdown("<div class='neix-caption'>Navegación por áreas y proyectos</div>", unsafe_allow_html=True)
st.divider()

if area == "Mesa":
    st.markdown("<div class='section-title'>Mesa</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Bonos, ONs y carteras</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=bonos" target="_self">Bonos</a>
          <a class="tool-btn" href="?tool=ons" target="_self">Obligaciones Negociables</a>
          <a class="tool-btn" href="?tool=cartera" target="_self">Carteras</a>
        </div>
        """,
        unsafe_allow_html=True
    )

elif area == "Comercial":
    st.markdown("<div class='section-title'>Comercial</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Seguimiento y herramientas comerciales</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=cartera" target="_self">Carteras</a>
          <a class="tool-btn" href="?tool=cauciones_mae" target="_self">Cauciones MAE</a>
          <a class="tool-btn" href="?tool=cauciones_byma" target="_self">Cauciones BYMA</a>
          <a class="tool-btn" href="?tool=cheques" target="_self">Cheques</a>
          <a class="tool-btn" href="?tool=alquileres" target="_self">Alquiler</a>
          <a class="tool-btn" href="?tool=tenencia" target="_self">Tenencia</a>
        </div>
        """,
        unsafe_allow_html=True
    )

else:  # Operaciones
    st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Procesos operativos y control</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=op_ppt_manana" target="_self">PPT Mañana</a>
          <a class="tool-btn" href="?tool=op_moc_tarde" target="_self">MOC Tarde</a>
          <a class="tool-btn" href="?tool=op_control_sliq" target="_self">Control SLIQ</a>
          <a class="tool-btn" href="?tool=op_acreditacion_mav" target="_self">Acreditación MAV</a>
          <a class="tool-btn" href="?tool=op_cauciones" target="_self">Cauciones</a>
          <a class="tool-btn" href="?tool=op_pendiente" target="_self">+ Pendiente (nuevo)</a>
        </div>
        """,
        unsafe_allow_html=True
    )

