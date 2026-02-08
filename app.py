import streamlit as st

# =========================
# IMPORTS
# =========================
from tools.mesa import cartera, ons, vencimientos, bonos
from tools.comerciales import cheques, cauciones_mae, cauciones_byma, alquileres
from tools.backoffice import cauciones, control_sliq, moc_tarde, ppt_manana, acreditacion_mav


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide"
)

# =========================================================
# ESTÉTICA PREMIUM
# =========================================================
st.markdown(
    """
    <style>
    /* ===== Contenedor general ===== */
    .block-container{
        padding-top: 1.4rem;
        max-width: 1240px;
    }

    /* ===== Header ===== */
    .neix-title{
        font-weight: 900;
        letter-spacing: .12em;
        font-size: 1.55rem;
        margin-bottom: 4px;
    }
    .neix-caption{
        color:#6b7280;
        font-size:.95rem;
        margin-bottom: 18px;
    }

    /* ===== Tabs arriba, izquierda ===== */
    .stTabs [data-baseweb="tab-list"]{
        justify-content: flex-start;
        gap: 6px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        padding-left: 2px;
    }

    .stTabs [data-baseweb="tab"]{
        background: transparent;
        border: none;
        font-weight: 700;
        color: #6b7280;
        padding: 10px 14px;
        font-size: .95rem;
    }

    .stTabs [data-baseweb="tab"]:hover{
        color:#111827;
        background: transparent;
    }

    .stTabs [aria-selected="true"]{
        color:#111827;
        border-bottom: 3px solid #ef4444; /* rojo NEIX */
    }

    /* ===== Secciones ===== */
    .section-title{
        font-size:1.35rem;
        font-weight:800;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .section-sub{
        color:#6b7280;
        font-size:.92rem;
        margin-bottom:14px;
    }

    /* ===== Cards ===== */
    .tool-grid{
        display:flex;
        gap:14px;
        flex-wrap:wrap;
        margin-top:6px;
    }

    .tool-btn{
        display:flex;
        align-items:center;
        justify-content:center;
        padding:14px 18px;
        border-radius:14px;
        border:1px solid rgba(0,0,0,0.08);
        background:white;
        text-decoration:none !important;
        font-weight:700;
        color:#0f172a;
        min-width:240px;
        box-shadow:0 2px 10px rgba(0,0,0,0.04);
        transition: all .08s ease;
    }

    .tool-btn:hover{
        transform: translateY(-1px);
        box-shadow:0 8px 22px rgba(0,0,0,0.08);
        border-color: rgba(239,68,68,.35);
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
        if st.button("← Volver", key=f"back_{tool_key}"):
            go_home()
    return _back


# =========================================================
# ROUTER
# =========================================================
tool = (st.query_params.get("tool") or "").lower().strip()

if tool:
    st.markdown("<div class='neix-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
    st.markdown("<div class='neix-caption'>Navegación por áreas y proyectos</div>", unsafe_allow_html=True)

    back_to_home = back_to_home_factory(tool)
    back_to_home()
    st.divider()

    if tool == "bonos":
        bonos.render(back_to_home)
    elif tool == "ons":
        ons.render(back_to_home)
    elif tool == "cartera":
        cartera.render(back_to_home)
    elif tool in ("tenencia", "vencimientos"):
        vencimientos.render(back_to_home)

    elif tool == "cheques":
        cheques.render(back_to_home)
    elif tool == "cauciones_mae":
        cauciones_mae.render(back_to_home)
    elif tool == "cauciones_byma":
        cauciones_byma.render(back_to_home)
    elif tool == "alquileres":
        alquileres.render(back_to_home)

    elif tool == "ppt_manana":
        ppt_manana.render(back_to_home)
    elif tool == "moc_tarde":
        moc_tarde.render(back_to_home)
    elif tool == "control_sliq":
        control_sliq.render(back_to_home)
    elif tool == "acreditacion_mav":
        acreditacion_mav.render(back_to_home)
    elif tool == "cauciones":
        cauciones.render(back_to_home)
    else:
        st.error("Herramienta no encontrada")

    st.stop()


# =========================================================
# HOME
# =========================================================
st.markdown("<div class='neix-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
st.markdown("<div class='neix-caption'>Navegación por áreas y proyectos</div>", unsafe_allow_html=True)

tabs = st.tabs(["Mesa", "Comercial", "Operaciones"])


# =======================
# MESA
# =======================
with tabs[0]:
    st.markdown("<div class='section-title'>Mesa</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Bonos, ONs y carteras</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=bonos">Bonos</a>
          <a class="tool-btn" href="?tool=ons">Obligaciones Negociables</a>
          <a class="tool-btn" href="?tool=cartera">Carteras</a>
        </div>
        """,
        unsafe_allow_html=True
    )


# =======================
# COMERCIAL
# =======================
with tabs[1]:
    st.markdown("<div class='section-title'>Comercial</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Seguimiento y herramientas comerciales</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=cartera">Carteras</a>
          <a class="tool-btn" href="?tool=cauciones_mae">Cauciones MAE</a>
          <a class="tool-btn" href="?tool=cauciones_byma">Cauciones BYMA</a>
          <a class="tool-btn" href="?tool=cheques">Cheques</a>
          <a class="tool-btn" href="?tool=alquileres">Alquileres</a>
          <a class="tool-btn" href="?tool=tenencia">Tenencia</a>
        </div>
        """,
        unsafe_allow_html=True
    )


# =======================
# OPERACIONES
# =======================
with tabs[2]:
    st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Procesos operativos y control</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=ppt_manana">PPT Mañana</a>
          <a class="tool-btn" href="?tool=moc_tarde">MOC Tarde</a>
          <a class="tool-btn" href="?tool=control_sliq">Control SLIQ</a>
          <a class="tool-btn" href="?tool=acreditacion_mav">Acreditación MAV</a>
          <a class="tool-btn" href="?tool=cauciones">Cauciones</a>
        </div>
        """,
        unsafe_allow_html=True
    )
