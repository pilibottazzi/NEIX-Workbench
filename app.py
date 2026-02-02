import streamlit as st

# =========================
# IMPORTS (nueva estructura)
# =========================
from tools.mesa import cartera, ons, vencimientos, bonos
from tools.comerciales import cheques, cauciones_mae, cauciones_byma, alquileres
from tools.backoffice import cauciones, control_sliq, moc_tarde, ppt_manana, acreditacion_mav


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide"
)


# =========================
# HELPERS
# =========================
def go_home():
    st.query_params.clear()
    st.rerun()

def get_tool_param() -> str:
    raw = st.query_params.get("tool", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return (raw or "").strip()


# =========================
# ROUTER
# =========================
TOOL_REGISTRY = {
    # Mesa / Trading
    "ons": ons.render,
    "bonos": bonos.render,
    "vencimientos": vencimientos.render,
    "cartera": cartera.render,

    # Middle Office
    "cheques": cheques.render,
    "cauciones_mae": cauciones_mae.render,
    "cauciones_byma": cauciones_byma.render,
    "alquileres": alquileres.render,

    # Backoffice
    "cauciones": cauciones.render,
    "control_sliq": control_sliq.render,
    "moc_tarde": moc_tarde.render,
    "ppt_manana": ppt_manana.render,
    "acreditacion_mav": acreditacion_mav.render,
}

tool = get_tool_param()

if tool:
    # Header simple + volver
    top = st.columns([1, 1, 6])
    with top[0]:
        if st.button("← Volver", use_container_width=True):
            go_home()

    fn = TOOL_REGISTRY.get(tool)
    if fn is None:
        st.error(f"Herramienta no encontrada: {tool}")
        st.stop()

    fn()
    st.stop()


# =========================
# HOME UI (Tabs minimalistas)
# =========================
st.markdown(
    """
<style>
/* Layout más compacto y “NEIX” */
.block-container{padding-top:2.2rem; padding-bottom:2rem;}
h1{margin-bottom:0.1rem;}
.small-sub{color:#6b7280; margin-top:0; margin-bottom:1.2rem;}

/* Tabs más limpias */
.stTabs [data-baseweb="tab-list"]{
  gap:10px;
}
.stTabs [data-baseweb="tab"]{
  height:42px;
  padding:0 14px;
  border-radius:12px;
  border:1px solid rgba(0,0,0,0.08);
  background:#fff;
}
.stTabs [aria-selected="true"]{
  border-color: rgba(0,0,0,0.18);
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

/* Botones estilo tarjeta */
.tool-btn{
  display:flex;
  align-items:center;
  justify-content:center;
  height:44px;
  border-radius:12px;
  border:1px solid rgba(0,0,0,0.08);
  background:#ffffff;
  font-weight:600;
  color:#111827;
  text-decoration:none !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.03);
  transition: transform .06s ease, box-shadow .06s ease;
}
.tool-btn:hover{
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(0,0,0,0.07);
}
</style>
""",
    unsafe_allow_html=True
)

st.title("NEIX Workbench")
st.markdown('<p class="small-sub">Herramientas internas • Mesa / Middle / Backoffice</p>', unsafe_allow_html=True)

tab_mesa, tab_middle, tab_bo = st.tabs(["Mesa / Trading", "Middle Office", "Backoffice"])

with tab_mesa:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<a class="tool-btn" href="?tool=ons" target="_self">ONs — Screener</a>', unsafe_allow_html=True)
    with c2:
        st.markdown('<a class="tool-btn" href="?tool=bonos" target="_self">Bonos</a>', unsafe_allow_html=True)
    with c3:
        st.markdown('<a class="tool-btn" href="?tool=vencimientos" target="_self">Tenencias</a>', unsafe_allow_html=True)
    with c4:
        st.markdown('<a class="tool-btn" href="?tool=cartera" target="_self">Carteras comerciales</a>', unsafe_allow_html=True)

with tab_middle:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<a class="tool-btn" href="?tool=cheques" target="_self">Cheques y Pagarés</a>', unsafe_allow_html=True)
    with c2:
        st.markdown('<a class="tool-btn" href="?tool=cauciones_mae" target="_self">Garantías MAE</a>', unsafe_allow_html=True)
    with c3:
        st.markdown('<a class="tool-btn" href="?tool=cauciones_byma" target="_self">Garantías BYMA</a>', unsafe_allow_html=True)
    with c4:
        st.markdown('<a class="tool-btn" href="?tool=alquileres" target="_self">Alquileres</a>', unsafe_allow_html=True)

with tab_bo:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<a class="tool-btn" href="?tool=cauciones" target="_self">Cauciones</a>', unsafe_allow_html=True)
    with c2:
        st.markdown('<a class="tool-btn" href="?tool=control_sliq" target="_self">Control SLIQ</a>', unsafe_allow_html=True)
    with c3:
        st.markdown('<a class="tool-btn" href="?tool=moc_tarde" target="_self">MOC Tarde</a>', unsafe_allow_html=True)
    with c4:
        st.markdown('<a class="tool-btn" href="?tool=ppt_manana" target="_self">PPT Mañana</a>', unsafe_allow_html=True)
    with c5:
        st.markdown('<a class="tool-btn" href="?tool=acreditacion_mav" target="_self">Acreditación MAV</a>', unsafe_allow_html=True)

