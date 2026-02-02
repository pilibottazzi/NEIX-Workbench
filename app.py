import streamlit as st

# =========================
# IMPORTS NUEVA ESTRUCTURA
# =========================
from tools.mesa import cartera, ons, vencimientos, bonos
from tools.comerciales import cheques, cauciones_mae, cauciones_byma, alquileres
from tools.backoffice import cauciones, control_sliq, moc_tarde, ppt_manana, acreditacion_mav

st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide"
)

# =========================
# ROUTER
# =========================
def run_tool(tool: str):
    if tool == "cartera":
        cartera.render()
    elif tool == "ons":
        ons.render()
    elif tool == "bonos":
        bonos.render()
    elif tool == "vencimientos":
        vencimientos.render()

    elif tool == "cheques":
        cheques.render()
    elif tool == "cauciones_mae":
        cauciones_mae.render()
    elif tool == "cauciones_byma":
        cauciones_byma.render()
    elif tool == "alquileres":
        alquileres.render()

    elif tool == "cauciones":
        cauciones.render()
    elif tool == "control_sliq":
        control_sliq.render()
    elif tool == "moc_tarde":
        moc_tarde.render()
    elif tool == "ppt_manana":
        ppt_manana.render()
    elif tool == "acreditacion_mav":
        acreditacion_mav.render()


# =========================
# HOME (tu estética original)
# =========================
tool = st.query_params.get("tool")

if tool:
    if st.button("← Volver"):
        st.query_params.clear()
        st.rerun()
    run_tool(tool)
    st.stop()

st.markdown("""
<style>
.tool-grid{
    display:flex;
    gap:14px;
    flex-wrap:wrap;
    margin-top:20px;
}
.tool-btn{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    padding:16px 22px;
    border-radius:14px;
    border:1px solid rgba(0,0,0,0.08);
    background: white;
    text-decoration:none !important;
    font-weight:600;
    color:#111827;
    min-width:260px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: transform .06s ease, box-shadow .06s ease;
}
.tool-btn:hover{
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
.section-title{
    margin-top:30px;
    font-size:22px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.title("NEIX Workbench")

st.markdown('<div class="section-title">Mesa / Trading</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tool-grid">
<a class="tool-btn" href="?tool=ons">ONs — Screener</a>
<a class="tool-btn" href="?tool=bonos">Bonos</a>
<a class="tool-btn" href="?tool=vencimientos">Tenencias</a>
<a class="tool-btn" href="?tool=cartera">Carteras comerciales</a>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Middle Office</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tool-grid">
<a class="tool-btn" href="?tool=cheques">Cheques y Pagarés</a>
<a class="tool-btn" href="?tool=cauciones_mae">Garantías MAE</a>
<a class="tool-btn" href="?tool=cauciones_byma">Garantías BYMA</a>
<a class="tool-btn" href="?tool=alquileres">Alquileres</a>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Backoffice</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tool-grid">
<a class="tool-btn" href="?tool=cauciones">Cauciones</a>
<a class="tool-btn" href="?tool=control_sliq">Control SLIQ</a>
<a class="tool-btn" href="?tool=moc_tarde">MOC Tarde</a>
<a class="tool-btn" href="?tool=ppt_manana">PPT Mañana</a>
<a class="tool-btn" href="?tool=acreditacion_mav">Acreditación MAV</a>
</div>
""", unsafe_allow_html=True)
