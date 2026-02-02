# app.py
import streamlit as st

# ==========================
# IMPORTS POR ÁREA (nuevo)
# ==========================
from tools.mesa import cartera, ons, vencimientos, bonos
from tools.backoffice import (
    moc_tarde,
    ppt_manana,
    acreditacion_mav,
    control_sliq,
    cauciones as bo_cauciones,
)
from tools.comerciales import (
    alquileres,
    cheques,
    cauciones_mae,
    cauciones_byma,
)

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NEIX Workbench", page_icon="🧰", layout="wide")

st.markdown(
    """
    <style>
      .tool-grid{ display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; }
      .tool-btn{
        display:inline-flex; align-items:center; justify-content:center;
        padding:14px 18px; border-radius:14px;
        border:1px solid rgba(0,0,0,0.08);
        background:white; text-decoration:none !important;
        font-weight:600; color:#111827; min-width:220px;
        box-shadow:0 2px 10px rgba(0,0,0,0.04);
        transition: transform .06s ease, box-shadow .06s ease;
      }
      .tool-btn:hover{ transform: translateY(-1px); box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
      .top-note{ color:#6b7280; font-size:0.92rem; }
      .back-link{ display:inline-block; margin:10px 0 18px 0; text-decoration:none; font-weight:600; }
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
        if st.button("← Volver", key=f"btn_back_{tool_key}"):
            go_home()
    return _back

def get_tool_param() -> str:
    raw = st.query_params.get("tool", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw).strip().lower()

# =========================================================
# REGISTRO DE HERRAMIENTAS
# =========================================================
TOOL_REGISTRY = {
    # Mesa
    "cartera": cartera.render,
    "bonos": bonos.render,
    "ons": ons.render,
    "vencimientos": vencimientos.render,

    # Backoffice
    "bo_ppt_manana": ppt_manana.render,
    "bo_moc_tarde": moc_tarde.render,
    "bo_control_sliq": control_sliq.render,
    "bo_acreditacion_mav": acreditacion_mav.render,
    "bo_cauciones": bo_cauciones.render,

    # Comerciales
    "cheques": cheques.render,
    "cauciones_mae": cauciones_mae.render,
    "cauciones-mae": cauciones_mae.render,
    "cauciones_byma": cauciones_byma.render,
    "cauciones-byma": cauciones_byma.render,
    "alquileres": alquileres.render,
}

# =========================================================
# ROUTER
# =========================================================
tool = get_tool_param()

if tool:
    st.markdown("<h2 style='text-align:center;'>N E I X &nbsp;&nbsp;Workbench</h2>", unsafe_allow_html=True)

    back_to_home = back_to_home_factory(tool_key=tool)
    back_to_home()
    st.divider()

    render_fn = TOOL_REGISTRY.get(tool)

    if render_fn is None:
        st.error("Herramienta no encontrada.")
        st.caption("Volvé a Home y verificá el link.")
        st.stop()

    try:
        render_fn(back_to_home)
    except Exception as e:
        st.error("Error cargando la herramienta.")
        st.exception(e)

    st.stop()

# =========================================================
# HOME
# =========================================================
st.markdown("<h2 style='text-align:center;'>N E I X &nbsp;&nbsp;Workbench</h2>", unsafe_allow_html=True)
st.caption("Navegación por áreas y proyectos")
st.divider()

tabs = st.tabs(["Mesa", "Backoffice", "Comerciales"])

with tabs[0]:
    st.subheader("Mesa")
    st.caption("Elegí una herramienta")
    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=cartera" target="_self">Carteras comerciales</a>
          <a class="tool-btn" href="?tool=bonos" target="_self">Bonos</a>
          <a class="tool-btn" href="?tool=ons" target="_self">Obligaciones negociables</a>
          <a class="tool-btn" href="?tool=vencimientos" target="_self">Vencimientos / Tenencias</a>
        </div>
        """,
        unsafe_allow_html=True
    )

with tabs[1]:
    st.subheader("Backoffice")
    st.caption("Elegí una herramienta")
    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=bo_ppt_manana" target="_self">PPT Mañana</a>
          <a class="tool-btn" href="?tool=bo_moc_tarde" target="_self">MOC Tarde</a>
          <a class="tool-btn" href="?tool=bo_control_sliq" target="_self">Control SLIQ</a>
          <a class="tool-btn" href="?tool=bo_acreditacion_mav" target="_self">Acreditación MAV</a>
          <a class="tool-btn" href="?tool=bo_cauciones" target="_self">Cauciones</a>
        </div>
        """,
        unsafe_allow_html=True
    )

with tabs[2]:
    st.subheader("Comerciales")
    st.caption("Elegí una herramienta")
    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=cheques" target="_self">Cheques</a>
          <a class="tool-btn" href="?tool=cauciones_mae" target="_self">Cauciones MAE</a>
          <a class="tool-btn" href="?tool=cauciones_byma" target="_self">Cauciones BYMA</a>
          <a class="tool-btn" href="?tool=alquileres" target="_self">Alquileres</a>
        </div>
        """,
        unsafe_allow_html=True
    )
