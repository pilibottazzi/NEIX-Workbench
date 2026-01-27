# app.py
import streamlit as st

from tools import ons
from tools import backoffice
from tools import cheques
from tools import cauciones_mae
from tools import cauciones_byma
from tools import control_sliq
from tools import alquileres 
from tools import moc_tarde


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
    # limpia query params y re-renderiza home
    st.query_params.clear()
    st.rerun()

def back_to_home_factory(tool_key: str):
    """
    Devuelve una función back_to_home() con key único por tool,
    evitando StreamlitDuplicateElementId.
    """
    def _back():
        if st.button("← Volver", key=f"btn_back_{tool_key}"):
            go_home()
    return _back

# =========================================================
# ROUTER
# =========================================================
q = st.query_params
tool = (q.get("tool") or "").strip().lower()

if tool:
    st.markdown("<h2 style='text-align:center;'>N E I X &nbsp;&nbsp;Workbench</h2>", unsafe_allow_html=True)

    # botón volver con key único por tool (widget)
    back_to_home = back_to_home_factory(tool_key=tool)
    back_to_home()  # <- SOLO acá (una vez). Las tools no lo deben renderizar "por defecto".

    st.divider()

    try:
        if tool == "ons":
            ons.render(back_to_home)

        elif tool == "bo_ppt_manana":
            backoffice.render_ppt_manana(back_to_home)
        elif tool == "bo_moc_tarde":
            moc_tarde.render(back_to_home)
        elif tool == "bo_control_sliq":
            control_sliq.render(back_to_home)
        elif tool == "bo_acreditacion_mav":
            backoffice.render_acreditacion_mav(back_to_home)
        elif tool == "bo_cauciones":
            backoffice.render_cauciones(back_to_home)

        elif tool == "cheques":
            cheques.render(back_to_home)

        elif tool in ("cauciones_mae", "cauciones-mae"):
            cauciones_mae.render(back_to_home)

        elif tool in ("cauciones_byma", "cauciones-byma"):
            cauciones_byma.render(back_to_home)

        elif tool == "alquileres":
            alquileres.render(back_to_home)
            
        else:
            st.error("Herramienta no encontrada.")
            st.caption("Volvé a Home y verificá el link.")
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
          <a class="tool-btn" href="?tool=ons" target="_self">ON’s</a>
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


