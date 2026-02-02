# app.py
import streamlit as st

# =========================
# IMPORTS (NUEVA ESTRUCTURA)
# =========================
from tools.mesa import cartera
from tools.mesa import ons
from tools.mesa import vencimientos
from tools.mesa import bonos

from tools.backoffice import cauciones as bo_cauciones
from tools.backoffice import control_sliq
from tools.backoffice import moc_tarde
from tools.backoffice import ppt_manana as bo_ppt_manana
from tools.backoffice import acreditacion_mav as bo_acreditacion_mav

from tools.comerciales import cheques
from tools.comerciales import cauciones_mae
from tools.comerciales import cauciones_byma
from tools.comerciales import alquileres


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide",
)

# =========================================================
# Helpers de navegación
# =========================================================
def go_home():
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

def get_tool_param() -> str:
    """
    Streamlit a veces devuelve str o list[str] en query_params.
    """
    raw = st.query_params.get("tool", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return (raw or "").strip()


# =========================================================
# REGISTRO DE HERRAMIENTAS
# =========================================================
TOOL_REGISTRY = {
    # Mesa
    "cartera": cartera.render,
    "bonos": bonos.render,
    "ons": ons.render,
    "vencimientos": vencimientos.render,

    # BackOffice
    "bo_ppt_manana": bo_ppt_manana.render,
    "bo_moc_tarde": moc_tarde.render,
    "bo_control_sliq": control_sliq.render,
    "bo_acreditacion_mav": bo_acreditacion_mav.render,
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
# CSS / Header (si ya lo tenías, dejalo igual)
# =========================================================
st.markdown(
    """
    <style>
      .tool-grid{
        display:flex;
        gap:14px;
        flex-wrap:wrap;
        margin-top:8px;
      }
      .tool-btn{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        padding:14px 18px;
        border-radius:14px;
        border:1px solid rgba(0,0,0,0.08);
        background: white;
        text-decoration:none !important;
        font-weight:600;
        color:#111827;
        min-width: 220px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        transition: transform .06s ease, box-shadow .06s ease;
      }
      .tool-btn:hover{
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
      }
      .section-title{
        margin-top: 18px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NEIX Workbench")
st.caption("Plataforma interna de herramientas")


# =========================================================
# HOME
# =========================================================
tool = get_tool_param()

if not tool:
    # --- Sección Mesa ---
    st.markdown("### 📈 Mesa", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=cartera" target="_self">Carteras comerciales</a>
          <a class="tool-btn" href="?tool=bonos" target="_self">Bonos</a>
          <a class="tool-btn" href="?tool=ons" target="_self">Obligaciones negociables</a>
          <a class="tool-btn" href="?tool=vencimientos" target="_self">Tenencias</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Sección Back Office ---
    st.markdown("### 🧾 Back Office", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tool-grid">
          <a class="tool-btn" href="?tool=bo_ppt_manana" target="_self">PPT Mañana</a>
          <a class="tool-btn" href="?tool=bo_moc_tarde" target="_self">MOC Tarde</a>
          <a class="tool-btn" href="?tool=bo_control_sliq" target="_self">Control SLIQ</a>
          <a class="tool-btn" href="?tool=bo_cauciones" target="_self">Cauciones</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Sección Comerciales / Middle ---
    st.markdown("### 🤝 Comerciales", unsafe_allow_html=True)
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

else:
    # =========================================================
    # TOOL VIEW
    # =========================================================
    fn = TOOL_REGISTRY.get(tool)
    if fn is None:
        st.error(f"Herramienta no registrada: {tool}")
        if st.button("← Volver"):
            go_home()
    else:
        back_to_home = back_to_home_factory(tool)

        # Si tu render acepta back_to_home, pasamos.
        # Si no acepta, lo llamamos sin args.
        try:
            fn(back_to_home=back_to_home)
        except TypeError:
            fn()
