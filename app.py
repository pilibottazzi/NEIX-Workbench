import streamlit as st
from tools.registry import TOOL_TABS, run_tool

st.set_page_config(page_title="NEIX Workbench", page_icon="🧰", layout="wide")

# =============== ESTÉTICA (TU CSS) ===============
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

# ==================================================
# Navegación
# ==================================================
def go_home():
    st.query_params.clear()
    st.rerun()

def back_to_home(tool_id: str):
    # ✅ key única por tool (evita StreamlitDuplicateElementKey)
    if st.button("← Volver al Workbench", key=f"btn_back__{tool_id}"):
        go_home()

q = st.query_params
tool = (q.get("tool") or "").strip().lower()

# =========================
# ROUTER (tool view)
# =========================
if tool:
    st.markdown("<h2 style='text-align:center;'>N E I X &nbsp;&nbsp;Workbench</h2>", unsafe_allow_html=True)
    st.markdown('<div class="top-note">Vista de herramienta</div>', unsafe_allow_html=True)

    st.markdown('<a class="back-link" href="?">🏠 Ir a Home</a>', unsafe_allow_html=True)
    st.divider()

    try:
        ok = run_tool(tool, lambda: back_to_home(tool))
        if not ok:
            back_to_home(tool)
            st.error("Herramienta no encontrada.")
    except Exception as e:
        back_to_home(tool)
        st.error("Error cargando la herramienta.")
        st.exception(e)

    st.stop()

# =========================
# HOME
# =========================
st.markdown("<h2 style='text-align:center;'>N E I X &nbsp;&nbsp;Workbench</h2>", unsafe_allow_html=True)
st.caption("Navegación por áreas y proyectos")
st.divider()

tab_names = list(TOOL_TABS.keys())
tabs = st.tabs(tab_names)

for i, tab_name in enumerate(tab_names):
    with tabs[i]:
        st.subheader(tab_name)
        st.caption("Elegí una herramienta")

        buttons_html = '<div class="tool-grid">'
        for t in TOOL_TABS[tab_name]:
            buttons_html += (
                f'<a class="tool-btn" href="?tool={t["id"]}" target="_blank">'
                f'{t["label"]}'
                f'</a>'
            )
        buttons_html += "</div>"

        st.markdown(buttons_html, unsafe_allow_html=True)

