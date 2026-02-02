# app.py
import streamlit as st

from tools.registry import TOOL_TABS, run_tool

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide",
)

# =========================================================
# HELPERS
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
# CSS (tu estética)
# =========================================================
st.markdown(
    """
    <style>
      .topbar{
        display:flex; align-items:center; justify-content:space-between;
        padding:10px 2px 16px 2px;
      }
      .brand{display:flex; gap:12px; align-items:center;}
      .logo{
        width:38px; height:38px; border-radius:14px;
        display:flex; align-items:center; justify-content:center;
        font-weight:800;
        background: #111827; color: white;
      }
      .tool-grid{
        display:flex; gap:14px; flex-wrap:wrap;
        margin-top:10px;
      }
      .tool-btn{
        display:inline-flex; align-items:center; justify-content:center;
        padding:14px 18px; border-radius:14px;
        border:1px solid rgba(0,0,0,0.10);
        background:white; text-decoration:none !important;
        font-weight:650; color:#111827;
        min-width: 240px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        transition: transform .06s ease, box-shadow .06s ease;
      }
      .tool-btn:hover{
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
      }
      .section-title{
        margin-top: 22px;
        font-size: 1.05rem;
        font-weight: 800;
        color: #111827;
      }
      .muted{color:#6b7280;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HOME UI
# =========================================================
def render_home():
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <div class="logo">N</div>
            <div>
              <div style="font-size:1.35rem; font-weight:900; color:#111827;">NEIX Workbench</div>
              <div class="muted">Herramientas internas • Mesa / Middle / Backoffice</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tabs -> secciones
    for tab_name, items in TOOL_TABS.items():
        st.markdown(f'<div class="section-title">{tab_name}</div>', unsafe_allow_html=True)
        st.markdown('<div class="tool-grid">', unsafe_allow_html=True)

        for it in items:
            tool_id = it["id"]
            label = it["label"]
            st.markdown(
                f'<a class="tool-btn" href="?tool={tool_id}" target="_self">{label}</a>',
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ROUTER
# =========================================================
tool = get_tool_param()

if not tool:
    render_home()
else:
    # intenta correr la tool con router nuevo (subcarpetas)
    back_to_home = back_to_home_factory(tool_key=tool)

    ok = run_tool(tool_id=tool, back_to_home=back_to_home)

    if not ok:
        st.error(f"No se encontró la herramienta '{tool}'. Revisá TOOL_MODULES en tools/registry.py.")
        if st.button("← Volver", key="btn_back_fallback"):
            go_home()
