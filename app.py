# app.py
import streamlit as st

# Router / registro de herramientas
from tools.registry import TOOL_TABS, run_tool


# =========================
# Config general
# =========================
st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide"
)

# =========================
# CSS NEIX (minimal)
# =========================
st.markdown(
    """
    <style>
      .topbar{
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding: 10px 6px 2px 6px;
      }
      .brand{
        display:flex;
        gap:10px;
        align-items:center;
      }
      .logo{
        width:40px;height:40px;border-radius:12px;
        display:flex;align-items:center;justify-content:center;
        font-weight:800;background:#111827;color:#fff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      }
      .subtitle{color:#6b7280;margin-top:-6px}
      .tool-grid{
        display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;
      }
      .tool-btn{
        display:inline-flex;align-items:center;justify-content:center;
        padding:14px 18px;border-radius:14px;border:1px solid rgba(0,0,0,0.08);
        background:#fff;text-decoration:none !important;font-weight:650;color:#111827;
        min-width:220px;box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        transition: transform .06s ease, box-shadow .06s ease;
        cursor:pointer;
      }
      .tool-btn:hover{
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.10);
      }
      .card{
        background:#fff;border:1px solid rgba(0,0,0,0.06);
        border-radius:18px;padding:16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
      }
      .muted{color:#6b7280}
      .hr{height:1px;background:rgba(0,0,0,0.06);margin:12px 0}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar: selector de tool
# =========================
st.sidebar.markdown("### 🧰 NEIX Workbench")
tab_names = list(TOOL_TABS.keys())

default_tab = tab_names[0] if tab_names else None
selected_tab = st.sidebar.selectbox("Área", tab_names, index=0 if default_tab else None)

tool_names = TOOL_TABS.get(selected_tab, [])
selected_tool = st.sidebar.selectbox("Herramienta", tool_names, index=0 if tool_names else None)

st.sidebar.markdown("---")
st.sidebar.caption("NEIX • Workbench")

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="logo">N</div>
        <div>
          <h2 style="margin:0">NEIX Workbench</h2>
          <div class="subtitle">Herramientas internas • Mesa • BackOffice • Comerciales</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Render tool
# =========================
if not selected_tool:
    st.info("Seleccioná una herramienta desde la barra lateral.")
else:
    # Ejecuta la herramienta vía registry (evita imports rotos en app.py)
    run_tool(selected_tool)
