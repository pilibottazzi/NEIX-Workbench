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
    return (raw or "").strip()


# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <style>
      .tool-grid{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;}
      .tool-btn{
        display:inline-flex;align-items:center;justify-content:center;
        padding:14px 18px;border-radius:14px;
        border:1px solid rgba(0,0,0,0.08);background:white;
        text-decoration:none !important;font-weight:600;color:#111827;
        min-width:240px;box-shadow:0 2px 10px rgba(0,0,0,0.04);
        transition:transform .06s ease, box-shadow .06s ease;
      }
      .tool-btn:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(0,0,0,0.06);}
      .topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 2px;margin-bottom:8px;}
      .brand{display:flex;gap:10px;align-items:center;}
      .logo{
        width:40px;height:40px;border-radius:12px;
        display:flex;align-items:center;justify-content:center;
        font-weight:800;background:#111827;color:white;
      }
      .muted{color:rgba(17,24,39,0.65);margin-top:-6px;}
      .section-title{margin-top:18px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="logo">N</div>
        <div>
          <div style="font-size:22px;font-weight:800;">NEIX Workbench</div>
          <div class="muted">Herramientas internas</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

tool = get_tool_param()

if tool:
    ok = run_tool(tool, back_to_home=back_to_home_factory(tool))
    if not ok:
        st.error(f"No pude abrir la herramienta: '{tool}'. Revisá registry.py y que el módulo tenga render().")
        if st.button("← Volver al inicio", key="btn_back_fallback"):
            go_home()
else:
    for section, items in TOOL_TABS.items():
        st.markdown(f"<h3 class='section-title'>{section}</h3>", unsafe_allow_html=True)
        html = ["<div class='tool-grid'>"]
        for it in items:
            html.append(f"<a class='tool-btn' href='?tool={it['id']}' target='_self'>{it['label']}</a>")
        html.append("</div>")
        st.markdown("\n".join(html), unsafe_allow_html=True)
