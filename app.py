import streamlit as st
from tools.registry import TOOL_TABS, run_tool

st.set_page_config(page_title="NEIX Workbench", page_icon="🧰", layout="wide")

# -------------------------
# Helpers
# -------------------------
def go_home():
    st.query_params.clear()
    st.rerun()

def get_tool_param() -> str:
    raw = st.query_params.get("tool", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return (raw or "").strip()

def back_to_home(tool_key: str):
    # key única por herramienta (evita duplicate id)
    if st.button("← Volver", key=f"btn_back_{tool_key}", use_container_width=True):
        go_home()

# -------------------------
# CSS (minimal NEIX)
# -------------------------
st.markdown(
    """
<style>
.block-container{padding-top:2.2rem; padding-bottom:2rem; max-width:1100px;}
.small-sub{color:#6b7280; margin-top:0; margin-bottom:1.2rem;}

.stTabs [data-baseweb="tab-list"]{gap:10px;}
.stTabs [data-baseweb="tab"]{
  height:42px; padding:0 14px; border-radius:12px;
  border:1px solid rgba(0,0,0,0.08); background:#fff;
}
.stTabs [aria-selected="true"]{
  border-color: rgba(0,0,0,0.18);
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.tool-grid{display:flex; gap:12px; flex-wrap:wrap; margin-top:10px;}
.tool-btn{
  display:flex; align-items:center; justify-content:center;
  height:44px; min-width:220px;
  border-radius:12px;
  border:1px solid rgba(0,0,0,0.08);
  b
