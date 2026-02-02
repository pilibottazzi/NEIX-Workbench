# app.py
import streamlit as st
from tools.registry import TOOL_TABS, run_tool

st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide"
)

# Sidebar
st.sidebar.markdown("### 🧰 NEIX Workbench")
tab_names = list(TOOL_TABS.keys())
selected_tab = st.sidebar.selectbox("Área", tab_names, index=0)

tool_names = TOOL_TABS.get(selected_tab, [])
selected_tool = st.sidebar.selectbox("Herramienta", tool_names, index=0 if tool_names else None)

st.sidebar.markdown("---")
st.sidebar.caption("NEIX • Workbench")

# Header simple
st.title("NEIX Workbench")
st.caption("Herramientas internas • Mesa • BackOffice • Comerciales")

st.divider()

if not selected_tool:
    st.info("Seleccioná una herramienta desde la barra lateral.")
else:
    run_tool(selected_tool)
