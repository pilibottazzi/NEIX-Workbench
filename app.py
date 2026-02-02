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
# SI ESTÁ EN UNA TOOL
# =========================
tool = st.query_params.get("tool")

if tool:
    cols = st.columns([1,6,1])
    with cols[0]:
        if st.button("← Volver", use_container_width=True):
            st.query_params.clear()
            st.rerun()

    run_tool(tool)
    st.stop()


# =========================
# CSS NEIX CORPORATE
# =========================
st.markdown("""
<style>
.block-container{
  padding-top: 2.2rem;
  padding-bottom: 2.8rem;
  max-width: 1400px;
}

.neix-header{
  text-align:center;
  margin: 6px 0 18px 0;
}
.neix-title{
  font-size: 46px;
  font-weight: 850;
  color: #0f172a;
  letter-spacing: -0.03em;
  line-height: 1.10;
}
.neix-subtitle{
  font-size: 15px;
  color: rgba(15, 23, 42, 0.62);
  margin-top: 10px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap: 12px;
  margin-top: 10px;
}
.stTabs [data-baseweb="tab"]{
  height: 52px;
  padding: 0 18px;
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.12);
  background: #ffffff;
  font-size: 16px;
  font-weight: 650;
}
.stTabs [aria-selected="true"]{
  border-color: rgba(15,23,42,0.22);
  box-shadow: 0 10px 26px rgba(15,23,42,0.08);
}

/* Panel */
.neix-panel{
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 20px;
  padding: 22px;
  background: #ffffff;
  box-shadow: 0 10px 28px rgba(15,23,42,0.05);
  margin-top: 10px;
}

.neix-section{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom: 16px;
}
.neix-section h3{
  font-size: 18px;
  font-weight: 750;
  margin: 0;
}
.neix-section span{
  font-size: 13px;
  color: rgba(15,23,42,0.55);
}

/* Cards */
.tool-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.tool-card{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 14px;

  padding: 18px;
  border-radius: 16px;

  border: 1px solid rgba(15,23,42,0.12);
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  text-decoration: none !important;

  color: #0f172a;
  font-weight: 700;

  box-shadow: 0 4px 16px rgba(15,23,42,0.06);
  transition: transform .10s ease, box-shadow .10s ease;
}
.tool-card:hover{
  transform: translateY(-2px);
  box-shadow: 0 16px 34px rgba(15,23,42,0.10);
}

.tool-left{
  display:flex;
  flex-direction:column;
}
.tool-name{
  font-size: 16px;
}
.tool-meta{
  font-size: 13px;
  color: rgba(15,23,42,0.58);
  margin-top: 5px;
}
.tool-right{
  font-size: 20px;
  color: rgba(15,23,42,0.45);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="neix-header">
  <div class="neix-title">NEIX Workbench</div>
  <div class="neix-subtitle">Herramientas internas · Mesa · Comercial · Backoffice</div>
</div>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab_mesa, tab_comercial, tab_backoffice = st.tabs(
    ["Mesa", "Comercial", "Backoffice"]
)

# =========================
# MESA
# =========================
with tab_mesa:
    st.markdown("""
    <div class="neix-panel">
      <div class="neix-section">
        <h3>Mesa / Trading</h3>
        <span>Research · Pricing · Tenencias</span>
      </div>
      <div class="tool-grid">
        <a class="tool-card" href="?tool=ons">
          <div class="tool-left">
            <div class="tool-name">ONs — Screener</div>
            <div class="tool-meta">Curvas, TIR, duration, spreads</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=bonos">
          <div class="tool-left">
            <div class="tool-name">Bonos</div>
            <div class="tool-meta">Pricing, métricas y comparables</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=vencimientos">
          <div class="tool-left">
            <div class="tool-name">Tenencias</div>
            <div class="tool-meta">Vencimientos y composición</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=cartera">
          <div class="tool-left">
            <div class="tool-name">Carteras comerciales</div>
            <div class="tool-meta">Seguimiento y resumen</div>
          </div>
          <div class="tool-right">›</div>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# COMERCIAL
# =========================
with tab_comercial:
    st.markdown("""
    <div class="neix-panel">
      <div class="neix-section">
        <h3>Comercial / Middle Office</h3>
        <span>Controles · Garantías · Operativa</span>
      </div>
      <div class="tool-grid">
        <a class="tool-card" href="?tool=cheques">
          <div class="tool-left">
            <div class="tool-name">Cheques y Pagarés</div>
            <div class="tool-meta">Cruce, estado, alertas</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=cauciones_mae">
          <div class="tool-left">
            <div class="tool-name">Garantías MAE</div>
            <div class="tool-meta">Aforos y validaciones</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=cauciones_byma">
          <div class="tool-left">
            <div class="tool-name">Garantías BYMA</div>
            <div class="tool-meta">Validaciones y resumen</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=alquileres">
          <div class="tool-left">
            <div class="tool-name">Alquileres</div>
            <div class="tool-meta">Control y seguimiento mensual</div>
          </div>
          <div class="tool-right">›</div>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# BACKOFFICE
# =========================
with tab_backoffice:
    st.markdown("""
    <div class="neix-panel">
      <div class="neix-section">
        <h3>Backoffice</h3>
        <span>Procesos · Control · Reportes</span>
      </div>
      <div class="tool-grid">
        <a class="tool-card" href="?tool=cauciones">
          <div class="tool-left">
            <div class="tool-name">Cauciones</div>
            <div class="tool-meta">Control operativo</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=control_sliq">
          <div class="tool-left">
            <div class="tool-name">Control SLIQ</div>
            <div class="tool-meta">Chequeos y validaciones</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=moc_tarde">
          <div class="tool-left">
            <div class="tool-name">MOC Tarde</div>
            <div class="tool-meta">Papel de trabajo</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=ppt_manana">
          <div class="tool-left">
            <div class="tool-name">PPT Mañana</div>
            <div class="tool-meta">Generación y control</div>
          </div>
          <div class="tool-right">›</div>
        </a>

        <a class="tool-card" href="?tool=acreditacion_mav">
          <div class="tool-left">
            <div class="tool-name">Acreditación MAV</div>
            <div class="tool-meta">Monitoreo y estados</div>
          </div>
          <div class="tool-right">›</div>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

