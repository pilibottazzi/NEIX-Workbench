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
# HOME (estética NEIX minimal / corporate + tabs por área)
# =========================
tool = st.query_params.get("tool")

if tool:
    top = st.columns([1, 6, 1])
    with top[0]:
        if st.button("← Volver", use_container_width=True):
            st.query_params.clear()
            st.rerun()
    run_tool(tool)
    st.stop()

st.markdown("""
<style>
/* ---- Layout general ---- */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1180px; }
h1, h2, h3, h4 { letter-spacing: -0.02em; }

/* ---- Header centrado ---- */
.neix-header{
  text-align:center;
  margin: 6px 0 18px 0;
}
.neix-title{
  font-size: 36px;
  font-weight: 800;
  color: #0f172a;
  margin-bottom: 4px;
}
.neix-subtitle{
  font-size: 14px;
  color: rgba(15, 23, 42, 0.65);
  margin-top: 0px;
}

/* ---- Card contenedor de cada tab ---- */
.neix-panel{
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  padding: 18px 18px 8px 18px;
  background: #ffffff;
  box-shadow: 0 6px 20px rgba(15,23,42,0.04);
}

/* ---- Título de sección dentro del panel ---- */
.neix-section{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom: 12px;
}
.neix-section h3{
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
  margin: 0;
}
.neix-section span{
  font-size: 12px;
  color: rgba(15,23,42,0.55);
}

/* ---- Grid de tools ---- */
.tool-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin: 12px 0 6px 0;
}
@media (max-width: 900px){
  .tool-grid{ grid-template-columns: 1fr; }
}

/* ---- Botón-card ---- */
.tool-card{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 12px;

  padding: 14px 16px;
  border-radius: 14px;

  border: 1px solid rgba(15,23,42,0.10);
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  text-decoration: none !important;

  color: #0f172a;
  font-weight: 650;

  box-shadow: 0 2px 10px rgba(15,23,42,0.04);
  transition: transform .08s ease, box-shadow .08s ease, border-color .08s ease;
}
.tool-card:hover{
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(15,23,42,0.08);
  border-color: rgba(15,23,42,0.18);
}

/* ---- Texto secundario dentro de la card ---- */
.tool-meta{
  font-size: 12px;
  font-weight: 500;
  color: rgba(15,23,42,0.55);
  margin-top: 2px;
}
.tool-left{
  display:flex;
  flex-direction:column;
  line-height: 1.1;
}
.tool-right{
  font-size: 16px;
  color: rgba(15,23,42,0.45);
}

/* ---- Ajuste tabs Streamlit ---- */
.stTabs [data-baseweb="tab-list"]{
  gap: 10px;
}
.stTabs [data-baseweb="tab"]{
  height: 44px;
  padding: 0 14px;
  border-radius: 12px;
  border: 1px solid rgba(15,23,42,0.10);
  background: #ffffff;
}
.stTabs [aria-selected="true"]{
  border-color: rgba(15,23,42,0.22);
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="neix-header">
  <div class="neix-title">NEIX Workbench</div>
  <div class="neix-subtitle">Herramientas internas · Mesa · Comercial · Backoffice</div>
</div>
""", unsafe_allow_html=True)

tab_mesa, tab_comercial, tab_backoffice = st.tabs(["Mesa", "Comercial", "Backoffice"])

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
            ONs — Screener
            <div class="tool-meta">Curvas, TIR, duration, spreads</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=bonos">
          <div class="tool-left">
            Bonos
            <div class="tool-meta">Pricing, métricas y comparables</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=vencimientos">
          <div class="tool-left">
            Tenencias
            <div class="tool-meta">Vencimientos y composición</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=cartera">
          <div class="tool-left">
            Carteras comerciales
            <div class="tool-meta">Seguimiento y resumen</div>
          </div>
          <div class="tool-right">›</div>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
            Cheques y Pagarés
            <div class="tool-meta">Cruce, estado, alertas</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=cauciones_mae">
          <div class="tool-left">
            Garantías MAE
            <div class="tool-meta">Aforos, elegibles, inconsistencias</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=cauciones_byma">
          <div class="tool-left">
            Garantías BYMA
            <div class="tool-meta">Validaciones, cortes y resumen</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=alquileres">
          <div class="tool-left">
            Alquileres
            <div class="tool-meta">Control y seguimiento mensual</div>
          </div>
          <div class="tool-right">›</div>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
            Cauciones
            <div class="tool-meta">Control operativo y conciliación</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=control_sliq">
          <div class="tool-left">
            Control SLIQ
            <div class="tool-meta">Chequeos y validaciones</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=moc_tarde">
          <div class="tool-left">
            MOC Tarde
            <div class="tool-meta">Papel de trabajo</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=ppt_manana">
          <div class="tool-left">
            PPT Mañana
            <div class="tool-meta">Generación y control</div>
          </div>
          <div class="tool-right">›</div>
        </a>
        <a class="tool-card" href="?tool=acreditacion_mav">
          <div class="tool-left">
            Acreditación MAV
            <div class="tool-meta">Monitoreo y estados</div>
          </div>
          <div class="tool-right">›</div>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)


