import streamlit as st

from tools.mesa import cartera, ons, vencimientos, bonos, cartera2
from tools.comerciales import cauciones_mae, cauciones_byma, alquileres, cn, transactions_analyzer


# =========================
# URLS
# =========================
BACKOFFICE_URL = "https://neix-workbench-bo.streamlit.app/"
BI_BANCA_PRIVADA = "https://lookerstudio.google.com/reporting/75c2a6d0-0086-491f-b112-88fe3d257ef9"
BI_BANCA_CORP = "https://lookerstudio.google.com/reporting/4f70efa8-2b86-4134-a9cb-9e6f90117f3b"
BI_MIDDLE = "https://lookerstudio.google.com/reporting/5b834e5f-aeef-4042-ac0f-e1ed3564a010"

SP_MKT_INSTRUCTIVOS = "https://neixcom.sharepoint.com/sites/NEIXSOCIEDADDEBOLSAS.A-Marketingprueba/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FNEIXSOCIEDADDEBOLSAS%2EA%2DMarketingprueba%2FShared%20Documents%2FMarketing%2FInstructivos&viewid=74e4d9a3%2Dd2c9%2D4f09%2D9bc8%2Deb59e613117f&p=true"
SP_MKT_MATERIALES = "https://neixcom.sharepoint.com/sites/NEIXSOCIEDADDEBOLSAS.A-Marketingprueba/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FNEIXSOCIEDADDEBOLSAS%2EA%2DMarketingprueba%2FShared%20Documents%2FMarketing%2FMateriales%20de%20Marketing&viewid=74e4d9a3%2Dd2c9%2D4f09%2D9bc8%2Deb59e613117f&p=true"
SP_MKT_PRESENTACIONES = "https://neixcom.sharepoint.com/sites/NEIXSOCIEDADDEBOLSAS.A-Marketingprueba/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FNEIXSOCIEDADDEBOLSAS%2EA%2DMarketingprueba%2FShared%20Documents%2FMarketing%2FPresentaciones&viewid=74e4d9a3%2Dd2c9%2D4f09%2D9bc8%2Deb59e613117f&p=true"


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="NEIX Workbench",
    page_icon="🧰",
    layout="wide"
)


# =========================
# PASSWORD
# =========================
APP_PASSWORD = st.secrets.get("app_password")

if not APP_PASSWORD:
    st.error("No se encontró 'app_password' en st.secrets.")
    st.stop()


# =========================
# CSS MINIMAL
# =========================
st.markdown("""
<style>

.block-container{
    max-width:1240px;
    padding-top:2rem;
}

header[data-testid="stHeader"]{
    visibility:hidden;
}

/* HEADER */

.neix-title{
    text-align:center;
    font-weight:900;
    letter-spacing:.14em;
    font-size:1.6rem;
}

.neix-caption{
    text-align:center;
    color:#6b7280;
    font-size:.9rem;
}

.neix-line{
    width:60px;
    height:3px;
    background:#ef4444;
    margin:12px auto 26px auto;
    border-radius:999px;
}

/* LOGIN */

.login-wrap{
    max-width:360px;
    margin:140px auto 0 auto;
    padding:30px 28px;
    border-radius:18px;
    border:1px solid rgba(0,0,0,0.08);
    background:white;
    box-shadow:0 10px 26px rgba(0,0,0,0.05);
}

.login-title{
    text-align:center;
    font-weight:900;
    letter-spacing:.12em;
    font-size:1.25rem;
}

.login-sub{
    text-align:center;
    color:#6b7280;
    font-size:.9rem;
    margin-bottom:14px;
}

.login-footer{
    text-align:center;
    color:#94a3b8;
    font-size:.8rem;
    margin-top:10px;
}

/* INPUT */

div[data-testid="stTextInput"]{
    margin-top:0 !important;
}

div[data-testid="stTextInput"] > div{
    background:transparent !important;
}

div[data-testid="stTextInput"] input{
    border-radius:12px;
    padding:12px;
    border:1px solid rgba(0,0,0,0.08);
}

/* BOTONES */

div[data-testid="stButton"] > button{
    width:100%;
    border-radius:12px;
    height:42px;
    font-weight:700;
}

/* TABS */

.stTabs [data-baseweb="tab"]{
    font-weight:700;
    color:#64748b;
}

.stTabs [aria-selected="true"]{
    border-bottom:3px solid #ef4444;
    color:#1e3a8a !important;
}

/* CARDS */

.tool-grid{
    display:flex;
    flex-wrap:wrap;
    gap:14px;
}

.tool-btn{
    padding:12px 18px;
    border-radius:14px;
    border:1px solid rgba(0,0,0,0.08);
    background:white;
    text-decoration:none !important;
    color:#1e3a8a !important;
    font-weight:700;
    min-width:240px;
}

.tool-btn-primary{
    background:#ef4444 !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
def _header():

    col1, col2 = st.columns([12,1])

    with col1:
        st.markdown("<div class='neix-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
        st.markdown("<div class='neix-caption'>Navegación por áreas y proyectos</div>", unsafe_allow_html=True)
        st.markdown("<div class='neix-line'></div>", unsafe_allow_html=True)

    with col2:
        if st.session_state.get("logged_in", False):
            if st.button("Salir"):
                st.session_state.logged_in = False
                st.rerun()


# =========================
# LOGIN
# =========================
def check_password():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    col1,col2,col3 = st.columns([1,1.2,1])

    with col2:

        st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)

        st.markdown("<div class='login-title'>NEIX Workbench</div>", unsafe_allow_html=True)
        st.markdown("<div class='login-sub'>Ingresá la clave para continuar</div>", unsafe_allow_html=True)
        st.markdown("<div class='neix-line'></div>", unsafe_allow_html=True)

        password = st.text_input(
            "Clave",
            type="password",
            placeholder="Clave",
            label_visibility="collapsed"
        )

        if st.button("Ingresar"):
            if password == APP_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Clave incorrecta")

        st.markdown("<div class='login-footer'>Acceso interno</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    return False


# =========================
# LOGIN CHECK
# =========================
if not check_password():
    st.stop()


# =========================
# ROUTER
# =========================
tool = (st.query_params.get("tool") or "").lower().strip()

if tool:

    _header()

    if tool == "bonos":
        bonos.render(None)

    elif tool == "ons":
        ons.render(None)

    elif tool == "cartera":
        cartera.render(None)

    elif tool == "cartera2":
        cartera2.render(None)

    elif tool in ("tenencia","tenencias","vencimientos"):
        vencimientos.render(None)

    elif tool == "cauciones_mae":
        cauciones_mae.render(None)

    elif tool == "cn":
        cn.render(None)

    elif tool == "cauciones_byma":
        cauciones_byma.render(None)

    elif tool == "alquileres":
        alquileres.render(None)

    elif tool == "transactions_analyzer":
        transactions_analyzer.render()

    st.stop()


# =========================
# HOME
# =========================
_header()

tabs = st.tabs(["Comercial","Operaciones","Mesa","Performance · BI","Marketing"])


# =========================
# COMERCIAL
# =========================
with tabs[0]:

    st.markdown("### Comercial")

    st.markdown("""
    <div class="tool-grid">
        <a class="tool-btn" href="?tool=cauciones_mae">Cauciones MAE</a>
        <a class="tool-btn" href="?tool=cauciones_byma">Cauciones BYMA</a>
        <a class="tool-btn" href="?tool=alquileres">Alquileres</a>
        <a class="tool-btn" href="?tool=tenencia">Tenencia</a>
        <a class="tool-btn" href="?tool=cn">CN</a>
        <a class="tool-btn" href="?tool=transactions_analyzer">Movimientos CV</a>
    </div>
    """, unsafe_allow_html=True)


# =========================
# OPERACIONES
# =========================
with tabs[1]:

    st.markdown("### Operaciones")

    st.markdown(f"""
    <div class="tool-grid">
        <a class="tool-btn tool-btn-primary" href="{BACKOFFICE_URL}" target="_blank">
        Abrir Backoffice
        </a>
    </div>
    """, unsafe_allow_html=True)


# =========================
# MESA
# =========================
with tabs[2]:

    st.markdown("### Mesa")

    st.markdown("""
    <div class="tool-grid">
        <a class="tool-btn" href="?tool=bonos">Bonos</a>
        <a class="tool-btn" href="?tool=ons">Obligaciones Negociables</a>
        <a class="tool-btn" href="?tool=cartera">Carteras (rendimiento)</a>
        <a class="tool-btn" href="?tool=cartera2">Carteras (ARG)</a>
    </div>
    """, unsafe_allow_html=True)


# =========================
# BI
# =========================
with tabs[3]:

    st.markdown("### Performance · BI")

    st.markdown(f"""
    <div class="tool-grid">
        <a class="tool-btn" href="{BI_BANCA_PRIVADA}" target="_blank">Banca Privada</a>
        <a class="tool-btn" href="{BI_BANCA_CORP}" target="_blank">Banca Corporativa</a>
        <a class="tool-btn" href="{BI_MIDDLE}" target="_blank">Middle Office</a>
    </div>
    """, unsafe_allow_html=True)


# =========================
# MARKETING
# =========================
with tabs[4]:

    st.markdown("### Marketing")

    st.markdown("""
    <div class="tool-grid">
        <a class="tool-btn" href="?tool=mkt_instructivos">Instructivos</a>
        <a class="tool-btn" href="?tool=mkt_materiales">Materiales de Marketing</a>
        <a class="tool-btn" href="?tool=mkt_presentaciones">Presentaciones</a>
    </div>
    """, unsafe_allow_html=True)
