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

# SharePoint Marketing
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
# PASSWORD DESDE SECRETS
# =========================
APP_PASSWORD = st.secrets["app_password"]


# =========================
# ESTADO INICIAL
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# =========================
# ESTÉTICA
# =========================
st.markdown(
    """
    <style>
    .block-container{
        padding-top: 2.2rem;
        max-width: 1240px;
    }

    header[data-testid="stHeader"]{
        visibility: hidden;
        height: 3rem;
    }

    /* Header */
    .neix-title{
        text-align:center;
        font-weight:900;
        letter-spacing:.14em;
        font-size:1.6rem;
        margin-bottom:4px;
        color:#111827;
    }

    .neix-caption{
        text-align:center;
        color:#6b7280;
        font-size:.95rem;
        margin-bottom:10px;
    }

    .neix-line{
        width:60px;
        height:3px;
        background:#ef4444;
        margin:0 auto 22px auto;
        border-radius:4px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"]{
        justify-content:flex-start;
        gap:8px;
        border-bottom:1px solid rgba(0,0,0,0.08);
        padding-left:2px;
        margin-top:4px;
    }

    .stTabs [data-baseweb="tab"]{
        background:transparent;
        border:none;
        font-weight:700;
        color:#64748b;
        padding:10px 14px;
        font-size:.95rem;
    }

    .stTabs [data-baseweb="tab"]:hover{
        color:#1e3a8a;
    }

    .stTabs [aria-selected="true"]{
        color:#1e3a8a;
        border-bottom:3px solid #ef4444;
    }

    /* Titles */
    .section-title{
        font-size:1.35rem;
        font-weight:800;
        margin-top:6px;
        margin-bottom:2px;
        color:#111827;
    }

    .section-sub{
        color:#6b7280;
        font-size:.92rem;
        margin-bottom:14px;
    }

    /* Login */
    .login-wrap{
        max-width:420px;
        margin:90px auto 0 auto;
        padding:34px 32px 28px 32px;
        border-radius:18px;
        border:1px solid rgba(0,0,0,0.08);
        background:white;
        box-shadow:0 10px 30px rgba(0,0,0,0.06);
    }

    .login-title{
        text-align:center;
        font-weight:900;
        letter-spacing:.12em;
        font-size:1.35rem;
        margin-bottom:6px;
        color:#111827;
    }

    .login-sub{
        text-align:center;
        color:#6b7280;
        font-size:.95rem;
        margin-bottom:18px;
    }

    .login-line{
        width:56px;
        height:3px;
        background:#ef4444;
        margin:0 auto 22px auto;
        border-radius:4px;
    }

    .login-footer{
        text-align:center;
        color:#94a3b8;
        font-size:.82rem;
        margin-top:10px;
    }

    div[data-testid="stButton"] > button{
        width:100%;
        border-radius:12px;
        font-weight:700;
        min-height:44px;
        border:1px solid rgba(0,0,0,0.08);
    }

    div[data-testid="stLinkButton"] a{
        width:100%;
        border-radius:12px !important;
        font-weight:700 !important;
        min-height:44px !important;
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HELPERS
# =========================
def _header():
    st.markdown("<div class='neix-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
    st.markdown("<div class='neix-caption'>Navegación por áreas y proyectos</div>", unsafe_allow_html=True)
    st.markdown("<div class='neix-line'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)


def go_tool(tool_name: str):
    st.query_params["tool"] = tool_name


def clear_tool():
    st.query_params.clear()


def render_internal_cards(items, cols_per_row=3, key_prefix="nav"):
    """
    items: list of tuples -> (label, tool_name)
    """
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for i in range(cols_per_row):
            with cols[i]:
                if i < len(row):
                    label, tool_name = row[i]
                    st.button(
                        label,
                        key=f"{key_prefix}_{tool_name}",
                        use_container_width=True,
                        on_click=go_tool,
                        args=(tool_name,),
                    )
                else:
                    st.empty()


def render_external_cards(items, cols_per_row=3):
    """
    items: list of tuples -> (label, url)
    """
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for i in range(cols_per_row):
            with cols[i]:
                if i < len(row):
                    label, url = row[i]
                    st.link_button(label, url, use_container_width=True)
                else:
                    st.empty()


def check_password():
    if st.session_state.logged_in:
        return True

    st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>N E I X &nbsp;&nbsp;Workbench</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-sub'>Ingresá la clave para continuar</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-line'></div>", unsafe_allow_html=True)

    password = st.text_input(
        "Clave",
        type="password",
        placeholder="Ingrese la clave",
        key="login_password"
    )

    if st.button("Ingresar", key="login_btn"):
        if password == APP_PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Clave incorrecta")

    st.markdown("<div class='login-footer'>Acceso interno</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    return False


# =========================
# LOGIN
# =========================
if not check_password():
    st.stop()


# =========================
# ROUTER (?tool=...)
# =========================
tool = (st.query_params.get("tool") or "").lower().strip()

if tool:
    _header()

    col_back, _ = st.columns([2, 10])
    with col_back:
        if st.button("← Volver", key="volver_home"):
            clear_tool()
            st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    try:
        # -------------------------
        # Mesa
        # -------------------------
        if tool == "bonos":
            bonos.render(None)
            st.stop()

        elif tool == "ons":
            ons.render(None)
            st.stop()

        elif tool == "cartera":
            cartera.render(None)
            st.stop()

        elif tool == "cartera2":
            cartera2.render(None)
            st.stop()

        elif tool in ("tenencia", "tenencias", "vencimientos"):
            vencimientos.render(None)
            st.stop()

        # -------------------------
        # Comercial
        # -------------------------
        elif tool == "cauciones_mae":
            cauciones_mae.render(None)
            st.stop()

        elif tool == "cn":
            cn.render(None)
            st.stop()

        elif tool == "cauciones_byma":
            cauciones_byma.render(None)
            st.stop()

        elif tool == "alquileres":
            alquileres.render(None)
            st.stop()

        elif tool == "transactions_analyzer":
            transactions_analyzer.render()
            st.stop()

        # -------------------------
        # Marketing
        # -------------------------
        elif tool == "mkt_instructivos":
            st.markdown("<div class='section-title'>Marketing · Instructivos</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Carpeta compartida en SharePoint</div>", unsafe_allow_html=True)
            st.link_button("Abrir Instructivos", SP_MKT_INSTRUCTIVOS, use_container_width=False)
            st.stop()

        elif tool == "mkt_materiales":
            st.markdown("<div class='section-title'>Marketing · Materiales</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Carpeta compartida en SharePoint</div>", unsafe_allow_html=True)
            st.link_button("Abrir Materiales de Marketing", SP_MKT_MATERIALES, use_container_width=False)
            st.stop()

        elif tool == "mkt_presentaciones":
            st.markdown("<div class='section-title'>Marketing · Presentaciones</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Carpeta compartida en SharePoint</div>", unsafe_allow_html=True)
            st.link_button("Abrir Presentaciones", SP_MKT_PRESENTACIONES, use_container_width=False)
            st.stop()

        # -------------------------
        # Operaciones / Backoffice
        # -------------------------
        elif tool in ("operaciones", "backoffice"):
            st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Backoffice se abre en una web externa</div>", unsafe_allow_html=True)
            st.link_button("Abrir Backoffice", BACKOFFICE_URL, use_container_width=False)
            st.stop()

        else:
            st.error("Herramienta no encontrada")
            st.stop()

    except Exception as e:
        st.error("Error cargando la herramienta.")
        st.exception(e)
        st.stop()


# =========================
# HOME
# =========================
_header()

tabs = st.tabs(["Comercial", "Operaciones", "Mesa", "Performance · BI", "Marketing"])


# =========================
# COMERCIAL
# =========================
with tabs[0]:
    st.markdown("<div class='section-title'>Comercial</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Seguimiento y herramientas comerciales</div>", unsafe_allow_html=True)

    render_internal_cards(
        [
            ("Cauciones MAE", "cauciones_mae"),
            ("Cauciones BYMA", "cauciones_byma"),
            ("Alquileres", "alquileres"),
            ("Tenencia", "tenencia"),
            ("CN", "cn"),
            ("Movimientos CV", "transactions_analyzer"),
        ],
        cols_per_row=3,
        key_prefix="comercial"
    )


# =========================
# OPERACIONES
# =========================
with tabs[1]:
    st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Acceso al entorno externo de Backoffice</div>", unsafe_allow_html=True)

    render_external_cards(
        [
            ("Abrir Backoffice", BACKOFFICE_URL),
        ],
        cols_per_row=3
    )


# =========================
# MESA
# =========================
with tabs[2]:
    st.markdown("<div class='section-title'>Mesa</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Bonos, ONs y carteras</div>", unsafe_allow_html=True)

    render_internal_cards(
        [
            ("Bonos", "bonos"),
            ("Obligaciones Negociables", "ons"),
            ("Carteras (rendimiento)", "cartera"),
            ("Carteras (ARG)", "cartera2"),
        ],
        cols_per_row=3,
        key_prefix="mesa"
    )


# =========================
# PERFORMANCE · BI
# =========================
with tabs[3]:
    st.markdown("<div class='section-title'>Performance · BI</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Dashboards de performance y seguimiento por área</div>", unsafe_allow_html=True)

    render_external_cards(
        [
            ("Banca Privada", BI_BANCA_PRIVADA),
            ("Banca Corporativa", BI_BANCA_CORP),
            ("Middle Office", BI_MIDDLE),
        ],
        cols_per_row=3
    )


# =========================
# MARKETING
# =========================
with tabs[4]:
    st.markdown("<div class='section-title'>Marketing</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Acceso a carpetas e instructivos compartidos</div>", unsafe_allow_html=True)

    render_internal_cards(
        [
            ("Instructivos", "mkt_instructivos"),
            ("Materiales de Marketing", "mkt_materiales"),
            ("Presentaciones", "mkt_presentaciones"),
        ],
        cols_per_row=3,
        key_prefix="marketing"
    )
