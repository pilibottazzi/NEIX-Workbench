import streamlit as st

from tools.mesa import cartera, ons, vencimientos, bonos, cartera2
from tools.comerciales import (
    cauciones_mae,
    cauciones_byma,
    alquileres,
    cn,
    filtro_especies_exterior,
    cartera_propia,
)

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
    layout="wide",
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
    :root{
        --bg: #ffffff;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --line: rgba(15,23,42,0.08);
        --line-strong: rgba(15,23,42,0.11);
        --red: #ef4444;
        --shadow-sm: 0 6px 22px rgba(15,23,42,.035);
        --shadow-md: 0 10px 30px rgba(15,23,42,.055);
        --shadow-hover:
            0 16px 38px rgba(15,23,42,.07),
            0 10px 26px rgba(239,68,68,.10);
        --radius: 18px;
    }

    /* ===== Layout general ===== */
    .stApp{
        background: #ffffff;
    }

    .block-container{
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    header[data-testid="stHeader"]{
        visibility: hidden;
        height: 0rem;
    }

    div[data-testid="stToolbar"]{
        display: none !important;
    }

    #MainMenu{
        visibility: hidden;
    }

    footer{
        visibility: hidden;
    }

    /* ===== Header general ===== */
    .neix-wrap{
        text-align:center;
        margin-top: .2rem;
        margin-bottom: 1.25rem;
    }

    .neix-title{
        font-weight: 850;
        letter-spacing: .16em;
        font-size: 1.62rem;
        color: var(--text);
        margin-bottom: .18rem;
        line-height: 1.15;
    }

    .neix-caption{
        color: var(--muted);
        font-size: .93rem;
        margin-bottom: .65rem;
    }

    .neix-accent{
        width: 64px;
        height: 2px;
        background: linear-gradient(90deg, rgba(239,68,68,.95), rgba(239,68,68,.30));
        border-radius: 999px;
        margin: 0 auto;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"]{
        gap: 10px;
        border-bottom: 1px solid var(--line);
        padding-left: 0;
        margin-top: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .stTabs [data-baseweb="tab"]{
        background: transparent;
        border: none;
        color: var(--muted);
        font-weight: 700;
        font-size: .95rem;
        padding: 10px 16px 12px 16px;
    }

    .stTabs [data-baseweb="tab"]:hover{
        color: var(--text);
    }

    .stTabs [aria-selected="true"]{
        color: var(--text);
        border-bottom: 3px solid var(--red);
    }

    /* ===== Titles ===== */
    .section-title{
        font-size:1.28rem;
        font-weight:800;
        margin-top:6px;
        margin-bottom:4px;
        color:#111827;
        letter-spacing:-0.01em;
    }

    .section-sub{
        color:#6b7280;
        font-size:.94rem;
        margin-bottom:16px;
    }

    /* ===== Login ===== */
    .login-page-wrap{
        text-align:center;
        margin-top: .2rem;
        margin-bottom: .55rem;
    }

    .login-head-title{
        font-weight: 850;
        letter-spacing: .16em;
        font-size: 1.62rem;
        color: var(--text);
        margin-bottom: .18rem;
        line-height: 1.15;
    }

    .login-head-sub{
        color: var(--muted);
        font-size: .93rem;
        margin-bottom: .65rem;
    }

    .login-head-line{
        width: 64px;
        height: 2px;
        background: linear-gradient(90deg, rgba(239,68,68,.95), rgba(239,68,68,.30));
        border-radius: 999px;
        margin: 0 auto;
    }

    .login-wrap{
        max-width: 420px;
        margin: 86px auto 0 auto;
        padding: 26px 24px 22px 24px;
        border: 1px solid rgba(15,23,42,0.10);
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(15,23,42,.04);
        text-align: center;
    }

    .login-footer{
        text-align:center;
        color:#94a3b8;
        font-size:.82rem;
        margin-top:12px;
    }

    /* ===== Botones ===== */
    div[data-testid="stButton"] > button{
        width:100%;
        border-radius:14px !important;
        font-weight:700 !important;
        min-height:46px !important;
        border:1px solid rgba(15,23,42,0.10) !important;
        background:#ffffff !important;
        color:#0f172a !important;
        box-shadow:0 2px 8px rgba(15,23,42,0.03) !important;
        transition:
            transform .16s ease,
            box-shadow .16s ease,
            border-color .16s ease !important;
    }

    div[data-testid="stButton"] > button:hover{
        border-color:rgba(239,68,68,.35) !important;
        box-shadow:
            0 10px 24px rgba(15,23,42,.05),
            0 6px 18px rgba(239,68,68,.08) !important;
    }

    div[data-testid="stLinkButton"] a{
        width:100%;
        border-radius:14px !important;
        font-weight:700 !important;
        min-height:46px !important;
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
    }

    /* ===== Inputs ===== */
    div[data-testid="stTextInput"] input{
        border-radius: 14px !important;
    }

    /* ===== Botón principal rojo ===== */
    .primary-link{
        display:flex;
        align-items:center;
        justify-content:center;
        width:100%;
        min-height:48px;
        border-radius:14px;
        background:#ef4444;
        color:white !important;
        text-decoration:none !important;
        font-weight:800;
        box-shadow:0 8px 22px rgba(239,68,68,.18);
    }

    .primary-link:hover{
        filter:brightness(.97);
    }

    /* ===== Espaciados ===== */
    .top-gap{
        height:4px;
    }

    @media (max-width: 900px){
        .neix-title,
        .login-head-title{
            font-size: 1.42rem;
            letter-spacing: .14em;
        }

        .neix-caption,
        .login-head-sub{
            font-size: .9rem;
        }

        .login-wrap{
            margin-top: 64px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def _header():
    st.markdown(
        """
        <div class="neix-wrap">
            <div class="neix-title">N E I X&nbsp;&nbsp;Workbench</div>
            <div class="neix-caption">Navegación por áreas y proyectos</div>
            <div class="neix-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_tool(tool_name: str):
    st.query_params["tool"] = tool_name


def clear_tool():
    st.query_params.clear()


def render_internal_cards(items, cols_per_row=3, key_prefix="nav"):
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")

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
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")

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

    st.markdown(
        """
        <div class="login-page-wrap">
            <div class="login-head-title">N E I X&nbsp;&nbsp;Workbench</div>
            <div class="login-head-sub">Ingreso restringido</div>
            <div class="login-head-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2, c3 = st.columns([1.15, 1, 1.15])

    with c2:
        st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)

        password = st.text_input(
            "Clave",
            type="password",
            placeholder="Clave",
            label_visibility="collapsed",
            key="login_password",
        )

        if st.button("Ingresar", key="login_btn", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Clave incorrecta")

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

        elif tool == "filtro_especies_exterior":
            filtro_especies_exterior.render()
            st.stop()

        elif tool == "cartera_propia":
            cartera_propia.render()
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
            st.markdown(
                f'<a class="primary-link" href="{BACKOFFICE_URL}" target="_blank">Abrir Backoffice</a>',
                unsafe_allow_html=True,
            )
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
            ("Filtro Especies Exterior", "filtro_especies_exterior"),
            ("Cartera Propia", "cartera_propia"),
        ],
        cols_per_row=3,
        key_prefix="comercial",
    )

# =========================
# OPERACIONES
# =========================
with tabs[1]:
    st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Acceso al entorno externo de Backoffice</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="max-width:420px;">
            <a class="primary-link" href="{BACKOFFICE_URL}" target="_blank">Abrir Backoffice</a>
        </div>
        """,
        unsafe_allow_html=True,
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
        key_prefix="mesa",
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
        cols_per_row=3,
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
        key_prefix="marketing",
    )
