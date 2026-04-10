import hmac
import logging

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

logger = logging.getLogger(__name__)

# =========================================================
# URLS externas
# =========================================================
BACKOFFICE_URL = "https://neix-workbench-bo.streamlit.app/"
BI_BANCA_PRIVADA = "https://lookerstudio.google.com/reporting/75c2a6d0-0086-491f-b112-88fe3d257ef9"
BI_BANCA_CORP = "https://lookerstudio.google.com/reporting/4f70efa8-2b86-4134-a9cb-9e6f90117f3b"
BI_MIDDLE = "https://lookerstudio.google.com/reporting/5b834e5f-aeef-4042-ac0f-e1ed3564a010"

# SharePoint Marketing
SP_MKT_INSTRUCTIVOS = "https://neixcom.sharepoint.com/sites/NEIXSOCIEDADDEBOLSAS.A-Marketingprueba/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FNEIXSOCIEDADDEBOLSAS%2EA%2DMarketingprueba%2FShared%20Documents%2FMarketing%2FInstructivos&viewid=74e4d9a3%2Dd2c9%2D4f09%2D9bc8%2Deb59e613117f&p=true"
SP_MKT_MATERIALES = "https://neixcom.sharepoint.com/sites/NEIXSOCIEDADDEBOLSAS.A-Marketingprueba/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FNEIXSOCIEDADDEBOLSAS%2EA%2DMarketingprueba%2FShared%20Documents%2FMarketing%2FMateriales%20de%20Marketing&viewid=74e4d9a3%2Dd2c9%2D4f09%2D9bc8%2Deb59e613117f&p=true"
SP_MKT_PRESENTACIONES = "https://neixcom.sharepoint.com/sites/NEIXSOCIEDADDEBOLSAS.A-Marketingprueba/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FNEIXSOCIEDADDEBOLSAS%2EA%2DMarketingprueba%2FShared%20Documents%2FMarketing%2FPresentaciones&viewid=74e4d9a3%2Dd2c9%2D4f09%2D9bc8%2Deb59e613117f&p=true"

# =========================================================
# ENV / CONSTANTS
# =========================================================
APP_ENV = st.secrets.get("app_env", "prod").strip().lower()
IS_DEV = APP_ENV == "dev"

APP_TITLE = "NEIX Workbench (DEV)" if IS_DEV else "NEIX Workbench"
HEADER_TITLE = "N E I X  Workbench (DEV)" if IS_DEV else "N E I X  Workbench"
ACCENT_COLOR = "#1e40af" if IS_DEV else "#b91c1c"
ACCENT_COLOR_SOFT = "rgba(30,64,175,.25)" if IS_DEV else "rgba(185,28,28,.25)"
ACCENT_COLOR_BORDER = "rgba(30,64,175,.35)" if IS_DEV else "rgba(185,28,28,.35)"
ACCENT_COLOR_BORDER_SOFT = "rgba(30,64,175,.28)" if IS_DEV else "rgba(185,28,28,.28)"
ACCENT_COLOR_SHADOW = "rgba(30,64,175,.10)" if IS_DEV else "rgba(185,28,28,.10)"
ACCENT_COLOR_SHADOW_SOFT = "rgba(30,64,175,.07)" if IS_DEV else "rgba(185,28,28,.07)"
APP_BG = "#f0f4ff" if IS_DEV else "#faf8f5"
CARD_BG = "#f8fbff" if IS_DEV else "#ffffff"
CARD_BORDER = "rgba(30,64,175,.15)" if IS_DEV else "rgba(15,23,42,0.07)"

APP_PASSWORD = st.secrets.get("app_password")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧰",
    layout="wide",
)

# =========================================================
# STYLES
# =========================================================
st.markdown(
    f"""
<style>
:root {{
    --text: #0f172a;
    --muted: #64748b;
    --line: rgba(15,23,42,0.08);
    --line-strong: rgba(15,23,42,0.11);
    --accent: {ACCENT_COLOR};
    --accent-soft: {ACCENT_COLOR_SOFT};
    --accent-border: {ACCENT_COLOR_BORDER};
    --accent-border-soft: {ACCENT_COLOR_BORDER_SOFT};
    --accent-shadow: {ACCENT_COLOR_SHADOW};
    --accent-shadow-soft: {ACCENT_COLOR_SHADOW_SOFT};
    --app-bg: {APP_BG};
    --card-bg: {CARD_BG};
    --card-border: {CARD_BORDER};
    --shadow-sm: 0 4px 16px rgba(15,23,42,.04);
    --shadow-hover: 0 8px 24px rgba(15,23,42,.08), 0 4px 12px var(--accent-shadow);
}}
.stApp {{ background: var(--app-bg); }}
.block-container {{ max-width: 1320px; padding-top: 1.2rem; padding-bottom: 2rem; }}
header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
div[data-testid="stToolbar"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* ---------- Header ---------- */
.neix-wrap {{ text-align: center; margin-top: .2rem; margin-bottom: 1.1rem; }}
.neix-title {{ font-weight: 700; letter-spacing: .18em; font-size: 1.5rem; color: var(--text); margin-bottom: .15rem; }}
.neix-caption {{ color: var(--muted); font-size: .88rem; margin-bottom: .55rem; }}
.neix-accent {{ width: 48px; height: 2px; background: var(--accent); border-radius: 999px; margin: 0 auto; opacity: .7; }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid var(--line); padding-left: 0; margin-top: .3rem; margin-bottom: .3rem; }}
.stTabs [data-baseweb="tab"] {{ background: transparent; border: none; color: var(--muted); font-weight: 600; font-size: .9rem; padding: 8px 14px 10px; transition: color .12s ease; }}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
.stTabs [aria-selected="true"] {{ color: var(--text); font-weight: 700; border-bottom: 2px solid var(--accent); }}

/* ---------- Section titles ---------- */
.section-title {{ font-size: 1.1rem; font-weight: 700; margin-top: 6px; margin-bottom: 4px; color: var(--text); letter-spacing: -0.01em; }}
.section-sub {{ color: var(--muted); font-size: .88rem; margin-top: .5rem; margin-bottom: 1rem; }}

/* ---------- Default buttons (back button, etc.) ---------- */
div[data-testid="stButton"] > button {{
    border-radius: 10px !important; font-weight: 600 !important;
    min-height: 38px !important; font-size: .88rem !important;
    background: var(--card-bg) !important; color: var(--text) !important;
    border: 1px solid var(--card-border) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all .14s ease !important;
}}
div[data-testid="stButton"] > button:hover {{
    border-color: var(--accent-border-soft) !important;
    box-shadow: var(--shadow-hover) !important;
}}

/* ---------- Home cards (buttons inside tabs) ---------- */
.stTabs div[data-testid="stButton"] > button {{
    min-height: 72px !important;
    font-weight: 700 !important;
    font-size: .92rem !important;
    border-radius: 12px !important;
}}
.stTabs div[data-testid="stButton"] > button:hover {{
    transform: translateY(-2px) !important;
}}

/* ---------- Link buttons (external, inside tabs) ---------- */
.stTabs div[data-testid="stLinkButton"] a {{
    width: 100%;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 72px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--card-border) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all .14s ease !important;
    text-decoration: none !important;
}}
.stTabs div[data-testid="stLinkButton"] a:hover {{
    transform: translateY(-2px) !important;
    border-color: var(--accent-border-soft) !important;
    box-shadow: var(--shadow-hover) !important;
}}

/* ---------- Inputs ---------- */
div[data-testid="stTextInput"] input {{ border-radius: 10px !important; }}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {{ border-radius: 10px; }}
hr {{ border: none; border-top: 1px solid var(--line); margin: .2rem 0 .8rem; }}

@media (max-width: 900px) {{ .neix-title {{ font-size: 1.32rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def _header(caption: str = "Navegación por áreas y proyectos"):
    cap = f"<div class='neix-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div class="neix-wrap">
            <div class="neix-title">{HEADER_TITLE.replace(" ", "&nbsp;")}</div>
            {cap}
            <div class="neix-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_tool(tool_name: str) -> None:
    st.query_params["tool"] = tool_name


def clear_tool() -> None:
    st.query_params.clear()


def render_internal_cards(items, cols_per_row: int = 3, key_prefix: str = "nav") -> None:
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


def render_external_cards(items, cols_per_row: int = 3) -> None:
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


def _run_tool(module, label: str) -> None:
    """Call render() on a tool module with logging."""
    logger.info("Ejecutando herramienta '%s' (módulo: %s)", label, module.__name__)
    try:
        module.render()
        logger.info("Herramienta '%s' ejecutada correctamente", label)
    except Exception as e:
        logger.exception("Error ejecutando herramienta '%s'", label)
        st.error(f"Error cargando {label}: {e}")


# =========================================================
# LOGIN
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def check_password() -> bool:
    if st.session_state.logged_in:
        return True

    if not APP_PASSWORD:
        st.error("No se encontró 'app_password' en st.secrets.")
        st.stop()

    _header("Ingreso restringido")

    _, col, _ = st.columns([1.5, 1, 1.5])
    with col:
        password = st.text_input(
            "Clave",
            type="password",
            placeholder="Clave",
            label_visibility="collapsed",
            key="login_password",
        )
        if st.button("Ingresar", key="login_btn", use_container_width=True):
            if hmac.compare_digest(password, APP_PASSWORD):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Clave incorrecta")

    return False


if not check_password():
    st.stop()

# =========================================================
# ROUTER (?tool=...)
# =========================================================
tool = (st.query_params.get("tool") or "").lower().strip()

# Map de tools: slug -> (module, label)
TOOLS = {
    "bonos": (bonos, "Bonos"),
    "ons": (ons, "Obligaciones Negociables"),
    "cartera": (cartera, "Carteras (rendimiento)"),
    "cartera2": (cartera2, "Carteras (ARG)"),
    "tenencia": (vencimientos, "Tenencia"),
    "tenencias": (vencimientos, "Tenencia"),
    "vencimientos": (vencimientos, "Vencimientos"),
    "cauciones_mae": (cauciones_mae, "Cauciones MAE"),
    "cauciones_byma": (cauciones_byma, "Cauciones BYMA"),
    "alquileres": (alquileres, "Alquileres"),
    "cn": (cn, "CN"),
    "filtro_especies_exterior": (filtro_especies_exterior, "Filtro Especies Exterior"),
    "cartera_propia": (cartera_propia, "Cartera Propia"),
}

MARKETING_TOOLS = {
    "mkt_instructivos": ("Marketing · Instructivos", "Carpeta compartida en SharePoint", "Abrir Instructivos", SP_MKT_INSTRUCTIVOS),
    "mkt_materiales": ("Marketing · Materiales", "Carpeta compartida en SharePoint", "Abrir Materiales de Marketing", SP_MKT_MATERIALES),
    "mkt_presentaciones": ("Marketing · Presentaciones", "Carpeta compartida en SharePoint", "Abrir Presentaciones", SP_MKT_PRESENTACIONES),
}

if tool:
    _header()

    col_back, _ = st.columns([2, 10])
    with col_back:
        if st.button("← Volver", key="volver_home"):
            clear_tool()
            st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    if tool in TOOLS:
        module, label = TOOLS[tool]
        _run_tool(module, label)
        st.stop()

    if tool in MARKETING_TOOLS:
        title, subtitle, btn_label, url = MARKETING_TOOLS[tool]
        st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)
        st.link_button(btn_label, url)
        st.stop()

    if tool in ("operaciones", "backoffice"):
        st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Backoffice se abre en una web externa</div>", unsafe_allow_html=True)
        st.link_button("Abrir Backoffice", BACKOFFICE_URL)
        st.stop()

    st.error("Herramienta no encontrada")
    st.stop()

# =========================================================
# HOME
# =========================================================
_header()

tabs = st.tabs(["Comercial", "Operaciones", "Mesa", "Performance · BI", "Marketing"])

# ---------- Comercial ----------
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

# ---------- Operaciones ----------
with tabs[1]:
    st.markdown("<div class='section-title'>Operaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Acceso al entorno externo de Backoffice</div>", unsafe_allow_html=True)
    render_external_cards(
        [
            ("Abrir Backoffice", BACKOFFICE_URL),
        ],
        cols_per_row=3,
    )

# ---------- Mesa ----------
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

# ---------- Performance · BI ----------
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

# ---------- Marketing ----------
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
