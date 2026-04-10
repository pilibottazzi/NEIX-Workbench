# tools/cauciones_byma.py
import logging
import os

import pandas as pd
import streamlit as st

from tools._ui import inject_tool_css
from tools._parsers import parse_float

logger = logging.getLogger(__name__)


# =========================
# Config
# =========================
DATA_DIR = "data"
DATA_FILENAME_CANDIDATES = [
    "Garantia Byma.xlsx",
    "Garantia BYMA.xlsx",
    "GARANTIA BYMA.xlsx",
    "Garantía Byma.xlsx",
    "Garantía BYMA.xlsx",
]
REQUIRED_COLS = ["ESPECIE", "AFORO", "MARGEN", "MÁXIMO POR ESPECIE", "LISTA"]


# =========================
# Helpers
# =========================
def _fmt_ars(x) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "-"


def _fmt_pct(x) -> str:
    try:
        return f"{int(round(float(x) * 100))}%"
    except Exception:
        return "-"


def _normalize_colname(c: str) -> str:
    return str(c).strip().upper()


def _strip_accents(s: str) -> str:
    # Normalización simple para matchear con/sin tildes
    return (
        s.replace("Ó", "O").replace("Á", "A").replace("É", "E")
         .replace("Í", "I").replace("Ú", "U")
         .replace("Ü", "U").replace("Ñ", "N")
    )


def _find_col(df_cols, target: str):
    target_n = _normalize_colname(target)

    # match exacto
    for c in df_cols:
        if _normalize_colname(c) == target_n:
            return c

    # match “sin tildes”
    base = _strip_accents(target_n)
    for c in df_cols:
        cn = _strip_accents(_normalize_colname(c))
        if cn == base:
            return c

    return None


def _resolve_data_path() -> str:
    """
    En Cloud (Linux) el nombre del archivo es case-sensitive.
    Esto busca el Excel real dentro de /data con varios nombres comunes.
    """
    # 1) candidatos directos
    for name in DATA_FILENAME_CANDIDATES:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p

    # 2) fallback: buscar por contains "garantia" y "byma" ignorando mayúsc/minúsc
    if os.path.isdir(DATA_DIR):
        for fn in os.listdir(DATA_DIR):
            low = fn.lower()
            if "garantia" in low and "byma" in low and low.endswith(".xlsx"):
                return os.path.join(DATA_DIR, fn)

    # si no encuentra, devolvemos el “default” (para que el error sea claro)
    return os.path.join(DATA_DIR, DATA_FILENAME_CANDIDATES[0])


def _is_divide_by_100(tipo_activo: str) -> bool:
    """
    Regla BYMA:
    - Divide por 100 si es: TÍTULO/OBLIGACIÓN/BONO/LETRA/LECAP
    - NO divide si es CEDEAR o ACCIONES
    """
    t = (tipo_activo or "").upper().strip()

    if ("CEDEAR" in t) or ("ACCIÓN" in t) or ("ACCIONES" in t) or ("ACCION" in t):
        return False

    activos_base_100 = ["TÍTULO", "TITULO", "OBLIGACIÓN", "OBLIGACION", "BONO", "LETRA", "LECAP"]
    return any(a in t for a in activos_base_100)


@st.cache_data(show_spinner=False)
def cargar_aforos_byma() -> pd.DataFrame:
    data_path = _resolve_data_path()

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No existe el archivo '{data_path}'.\n"
            f"En Streamlit Cloud el nombre es case-sensitive.\n"
            f"Revisá que el Excel esté en '{DATA_DIR}/' y se llame como corresponde."
        )

    df = pd.read_excel(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    # detectar columnas robusto (con/sin tildes)
    col_especie = _find_col(df.columns, "ESPECIE")
    col_aforo = _find_col(df.columns, "AFORO")
    col_margen = _find_col(df.columns, "MARGEN")
    col_max = _find_col(df.columns, "MÁXIMO POR ESPECIE")
    col_lista = _find_col(df.columns, "LISTA")

    missing = []
    if col_especie is None: missing.append("ESPECIE")
    if col_aforo is None: missing.append("AFORO")
    if col_margen is None: missing.append("MARGEN")
    if col_max is None: missing.append("MÁXIMO POR ESPECIE")
    if col_lista is None: missing.append("LISTA")

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas {missing}. Columnas disponibles: {list(df.columns)}"
        )

    df = df.rename(columns={
        col_especie: "ESPECIE",
        col_aforo: "AFORO",
        col_margen: "MARGEN",
        col_max: "MÁXIMO POR ESPECIE",
        col_lista: "LISTA",
    })

    df["ESPECIE"] = df["ESPECIE"].astype(str).str.upper().str.strip()
    df["AFORO"] = pd.to_numeric(df["AFORO"], errors="coerce")
    df["MARGEN"] = pd.to_numeric(df["MARGEN"], errors="coerce")
    df["MÁXIMO POR ESPECIE"] = pd.to_numeric(df["MÁXIMO POR ESPECIE"], errors="coerce")
    df["LISTA"] = df["LISTA"].astype(str)

    df = df.dropna(subset=["ESPECIE", "AFORO"])
    return df


# =========================
# Main render
# =========================
def render():
    inject_tool_css()
    st.caption("Calculá garantía admitida por especie según aforos BYMA (Excel pre-cargado en el repo).")

    try:
        df_aforos = cargar_aforos_byma()
    except Exception as e:
        logger.exception("No pude cargar el Excel de aforos BYMA")
        st.error(f"No pude cargar el Excel de aforos BYMA: {e}")
        st.stop()

    if "byma_operaciones" not in st.session_state:
        st.session_state.byma_operaciones = []

    metodo = st.radio(
        "¿Cómo querés ingresar el valor?",
        ["Por monto", "Por precio y nominales"],
        horizontal=True,
        key="byma_metodo"
    )

    with st.form("form_byma_operacion", clear_on_submit=True):
        especies = [""] + sorted(df_aforos["ESPECIE"].dropna().unique().tolist())
        especie = st.selectbox("Seleccioná la especie", options=especies, index=0, key="byma_especie")

        monto = None
        tipo_activo = ""
        dividir_por_100 = True

        if especie:
            row = df_aforos[df_aforos["ESPECIE"] == especie]
            if not row.empty:
                tipo_activo = str(row.iloc[0]["LISTA"])
                dividir_por_100 = _is_divide_by_100(tipo_activo)

        if metodo == "Por monto":
            monto_txt = st.text_input("Monto (AR$)", placeholder="Ej: 1.000.000", key="byma_monto")
            monto = parse_float(monto_txt)
        else:
            c1, c2 = st.columns(2)
            precio_txt = c1.text_input("Precio", placeholder="Ej: 68,75", key="byma_precio")
            nominales_txt = c2.text_input("Nominales", placeholder="Ej: 100.000", key="byma_nominales")

            precio = parse_float(precio_txt)
            nominales = parse_float(nominales_txt)

            if precio is not None and nominales is not None:
                monto = precio * nominales
                if dividir_por_100:
                    monto = monto / 100.0

        if especie:
            st.caption(
                f"Tipo: **{tipo_activo or '-'}** · "
                f"{'Se divide' if dividir_por_100 else 'No se divide'} por 100 para valor de mercado (regla BYMA)."
            )

        submitted = st.form_submit_button("Agregar")

        if submitted:
            if not especie:
                st.warning("Seleccioná una especie válida.")
            elif monto is None or monto <= 0:
                st.warning("Ingresá valores válidos.")
            else:
                row = df_aforos[df_aforos["ESPECIE"] == especie]
                if row.empty:
                    st.warning("La especie seleccionada no se encontró en el Excel.")
                else:
                    datos = row.iloc[0]
                    aforo = float(datos["AFORO"])
                    margen = float(datos["MARGEN"]) if pd.notna(datos["MARGEN"]) else None
                    maximo = float(datos["MÁXIMO POR ESPECIE"]) if pd.notna(datos["MÁXIMO POR ESPECIE"]) else None
                    garantia = float(monto) * aforo

                    st.session_state.byma_operaciones.append({
                        "Especie": especie,
                        "Tipo de activo": str(datos["LISTA"]),
                        "Método": metodo,
                        "Monto": float(monto),
                        "Aforo": aforo,
                        "Margen": margen,
                        "Máximo permitido": maximo,
                        "Garantía admitida": float(garantia),
                    })

    st.divider()

    ops = st.session_state.byma_operaciones
    if ops:
        st.subheader("Resultado del cálculo")
        df_res = pd.DataFrame(ops)

        show = df_res.copy()
        show["Monto"] = show["Monto"].map(_fmt_ars)
        show["Aforo"] = show["Aforo"].map(_fmt_pct)
        if "Margen" in show.columns:
            show["Margen"] = show["Margen"].map(_fmt_pct)
        if "Máximo permitido" in show.columns:
            show["Máximo permitido"] = show["Máximo permitido"].map(_fmt_ars)
        show["Garantía admitida"] = show["Garantía admitida"].map(_fmt_ars)

        st.dataframe(show, use_container_width=True, hide_index=True)

        total = df_res["Garantía admitida"].sum()
        st.markdown(f"### Garantía total admitida: **AR$ {_fmt_ars(total)}**")

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar operaciones (CSV)",
            csv,
            file_name="garantias_byma_operaciones.csv",
            mime="text/csv",
            key="byma_download"
        )
    else:
        st.info("Todavía no agregaste operaciones.")

    if st.button("Reiniciar cálculo", key="byma_reset"):
        st.session_state.byma_operaciones = []
        st.rerun()

