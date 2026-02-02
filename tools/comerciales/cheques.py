# tools/cheques.py
import os
import re
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
MANAGERS_PATH = os.path.join("data", "managers_neix.xlsx")


# =========================
# Helpers
# =========================
def limpiar_monto(x):
    if pd.isna(x):
        return pd.NA
    s = str(x)
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA


def _fmt_money(x):
    try:
        return f"${float(x):,.0f}".replace(",", ".")
    except Exception:
        return ""


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    """
    Espera un Excel en: data/managers_neix.xlsx
    Columnas mínimas esperadas (idealmente):
      NumeroComitente, Comitente (o Cliente), Manager, Oficial
    (Si además trae NumeroManager/NumeroOficial, no molesta)
    """
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(
            f"No existe '{MANAGERS_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # Normalizaciones de nombres por si tu excel usa Cliente en vez de Comitente
    if "Comitente" not in df.columns and "Cliente" in df.columns:
        df = df.rename(columns={"Cliente": "Comitente"})

    required = ["NumeroComitente"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"El archivo managers_neix.xlsx no tiene columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df["NumeroComitente"] = (
        pd.to_numeric(df["NumeroComitente"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.strip()
    )

    # Asegurar que existan estas columnas (aunque sea vacías)
    for col in ["Comitente", "Manager", "Oficial"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _require_password_gate():
    expected = st.secrets.get("app_password", "ciclon")
    with st.expander("🔐 Sección protegida por clave"):
        pwd = st.text_input("Ingresá la clave para acceder:", type="password", key="cheques_pwd")
        if pwd != expected:
            st.warning("⚠️ Acceso restringido. Ingresá la clave correcta para ver el contenido.")
            st.stop()


# =========================
# Render
# =========================
def render(back_to_home=None):
    st.markdown("## Dashboard cheques y pagarés")
    st.caption("Carga obligatoria de Excel + cruce con Managers (Excel: data/managers_neix.xlsx)")

    _require_password_gate()

    st.subheader("📎 Carga de archivo (obligatoria)")
    archivo = st.file_uploader(
        "Subí el Excel de cheques / pagarés para continuar",
        type=["xlsx"],
        accept_multiple_files=False,
        key="cheques_uploader",
    )
    if archivo is None:
        st.info("Debés subir un archivo Excel para habilitar el dashboard.")
        st.stop()

    if not st.button("Cargar datos", type="primary", key="cheques_cargar"):
        st.caption("Una vez subido el archivo, tocá **Cargar datos**.")
        st.stop()

    try:
        df = pd.read_excel(archivo, header=0)
    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        st.stop()

    df.columns = df.columns.astype(str).str.strip()

    COL_TIPO = "TIPO INSTRUMENTO"
    COL_ESTADO = "ESTADO"
    COL_MONTO = "MONTO"
    COL_MONEDA = "MONEDA"
    COL_FECHA_PAGO = "FECHA PAGO"
    COL_FECHA_COBRO = "FECHA COBRO"
    COL_FECHA_INGRESO = "FECHA INGRESO"
    COL_CHEQUE = "NRO.CHEQUE/PAGARE"
    COL_TENEDOR = "COMITENTE TENEDOR"
    COL_INGRESANTE = "COMITENTE INGRESANTE"

    required = [COL_TIPO, COL_ESTADO, COL_MONTO, COL_MONEDA, COL_FECHA_PAGO, COL_TENEDOR, COL_INGRESANTE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en el Excel: {missing}")
        st.stop()

    for c in [COL_FECHA_PAGO, COL_FECHA_COBRO, COL_FECHA_INGRESO]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    df[COL_MONTO] = df[COL_MONTO].apply(limpiar_monto)

    for c in [COL_TENEDOR, COL_INGRESANTE]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str).str.strip()

    # =========================
    # Cruce managers desde EXCEL
    # =========================
    try:
        with st.spinner("Cargando Managers desde managers_neix.xlsx…"):
            df_manager = cargar_managers_excel()
    except Exception as e:
        st.error("No pude cargar managers_neix.xlsx")
        st.exception(e)
        df_manager = pd.DataFrame()

    if not df_manager.empty:
        base_cols = ["NumeroComitente", "Comitente", "Manager", "Oficial"]
        base_cols = [c for c in base_cols if c in df_manager.columns]

        df = (
            df.merge(
                df_manager[base_cols],
                left_on=COL_TENEDOR,
                right_on="NumeroComitente",
                how="left",
            )
            .rename(
                columns={
                    "Comitente": "Nombre Comitente TENEDOR",
                    "Manager": "Manager TENEDOR",
                    "Oficial": "Oficial TENEDOR",
                }
            )
            .drop(columns=["NumeroComitente"], errors="ignore")
        )

        df = (
            df.merge(
                df_manager[base_cols],
                left_on=COL_INGRESANTE,
                right_on="NumeroComitente",
                how="left",
                suffixes=("", "_ING"),
            )
            .rename(
                columns={
                    "Comitente": "Nombre Comitente INGRESANTE",
                    "Manager": "Manager INGRESANTE",
                    "Oficial": "Oficial INGRESANTE",
                }
            )
            .drop(columns=["NumeroComitente"], errors="ignore")
        )

    # =========================
    # Vista
    # =========================
    st.subheader("📊 Vista general")

    show_cols = [c for c in [
        COL_TIPO, COL_MONEDA, COL_CHEQUE,
        "Nombre Comitente TENEDOR", COL_TENEDOR,
        "Manager TENEDOR", "Oficial TENEDOR",
        COL_MONTO, COL_FECHA_PAGO, COL_ESTADO,
        "Nombre Comitente INGRESANTE", COL_INGRESANTE,
        "Manager INGRESANTE", "Oficial INGRESANTE",
    ] if c in df.columns]

    out = df[show_cols].copy()
    if COL_FECHA_PAGO in out.columns:
        out[COL_FECHA_PAGO] = pd.to_datetime(out[COL_FECHA_PAGO], errors="coerce").dt.date
    out[COL_MONTO] = out[COL_MONTO].fillna(0).apply(_fmt_money)

    st.dataframe(out, use_container_width=True, hide_index=True)

    if st.button("Reiniciar", key="cheques_reset"):
        st.rerun()
