# tools/cheques.py
import re
from datetime import datetime, timedelta

import mysql.connector
import pandas as pd
import streamlit as st


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


def get_db_config() -> dict:
    try:
        return {
            "host": st.secrets["mysql"]["host"],
            "user": st.secrets["mysql"]["user"],
            "password": st.secrets["mysql"]["password"],
            "database": st.secrets["mysql"]["database"],
            "port": int(st.secrets["mysql"].get("port", 3306)),
            "allow_local_infile": True,
        }
    except Exception:
        st.error("Faltan secrets de MySQL. Configurá [mysql] en secrets (local o Streamlit Cloud).")
        st.stop()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def obtener_managers_cached() -> pd.DataFrame:
    db_config = get_db_config()
    cn = mysql.connector.connect(**db_config)
    cur = cn.cursor(dictionary=True)

    q = """
        WITH ultimo AS (
            SELECT t.*
            FROM neix.principalLog t
            JOIN (
                SELECT principal_id, MAX(created_at) AS max_created
                FROM neix.principalLog
                GROUP BY principal_id
            ) sub ON t.principal_id = sub.principal_id AND t.created_at = sub.max_created
        )
        SELECT 
            p.id as NumeroComitente, 
            p.description AS Comitente,
            m.id AS NumeroManager,
            m.description AS Manager,
            o.id AS NumeroOficial,
            o.description AS Oficial
        FROM ultimo u
        LEFT JOIN neix.principal as p ON u.principal_id = p.id
        LEFT JOIN neix.manager as m ON u.manager_id = m.id
        LEFT JOIN neix.officer as o ON u.officer_id = o.id
    """
    cur.execute(q)
    rows = cur.fetchall()
    cur.close()
    cn.close()

    df_manager = pd.DataFrame(rows)
    if df_manager.empty:
        return df_manager

    df_manager.columns = df_manager.columns.astype(str).str.strip()
    df_manager["NumeroComitente"] = (
        pd.to_numeric(df_manager["NumeroComitente"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.strip()
    )
    return df_manager


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
    # IMPORTANTE: NO LLAMAR back_to_home() ACÁ.
    # El botón volver ya lo renderiza app.py

    st.markdown("## Dashboard cheques y pagarés")
    st.caption("Carga obligatoria de Excel + cruce con Managers (MySQL)")

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

    # Cruce managers
    with st.spinner("Cargando Managers desde MySQL…"):
        df_manager = obtener_managers_cached()

    if not df_manager.empty:
        df = (
            df.merge(
                df_manager[["NumeroComitente", "Comitente", "NumeroManager", "Manager", "NumeroOficial", "Oficial"]],
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
            .drop(columns=["NumeroComitente", "NumeroManager", "NumeroOficial"], errors="ignore")
        )

        df = (
            df.merge(
                df_manager[["NumeroComitente", "Comitente", "NumeroManager", "Manager", "NumeroOficial", "Oficial"]],
                left_on=COL_INGRESANTE,
                right_on="NumeroComitente",
                how="left",
            )
            .rename(
                columns={
                    "Comitente": "Nombre Comitente INGRESANTE",
                    "Manager": "Manager INGRESANTE",
                    "Oficial": "Oficial INGRESANTE",
                }
            )
            .drop(columns=["NumeroComitente", "NumeroManager", "NumeroOficial"], errors="ignore")
        )

    st.subheader("📊 Vista general")
    show_cols = [c for c in [
        COL_TIPO, COL_MONEDA, COL_CHEQUE,
        "Nombre Comitente TENEDOR", COL_TENEDOR,
        COL_MONTO, COL_FECHA_PAGO, COL_ESTADO,
        "Nombre Comitente INGRESANTE", COL_INGRESANTE,
    ] if c in df.columns]

    out = df[show_cols].copy()
    if COL_FECHA_PAGO in out.columns:
        out[COL_FECHA_PAGO] = pd.to_datetime(out[COL_FECHA_PAGO], errors="coerce").dt.date
    out[COL_MONTO] = out[COL_MONTO].fillna(0).apply(_fmt_money)

    st.dataframe(out, use_container_width=True, hide_index=True)

    if st.button("Reiniciar"):
        st.rerun()

