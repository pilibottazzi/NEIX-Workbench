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
    """Normaliza montos con símbolos y separadores (AR/US). Devuelve float o NA."""
    if pd.isna(x):
        return pd.NA
    s = str(x)
    s = re.sub(r"[^\d,.\-]", "", s)  # deja solo dígitos/coma/punto/signo
    if "," in s and "." in s:
        # asume '.' miles y ',' decimal
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        # solo coma: coma decimal
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
    """
    Lee la config de MySQL desde st.secrets.
    Espera:
      [mysql]
      host, user, password, database, (opcional port)
    """
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
        st.error("Faltan secrets de MySQL. Configurá st.secrets['mysql'] en .streamlit/secrets.toml o en Cloud.")
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
    """Gate simple por clave usando st.secrets (con fallback)."""
    expected = st.secrets.get("app_password", "ciclon")
    with st.expander("🔐 Sección protegida por clave"):
        pwd = st.text_input("Ingresá la clave para acceder:", type="password")
        if pwd != expected:
            st.warning("⚠️ Acceso restringido. Ingresá la clave correcta para ver el contenido.")
            st.stop()


# =========================
# Render (Workbench)
# =========================
def render(back_to_home=None):
    # Botón volver (si te lo pasan desde app.py)
    if callable(back_to_home):
        back_to_home()

    st.markdown("## Cheques y Pagarés")
    st.caption("Carga obligatoria de Excel + cruce con Managers (MySQL)")

    # Gate
    _require_password_gate()

    # ========= Carga obligatoria =========
    st.subheader("📎 Carga de archivo (obligatoria)")
    archivo = st.file_uploader(
        "Subí el Excel de cheques / pagarés para continuar",
        type=["xlsx"],
        accept_multiple_files=False,
    )

    if archivo is None:
        st.info("Debés subir un archivo Excel para habilitar el dashboard.")
        st.stop()

    btn = st.button("Cargar datos", type="primary")
    if not btn:
        st.caption("Una vez subido el archivo, tocá **Cargar datos**.")
        st.stop()

    # ========= Leer Excel =========
    try:
        df = pd.read_excel(archivo, header=0)
    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        st.stop()

    df.columns = df.columns.astype(str).str.strip()

    # ========= Columnas esperadas =========
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

    # ========= Limpieza =========
    for c in [COL_FECHA_PAGO, COL_FECHA_COBRO, COL_FECHA_INGRESO]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    df[COL_MONTO] = df[COL_MONTO].apply(limpiar_monto)

    for c in [COL_TENEDOR, COL_INGRESANTE]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str).str.strip()

    df["Mes Pago"] = df[COL_FECHA_PAGO].dt.month

    # ========= Cruce Managers (MySQL) =========
    with st.spinner("Cargando Managers desde MySQL…"):
        df_manager = obtener_managers_cached()

    if not df_manager.empty:
        # TENEDOR
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
                    "NumeroManager": "NumeroManager TENEDOR",
                    "Manager": "Manager TENEDOR",
                    "NumeroOficial": "NumeroOficial TENEDOR",
                    "Oficial": "Oficial TENEDOR",
                }
            )
            .drop(columns=["NumeroComitente"], errors="ignore")
        )

        # INGRESANTE
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
                    "NumeroManager": "NumeroManager INGRESANTE",
                    "Manager": "Manager INGRESANTE",
                    "NumeroOficial": "NumeroOficial INGRESANTE",
                    "Oficial": "Oficial INGRESANTE",
                }
            )
            .drop(columns=["NumeroComitente"], errors="ignore")
        )
    else:
        st.warning("No se pudo traer la tabla de Managers desde MySQL (vacía). Se mostrará sin cruce.")

    # ========= Fechas base =========
    hoy = datetime.today().date()
    maniana = hoy + timedelta(days=1)
    lunes = hoy - timedelta(days=hoy.weekday())
    domingo = lunes + timedelta(days=6)
    inicio_mes = hoy.replace(day=1)
    fin_mes = (pd.Timestamp(inicio_mes) + pd.offsets.MonthEnd(1)).date()

    # ========= Vista por Fecha de Pago =========
    st.subheader("Vencimientos por **Fecha de PAGO**")
    opciones_fecha = ["Hoy", "Mañana", "Esta Semana", "Este Mes"]
    vista_pago = st.selectbox("Elegí la vista", opciones_fecha, key="vista_pago")

    if vista_pago == "Hoy":
        df_pago = df[df[COL_FECHA_PAGO].dt.date == hoy].copy()
    elif vista_pago == "Mañana":
        df_pago = df[df[COL_FECHA_PAGO].dt.date == maniana].copy()
    elif vista_pago == "Esta Semana":
        df_pago = df[(df[COL_FECHA_PAGO].dt.date >= lunes) & (df[COL_FECHA_PAGO].dt.date <= domingo)].copy()
    else:
        df_pago = df[(df[COL_FECHA_PAGO].dt.date >= inicio_mes) & (df[COL_FECHA_PAGO].dt.date <= fin_mes)].copy()

    # ========= Filtros =========
    col1, col2, col3 = st.columns(3)
    monedas = sorted(df[COL_MONEDA].dropna().astype(str).unique().tolist())
    estados = sorted(df[COL_ESTADO].dropna().astype(str).unique().tolist())
    instrumentos = sorted(df[COL_TIPO].dropna().astype(str).unique().tolist())

    with col1:
        filtro_monedas = st.multiselect("Moneda", ["Todos"] + monedas, default=["Todos"])
    with col2:
        filtro_estados = st.multiselect("Estado", ["Todos"] + estados, default=["Todos"])
    with col3:
        filtro_instrumento = st.multiselect("Tipo Instrumento", ["Todos"] + instrumentos, default=["Todos"])

    df_pago_f = df_pago.copy()
    if "Todos" not in filtro_monedas:
        df_pago_f = df_pago_f[df_pago_f[COL_MONEDA].astype(str).isin(filtro_monedas)]
    if "Todos" not in filtro_estados:
        df_pago_f = df_pago_f[df_pago_f[COL_ESTADO].astype(str).isin(filtro_estados)]
    if "Todos" not in filtro_instrumento:
        df_pago_f = df_pago_f[df_pago_f[COL_TIPO].astype(str).isin(filtro_instrumento)]

    # ========= Columnas a mostrar =========
    venc_cols = [
        c for c in [
            COL_FECHA_INGRESO, COL_TIPO, COL_MONEDA, COL_CHEQUE,
            "Nombre Comitente TENEDOR", COL_TENEDOR,
            "Manager TENEDOR", "Oficial TENEDOR",
            COL_MONTO, COL_FECHA_PAGO, COL_FECHA_COBRO,
            "Nombre Comitente INGRESANTE", COL_INGRESANTE,
            "Manager INGRESANTE", "Oficial INGRESANTE",
            COL_ESTADO,
        ] if c in df_pago_f.columns
    ]

    # formateo display
    disp = df_pago_f.copy()
    for c in [COL_FECHA_PAGO, COL_FECHA_COBRO, COL_FECHA_INGRESO]:
        if c in disp.columns:
            disp[c] = pd.to_datetime(disp[c], errors="coerce").dt.date
    disp[COL_MONTO] = disp[COL_MONTO].fillna(0).apply(_fmt_money)

    st.dataframe(
        disp[venc_cols] if not disp.empty else pd.DataFrame(columns=venc_cols),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ========= Botón reiniciar =========
    if st.button("Reiniciar"):
        st.rerun()

