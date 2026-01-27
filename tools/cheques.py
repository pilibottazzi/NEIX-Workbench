# tools/cheques.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import mysql.connector
import re

# =====================================================
# CONFIG INICIAL
# =====================================================
st.set_page_config(page_title="Dashboard cheques y pagarés", layout="wide")
st.title("Dashboard cheques y pagarés")

# =====================================================
# PROTECCIÓN POR CLAVE (simple)
# =====================================================
with st.expander("🔐 Sección protegida por clave"):
    password = st.text_input("Ingresá la clave para acceder:", type="password")
    if password != st.secrets.get("app_password", "ciclon"):
        st.warning("⚠️ Acceso restringido. Ingresá la clave correcta para ver el contenido.")
        st.stop()

# =====================================================
# DB CONFIG (desde Secrets)
# =====================================================
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
        st.error("No están configurados los Secrets de MySQL (st.secrets['mysql']).")
        st.stop()

def obtener_managers() -> pd.DataFrame:
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
    return pd.DataFrame(rows)

# =====================================================
# HELPERS
# =====================================================
def limpiar_monto(x):
    """Normaliza montos con símbolos y separadores (formato AR/US). Devuelve float o NA."""
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

meses_nombres = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# =====================================================
# CARGA OBLIGATORIA DE ARCHIVO
# =====================================================
st.subheader("📎 Carga de archivo (obligatoria)")

archivo_subido = st.file_uploader(
    "Subí el Excel de cheques / pagarés para continuar",
    type=["xlsx"],
    accept_multiple_files=False
)

if archivo_subido is None:
    st.info("Debés subir un archivo Excel para habilitar el dashboard.")
    st.stop()

btn_cargar = st.button("Cargar datos", type="primary")
if not btn_cargar:
    st.caption("Una vez subido el archivo, tocá **Cargar datos**.")
    st.stop()

# =====================================================
# LECTURA EXCEL
# =====================================================
try:
    df = pd.read_excel(archivo_subido, header=0)
except Exception as e:
    st.error(f"Error leyendo el archivo: {e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

# =====================================================
# DEFINICIÓN COLUMNAS
# =====================================================
COL_ID = "IDENT"
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

# Validación mínima (para que no explote raro)
required_excel_cols = [COL_TIPO, COL_MONEDA, COL_MONTO, COL_FECHA_PAGO, COL_ESTADO, COL_TENEDOR, COL_INGRESANTE]
missing_excel = [c for c in required_excel_cols if c not in df.columns]
if missing_excel:
    st.error(f"Faltan columnas en el Excel: {missing_excel}")
    st.stop()

# =====================================================
# LIMPIEZA
# =====================================================
for c in [COL_FECHA_PAGO, COL_FECHA_COBRO, COL_FECHA_INGRESO]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

df[COL_MONTO] = df[COL_MONTO].apply(limpiar_monto)

for c in [COL_TENEDOR, COL_INGRESANTE]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str)

# =====================================================
# CRUCE MANAGERS (DB)
# =====================================================
with st.spinner("Cargando datos de Managers desde MySQL…"):
    df_manager = obtener_managers()

if not df_manager.empty:
    df_manager["NumeroComitente"] = (
        pd.to_numeric(df_manager["NumeroComitente"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.strip()
    )

    # TENEDOR
    df = df.merge(
        df_manager[["NumeroComitente", "Comitente", "NumeroManager", "Manager", "NumeroOficial", "Oficial"]],
        left_on=COL_TENEDOR,
        right_on="NumeroComitente",
        how="left",
    ).rename(columns={
        "Comitente": "Nombre Comitente TENEDOR",
        "NumeroManager": "NumeroManager TENEDOR",
        "Manager": "Manager TENEDOR",
        "NumeroOficial": "NumeroOficial TENEDOR",
        "Oficial": "Oficial TENEDOR",
    }).drop(columns=["NumeroComitente"], errors="ignore")

    # INGRESANTE
    df = df.merge(
        df_manager[["NumeroComitente", "Comitente", "NumeroManager", "Manager", "NumeroOficial", "Oficial"]],
        left_on=COL_INGRESANTE,
        right_on="NumeroComitente",
        how="left",
    ).rename(columns={
        "Comitente": "Nombre Comitente INGRESANTE",
        "NumeroManager": "NumeroManager INGRESANTE",
        "Manager": "Manager INGRESANTE",
        "NumeroOficial": "NumeroOficial INGRESANTE",
        "Oficial": "Oficial INGRESANTE",
    }).drop(columns=["NumeroComitente"], errors="ignore")

# =====================================================
# CAMPOS AUX
# =====================================================
df["Mes Pago"] = df[COL_FECHA_PAGO].dt.month

hoy = datetime.today().date()
maniana = hoy + timedelta(days=1)
lunes = hoy - timedelta(days=hoy.weekday())
domingo = lunes + timedelta(days=6)
inicio_mes = hoy.replace(day=1)
fin_mes = (pd.Timestamp(inicio_mes) + pd.offsets.MonthEnd(1)).date()

# =====================================================
# VENCIMIENTOS POR FECHA DE PAGO
# =====================================================
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

# Filtros
col1, col2, col3 = st.columns(3)
monedas = sorted(df[COL_MONEDA].dropna().astype(str).unique())
estados = sorted(df[COL_ESTADO].dropna().astype(str).unique())
instrumentos = sorted(df[COL_TIPO].dropna().astype(str).unique())

with col1:
    filtro_monedas = st.multiselect("Moneda", ["Todos"] + monedas, default=["Todos"])
with col2:
    filtro_estados = st.multiselect("Estado", ["Todos"] + estados, default=["Todos"])
with col3:
    filtro_instrumento = st.multiselect("Tipo Instrumento", ["Todos"] + instrumentos, default=["Todos"])

df_pago_filtrado = df_pago.copy()
if "Todos" not in filtro_monedas:
    df_pago_filtrado = df_pago_filtrado[df_pago_filtrado[COL_MONEDA].astype(str).isin(filtro_monedas)]
if "Todos" not in filtro_estados:
    df_pago_filtrado = df_pago_filtrado[df_pago_filtrado[COL_ESTADO].astype(str).isin(filtro_estados)]
if "Todos" not in filtro_instrumento:
    df_pago_filtrado = df_pago_filtrado[df_pago_filtrado[COL_TIPO].astype(str).isin(filtro_instrumento)]

# columnas a mostrar
venc_cols = [c for c in [
    COL_FECHA_INGRESO, COL_TIPO, COL_MONEDA, COL_CHEQUE,
    "Nombre Comitente TENEDOR", COL_TENEDOR,
    "Manager TENEDOR", "Oficial TENEDOR",
    COL_MONTO, COL_FECHA_PAGO, COL_FECHA_COBRO,
    "Nombre Comitente INGRESANTE", COL_INGRESANTE,
    "Manager INGRESANTE", "Oficial INGRESANTE",
    COL_ESTADO,
] if c in df_pago_filtrado.columns]

# formateo
if not df_pago_filtrado.empty:
    for c in [COL_FECHA_PAGO, COL_FECHA_COBRO, COL_FECHA_INGRESO]:
        if c in df_pago_filtrado.columns:
            df_pago_filtrado[c] = df_pago_filtrado[c].dt.date
    df_pago_filtrado[COL_MONTO] = df_pago_filtrado[COL_MONTO].fillna(0).apply(
        lambda x: f"${x:,.0f}".replace(",", ".")
    )

st.dataframe(
    df_pago_filtrado[venc_cols] if not df_pago_filtrado.empty else pd.DataFrame(columns=venc_cols),
    hide_index=True,
    use_container_width=True
)

st.divider()

# =====================================================
# Botón reiniciar (opcional)
# =====================================================
if st.button("Reiniciar"):
    st.rerun()

