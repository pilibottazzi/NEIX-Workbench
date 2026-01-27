# tools/alquileres.py
import pandas as pd
import streamlit as st
import mysql.connector


# =========================
# DB CONFIG (desde secrets)
# =========================
def get_db_config() -> dict:
    """
    Espera en secrets:
      [mysql]
      user, password, host, database, (opcional port)
    """
    try:
        return {
            "user": st.secrets["mysql"]["user"],
            "password": st.secrets["mysql"]["password"],
            "host": st.secrets["mysql"]["host"],
            "database": st.secrets["mysql"]["database"],
            "port": int(st.secrets["mysql"].get("port", 3306)),
            "allow_local_infile": True,
        }
    except Exception:
        st.error(
            "Faltan secrets de MySQL. Configurá st.secrets['mysql'] (local o Streamlit Cloud)."
        )
        st.stop()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def obtener_managers_cached() -> pd.DataFrame:
    db_config = get_db_config()
    cn = mysql.connector.connect(**db_config)
    cur = cn.cursor(dictionary=True)

    cur.execute(
        """
        WITH ultimo AS (
            SELECT t.*
            FROM neix.principalLog t
            JOIN (
                SELECT principal_id, MAX(created_at) AS max_created
                FROM neix.principalLog
                GROUP BY principal_id
            ) sub
            ON t.principal_id = sub.principal_id AND t.created_at = sub.max_created
        )
        SELECT
            p.id AS NumeroComitente,
            p.description AS Cliente,
            m.id AS NumeroManager,
            m.description AS Manager,
            o.id AS NumeroOficial,
            o.description AS Oficial
        FROM ultimo u
        LEFT JOIN neix.principal AS p ON u.principal_id = p.id
        LEFT JOIN neix.manager  AS m ON u.manager_id   = m.id
        LEFT JOIN neix.officer  AS o ON u.officer_id   = o.id
        """
    )

    rows = cur.fetchall()
    cur.close()
    cn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalizar clave merge
    df["NumeroComitente"] = (
        pd.to_numeric(df["NumeroComitente"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.strip()
    )
    return df


# =========================
# Render (Workbench)
# =========================
def render(back_to_home=None):
    # IMPORTANTE: NO llamar back_to_home() acá.
    # El botón volver lo dibuja app.py

    st.markdown("## Alquileres")
    st.caption("Subí el Excel y cruzamos con Manager / Oficial desde MySQL.")

    archivo = st.file_uploader(
        "Subí el archivo Excel",
        type=["xlsx"],
        accept_multiple_files=False,
        key="alquileres_uploader",
    )

    if archivo is None:
        st.info("Esperando archivo Excel…")
        return

    # Lee excel (mantengo tu header=1)
    try:
        df_excel = pd.read_excel(archivo, header=1)
    except Exception as e:
        st.error("No pude leer el Excel.")
        st.exception(e)
        return

    df_excel.columns = df_excel.columns.astype(str).str.strip()

    # Renombrar "Cliente" si existe (para no pisar la de DB)
    if "Cliente" in df_excel.columns:
        df_excel = df_excel.rename(columns={"Cliente": "Nombre del cliente"})

    if "Neix" not in df_excel.columns:
        st.error("El archivo debe contener una columna llamada 'Neix'.")
        return

    # Columna para merge
    df_excel["Comitente"] = (
        df_excel["Neix"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    # Traer managers
    with st.spinner("Cargando Managers desde MySQL…"):
        df_managers = obtener_managers_cached()

    if df_managers.empty:
        st.warning("No se pudo traer Managers (tabla vacía). Se mostrará el Excel sin cruce.")
        df_cruce = df_excel.copy()
    else:
        df_cruce = df_excel.merge(
            df_managers,
            left_on="Comitente",
            right_on="NumeroComitente",
            how="left",
        )

    # =========================
    # Filtros
    # =========================
    st.markdown("### Filtros")
    col1, col2 = st.columns(2)

    with col1:
        if "Manager" in df_cruce.columns:
            managers = df_cruce["Manager"].dropna().astype(str).unique().tolist()
            managers_sel = st.multiselect(
                "Filtrar por Manager",
                options=["Todos"] + sorted(managers),
                default=["Todos"],
                key="alquileres_filtro_manager",
            )
        else:
            managers_sel = ["Todos"]

    with col2:
        comitentes = df_cruce["Neix"].dropna().astype(str).unique().tolist()
        comitentes_sel = st.multiselect(
            "Filtrar por Cliente (Neix)",
            options=["Todos"] + sorted(comitentes),
            default=["Todos"],
            key="alquileres_filtro_neix",
        )

    df_filtrado = df_cruce.copy()
    if "Todos" not in managers_sel and "Manager" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["Manager"].astype(str).isin(managers_sel)]
    if "Todos" not in comitentes_sel:
        df_filtrado = df_filtrado[df_filtrado["Neix"].astype(str).isin(comitentes_sel)]

    # =========================
    # Tabla
    # =========================
    st.markdown("### Alquileres")

    # Fecha
    if "F.Inicio" in df_filtrado.columns:
        df_filtrado["F.Inicio"] = pd.to_datetime(df_filtrado["F.Inicio"], errors="coerce").dt.date

    columnas_mostrar = [
        col for col in [
            "F.Inicio",
            "Nombre del cliente",
            "Neix",
            "Cartera Neix",
            "VN",
            "Activo",
            "Especie",
            "Dias",
            "Moneda",
            "Manager",
            "Oficial",
        ]
        if col in df_filtrado.columns
    ]

    st.dataframe(
        df_filtrado[columnas_mostrar] if columnas_mostrar else df_filtrado,
        use_container_width=True,
        hide_index=True,
    )
