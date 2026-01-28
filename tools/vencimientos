#tools/vencimientos.py
import os
import pandas as pd
import streamlit as st

MANAGERS_PATH = os.path.join("data", "managers_neix.xlsx")


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(
            f"No existe '{MANAGERS_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    if "Comitente" not in df.columns and "Cliente" in df.columns:
        df = df.rename(columns={"Cliente": "Comitente"})

    if "NumeroComitente" not in df.columns:
        raise ValueError(
            f"managers_neix.xlsx debe tener 'NumeroComitente'. Columnas: {list(df.columns)}"
        )

    df["NumeroComitente"] = (
        pd.to_numeric(df["NumeroComitente"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.strip()
    )

    for col in ["Comitente", "Manager", "Oficial"]:
        if col not in df.columns:
            df[col] = ""

    return df


def render(back_to_home=None):
    st.markdown("## Alquileres")
    st.caption("Subí el Excel y cruzamos con Manager / Oficial desde managers_neix.xlsx (sin SQL).")

    archivo = st.file_uploader(
        "Subí el archivo Excel",
        type=["xlsx"],
        accept_multiple_files=False,
        key="alquileres_uploader",
    )

    if archivo is None:
        st.info("Esperando archivo Excel…")
        return

    try:
        df_excel = pd.read_excel(archivo, header=1)  # dejo tu header=1
    except Exception as e:
        st.error("No pude leer el Excel.")
        st.exception(e)
        return

    df_excel.columns = df_excel.columns.astype(str).str.strip()

    if "Cliente" in df_excel.columns:
        df_excel = df_excel.rename(columns={"Cliente": "Nombre del cliente"})

    if "Neix" not in df_excel.columns:
        st.error("El archivo debe contener una columna llamada 'Neix'.")
        return

    df_excel["Comitente"] = (
        df_excel["Neix"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    try:
        with st.spinner("Cargando Managers desde managers_neix.xlsx…"):
            df_managers = cargar_managers_excel()
    except Exception as e:
        st.error("No pude cargar managers_neix.xlsx")
        st.exception(e)
        df_managers = pd.DataFrame()

    if df_managers.empty:
        st.warning("Managers vacío / no disponible. Se muestra el Excel sin cruce.")
        df_cruce = df_excel.copy()
    else:
        df_cruce = df_excel.merge(
            df_managers[["NumeroComitente", "Comitente", "Manager", "Oficial"]],
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
