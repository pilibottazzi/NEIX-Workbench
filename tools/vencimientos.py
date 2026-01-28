# tools/vencimientos.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

MANAGERS_PATH = os.path.join("data", "managers_neix.xlsx")


# =========================
# Helpers
# =========================
def _norm_comitente(s: pd.Series) -> pd.Series:
    """
    Normaliza comitente para poder cruzar:
    - convierte a string
    - elimina .0 típico de Excel
    - strip
    """
    return (
        s.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    """
    Lee data/managers_neix.xlsx y deja una clave única "NumeroComitente" (string).
    Acepta que el archivo traiga:
      - NumeroComitente (ideal) o
      - Comitente (como en tu captura) o
      - Cliente (fallback)
    """
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(
            f"No existe '{MANAGERS_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # Tolerancia a nombres
    if "NumeroComitente" not in df.columns:
        if "Comitente" in df.columns:
            df = df.rename(columns={"Comitente": "NumeroComitente"})
        elif "Cliente" in df.columns:
            df = df.rename(columns={"Cliente": "NumeroComitente"})

    if "NumeroComitente" not in df.columns:
        raise ValueError(
            f"managers_neix.xlsx debe tener 'NumeroComitente' o 'Comitente'. Columnas: {list(df.columns)}"
        )

    df["NumeroComitente"] = _norm_comitente(df["NumeroComitente"])

    # Asegurar columnas de salida
    for col in ["Nombre", "Oficial", "Productor", "Acciones"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _cargar_archivo_subido(archivo) -> pd.DataFrame:
    """
    Tu Excel tiene una fila "Neix" como encabezado visual (merge) y los headers reales en la fila 2.
    Por eso: header=1 (fila 2 en Excel).
    """
    df = pd.read_excel(archivo, header=1)
    df.columns = df.columns.astype(str).str.strip()

    # Validaciones mínimas según tu captura
    required_any = {"Comitente", "Nombre"}
    missing = required_any - set(df.columns)
    if missing:
        raise ValueError(f"El Excel subido no tiene columnas requeridas {sorted(missing)}. Trae: {list(df.columns)}")

    # Normalizar comitente
    df["Comitente"] = _norm_comitente(df["Comitente"])

    # Orden/campos que suelen venir
    for col in ["Oficial", "Productor", "Acciones"]:
        if col not in df.columns:
            df[col] = ""

    # Limpiar filas vacías
    df = df[df["Comitente"].notna() & (df["Comitente"].str.len() > 0)].copy()
    return df


# =========================
# UI
# =========================
def render(back_to_home=None):
    st.markdown("## Vencimientos / Acciones (Excel)")
    st.caption("Subís el Excel (con fila visual 'Neix') y lo cruzamos por **Comitente** con `data/managers_neix.xlsx`.")

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    archivo = st.file_uploader(
        "Subí el archivo Excel",
        type=["xlsx"],
        accept_multiple_files=False,
        key="vencimientos_uploader",
    )

    if archivo is None:
        st.info("Esperando archivo Excel…")
        return

    # 1) Leer excel subido (ignora la fila visual 'Neix' usando header=1)
    try:
        df_excel = _cargar_archivo_subido(archivo)
    except Exception as e:
        st.error("No pude leer el Excel subido (revisá que los headers estén en la fila 2).")
        st.exception(e)
        return

    # 2) Cargar managers y cruzar
    try:
        with st.spinner("Cargando managers_neix.xlsx…"):
            df_managers = cargar_managers_excel()
    except Exception as e:
        st.warning("No pude cargar managers_neix.xlsx. Muestro solo el Excel subido.")
        st.exception(e)
        df_managers = pd.DataFrame()

    if df_managers.empty:
        df_cruce = df_excel.copy()
    else:
        # Elegimos columnas disponibles para enriquecer
        cols_mgr = [c for c in ["NumeroComitente", "Nombre", "Oficial", "Productor", "Acciones"] if c in df_managers.columns]
        df_cruce = df_excel.merge(
            df_managers[cols_mgr],
            left_on="Comitente",
            right_on="NumeroComitente",
            how="left",
            suffixes=("", "_mgr"),
        )

        # Si el Excel subido trae vacíos y managers trae data, completamos
        for c in ["Nombre", "Oficial", "Productor", "Acciones"]:
            if c in df_cruce.columns and f"{c}_mgr" in df_cruce.columns:
                df_cruce[c] = df_cruce[c].where(df_cruce[c].astype(str).str.strip().ne(""), df_cruce[f"{c}_mgr"])
        # Limpieza
        for c in ["NumeroComitente"] + [f"{x}_mgr" for x in ["Nombre", "Oficial", "Productor", "Acciones"]]:
            if c in df_cruce.columns:
                df_cruce = df_cruce.drop(columns=[c])

    # =========================
    # Filtros
    # =========================
    st.markdown("### Filtros")
    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

    with c1:
        oficiales = sorted([x for x in df_cruce["Oficial"].dropna().astype(str).unique().tolist() if x.strip() and x.strip() != "---"])
        oficiales_sel = st.multiselect("Oficial", options=["Todos"] + oficiales, default=["Todos"])

    with c2:
        productores = sorted([x for x in df_cruce["Productor"].dropna().astype(str).unique().tolist() if x.strip() and x.strip() != "---"])
        productores_sel = st.multiselect("Productor", options=["Todos"] + productores, default=["Todos"])

    with c3:
        acciones = sorted([x for x in df_cruce["Acciones"].dropna().astype(str).unique().tolist() if x.strip()])
        acciones_sel = st.multiselect("Acciones", options=["Todas"] + acciones, default=["Todas"])

    # Aplicar filtros
    df_filtrado = df_cruce.copy()

    if "Todos" not in oficiales_sel:
        df_filtrado = df_filtrado[df_filtrado["Oficial"].astype(str).isin(oficiales_sel)]

    if "Todos" not in productores_sel:
        df_filtrado = df_filtrado[df_filtrado["Productor"].astype(str).isin(productores_sel)]

    if "Todas" not in acciones_sel:
        df_filtrado = df_filtrado[df_filtrado["Acciones"].astype(str).isin(acciones_sel)]

    # =========================
    # Tabla
    # =========================
    st.markdown("### Tabla")
    columnas_mostrar = [c for c in ["Comitente", "Nombre", "Oficial", "Productor", "Acciones"] if c in df_filtrado.columns]

    st.dataframe(
        df_filtrado[columnas_mostrar] if columnas_mostrar else df_filtrado,
        use_container_width=True,
        hide_index=True,
    )
