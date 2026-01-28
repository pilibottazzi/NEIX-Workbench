# tools/vencimientos.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

MANAGERS_PATH = os.path.join("data", "managers_neix.xlsx")


# ======================================================
# Datos base: managers_neix.xlsx
# (esperado: NumeroComitente, Comitente, Oficial, Productor, Acciones)
# ======================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(f"No existe '{MANAGERS_PATH}'. Subilo al repo dentro de la carpeta 'data/'.")

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # Normalizaciones típicas
    if "Cliente" in df.columns and "Comitente" not in df.columns:
        df = df.rename(columns={"Cliente": "Comitente"})

    # Tu excel (según captura) trae "Comitente" (número) + "Nombre" + "Oficial" + "Productor" + "Acciones"
    # Pero vos venías usando NumeroComitente. Lo soportamos igual.
    if "NumeroComitente" not in df.columns:
        if "Comitente" in df.columns:
            df["NumeroComitente"] = df["Comitente"]
        else:
            raise ValueError(
                f"managers_neix.xlsx debe tener 'NumeroComitente' o 'Comitente'. Columnas: {list(df.columns)}"
            )

    # Clave de merge como string limpio
    df["NumeroComitente"] = (
        pd.to_numeric(df["NumeroComitente"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.strip()
    )

    # Columnas opcionales
    for col in ["Nombre", "Comitente", "Oficial", "Productor", "Acciones"]:
        if col not in df.columns:
            df[col] = ""

    # Dejamos una tabla "lookup" sin duplicados por comitente
    lookup_cols = ["NumeroComitente", "Nombre", "Oficial", "Productor", "Acciones"]
    df = df[lookup_cols].copy()

    # Si hay repetidos, priorizamos el último no vacío
    df = (
        df.replace({None: "", pd.NA: ""})
        .sort_values(by=["NumeroComitente"])
        .groupby("NumeroComitente", as_index=False)
        .last()
    )

    return df


# ======================================================
# Helpers para detectar columna comitente del Excel subido
# ======================================================
def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip(): c for c in df.columns.astype(str)}
    for cand in candidates:
        for real in cols:
            if real.lower() == cand.lower():
                return cols[real]
    return None


def _normalize_comitente_series(s: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(s, errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


# ======================================================
# UI
# ======================================================
def _ui_css():
    st.markdown(
        """
    <style>
      .wrap{ max-width: 1200px; margin: 0 auto; }
      .title{ font-size:22px; font-weight:800; letter-spacing:.04em; color:#111827; }
      .sub{ color:rgba(17,24,39,.60); font-size:13px; margin-top:2px; }
      .card{
        border:1px solid rgba(17,24,39,0.08);
        border-radius:14px;
        padding:14px 14px;
        background:#fff;
        box-shadow: 0 8px 26px rgba(17,24,39,0.05);
      }
      div[data-baseweb="tag"]{ border-radius:999px !important; }
      .block-container { padding-top: 1.2rem; }
      .stButton > button { border-radius: 12px; padding: 0.60rem 1.0rem; }
      .stSelectbox div[data-baseweb="select"]{ border-radius: 12px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    h1, h2 = st.columns([0.78, 0.22])
    with h1:
        st.markdown('<div class="title">NEIX · Vencimientos</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub">Subí el Excel y cruzamos por comitente contra <b>data/managers_neix.xlsx</b>.</div>',
            unsafe_allow_html=True,
        )
    with h2:
        if back_to_home is not None:
            st.button("← Volver", on_click=back_to_home)

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    archivo = st.file_uploader(
        "Subí el archivo Excel",
        type=["xlsx"],
        accept_multiple_files=False,
        key="vencimientos_uploader",
    )

    if archivo is None:
        st.info("Esperando archivo Excel…")
        return

    # -------------------------
    # Leer excel subido
    # -------------------------
    try:
        # Si tu excel tiene 2 filas de header raras, podés cambiar a header=1.
        # Pero por defecto lo dejo en 0 para que no “rompa” con otros formatos.
        df_excel = pd.read_excel(archivo, header=0)
    except Exception:
        st.error("No pude leer el Excel.")
        st.stop()

    df_excel.columns = df_excel.columns.astype(str).str.strip()

    # -------------------------
    # Detectar columna comitente del archivo subido
    # -------------------------
    # Priorizamos: NumeroComitente / Comitente / Neix
    col_comit = _find_first_col(df_excel, ["NumeroComitente", "Comitente", "Neix"])
    if col_comit is None:
        st.error("No encontré columna de comitente en el Excel subido (busqué: NumeroComitente, Comitente o Neix).")
        st.stop()

    df_excel = df_excel.copy()
    df_excel["_NumeroComitente"] = _normalize_comitente_series(df_excel[col_comit])

    # -------------------------
    # Cargar managers lookup
    # -------------------------
    try:
        with st.spinner("Cargando managers_neix.xlsx…"):
            df_m = cargar_managers_excel()
    except Exception as e:
        st.error("No pude cargar managers_neix.xlsx")
        st.exception(e)
        df_m = pd.DataFrame()

    # -------------------------
    # Merge
    # -------------------------
    if df_m.empty:
        st.warning("Managers vacío / no disponible. Se muestra el Excel sin cruce.")
        df_cruce = df_excel.copy()
    else:
        df_cruce = df_excel.merge(
            df_m,
            left_on="_NumeroComitente",
            right_on="NumeroComitente",
            how="left",
        )

    # -------------------------
    # Filtros
    # -------------------------
    st.markdown("### Filtros")
    c1, c2, c3 = st.columns([0.40, 0.35, 0.25])

    with c1:
        productores = (
            df_cruce["Productor"].dropna().astype(str).replace({"": pd.NA}).dropna().unique().tolist()
            if "Productor" in df_cruce.columns
            else []
        )
        prod_sel = st.multiselect(
            "Productor",
            options=["Todos"] + sorted(productores),
            default=["Todos"],
            key="venc_prod",
        )

    with c2:
        oficiales = (
            df_cruce["Oficial"].dropna().astype(str).replace({"": pd.NA}).dropna().unique().tolist()
            if "Oficial" in df_cruce.columns
            else []
        )
        ofi_sel = st.multiselect(
            "Oficial",
            options=["Todos"] + sorted(oficiales),
            default=["Todos"],
            key="venc_ofi",
        )

    with c3:
        # filtro por comitente (numérico) – útil para buscar rápido
        comitentes = df_cruce["_NumeroComitente"].replace({"": pd.NA}).dropna().unique().tolist()
        com_sel = st.multiselect(
            "Comitente",
            options=["Todos"] + sorted(comitentes),
            default=["Todos"],
            key="venc_com",
        )

    df_f = df_cruce.copy()
    if "Todos" not in prod_sel and "Productor" in df_f.columns:
        df_f = df_f[df_f["Productor"].astype(str).isin(prod_sel)]
    if "Todos" not in ofi_sel and "Oficial" in df_f.columns:
        df_f = df_f[df_f["Oficial"].astype(str).isin(ofi_sel)]
    if "Todos" not in com_sel:
        df_f = df_f[df_f["_NumeroComitente"].astype(str).isin(com_sel)]

    st.divider()

    # -------------------------
    # Tabla (prolija)
    # -------------------------
    st.markdown("### Resultado")

    # Convertir fechas comunes si existen
    for possible_date in ["Fecha", "Vencimiento", "F.Inicio", "F. Inicio", "FInicio"]:
        if possible_date in df_f.columns:
            df_f[possible_date] = pd.to_datetime(df_f[possible_date], errors="coerce").dt.date

    # Columnas sugeridas (y las que existan)
    # - del lookup: NumeroComitente, Nombre, Oficial, Productor, Acciones
    # - del excel subido: mantenemos todo, pero priorizamos lo importante arriba
    preferred = []
    # Primero comitente
    if col_comit in df_f.columns:
        preferred.append(col_comit)
    preferred += ["Nombre", "Oficial", "Productor", "Acciones"]

    # Agregamos algunas típicas que suelen venir en vencimientos
    preferred += ["Fecha", "Vencimiento", "Especie", "Moneda", "VN", "Dias", "Activo"]

    # Armamos lista final sin repetir y solo existentes
    cols_out = []
    for c in preferred:
        if c in df_f.columns and c not in cols_out:
            cols_out.append(c)

    # Agregamos el resto (por si tu excel trae campos extra)
    for c in df_f.columns:
        if c not in cols_out and c not in {"_NumeroComitente"}:
            cols_out.append(c)

    # Mostrar más filas (alto dinámico)
    base = 420
    row_h = 28
    max_h = 900
    height_df = int(min(max_h, base + row_h * len(df_f)))

    st.dataframe(
        df_f[cols_out],
        use_container_width=True,
        hide_index=True,
        height=height_df,
        column_config={
            col_comit: st.column_config.TextColumn(col_comit),
            "Nombre": st.column_config.TextColumn("Nombre"),
            "Oficial": st.column_config.TextColumn("Oficial"),
            "Productor": st.column_config.TextColumn("Productor"),
            "Acciones": st.column_config.TextColumn("Acciones"),
            "Fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
            "VN": st.column_config.NumberColumn("VN", format="%.0f"),
            "Dias": st.column_config.NumberColumn("Días", format="%.0f"),
        },
    )
