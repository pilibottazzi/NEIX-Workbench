# tools/vencimientos.py
from __future__ import annotations

import os
import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

MANAGERS_PATH = os.path.join("data", "managers_neix.xlsx")


# =========================
# Helpers
# =========================
def _norm_num(s: pd.Series) -> pd.Series:
    """Normaliza ids tipo comitente: str, sin .0, sin espacios."""
    return (
        s.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


def _asset_from_filename(name: str) -> str:
    base = os.path.basename(name)
    return os.path.splitext(base)[0].strip()


def _detect_header_line(text: str) -> int:
    """
    En tus TXT hay una línea basura primero ("t;do;do;t;d")
    y luego el header real:
    "Cliente Nombre del Cliente;Saldo Tesoro;Saldo Caja Val.;Ob;Fecha"
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if "Cliente" in ln and ";" in ln:
            return i
    return 0


def _find_col(df: pd.DataFrame, includes: list[str]) -> str | None:
    """
    Busca una columna por 'includes' (case-insensitive), tolerando variaciones:
    - puntos, dobles espacios, texto extra
    """
    cols = list(df.columns.astype(str))
    norm = {c: re.sub(r"\s+", " ", c.strip().lower()) for c in cols}

    for c, c_norm in norm.items():
        ok = True
        for token in includes:
            token_norm = re.sub(r"\s+", " ", token.strip().lower())
            if token_norm not in c_norm:
                ok = False
                break
        if ok:
            return c
    return None


def _split_cliente_nombre(df: pd.DataFrame, col_raw: str) -> pd.DataFrame:
    """
    Recibe df con una columna tipo "1154 NOMBRE..." y genera:
    - Cliente
    - Nombre del Cliente
    """
    df = df.rename(columns={col_raw: "ClienteNombreRaw"}).copy()
    split = (
        df["ClienteNombreRaw"]
        .astype(str)
        .str.strip()
        .str.split(r"\s+", n=1, expand=True)
    )
    df["Cliente"] = split[0]
    df["Nombre del Cliente"] = split[1].fillna("")
    df = df.drop(columns=["ClienteNombreRaw"])
    df["Cliente"] = _norm_num(df["Cliente"])

    # limpiar "Total" si aparece
    df = df[~df["Cliente"].astype(str).str.lower().str.contains("total", na=False)].copy()
    return df


def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    df[col] = (
        df[col].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _read_txt_uploaded(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin1")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("El archivo TXT está vacío.")

    header_idx = _detect_header_line(text)
    data_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(
        io.StringIO(data_text),
        sep=";",
        engine="python",
        dtype=str,
    )
    df.columns = df.columns.astype(str).str.strip()

    # Columna combinada: "Cliente Nombre del Cliente"
    col_cliente_nombre = _find_col(df, ["cliente", "nombre"])
    if not col_cliente_nombre:
        raise ValueError(f"No encontré columna tipo 'Cliente Nombre del Cliente'. Columnas: {list(df.columns)}")

    df = _split_cliente_nombre(df, col_cliente_nombre)

    # Nominales = Saldo Caja Val.
    col_saldo_caja = _find_col(df, ["saldo", "caja"])
    if not col_saldo_caja:
        raise ValueError(f"No encontré columna tipo 'Saldo Caja Val.'. Columnas: {list(df.columns)}")

    # aseguramos numérico
    _coerce_numeric(df, col_saldo_caja)
    df = df.rename(columns={col_saldo_caja: "Nominales"})

    return df


def _read_excel_uploaded(uploaded_file) -> pd.DataFrame:
    # Streamlit UploadedFile funciona directo con read_excel
    df = pd.read_excel(uploaded_file, dtype=str)
    df.columns = df.columns.astype(str).str.strip()

    # Columna combinada: "Cliente Nombre del Cliente"
    col_cliente_nombre = _find_col(df, ["cliente", "nombre"])
    if not col_cliente_nombre:
        raise ValueError(f"No encontré columna tipo 'Cliente Nombre del Cliente' en Excel. Columnas: {list(df.columns)}")

    df = _split_cliente_nombre(df, col_cliente_nombre)

    # Nominales = Saldo Caja Val.
    col_saldo_caja = _find_col(df, ["saldo", "caja"])
    if not col_saldo_caja:
        raise ValueError(f"No encontré columna tipo 'Saldo Caja Val.' en Excel. Columnas: {list(df.columns)}")

    _coerce_numeric(df, col_saldo_caja)
    df = df.rename(columns={col_saldo_caja: "Nominales"})

    return df


def _read_uploaded_any(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "")
    ext = Path(name).suffix.lower()
    if ext == ".txt":
        return _read_txt_uploaded(uploaded_file)
    if ext in (".xlsx", ".xls"):
        return _read_excel_uploaded(uploaded_file)
    raise ValueError(f"Formato no soportado: {ext}. Subí .txt o .xlsx/.xls")


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(
            f"No existe '{MANAGERS_PATH}'. Guardalo en el repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    if "NumeroComitente" not in df.columns:
        raise ValueError(
            f"managers_neix.xlsx debe tener 'NumeroComitente'. Columnas: {list(df.columns)}"
        )

    df["NumeroComitente"] = _norm_num(df["NumeroComitente"])

    if "Comitente" in df.columns:
        df = df.rename(columns={"Comitente": "Nombre Comitente"})

    return df


def _merge_managers_strict(df_tenencia: pd.DataFrame, df_mgr: pd.DataFrame) -> pd.DataFrame:
    cols_keep = [
        c for c in ["NumeroComitente", "Nombre Comitente", "NumeroManager", "Manager", "NumeroOficial", "Oficial"]
        if c in df_mgr.columns
    ]

    out = df_tenencia.merge(
        df_mgr[cols_keep],
        left_on="Cliente",
        right_on="NumeroComitente",
        how="left",
    )
    out = out.drop(columns=[c for c in ["NumeroComitente"] if c in out.columns])
    return out


def _to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\*\?/\\:]", "-", name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe)
    output.seek(0)
    return output.read()


# =========================
# UI
# =========================
def render(back_to_home=None):
    st.markdown("## Vencimientos / Tenencia (por Activo)")
    st.caption(
        "Subís 1 o varios archivos **TXT o Excel**. "
        "El Activo se toma del nombre del archivo. "
        "Cruce por **NumeroComitente** con `data/managers_neix.xlsx`. "
        "Se usa: **Cliente Nombre del Cliente** + **Saldo Caja Val.**"
    )

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    # 1) managers obligatorio
    try:
        with st.spinner("Cargando managers_neix.xlsx…"):
            df_mgr = cargar_managers_excel()
    except Exception as e:
        st.error("No puedo seguir: el cruce con managers es obligatorio y el archivo no cargó.")
        st.exception(e)
        st.stop()

    # 2) uploader múltiples txt/xlsx
    archivos = st.file_uploader(
        "Subí uno o varios archivos (.txt / .xlsx / .xls)",
        type=["txt", "xlsx", "xls"],
        accept_multiple_files=True,
        key="vencimientos_any_uploader",
    )

    if not archivos:
        st.info("Esperando archivos…")
        return

    dfs_por_activo: dict[str, pd.DataFrame] = {}
    errores: list[tuple[str, str]] = []

    for f in archivos:
        activo = _asset_from_filename(getattr(f, "name", "ACTIVO"))
        try:
            df = _read_uploaded_any(f)

            # Activo + cruce
            df.insert(0, "Activo", activo)
            df = _merge_managers_strict(df, df_mgr)

            # Nos quedamos SOLO con lo que pediste (más lo de managers)
            # Base: Activo, Cliente, Nombre del Cliente, Nominales + columnas managers si existen
            prefer = [
                "Activo", "Cliente", "Nombre del Cliente", "Nominales",
                "NumeroManager", "Manager", "NumeroOficial", "Oficial", "Nombre Comitente"
            ]
            cols_keep = [c for c in prefer if c in df.columns]
            # por si alguna vez querés ver extras, dejalo comentado:
            # df = df[cols_keep + [c for c in df.columns if c not in cols_keep]]
            df = df[cols_keep].copy()

            dfs_por_activo[activo] = df

        except Exception as e:
            errores.append((activo, str(e)))

    if errores:
        st.error("Algunos archivos no se pudieron leer:")
        for activo, msg in errores:
            st.write(f"- **{activo}**: {msg}")

    if not dfs_por_activo:
        st.stop()

    df_all = pd.concat(dfs_por_activo.values(), ignore_index=True)

    # 3) indicador de match del cruce
    has_manager = "Manager" in df_all.columns
    has_oficial = "Oficial" in df_all.columns
    if has_manager or has_oficial:
        match_mask = pd.Series(False, index=df_all.index)
        if has_manager:
            match_mask = match_mask | df_all["Manager"].notna()
        if has_oficial:
            match_mask = match_mask | df_all["Oficial"].notna()

        matched = int(match_mask.sum())
        total = int(len(df_all))
        st.info(
            f"Cruce managers: **{matched} / {total}** filas matcheadas "
            f"({(matched/total*100):.1f}%). "
            "Si esto da bajo, esos comitentes no existen en managers_neix.xlsx o vienen con otro número."
        )

    # 4) filtros
    st.markdown("### Filtros (globales)")
    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

    with c1:
        activos = sorted(df_all["Activo"].dropna().astype(str).unique().tolist())
        activos_sel = st.multiselect("Activo", options=["Todos"] + activos, default=["Todos"])

    with c2:
        if "Manager" in df_all.columns:
            managers = sorted([x for x in df_all["Manager"].dropna().astype(str).unique().tolist() if x.strip() and x.strip() != "None"])
            managers_sel = st.multiselect("Manager", options=["Todos"] + managers, default=["Todos"])
        else:
            managers_sel = ["Todos"]

    with c3:
        if "Oficial" in df_all.columns:
            oficiales = sorted([x for x in df_all["Oficial"].dropna().astype(str).unique().tolist() if x.strip() and x.strip() != "None"])
            oficiales_sel = st.multiselect("Oficial", options=["Todos"] + oficiales, default=["Todos"])
        else:
            oficiales_sel = ["Todos"]

    df_f = df_all.copy()
    if "Todos" not in activos_sel:
        df_f = df_f[df_f["Activo"].astype(str).isin(activos_sel)]
    if "Manager" in df_f.columns and "Todos" not in managers_sel:
        df_f = df_f[df_f["Manager"].astype(str).isin(managers_sel)]
    if "Oficial" in df_f.columns and "Todos" not in oficiales_sel:
        df_f = df_f[df_f["Oficial"].astype(str).isin(oficiales_sel)]

    # 5) resumen
    st.markdown("### Resumen")
    if "Nominales" in df_f.columns:
        st.markdown("**Por Activo**")
        st.dataframe(
            df_f.groupby("Activo", as_index=False)[["Nominales"]].sum(numeric_only=True),
            use_container_width=True,
            hide_index=True
        )

        if "Manager" in df_f.columns:
            st.markdown("**Por Manager**")
            st.dataframe(
                df_f.groupby("Manager", as_index=False)[["Nominales"]].sum(numeric_only=True).sort_values("Nominales", ascending=False),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("No encontré la columna 'Nominales' para resumir.")

    # 6) tablas por activo
    st.markdown("### Tablas por Activo")
    tabs = st.tabs(sorted(dfs_por_activo.keys()))
    for activo, tab in zip(sorted(dfs_por_activo.keys()), tabs):
        with tab:
            df_tab = dfs_por_activo[activo].copy()

            if "Todos" not in managers_sel and "Manager" in df_tab.columns:
                df_tab = df_tab[df_tab["Manager"].astype(str).isin(managers_sel)]
            if "Todos" not in oficiales_sel and "Oficial" in df_tab.columns:
                df_tab = df_tab[df_tab["Oficial"].astype(str).isin(oficiales_sel)]

            st.dataframe(df_tab, use_container_width=True, hide_index=True)

    st.markdown("### Consolidado (todo junto)")
    st.dataframe(df_f, use_container_width=True, hide_index=True)

    # 7) export
    st.markdown("### Exportar")
    sheets = {"Consolidado": df_f}
    for activo, df in dfs_por_activo.items():
        sheets[activo] = df

    st.download_button(
        "📥 Descargar Excel",
        data=_to_excel_bytes(sheets),
        file_name="vencimientos_por_activo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
