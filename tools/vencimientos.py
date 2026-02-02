# tools/vencimientos.py
from __future__ import annotations

import os
import io
import re
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
        dtype=str
    )
    df.columns = df.columns.astype(str).str.strip()

    # Caso típico: primera columna es "Cliente Nombre del Cliente" y viene "1154 NOMBRE..."
    first_col = df.columns[0]
    if ("Cliente" in first_col) and ("Nombre" in first_col):
        df = df.rename(columns={first_col: "ClienteNombreRaw"})
        split = df["ClienteNombreRaw"].astype(str).str.strip().str.split(r"\s+", n=1, expand=True)
        df["Cliente"] = split[0]
        df["Nombre del Cliente"] = split[1].fillna("")
        df = df.drop(columns=["ClienteNombreRaw"])
    else:
        # tolerancia si alguna vez viniera separado
        if "Cliente" not in df.columns:
            raise ValueError(f"No encontré columna Cliente. Columnas: {list(df.columns)}")

    df["Cliente"] = _norm_num(df["Cliente"])

    # limpiar "Total" si aparece
    df = df[~df["Cliente"].astype(str).str.lower().str.contains("total", na=False)].copy()

    # convertir numéricas si existen
    for col in ["Saldo Tesoro", "Saldo Caja Val.", "Ob"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(
            f"No existe '{MANAGERS_PATH}'. Guardalo en el repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # Requisito: NumeroComitente sí o sí
    if "NumeroComitente" not in df.columns:
        raise ValueError(
            f"managers_neix.xlsx debe tener 'NumeroComitente'. Columnas: {list(df.columns)}"
        )

    df["NumeroComitente"] = _norm_num(df["NumeroComitente"])

    # dejamos SOLO columnas reales del archivo (no agregamos inventadas)
    # pero normalizamos por las dudas algunos nombres comunes
    rename_map = {}
    if "Comitente" in df.columns:
        rename_map["Comitente"] = "Nombre Comitente"
    df = df.rename(columns=rename_map)

    return df


def _merge_managers_strict(df_txt: pd.DataFrame, df_mgr: pd.DataFrame) -> pd.DataFrame:
    """
    Cruce obligatorio por Cliente (TXT) = NumeroComitente (Excel).
    """
    cols_keep = [c for c in ["NumeroComitente", "Nombre Comitente", "NumeroManager", "Manager", "NumeroOficial", "Oficial"] if c in df_mgr.columns]

    out = df_txt.merge(
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
    st.markdown("## Tenencias")
    st.caption("RUTA DE GALLO: BURSATIL -> CONSULTAS -> CONSULTAS DE TENENCIAS -> TENENCIAS -> POR ESPECIE")

    # 1) managers obligatorio
    try:
        with st.spinner("Cargando managers_neix.xlsx…"):
            df_mgr = cargar_managers_excel()
    except Exception as e:
        st.error("No puedo seguir: el cruce con managers es obligatorio y el archivo no cargó.")
        st.exception(e)
        st.stop()

    # 2) uploader múltiples txt
    archivos = st.file_uploader(
        "Subí uno o varios archivos .txt",
        type=["txt"],
        accept_multiple_files=True,
        key="vencimientos_txt_uploader",
    )

    if not archivos:
        st.info("Esperando archivos .txt…")
        return

    dfs_por_activo: dict[str, pd.DataFrame] = {}
    errores = []

    for f in archivos:
        activo = _asset_from_filename(getattr(f, "name", "ACTIVO"))
        try:
            df = _read_txt_uploaded(f)
            df.insert(0, "Activo", activo)
            df = _merge_managers_strict(df, df_mgr)
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
    # consideramos match si hay Manager u Oficial
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
        st.info(f"Cruce managers")

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


    # 6) tablas por activo
    st.markdown("### Tablas por Activo")
    tabs = st.tabs(sorted(dfs_por_activo.keys()))
    for activo, tab in zip(sorted(dfs_por_activo.keys()), tabs):
        with tab:
            df_tab = dfs_por_activo[activo]

            # aplicar mismos filtros al tab
            df_tab_f = df_tab.copy()
            if "Todos" not in managers_sel and "Manager" in df_tab_f.columns:
                df_tab_f = df_tab_f[df_tab_f["Manager"].astype(str).isin(managers_sel)]
            if "Todos" not in oficiales_sel and "Oficial" in df_tab_f.columns:
                df_tab_f = df_tab_f[df_tab_f["Oficial"].astype(str).isin(oficiales_sel)]

            prefer = [
                "Activo", "Cliente", "Nombre del Cliente",
                "Saldo Tesoro", "Saldo Caja Val.", "Ob", "Fecha",
                "NumeroManager", "Manager", "NumeroOficial", "Oficial",
                "Nombre Comitente",
            ]
            cols_show = [c for c in prefer if c in df_tab_f.columns] + [c for c in df_tab_f.columns if c not in prefer]
            st.dataframe(df_tab_f[cols_show], use_container_width=True, hide_index=True)

    st.markdown("### Consolidado (todo junto)")
    prefer_all = [
        "Activo", "Cliente", "Nombre del Cliente",
        "Saldo Tesoro", "Saldo Caja Val.", "Ob", "Fecha",
        "NumeroManager", "Manager", "NumeroOficial", "Oficial",
        "Nombre Comitente",
    ]
    cols_show_all = [c for c in prefer_all if c in df_f.columns] + [c for c in df_f.columns if c not in prefer_all]
    st.dataframe(df_f[cols_show_all], use_container_width=True, hide_index=True)

