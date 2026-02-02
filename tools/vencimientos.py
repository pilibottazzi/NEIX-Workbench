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
def _norm_comitente(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


def _asset_from_filename(name: str) -> str:
    # "AL30D.txt" -> "AL30D"
    base = os.path.basename(name)
    return os.path.splitext(base)[0].strip()


def _detect_header_line(text: str, min_cols: int = 3) -> int:
    """
    Busca la línea donde arranca el header real.
    En tu TXT típico aparece una línea basura primero y luego:
    "Cliente Nombre del Cliente;Saldo Tesoro;Saldo Caja Val.;Ob;Fecha"
    """
    lines = [ln.strip("\n\r") for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        # Heurística: header debe contener "Cliente" y separadores ;
        if "Cliente" in ln and ";" in ln:
            # Además, que tenga al menos min_cols columnas
            if ln.count(";") + 1 >= min_cols:
                return i
    # Fallback: si no encuentra, usa primera línea no vacía
    return 0


def _read_txt_uploaded(uploaded_file) -> pd.DataFrame:
    """
    Lee TXT separado por ';' con posible basura arriba del header.
    Soporta:
    - primera línea rara (ej: t;do;do;t;d)
    - header real en otra línea
    """
    raw = uploaded_file.getvalue()
    # Intentar decodificar; si falla, latin1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin1")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("El archivo está vacío.")

    header_idx = _detect_header_line(text)
    data_text = "\n".join(lines[header_idx:])

    # Leer como CSV con separador ;
    df = pd.read_csv(
        io.StringIO(data_text),
        sep=";",
        engine="python",
        dtype=str  # leemos todo como texto y luego convertimos lo numérico
    )

    # Normalizar nombres de columnas
    df.columns = df.columns.astype(str).str.strip()

    # En algunos casos el primer campo viene como "Cliente Nombre del Cliente"
    # y luego el contenido trae "1154 CACCIA DIEGO..." en una sola celda.
    # Si pasa eso, lo separamos.
    first_col = df.columns[0]
    if re.search(r"\bCliente\b", first_col) and re.search(r"\bNombre\b", first_col):
        # Renombrar a un estándar
        df = df.rename(columns={first_col: "ClienteNombreRaw"})
        # Split: primero número hasta primer espacio, resto nombre
        split = df["ClienteNombreRaw"].astype(str).str.strip().str.split(r"\s+", n=1, expand=True)
        df["Cliente"] = split[0]
        df["Nombre del Cliente"] = split[1].fillna("")
        df = df.drop(columns=["ClienteNombreRaw"])
    else:
        # Si el archivo ya trae "Cliente" y "Nombre del Cliente" separados
        # tolerancia a variantes
        if "Cliente" not in df.columns:
            # buscar algo parecido
            candidates = [c for c in df.columns if c.lower().strip() in ("cliente", "comitente", "nro cliente", "n°")]
            if candidates:
                df = df.rename(columns={candidates[0]: "Cliente"})

        if "Nombre del Cliente" not in df.columns:
            name_candidates = [c for c in df.columns if "nombre" in c.lower()]
            if name_candidates:
                df = df.rename(columns={name_candidates[0]: "Nombre del Cliente"})

    if "Cliente" not in df.columns:
        raise ValueError(f"No encontré la columna Cliente. Columnas: {list(df.columns)}")

    # Limpiar / normalizar Cliente
    df["Cliente"] = _norm_comitente(df["Cliente"])

    # Convertir algunas columnas numéricas si existen
    for col in ["Saldo Tesoro", "Saldo Caja Val.", "Ob"]:
        if col in df.columns:
            # quitar separadores raros y convertir
            df[col] = (
                df[col].astype(str)
                .str.replace(".", "", regex=False)   # por si hay miles con punto
                .str.replace(",", ".", regex=False)  # por si decimal con coma
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sacar fila "Total" si aparece (en tu ejemplo aparece "Total Disponible")
    if "Cliente" in df.columns:
        df = df[~df["Cliente"].astype(str).str.lower().str.contains("total", na=False)].copy()

    # Drop filas sin cliente
    df = df[df["Cliente"].notna() & (df["Cliente"].astype(str).str.len() > 0)].copy()

    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cargar_managers_excel() -> pd.DataFrame:
    if not os.path.exists(MANAGERS_PATH):
        raise FileNotFoundError(
            f"No existe '{MANAGERS_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(MANAGERS_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # Tolerancia
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

    # columnas esperadas
    for col in ["Nombre", "Oficial", "Productor", "Acciones"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _merge_managers(df: pd.DataFrame, df_managers: pd.DataFrame) -> pd.DataFrame:
    if df_managers is None or df_managers.empty:
        return df

    cols_mgr = [c for c in ["NumeroComitente", "Nombre", "Oficial", "Productor", "Acciones"] if c in df_managers.columns]
    out = df.merge(
        df_managers[cols_mgr],
        left_on="Cliente",
        right_on="NumeroComitente",
        how="left",
        suffixes=("", "_mgr"),
    )

    # completar si existiera "Nombre del Cliente" pero managers trae "Nombre"
    # y dejar ambos: "Nombre del Cliente" y "Nombre" (interno)
    out = out.drop(columns=[c for c in ["NumeroComitente"] if c in out.columns])

    return out


def _to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\*\?/\\:]", "-", name)[:31]  # límite Excel
            df.to_excel(writer, index=False, sheet_name=safe)
    output.seek(0)
    return output.read()


# =========================
# UI
# =========================
def render(back_to_home=None):
    st.markdown("## Vencimientos / TXT (por Activo)")
    st.caption(
        "Subís 1 o varios **.txt** (separados por `;`). "
        "El **Activo** se toma del nombre del archivo. "
        "Se cruza por **Cliente** con `data/managers_neix.xlsx`."
    )

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    archivos = st.file_uploader(
        "Subí uno o varios archivos .txt",
        type=["txt"],
        accept_multiple_files=True,
        key="vencimientos_txt_uploader",
    )

    if not archivos:
        st.info("Esperando archivos .txt…")
        return

    # Managers
    try:
        with st.spinner("Cargando managers_neix.xlsx…"):
            df_managers = cargar_managers_excel()
    except Exception as e:
        st.warning("No pude cargar managers_neix.xlsx. Voy a mostrar igual las tablas sin cruce.")
        st.exception(e)
        df_managers = pd.DataFrame()

    # Leer todos los txt
    dfs_por_activo: dict[str, pd.DataFrame] = {}
    errores = []

    for f in archivos:
        activo = _asset_from_filename(getattr(f, "name", "ACTIVO"))
        try:
            df = _read_txt_uploaded(f)
            df.insert(0, "Activo", activo)
            df = _merge_managers(df, df_managers)
            dfs_por_activo[activo] = df
        except Exception as e:
            errores.append((activo, str(e)))

    if errores:
        st.error("Algunos archivos no se pudieron leer:")
        for activo, msg in errores:
            st.write(f"- **{activo}**: {msg}")

    if not dfs_por_activo:
        st.stop()

    # Consolidado
    df_all = pd.concat(dfs_por_activo.values(), ignore_index=True)

    # =========================
    # Filtros globales
    # =========================
    st.markdown("### Filtros (globales)")
    c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])

    with c1:
        activos = sorted(df_all["Activo"].dropna().astype(str).unique().tolist())
        activos_sel = st.multiselect("Activo", options=["Todos"] + activos, default=["Todos"])

    with c2:
        oficiales = sorted([x for x in df_all.get("Oficial", pd.Series([])).dropna().astype(str).unique().tolist() if x.strip() and x.strip() != "---"])
        oficiales_sel = st.multiselect("Oficial", options=["Todos"] + oficiales, default=["Todos"])

    with c3:
        productores = sorted([x for x in df_all.get("Productor", pd.Series([])).dropna().astype(str).unique().tolist() if x.strip() and x.strip() != "---"])
        productores_sel = st.multiselect("Productor", options=["Todos"] + productores, default=["Todos"])

    with c4:
        acciones = sorted([x for x in df_all.get("Acciones", pd.Series([])).dropna().astype(str).unique().tolist() if x.strip()])
        acciones_sel = st.multiselect("Acciones", options=["Todas"] + acciones, default=["Todas"])

    df_f = df_all.copy()

    if "Todos" not in activos_sel:
        df_f = df_f[df_f["Activo"].astype(str).isin(activos_sel)]

    if "Oficial" in df_f.columns and "Todos" not in oficiales_sel:
        df_f = df_f[df_f["Oficial"].astype(str).isin(oficiales_sel)]

    if "Productor" in df_f.columns and "Todos" not in productores_sel:
        df_f = df_f[df_f["Productor"].astype(str).isin(productores_sel)]

    if "Acciones" in df_f.columns and "Todas" not in acciones_sel:
        df_f = df_f[df_f["Acciones"].astype(str).isin(acciones_sel)]

    # =========================
    # Resumen
    # =========================
    st.markdown("### Resumen")
    r1, r2 = st.columns([0.5, 0.5])

    # Resumen por Activo
    with r1:
        st.markdown("**Por Activo**")
        cols_sum = [c for c in ["Saldo Tesoro", "Saldo Caja Val.", "Ob"] if c in df_f.columns]
        if cols_sum:
            agg_activo = df_f.groupby("Activo", as_index=False)[cols_sum].sum(numeric_only=True)
            st.dataframe(agg_activo, use_container_width=True, hide_index=True)
        else:
            st.info("No encontré columnas numéricas (Saldo Tesoro / Saldo Caja Val. / Ob) para sumar.")

    # Resumen por Oficial/Productor (si existen)
    with r2:
        st.markdown("**Por Oficial / Productor**")
        if "Oficial" in df_f.columns and "Productor" in df_f.columns:
            cols_sum = [c for c in ["Saldo Tesoro", "Saldo Caja Val.", "Ob"] if c in df_f.columns]
            if cols_sum:
                agg_mgr = (
                    df_f.groupby(["Oficial", "Productor"], as_index=False)[cols_sum]
                    .sum(numeric_only=True)
                    .sort_values(cols_sum[0], ascending=False)
                )
                st.dataframe(agg_mgr, use_container_width=True, hide_index=True)
            else:
                st.info("No hay columnas numéricas para resumir.")
        else:
            st.info("Managers no cargados o columnas (Oficial/Productor) no disponibles.")

    # =========================
    # Tablas
    # =========================
    st.markdown("### Tablas por Activo")
    tabs = st.tabs(sorted(dfs_por_activo.keys()))

    for tab_name, tab in zip(sorted(dfs_por_activo.keys()), tabs):
        with tab:
            df_tab = dfs_por_activo[tab_name].copy()

            # Aplicar también los filtros globales sobre este activo
            # (si el usuario filtró a un set distinto, lo reflejamos)
            df_tab = df_tab[df_tab["Cliente"].isin(df_f["Cliente"])] if "Cliente" in df_f.columns else df_tab

            # Columnas recomendadas para mostrar
            prefer = [
                "Activo", "Cliente", "Nombre del Cliente",
                "Saldo Tesoro", "Saldo Caja Val.", "Ob", "Fecha",
                "Nombre", "Oficial", "Productor", "Acciones",
            ]
            cols_show = [c for c in prefer if c in df_tab.columns] + [c for c in df_tab.columns if c not in prefer]
            st.dataframe(df_tab[cols_show], use_container_width=True, hide_index=True)

    st.markdown("### Consolidado (todo junto)")
    prefer_all = [
        "Activo", "Cliente", "Nombre del Cliente",
        "Saldo Tesoro", "Saldo Caja Val.", "Ob", "Fecha",
        "Nombre", "Oficial", "Productor", "Acciones",
    ]
    cols_show_all = [c for c in prefer_all if c in df_f.columns] + [c for c in df_f.columns if c not in prefer_all]
    st.dataframe(df_f[cols_show_all], use_container_width=True, hide_index=True)

    # =========================
    # Export
    # =========================
    st.markdown("### Exportar")
    sheets = {"Consolidado": df_f}
    for activo, df in dfs_por_activo.items():
        sheets[f"{activo}"] = df

    xlsx_bytes = _to_excel_bytes(sheets)
    st.download_button(
        "📥 Descargar Excel (todas las tablas)",
        data=xlsx_bytes,
        file_name="vencimientos_por_activo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

