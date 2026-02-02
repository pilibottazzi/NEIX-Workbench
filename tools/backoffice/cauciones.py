# tools/cauciones.py
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st


# =========================
# Utils
# =========================
def _status(msg: str, kind: str = "info"):
    if kind == "success":
        st.success(msg)
    elif kind == "error":
        st.error(msg)
    elif kind == "warning":
        st.warning(msg)
    else:
        st.info(msg)


def extraer_numeros_instrumento(x) -> str:
    """
    Replica JS:
    - Si viene "(1234)" extrae 1234
    - Sino se queda con solo dígitos
    - Quita ceros a la izquierda
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"\((\d+)\)", s)
    if m:
        num = m.group(1)
    else:
        num = re.sub(r"\D", "", s)
    return num.lstrip("0")


def _to_int_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # elimina .0 típico de excel
    s = s.replace(".0", "")
    # solo dígitos
    s2 = re.sub(r"\D", "", s)
    return s2.lstrip("0") or s2 or s


def _read_excel_first_sheet(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, sheet_name=0, header=None)


def _find_header_row(df_raw: pd.DataFrame, must_have=("Comitente",)):
    """
    Busca una fila que contenga al menos alguno de los headers esperados.
    Devuelve índice de fila.
    """
    must = [m.lower() for m in must_have]
    for i in range(min(50, len(df_raw))):
        row = df_raw.iloc[i].astype(str).fillna("").tolist()
        joined = " | ".join([c.strip() for c in row]).lower()
        if all(m in joined for m in must):
            return i
    return 0


def _build_df_with_header(df_raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    header = df_raw.iloc[header_row].astype(str).str.strip().tolist()
    data = df_raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")
    return data


def _safe_float(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    # estilo AR: 1.234.567 -> 1234567
    s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


# =========================
# Core logic
# =========================
def procesar_garantias(xls_file) -> dict:
    """
    Lee hoja 'Resultado', arma map (comitente|codigo) -> disponible
    """
    df = pd.read_excel(xls_file, sheet_name="Resultado")
    df.columns = df.columns.astype(str).str.strip()

    required = ["Comitente", "Instrumento", "Disponible"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Garantías: faltan columnas {missing}. Columnas: {list(df.columns)}")

    out = {}
    for _, r in df.iterrows():
        com = _to_int_str(r["Comitente"])
        if not com:
            continue
        codigo = extraer_numeros_instrumento(r["Instrumento"])
        disp = _safe_float(r["Disponible"])
        key = f"{com}|{codigo}"
        out[key] = out.get(key, 0.0) + disp
    return out


def procesar_nasdaq(csv_file) -> tuple[dict, pd.DataFrame]:
    """
    NASDAQ: map (cuenta|instrumento) -> saldo (sumado)
    Además devuelve DF limpiado para exportar "Tenencias NASDAQ"
    """
    # PapaParse usaba encoding ISO-8859-1
    raw = csv_file.getvalue()
    try:
        text = raw.decode("ISO-8859-1")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    # intenta delim , (según tu JS)
    df = pd.read_csv(io.StringIO(text), sep=",", dtype=str).fillna("")
    # normaliza headers
    df.columns = (
        df.columns.astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.strip()
    )

    # columnas esperadas
    col_cuenta = None
    col_inst = None
    col_saldo = None
    for c in df.columns:
        lc = c.lower()
        if "numero de cuenta" in lc or "número de cuenta" in lc:
            col_cuenta = c
        if lc.strip() == "instrumento":
            col_inst = c
        if "saldo disponible" in lc:
            col_saldo = c

    if not (col_cuenta and col_inst and col_saldo):
        raise ValueError(f"NASDAQ: no encuentro columnas. Tengo: {list(df.columns)}")

    # map de cauciones (sum)
    out = {}
    for _, r in df.iterrows():
        cuenta = str(r.get(col_cuenta, "")).strip()
        cuenta = re.sub(r"^(7142/|142/|5992/)", "", cuenta).strip()
        cuenta = _to_int_str(cuenta)

        inst = _to_int_str(r.get(col_inst, ""))
        saldo = _safe_float(r.get(col_saldo, ""))

        if not cuenta or not inst or inst == "0":
            continue
        key = f"{cuenta}|{inst}"
        out[key] = out.get(key, 0.0) + saldo

    # limpieza para hoja "Tenencias NASDAQ" (igual a tu JS)
    columnas_eliminar = {
        "Tipo de cuenta", "Nombre del titular", "ID titular", "ISIN",
        "Central depositaria emisora", "Saldo liquidado total",
        "Saldo pendiente de entrega", "Saldo bloqueado", "Tipo de saldo",
        "Cuenta con restriccion", "Valida desde", "Valido hasta"
    }

    keep_cols = [c for c in df.columns if c.strip() not in columnas_eliminar]
    df_clean = df[keep_cols].copy()

    # convertir tipos de 3 columnas principales si existen en A,B,C
    # para que al exportar quede prolijo
    if col_cuenta in df_clean.columns:
        df_clean[col_cuenta] = df_clean[col_cuenta].apply(lambda v: _to_int_str(re.sub(r"^(7142/|142/|5992/)", "", str(v))))
    if col_inst in df_clean.columns:
        df_clean[col_inst] = df_clean[col_inst].apply(_to_int_str)
    if col_saldo in df_clean.columns:
        df_clean[col_saldo] = df_clean[col_saldo].apply(_safe_float).round(0).astype("Int64").astype(str)

    # filtrar instrumento != 0
    if col_inst in df_clean.columns:
        df_clean = df_clean[df_clean[col_inst] != "0"]

    return out, df_clean


def procesar_ventas(xlsx_file) -> tuple[dict, pd.DataFrame]:
    """
    Ventas: detecta header row, arma map (comitente|codigo)->ventas
    y devuelve df para exportar.
    """
    df_raw = _read_excel_first_sheet(xlsx_file)
    header_row = _find_header_row(df_raw, must_have=("Comitente",))
    df = _build_df_with_header(df_raw, header_row)
    df.columns = df.columns.astype(str).str.strip()

    # detectar columnas
    col_com = None
    col_cod = None
    col_ven = None
    for c in df.columns:
        lc = c.lower()
        if col_com is None and "comitente" in lc:
            col_com = c
        if col_cod is None and ("cod" in lc or "codigo" in lc):
            col_cod = c
        if col_ven is None and ("vent" in lc):
            col_ven = c

    if not (col_com and col_cod and col_ven):
        raise ValueError(f"VENTAS: no encuentro columnas. Tengo: {list(df.columns)}")

    m = {}
    for _, r in df.iterrows():
        com = _to_int_str(r.get(col_com, ""))
        cod = _to_int_str(r.get(col_cod, ""))
        ven = _safe_float(r.get(col_ven, 0))
        if com and cod:
            m[f"{com}|{cod}"] = m.get(f"{com}|{cod}", 0.0) + ven

    # df exportable con 3 columnas normalizadas
    df_out = pd.DataFrame({
        "Comitente": [k.split("|")[0] for k in m.keys()],
        "Codigo":    [k.split("|")[1] for k in m.keys()],
        "Ventas":    [m[k] for k in m.keys()],
    })

    return m, df_out


def construir_cauciones_df(gar_map: dict, nas_map: dict, ven_map: dict) -> pd.DataFrame:
    # universo garantías + nasdaq (como tu JS)
    keys = set(gar_map.keys()) | set(nas_map.keys())

    rows = []
    for key in keys:
        com, cod = key.split("|", 1)
        # filtro de códigos que empiezan con % * #
        if cod and re.match(r"^[%*#]", str(cod).strip()):
            continue

        rows.append({
            "Comitente": com,
            "Codigo": cod,
            "NETO PARA UTILIZAR": None,  # se arma como fórmula en Excel
            "Utilizado": 0,
            "NASDAQ": float(nas_map.get(key, 0.0) or 0.0),
            "Garantías Disponibles": float(gar_map.get(key, 0.0) or 0.0),
            "Compras": 0,
            "Ventas": float(ven_map.get(key, 0.0) or 0.0),
            "Alquileres": 0,
            "A3": 0,
            "Exterior": 0,
        })

    df = pd.DataFrame(rows)
    # orden opcional
    if not df.empty:
        df = df.sort_values(["Comitente", "Codigo"], kind="stable").reset_index(drop=True)
    return df


def exportar_excel(df_cauc: pd.DataFrame, df_ventas: pd.DataFrame, df_nasdaq_clean: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # --- CAUCIONES
        df_cauc.to_excel(writer, sheet_name="CAUCIONES", index=False, startrow=0)

        wb = writer.book
        ws = writer.sheets["CAUCIONES"]

        # formatos
        fmt_int = wb.add_format({"num_format": "0"})
        fmt_acc = wb.add_format({"num_format": "#,##0_);(#,##0)"})

        # aplica formatos por columnas (A,B enteros; resto contable)
        # 0-based: A=0
        ws.set_column(0, 0, 12, fmt_int)  # Comitente
        ws.set_column(1, 1, 12, fmt_int)  # Codigo
        ws.set_column(2, 10, 18, fmt_acc) # desde NETO...Exterior (contable)

        # fórmula de NETO PARA UTILIZAR (col C) desde fila 2
        # Excel rows: header=1, data starts row=2
        # C = SUM(E:G) - SUM(D, H:K)
        for i in range(len(df_cauc)):
            excel_row = i + 2
            ws.write_formula(
                i + 1, 2,
                f"=SUM(E{excel_row}:G{excel_row})-SUM(D{excel_row},H{excel_row}:K{excel_row})",
                fmt_acc
            )

        # --- VENTAS (exportable)
        if df_ventas is not None and not df_ventas.empty:
            df_ventas.to_excel(writer, sheet_name="VENTAS", index=False)
            ws_v = writer.sheets["VENTAS"]
            ws_v.set_column(0, 1, 12, fmt_int)
            ws_v.set_column(2, 2, 18, fmt_acc)

        # --- Tenencias NASDAQ (limpio)
        if df_nasdaq_clean is not None and not df_nasdaq_clean.empty:
            df_nasdaq_clean.to_excel(writer, sheet_name="Tenencias NASDAQ", index=False)

    return output.getvalue()


# =========================
# Streamlit render
# =========================
def render(back_to_home=None):
    st.markdown("## CAUCIONES")
    st.caption('Subí Garantías (hoja "Resultado"), Tenencias NASDAQ (CSV) y VENTAS (Excel).')

    c1, c2, c3 = st.columns(3)
    with c1:
        f_gar = st.file_uploader("1) Garantías Disponibles (XLS/XLSX)", type=["xls", "xlsx"], key="cau_gar")
    with c2:
        f_nas = st.file_uploader("2) Tenencias NASDAQ (CSV)", type=["csv"], key="cau_nas")
    with c3:
        f_ven = st.file_uploader("3) VENTAS (XLSX)", type=["xlsx"], key="cau_ven")

    if not (f_gar and f_nas and f_ven):
        st.info("Cargá los 3 archivos para habilitar el botón.")
        return

    if st.button("Generar Excel CAUCIONES", type="primary", key="cau_btn"):
        try:
            _status("Procesando archivos…", "info")

            gar_map = procesar_garantias(f_gar)
            nas_map, df_nas_clean = procesar_nasdaq(f_nas)
            ven_map, df_ven_out = procesar_ventas(f_ven)

            df_cauc = construir_cauciones_df(gar_map, nas_map, ven_map)

            hoy = datetime.now().strftime("%d-%m-%Y")
            nombre = f"CAUCIONES {hoy}.xlsx"

            xbytes = exportar_excel(df_cauc, df_ven_out, df_nas_clean)

            _status(f"Archivo generado OK: {nombre}", "success")
            st.download_button(
                "Descargar CAUCIONES.xlsx",
                data=xbytes,
                file_name=nombre,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="cau_download",
            )

            st.divider()
            st.subheader("Preview")
            st.dataframe(df_cauc.head(50), use_container_width=True, hide_index=True)

        except Exception as e:
            _status(f"Error al procesar: {e}", "error")
            st.exception(e)
