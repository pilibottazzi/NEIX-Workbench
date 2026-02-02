# tools/moc_tarde.py
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st

ACCT_FMT = '_-* #,##0_-;_-* (#,##0);_-* "-"??_-;_-@_-'

CTE_CANDS = ["comitente", "nro comitente", "nº comitente", "nro. comitente", "numero comitente", "cliente", "ctte"]
COD_CANDS = ["codigo tv", "codigo", "símbolo", "simbolo", "ticker", "cod tv"]
TOT_CANDS = ["total", "garantia total", "garantias total", "importe", "monto", "saldo", "sdo", "total garantia", "importe total"]
COMP_CANDS = ["compras", "compra"]
VENT_CANDS = ["ventas", "venta"]
NETO_CANDS = ["neto"]
CANT_CANDS = ["cantidad", "cant"]
SALDO_CANDS = ["saldo negativos", "saldo negativo", "saldo", "total", "importe", "monto", "sdo"]

def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_cell(x) -> str:
    s = "" if x is None else str(x)
    s = _strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_code(x) -> str:
    return ("" if x is None else str(x)).strip().upper()


def normalize_comitente(x) -> str:
    raw = re.sub(r"\D", "", "" if x is None else str(x))
    if raw in {"3", "777777777", "888888888"}:
        return "999"
    return raw or ("" if x is None else str(x)).strip()


def to_num(v) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == "":
        return 0.0
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


# ---------------------------
# Lectura de archivos (robusta)
# ---------------------------
def read_any_to_df(uploaded_file) -> pd.DataFrame:
    """
    Lee CSV/XLSX/XLSM. Para XLS (viejo) NO usamos xlrd (no lo queremos en cloud).
    """
    name = (uploaded_file.name or "").lower()

    if name.endswith(".csv"):
        raw = uploaded_file.getvalue()
        text = None
        for enc in ("utf-8", "latin1", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                pass
        if text is None:
            raise ValueError("No pude decodificar el CSV.")

        # detect delimiter por primera línea útil
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        first = ""
        for ln in lines:
            t = ln.strip()
            if not t:
                continue
            if re.fullmatch(r"[;,]+", t):
                continue
            first = ln
            break
        sc = first.count(";")
        cc = first.count(",")
        sep = ";" if sc >= cc else ","

        return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)

    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        return pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")

    if name.endswith(".xls"):
        raise ValueError(
            "Archivo .xls (Excel viejo) no soportado en este entorno.\n"
            "Solución rápida: abrilo y guardalo como .xlsx, y volvé a subirlo."
        )

    return pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")


def detect_header_positions_aoa(aoa: list[list], sets: list[list[str]]):
    max_scan = min(10, len(aoa))
    for r in range(max_scan):
        hdr = aoa[r] or []
        hdr_norm = [norm_cell(x) for x in hdr]

        idxs = []
        ok = True
        for cand_set in sets:
            found = -1
            for i, h in enumerate(hdr_norm):
                for c in cand_set:
                    cc = norm_cell(c)
                    if h == cc or cc in h:
                        found = i
                        break
                if found >= 0:
                    break
            if found < 0:
                ok = False
                break
            idxs.append(found)

        if ok:
            return {"rowIndex": r, "indexes": idxs, "header": hdr}
    return None


def read_to_aoa(uploaded_file) -> list[list]:
    """
    Devuelve array-of-arrays como en el HTML (sheet_to_json header:1).
    """
    name = (uploaded_file.name or "").lower()

    if name.endswith(".csv"):
        df = read_any_to_df(uploaded_file)
        return [df.columns.tolist()] + df.fillna("").values.tolist()

    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        df0 = pd.read_excel(uploaded_file, header=None, dtype=str, engine="openpyxl")
        df0 = df0.fillna("")
        return df0.values.tolist()

    if name.endswith(".xls"):
        raise ValueError(
            "Este archivo es .xls (Excel viejo). Guardalo como .xlsx (Guardar como) y volvé a subirlo."
        )

    df0 = pd.read_excel(uploaded_file, header=None, dtype=str, engine="openpyxl")
    df0 = df0.fillna("")
    return df0.values.tolist()


# ---------------------------
# MAE Map (comitente||codigo -> compra+venta)
# ---------------------------
def build_mae_map(aoa_mae: list[list]) -> dict:
    mp = {}
    if not aoa_mae:
        return mp

    header_row_idx = 0
    for i in range(min(20, len(aoa_mae))):
        row = aoa_mae[i] or []
        hc = [norm_cell(x) for x in row]
        if ("comitente" in hc and "codigo" in hc and any(x in hc for x in ["compra", "compras"]) and any(x in hc for x in ["venta", "ventas"])):
            header_row_idx = i
            break

    sliced = aoa_mae[header_row_idx:]
    if not sliced:
        return mp

    hdr = sliced[0] or []
    hdrn = [norm_cell(x) for x in hdr]

    def idx_of(cands):
        for i, h in enumerate(hdrn):
            for c in cands:
                cc = norm_cell(c)
                if h == cc or cc in h:
                    return i
        return -1

    i_com = idx_of(["comitente"])
    i_cod = idx_of(["codigo"])
    i_comp = idx_of(["compra", "compras"])
    i_vent = idx_of(["venta", "ventas"])
    if min(i_com, i_cod, i_comp, i_vent) < 0:
        return mp

    for r in range(1, len(sliced)):
        row = sliced[r] or []
        com = normalize_comitente(row[i_com] if i_com < len(row) else "")
        cod = normalize_code(row[i_cod] if i_cod < len(row) else "")
        if not com or not cod:
            continue
        compra = to_num(row[i_comp] if i_comp < len(row) else 0)
        venta = to_num(row[i_vent] if i_vent < len(row) else 0)
        key = f"{com}||{cod}"
        mp[key] = mp.get(key, 0.0) + (compra + venta)

    return mp

def _apply_col_widths(ws, widths: dict):
    """
    widths: {"A": 14, "B": 12, ...}
    """
    try:
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
    except Exception:
        # si algo cambia en engine, no rompemos la generación
        pass


def generate_moc_tarde(f_moc, f_mae, f_saldos, f_byma):
    # MAE (opcional)
    mae_map = {}
    has_mae = False
    if f_mae is not None:
        try:
            aoa_mae = read_to_aoa(f_mae)
            mae_map = build_mae_map(aoa_mae)
            has_mae = len(mae_map) > 0
        except Exception:
            has_mae = False
            mae_map = {}

    # --- MOC ---
    wb_moc_aoa = read_to_aoa(f_moc)
    moc_pos = detect_header_positions_aoa(wb_moc_aoa, [CTE_CANDS, COD_CANDS, COMP_CANDS, VENT_CANDS, NETO_CANDS])
    if not moc_pos:
        raise ValueError("No se encontraron encabezados esperados en MOC.")

    hrm = moc_pos["rowIndex"]
    i_cte, i_cod, i_comp, i_vent, _i_neto = moc_pos["indexes"]

    moc_cols = ["COMITENTE", "CODIGO", "COMPRAS", "VENTAS", "MAE", "NETO"] if has_mae else ["COMITENTE", "CODIGO", "COMPRAS", "VENTAS", "NETO"]
    moc_rows = []

    for r in range(hrm + 1, len(wb_moc_aoa)):
        row = wb_moc_aoa[r] or []
        com = normalize_comitente(row[i_cte] if i_cte < len(row) else "")
        cod = normalize_code(row[i_cod] if i_cod < len(row) else "")
        comp = row[i_comp] if i_comp < len(row) else ""
        vent = row[i_vent] if i_vent < len(row) else ""

        if has_mae:
            mae_val = mae_map.get(f"{com}||{cod}", 0.0)
            moc_rows.append([com, cod, comp, vent, mae_val, ""])
        else:
            moc_rows.append([com, cod, comp, vent, ""])

    df_moc = pd.DataFrame(moc_rows, columns=moc_cols)

    # --- BYMA ---
    aoa_byma = read_to_aoa(f_byma)
    byma_pos = detect_header_positions_aoa(aoa_byma, [CTE_CANDS, COD_CANDS, TOT_CANDS])
    if not byma_pos:
        raise ValueError("No se encontraron encabezados esperados en Lista BYMA.")
    hrb = byma_pos["rowIndex"]
    i_cte_b, i_cod_b, i_tot_b = byma_pos["indexes"]

    byma_map = {}
    for r in range(hrb + 1, len(aoa_byma)):
        row = aoa_byma[r] or []
        cte = normalize_comitente(row[i_cte_b] if i_cte_b < len(row) else "")
        cod = normalize_code(row[i_cod_b] if i_cod_b < len(row) else "")
        tot = to_num(row[i_tot_b] if i_tot_b < len(row) else 0)
        key = f"{cte}||{cod}"
        byma_map[key] = byma_map.get(key, 0.0) + tot

    # --- SALDOS NEGATIVOS ---
    aoa_sal = read_to_aoa(f_saldos)
    sal_pos = detect_header_positions_aoa(aoa_sal, [CTE_CANDS, COD_CANDS, NETO_CANDS, CANT_CANDS, SALDO_CANDS])
    if not sal_pos:
        raise ValueError("No se encontraron encabezados esperados en Saldos Negativos.")
    hrs = sal_pos["rowIndex"]
    i_cte_s, i_cod_s, i_neto_s, i_cant_s, i_saldo_s = sal_pos["indexes"]

    rows = []
    for r in range(hrs + 1, len(aoa_sal)):
        base = aoa_sal[r] or []
        cte = normalize_comitente(base[i_cte_s] if i_cte_s < len(base) else "")
        cod = normalize_code(base[i_cod_s] if i_cod_s < len(base) else "")
        neto = to_num(base[i_neto_s] if i_neto_s < len(base) else 0)
        cant = to_num(base[i_cant_s] if i_cant_s < len(base) else 0)
        saldo = to_num(base[i_saldo_s] if i_saldo_s < len(base) else 0)
        byma = byma_map.get(f"{cte}||{cod}", 0.0)
        otros = mae_map.get(f"{cte}||{cod}", 0.0) if has_mae else 0.0

        total_num = saldo + byma + otros
        rows.append([cte, cod, neto, cant, saldo, byma, otros, total_num])

    df_sal = pd.DataFrame(
        rows,
        columns=[
            "COMITENTE",
            "CODIGO",
            "NETO",
            "CANTIDAD",
            "SALDO NEGATIVOS",
            "Garantías BYMA (Total)",
            "OTROS",
            "_TOTALNUM",
        ],
    ).sort_values("_TOTALNUM", ascending=True).reset_index(drop=True)

    # armamos columnas finales iguales al HTML (TOTAL y Observación como fórmula)
    df_sal["TOTAL"] = ""
    df_sal["Observación"] = ""
    df_sal = df_sal.drop(columns=["_TOTALNUM"])

    # --- Excel output (openpyxl) ---
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_moc.to_excel(writer, sheet_name="MOC tarde", index=False)
        df_sal.to_excel(writer, sheet_name="SALDOS NEGATIVOS tarde", index=False)

        ws1 = writer.sheets["MOC tarde"]
        ws2 = writer.sheets["SALDOS NEGATIVOS tarde"]

        # Anchos (opcional, no rompe)
        if has_mae:
            _apply_col_widths(ws1, {"A": 12, "B": 12, "C": 14, "D": 14, "E": 14, "F": 14})
        else:
            _apply_col_widths(ws1, {"A": 12, "B": 12, "C": 14, "D": 14, "E": 14})
        _apply_col_widths(ws2, {"A": 12, "B": 12, "C": 14, "D": 12, "E": 16, "F": 18, "G": 12, "H": 14, "I": 14})

        # Fórmulas NETO en MOC (igual HTML)
        nrows_moc = len(df_moc)
        if has_mae:
            # columnas: A COMITENTE, B CODIGO, C COMPRAS, D VENTAS, E MAE, F NETO
            for i in range(2, nrows_moc + 2):  # Excel row
                ws1[f"F{i}"].value = f"=C{i}+D{i}+E{i}"
        else:
            # columnas: A COMITENTE, B CODIGO, C COMPRAS, D VENTAS, E NETO
            for i in range(2, nrows_moc + 2):
                ws1[f"E{i}"].value = f"=C{i}+D{i}"

        # Fórmulas TOTAL + Observación en SALDOS (igual HTML)
        nrows_sal = len(df_sal)
        # columnas: A COMITENTE, B CODIGO, C NETO, D CANTIDAD, E SALDO, F BYMA, G OTROS, H TOTAL, I OBS
        for i in range(2, nrows_sal + 2):
            ws2[f"H{i}"].value = f"=E{i}+F{i}+G{i}"
            ws2[f"I{i}"].value = f'=IF(H{i}>=0,"OK","REVISAR")'

    out.seek(0)
    return out.getvalue(), has_mae


def generate_ventas(f_moc, f_mae):
    mae_map = {}
    has_mae = False
    if f_mae is not None:
        try:
            aoa_mae = read_to_aoa(f_mae)
            mae_map = build_mae_map(aoa_mae)
            has_mae = len(mae_map) > 0
        except Exception:
            has_mae = False
            mae_map = {}

    aoa_moc = read_to_aoa(f_moc)
    moc_pos = detect_header_positions_aoa(aoa_moc, [CTE_CANDS, COD_CANDS, COMP_CANDS, VENT_CANDS, NETO_CANDS])
    if not moc_pos:
        raise ValueError("No se encontraron encabezados esperados en MOC.")

    hrm = moc_pos["rowIndex"]
    i_cte, i_cod, _i_comp, i_vent, _i_neto = moc_pos["indexes"]

    rows = []
    for r in range(hrm + 1, len(aoa_moc)):
        row = aoa_moc[r] or []
        cte = normalize_comitente(row[i_cte] if i_cte < len(row) else "")
        cod = normalize_code(row[i_cod] if i_cod < len(row) else "")
        vent = to_num(row[i_vent] if i_vent < len(row) else 0)
        if vent == 0:
            continue
        mae_val = mae_map.get(f"{cte}||{cod}", 0.0) if has_mae else 0.0
        if has_mae:
            rows.append([cte, cod, vent, mae_val, ""])
        else:
            rows.append([cte, cod, vent, ""])

    cols = ["COMITENTE", "CODIGO", "VENTAS", "MAE", "NETO"] if has_mae else ["COMITENTE", "CODIGO", "VENTAS", "NETO"]
    df = pd.DataFrame(rows, columns=cols)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="VENTAS", index=False)
        ws = writer.sheets["VENTAS"]

        # anchos (opcional)
        if has_mae:
            _apply_col_widths(ws, {"A": 12, "B": 12, "C": 14, "D": 14, "E": 14})
        else:
            _apply_col_widths(ws, {"A": 12, "B": 12, "C": 14, "D": 14})

        # fórmula NETO (igual HTML)
        n = len(df)
        if has_mae:
            # A COM, B COD, C VENTAS, D MAE, E NETO
            for i in range(2, n + 2):
                ws[f"E{i}"].value = f"=C{i}+D{i}"
        else:
            # A COM, B COD, C VENTAS, D NETO
            for i in range(2, n + 2):
                ws[f"D{i}"].value = f"=C{i}"

    out.seek(0)
    return out.getvalue(), has_mae

def render(back_to_home=None):
    st.markdown("## MOC TARDE")
    st.caption("Genera el Excel de MOC tarde + un export de Ventas.")

    c1, c2 = st.columns(2)
    f_moc = c1.file_uploader("MOC Dashboard", type=["xlsx", "xls", "csv"], key="moc_file")
    f_mae = c2.file_uploader("MOC MAE GALLO (opcional)", type=["csv"], key="mae_file")

    c3, c4 = st.columns(2)
    f_saldos = c3.file_uploader("NEGATIVOS Dashboard", type=["xlsx", "xls", "csv"], key="neg_file")
    f_byma = c4.file_uploader("Lista de Saldos BYMA", type=["xlsx", "xls", "csv"], key="byma_file")

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        if st.button("Generar MOC TARDE", type="primary", key="btn_moc_tarde_run"):
            if not (f_moc and f_saldos and f_byma):
                st.error("Faltan archivos: cargá MOC, Saldos Negativos y Lista BYMA. (MAE es opcional)")
            else:
                try:
                    if f_mae is None:
                        st.warning("MOC MAE no adjuntado: se continúa SIN MAE (sin merge).")
                    with st.spinner("Procesando..."):
                        xls_bytes, has_mae = generate_moc_tarde(f_moc, f_mae, f_saldos, f_byma)
                    hoy = datetime.now().strftime("%d-%m-%Y")
                    name = f"MOC TARDE {hoy}.xlsx"
                    st.success(f"OK. Generado {'con' if has_mae else 'sin'} MAE.")
                    st.download_button(
                        "Descargar MOC TARDE",
                        data=xls_bytes,
                        file_name=name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_moc_tarde",
                    )
                except Exception as e:
                    st.error("Error generando MOC TARDE.")
                    st.exception(e)

    with colB:
        if st.button("Generar VENTAS", key="btn_moc_tarde_ventas"):
            if not f_moc:
                st.error("Falta archivo MOC.")
            else:
                try:
                    if f_mae is None:
                        st.warning("MOC MAE no adjuntado: se continúa SIN MAE (sin merge).")
                    with st.spinner("Procesando ventas..."):
                        xls_bytes, has_mae = generate_ventas(f_moc, f_mae)
                    hoy = datetime.now().strftime("%d-%m-%Y")
                    name = f"VENTAS TARDE {hoy}.xlsx"
                    st.success(f"OK. Ventas generado {'con' if has_mae else 'sin'} MAE.")
                    st.download_button(
                        "Descargar VENTAS TARDE",
                        data=xls_bytes,
                        file_name=name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_moc_tarde_ventas",
                    )
                except Exception as e:
                    st.error("Error generando VENTAS.")
                    st.exception(e)
