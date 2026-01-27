# tools/bo_ppt_manana.py
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers


# =========================
# Utils
# =========================
def canon(x) -> str:
    s = "" if x is None else str(x)
    s = s.strip().upper()
    # sacar acentos básico
    s = (
        s.replace("Á", "A").replace("É", "E").replace("Í", "I")
        .replace("Ó", "O").replace("Ú", "U").replace("Ü", "U")
        .replace("Ñ", "N")
    )
    s = re.sub(r"\s+", " ", s)
    return s


def coerce_number(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    # 1.234.567 -> 1234567 ; 1,23 -> 1.23
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"\s+", "", s)
    try:
        n = float(s)
        return n
    except Exception:
        return None


def format_date_short(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return ""
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if hasattr(v, "strftime"):
        return v.strftime("%d/%m/%Y")
    # si ya viene string dd/mm/yyyy
    s = str(v)
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return s


def find_header_row(df_raw: pd.DataFrame, must=("COMITENTE",)) -> int:
    must = [m.lower() for m in must]
    max_scan = min(80, len(df_raw))
    for i in range(max_scan):
        row = df_raw.iloc[i].astype(str).fillna("").tolist()
        joined = " | ".join([c.strip() for c in row]).lower()
        if all(m in joined for m in must):
            return i
    return 0


def df_from_header_row(df_raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    header = df_raw.iloc[header_row].astype(str).str.strip().tolist()
    data = df_raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")
    return data


def read_excel_any(uploaded, sheet_name=0) -> pd.DataFrame:
    return pd.read_excel(uploaded, sheet_name=sheet_name, header=None)


def read_excel_sheet(uploaded, sheet_name=0) -> pd.DataFrame:
    return pd.read_excel(uploaded, sheet_name=sheet_name)


def read_csv_robusto(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    # tu JS leía como texto y detectaba ; vs ,
    for enc in ("utf-8-sig", "ISO-8859-1", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = raw.decode("utf-8", errors="replace")

    # detectar delimitador por primera línea útil
    lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
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
    delim = ";" if sc >= cc else ","

    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df


def write_aoa(ws, aoa):
    for r_idx, row in enumerate(aoa, start=1):
        for c_idx, val in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=val)


def freeze_header(ws):
    ws.freeze_panes = "A2"


def set_accounting(ws, header, colnames):
    if not header:
        return
    H = [canon(h) for h in header]
    targets = []
    for name in colnames:
        try:
            targets.append(H.index(canon(name)) + 1)
        except ValueError:
            pass
    if not targets:
        return
    acc = numbers.FORMAT_NUMBER_COMMA_SEPARATED1
    for col in targets:
        for r in range(2, ws.max_row + 1):
            cell = ws.cell(r, col)
            if isinstance(cell.value, (int, float)) or (isinstance(cell.value, str) and cell.value and cell.value.replace(".", "").replace(",", "").isdigit()):
                cell.number_format = acc


# =========================
# Transformaciones (equivalente JS)
# =========================
def transform_MOC_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    # withHeaderRow2: usa fila 2 como header (index 1)
    if len(df_raw) < 2:
        return pd.DataFrame()
    df = df_raw.copy()
    header = df.iloc[1].astype(str).str.strip().tolist()
    body = df.iloc[2:].copy()
    body.columns = header
    body = body.dropna(how="all")
    H = [canon(c) for c in body.columns]

    def idx(name): 
        try: return H.index(canon(name))
        except ValueError: return -1

    i_cod = idx("CODIGO")
    i_com = idx("COMITENTE")
    i_comp = idx("COMPRAS")
    i_vent = idx("VENTAS")

    out = []
    for _, r in body.iterrows():
        com = coerce_number(r.iloc[i_com]) if i_com >= 0 else None
        cod = r.iloc[i_cod] if i_cod >= 0 else ""
        comp = coerce_number(r.iloc[i_comp]) if i_comp >= 0 else None
        vent = coerce_number(r.iloc[i_vent]) if i_vent >= 0 else None
        out.append([com, cod, comp, vent, 0, 0, ""])

    return pd.DataFrame(out, columns=["Comitente", "Codigo", "Compras", "Ventas", "MAE", "Neto", "Observaciones"])


def build_mae_map(df_mae: pd.DataFrame) -> dict:
    # busca headers COMITENTE CODIGO COMPRA VENTA (en cualquier fila)
    df_raw = df_mae.copy()
    # si viene con headers correctos ya, lo usamos
    cols = [canon(c) for c in df_raw.columns]
    needed = {"COMITENTE", "CODIGO", "COMPRA", "VENTA"}
    if not needed.issubset(set(cols)):
        # intentar: tomar primera fila como header en vez de columns
        # (si csv venía con basura arriba, se complica; lo resolvemos buscando la fila)
        raw = df_mae.copy()
        # fallback: si no están, devolvemos vacío
        return {}

    # map key: comitente|CODIGO_CANON
    c_com = df_raw.columns[cols.index("COMITENTE")]
    c_cod = df_raw.columns[cols.index("CODIGO")]
    c_comp = df_raw.columns[cols.index("COMPRA")]
    c_vent = df_raw.columns[cols.index("VENTA")]

    m = {}
    for _, r in df_raw.iterrows():
        com = coerce_number(r[c_com])
        cod_raw = r[c_cod]
        if com is None or cod_raw in (None, ""):
            continue
        key = f"{int(com)}|{canon(cod_raw)}"
        compra = coerce_number(r[c_comp]) or 0
        venta = coerce_number(r[c_vent]) or 0
        mae = compra + venta
        m[key] = m.get(key, 0) + mae
    return m


def apply_mae_to_moc(df_moc: pd.DataFrame, mae_map: dict) -> pd.DataFrame:
    if df_moc.empty:
        return df_moc
    df = df_moc.copy()
    maes = []
    netos = []
    for _, r in df.iterrows():
        com = r["Comitente"]
        cod = r["Codigo"]
        key = f"{int(com) if pd.notna(com) and com is not None else ''}|{canon(cod)}"
        mae = mae_map.get(key, 0) if mae_map else 0
        maes.append(mae)
        comp = r["Compras"] if pd.notna(r["Compras"]) and r["Compras"] is not None else 0
        vent = r["Ventas"] if pd.notna(r["Ventas"]) and r["Ventas"] is not None else 0
        netos.append((comp or 0) + (vent or 0) + (mae or 0))
    df["MAE"] = maes
    df["Neto"] = netos
    return df


def transform_NEG_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    if len(df_raw) < 2:
        return pd.DataFrame()
    header = df_raw.iloc[1].astype(str).str.strip().tolist()
    body = df_raw.iloc[2:].copy()
    body.columns = header
    body = body.dropna(how="all")
    H = [canon(c) for c in body.columns]

    def idx(name):
        try: return H.index(canon(name))
        except ValueError: return -1

    i_cod = idx("CODIGO")
    i_com = idx("COMITENTE")
    i_cant = idx("CANTIDAD")
    i_neto = idx("NETO")
    i_sneg = idx("SALDO NEGATIVOS")

    rows = []
    for _, r in body.iterrows():
        rows.append([
            coerce_number(r.iloc[i_com]) if i_com >= 0 else None,
            coerce_number(r.iloc[i_cod]) if i_cod >= 0 else None,
            coerce_number(r.iloc[i_neto]) if i_neto >= 0 else None,
            coerce_number(r.iloc[i_cant]) if i_cant >= 0 else None,
            coerce_number(r.iloc[i_sneg]) if i_sneg >= 0 else None,
        ])
    return pd.DataFrame(rows, columns=["Comitente", "Codigo", "Neto", "Cantidad", "Saldo Negativos"])


def transform_ASIGN(aoa_df_raw: pd.DataFrame) -> pd.DataFrame:
    # equivalente REQUIRED_ASIG y agrega Observacion
    REQUIRED = [
        "Nro. Mercado Corto","Moneda","Cantidad","Monto","Fecha Concertación","Fecha Liquidación","CUIT",
        "Negociación","Participante contraparte","Compra - Venta","Código Especie","Código CV","Código Liquidación"
    ]
    raw = aoa_df_raw.copy()
    # toma primera fila como header real
    header_row = 0
    df = df_from_header_row(raw, header_row)
    df.columns = [c.strip() for c in df.columns]

    # map columnas por canon
    H = [canon(c) for c in df.columns]
    out = {}
    for req in REQUIRED:
        key = canon(req)
        if key in H:
            out[req] = df.iloc[:, H.index(key)]
        else:
            out[req] = ""

    out_df = pd.DataFrame(out)
    out_df["Observacion"] = ""

    # numeric + dates como tu JS
    for col in ["Cantidad", "Monto", "Código CV", "Código Liquidación"]:
        out_df[col] = out_df[col].apply(coerce_number)

    for col in ["Fecha Concertación", "Fecha Liquidación"]:
        out_df[col] = out_df[col].apply(format_date_short)

    return out_df


def build_byma_map(df_raw: pd.DataFrame) -> dict:
    # withHeaderRow2: usa fila 2 como header
    if len(df_raw) < 2:
        return {}
    header = df_raw.iloc[1].astype(str).str.strip().tolist()
    body = df_raw.iloc[2:].copy()
    body.columns = header
    body = body.dropna(how="all")
    H = [canon(c) for c in body.columns]

    def idx(name):
        try: return H.index(canon(name))
        except ValueError: return -1

    i_com = idx("COMITENTE")
    i_cod = idx("CODIGO TV")
    i_tot = idx("TOTAL")

    def map_comitente(c):
        n = coerce_number(c)
        if n in (3, 777777777, 888888888):
            return 999
        return int(n) if n is not None else None

    m = {}
    for _, r in body.iterrows():
        com = map_comitente(r.iloc[i_com]) if i_com >= 0 else None
        cod = coerce_number(r.iloc[i_cod]) if i_cod >= 0 else None
        tot = coerce_number(r.iloc[i_tot]) if i_tot >= 0 else 0
        if com is None or cod is None:
            continue
        key = f"{com}|{int(cod)}"
        m[key] = m.get(key, 0) + (tot or 0)
    return m


def merge_byma_to_neg(df_neg: pd.DataFrame, byma_map: dict, mae_map: dict, use_mae_for_otros: bool) -> pd.DataFrame:
    if df_neg.empty:
        return df_neg
    df = df_neg.copy()
    garantias = []
    otros = []
    total = []
    obs = []

    for _, r in df.iterrows():
        com = int(r["Comitente"]) if pd.notna(r["Comitente"]) and r["Comitente"] is not None else None
        cod = int(r["Codigo"]) if pd.notna(r["Codigo"]) and r["Codigo"] is not None else None

        byma = byma_map.get(f"{com}|{cod}", 0) if (byma_map and com is not None and cod is not None) else 0
        saldo = r["Saldo Negativos"] if pd.notna(r["Saldo Negativos"]) and r["Saldo Negativos"] is not None else 0

        o = 0
        if use_mae_for_otros and mae_map and com is not None:
            # match por comitente y codigo canon (como tu JS)
            key_mae = f"{com}|{canon(cod)}"
            if key_mae in mae_map:
                o = mae_map[key_mae]

        t = (saldo or 0) + (byma or 0) + (o or 0)

        garantias.append(byma)
        otros.append(o)
        total.append(t)
        obs.append("OK" if t >= 0 else "REVISAR")

    df["Garantías BYMA"] = garantias
    df["OTROS"] = otros
    df["Total"] = total
    df["Observación"] = obs
    return df


def build_neix_map(df_raw: pd.DataFrame) -> dict:
    # withHeaderRow2
    if len(df_raw) < 2:
        return {}
    header = df_raw.iloc[1].astype(str).str.strip().tolist()
    body = df_raw.iloc[2:].copy()
    body.columns = header
    body = body.dropna(how="all")
    H = [canon(c) for c in body.columns]

    def idx(name):
        try: return H.index(canon(name))
        except ValueError: return -1

    i_code = idx("CSVA CODE")
    i_sym  = idx("SYMBOL")
    i_qty  = idx("QTY BYMA")
    i_tipo = idx("TIPO")
    i_op   = idx("OPERACION")

    m = {}
    for _, r in body.iterrows():
        code = coerce_number(r.iloc[i_code]) if i_code >= 0 else None
        if code is None:
            continue
        code = int(code)
        tipo = canon(r.iloc[i_tipo]) if i_tipo >= 0 else ""
        op = canon(r.iloc[i_op]) if i_op >= 0 else ""

        allowed = (tipo == "CEDEAR" and op == "ISSUE") or (tipo == "ADR" and op == "CXL")
        qty_src = coerce_number(r.iloc[i_qty]) if i_qty >= 0 else 0
        qty_row = abs(qty_src or 0) if allowed else 0

        prev = m.get(code, {"symbol": "", "qty": 0})
        sym = prev["symbol"] or (r.iloc[i_sym] if i_sym >= 0 else "")
        m[code] = {"symbol": sym, "qty": prev["qty"] + qty_row}
    return m


def apply_neix_to_neg992(df_neg992: pd.DataFrame, neix_map: dict) -> pd.DataFrame:
    if df_neg992.empty:
        return df_neg992
    df = df_neg992.copy()
    sym = []
    qty = []
    for _, r in df.iterrows():
        cod = int(r["Codigo"]) if pd.notna(r["Codigo"]) and r["Codigo"] is not None else None
        item = neix_map.get(cod, {"symbol": "", "qty": 0}) if neix_map else {"symbol": "", "qty": 0}
        sym.append(item.get("symbol", ""))
        qty.append(abs(item.get("qty", 0) or 0))
    df["Symbol"] = sym
    df["Qty BYMA"] = qty
    return df


def reorder_neg992(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Comitente","Symbol","Codigo","Neto","Cantidad","Saldo Negativos","Qty BYMA","Garantías BYMA","OTROS","Total","Observación"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def generate_ventas(df_moc: pd.DataFrame) -> pd.DataFrame:
    if df_moc.empty:
        return pd.DataFrame()
    df = df_moc.copy()
    if "Ventas" not in df.columns:
        return pd.DataFrame()
    df = df[df["Ventas"].fillna(0) != 0]
    if df.empty:
        return pd.DataFrame()
    return df[["Comitente", "Codigo", "Ventas"]].copy()


# =========================
# Export Excel
# =========================
def export_definitivo(
    df_moc: pd.DataFrame,
    df_neg: pd.DataFrame,
    df_neg992: pd.DataFrame,
    df_asig: pd.DataFrame | None,
    df_sliq: pd.DataFrame | None,
) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)

    # ---- MOC
    ws_moc = wb.create_sheet("MOC")
    write_aoa(ws_moc, [df_moc.columns.tolist()] + df_moc.values.tolist())
    freeze_header(ws_moc)

    # fórmula Neto = Compras + Ventas + MAE (igual que tu JS)
    H = [canon(h) for h in df_moc.columns.tolist()]
    if "NETO" in H and "COMPRAS" in H and "VENTAS" in H and "MAE" in H:
        cN = H.index("NETO") + 1
        cC = H.index("COMPRAS") + 1
        cV = H.index("VENTAS") + 1
        cM = H.index("MAE") + 1
        for r in range(2, ws_moc.max_row + 1):
            ws_moc.cell(r, cN).value = f"={get_column_letter(cC)}{r}+{get_column_letter(cV)}{r}+{get_column_letter(cM)}{r}"

    # ---- ASIGNACIONES
    if df_asig is not None and not df_asig.empty:
        ws = wb.create_sheet("ASIGNACIONES_PENDIENTES")
        write_aoa(ws, [df_asig.columns.tolist()] + df_asig.values.tolist())
        freeze_header(ws)

    # ---- SALDOS NEGATIVOS
    ws_neg = wb.create_sheet("SALDOS NEGATIVOS")
    write_aoa(ws_neg, [df_neg.columns.tolist()] + df_neg.values.tolist())
    freeze_header(ws_neg)
    set_accounting(ws_neg, df_neg.columns.tolist(), ["Neto","Cantidad","Saldo Negativos","Garantías BYMA","OTROS","Total"])

    # ---- SALDOS NEGATIVOS 992
    ws_992 = wb.create_sheet("SALDOS NEGATIVOS 992")
    write_aoa(ws_992, [df_neg992.columns.tolist()] + df_neg992.values.tolist())
    freeze_header(ws_992)
    set_accounting(ws_992, df_neg992.columns.tolist(), ["Neto","Cantidad","Saldo Negativos","Qty BYMA","Garantías BYMA","OTROS","Total"])

    # ---- Control SLIQ (si viene)
    if df_sliq is not None and not df_sliq.empty:
        ws = wb.create_sheet("Control SLIQ")
        write_aoa(ws, [df_sliq.columns.tolist()] + df_sliq.values.tolist())
        freeze_header(ws)

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# =========================
# Streamlit render
# =========================
def render(back_to_home=None):
    st.markdown("## ☀️ MOC MAÑANA — Papel de Trabajo Definitivo")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        f_moc = st.file_uploader("MOC (xlsx)", type=["xlsx"], key="ppt_moc")
    with c2:
        f_mae = st.file_uploader("MOC MAE (csv) (opcional)", type=["csv"], key="ppt_mae")
    with c3:
        f_neg = st.file_uploader("Saldos Negativos (xlsx)", type=["xlsx"], key="ppt_neg")
    with c4:
        f_asig = st.file_uploader("Asignaciones Pendientes (opcional)", type=["xls", "xlsx", "xlsb"], key="ppt_asig")

    c5, c6, c7 = st.columns(3)
    with c5:
        f_byma = st.file_uploader("Lista de Saldos BYMA (opcional)", type=["xls", "xlsx", "xlsb"], key="ppt_byma")
    with c6:
        f_992 = st.file_uploader("Emisiones 992 (opcional)", type=["xlsx"], key="ppt_992")
    with c7:
        f_sliq = st.file_uploader("Control SLIQ (opcional)", type=["csv"], key="ppt_sliq")

    # estado en session
    if "ppt_state" not in st.session_state:
        st.session_state["ppt_state"] = {}

    colA, colB, colC = st.columns(3)
    with colA:
        do_validar = st.button("Validar", type="primary", key="ppt_validar")
    with colB:
        do_export = st.button("Exportar MOC", key="ppt_export")
    with colC:
        do_export_ventas = st.button("Exportar VENTAS", key="ppt_export_ventas")

    if do_validar:
        st.session_state["ppt_state"] = {}
        tags = []

        if not f_moc:
            tags.append(("Falta archivo MOC (xlsx)", "error"))
        if not f_neg:
            tags.append(("Falta archivo Saldos Negativos (xlsx)", "error"))

        if tags:
            for t, k in tags:
                (st.error if k == "error" else st.info)(t)
            st.stop()

        # ---- MOC
        try:
            df_raw = read_excel_any(f_moc, sheet_name=0)
            df_moc = transform_MOC_v2(df_raw)
            tags.append(("MOC procesado.", "success"))
        except Exception as e:
            st.error(f"Error leyendo MOC: {e}")
            st.stop()

        # ---- MAE opcional
        mae_map = {}
        if f_mae:
            try:
                df_mae = read_csv_robusto(f_mae)
                mae_map = build_mae_map(df_mae)
                tags.append((f"MOC MAE leído (keys: {len(mae_map)}).", "success"))
            except Exception as e:
                tags.append((f"Error leyendo MOC MAE: {e}", "warning"))
        else:
            tags.append(("MOC MAE no adjuntado: MAE = 0.", "warning"))

        df_moc = apply_mae_to_moc(df_moc, mae_map)

        # ---- NEGATIVOS
        try:
            df_raw = read_excel_any(f_neg, sheet_name=0)
            df_neg_all = transform_NEG_v2(df_raw)
            df_neg_992 = df_neg_all[df_neg_all["Comitente"] == 992].copy()
            df_neg_otros = df_neg_all[df_neg_all["Comitente"] != 992].copy()
            tags.append(("Saldos Negativos procesado (split 992).", "success"))
        except Exception as e:
            st.error(f"Error leyendo Saldos Negativos: {e}")
            st.stop()

        # ---- ASIGNACIONES
        df_asig = None
        if f_asig:
            try:
                df_raw = read_excel_any(f_asig, sheet_name=0)
                df_asig = transform_ASIGN(df_raw)
                tags.append(("Asignaciones procesado.", "success"))
            except Exception as e:
                tags.append((f"Error leyendo Asignaciones: {e}", "warning"))
        else:
            tags.append(("Asignaciones no adjuntado.", "warning"))

        # ---- BYMA + MAE->OTROS solo si match (en OTROS y solo para NEG_OTROS)
        byma_map = {}
        if f_byma:
            try:
                df_raw = read_excel_any(f_byma, sheet_name=0)
                byma_map = build_byma_map(df_raw)
                df_neg_otros = merge_byma_to_neg(df_neg_otros, byma_map, mae_map, True)
                df_neg_992 = merge_byma_to_neg(df_neg_992, byma_map, mae_map, False)
                tags.append(("BYMA merge aplicado.", "success"))
            except Exception as e:
                tags.append((f"Error leyendo/merging BYMA: {e}", "warning"))
        else:
            tags.append(("Lista BYMA no adjuntada: negativos quedan sin BYMA/OTROS/Total.", "warning"))

        # ---- Emisiones 992
        if f_992:
            try:
                df_raw = read_excel_any(f_992, sheet_name=0)
                neix_map = build_neix_map(df_raw)
                df_neg_992 = apply_neix_to_neg992(df_neg_992, neix_map)
                tags.append(("Emisiones 992: agregado Symbol y Qty BYMA.", "success"))
            except Exception as e:
                tags.append((f"Error leyendo Emisiones 992: {e}", "warning"))
        else:
            tags.append(("Emisiones 992 no adjuntado.", "warning"))

        # reorder 992
        if "Garantías BYMA" in df_neg_992.columns:
            df_neg_992 = reorder_neg992(df_neg_992)

        # ---- SLIQ (por ahora solo carga como DF y lo exporta; si querés misma lógica full, te la armo)
        df_sliq = None
        if f_sliq:
            try:
                df_sliq = read_csv_robusto(f_sliq)
                tags.append((f"Control SLIQ leído ({len(df_sliq)} filas).", "success"))
            except Exception as e:
                tags.append((f"Error leyendo SLIQ: {e}", "warning"))
        else:
            tags.append(("Control SLIQ no adjuntado.", "warning"))

        # sort como tu JS (simple)
        df_moc = df_moc.sort_values(["Codigo", "Comitente"], kind="stable")
        if "Total" in df_neg_otros.columns:
            df_neg_otros = df_neg_otros.sort_values(["Total"], kind="stable")
        if all(c in df_neg_992.columns for c in ["Saldo Negativos","Qty BYMA","Garantías BYMA","OTROS"]):
            df_neg_992["_sortTotal"] = (
                df_neg_992["Saldo Negativos"].fillna(0)
                + df_neg_992["Qty BYMA"].fillna(0)
                + df_neg_992["Garantías BYMA"].fillna(0)
                + df_neg_992["OTROS"].fillna(0)
            )
            df_neg_992 = df_neg_992.sort_values(["_sortTotal"], kind="stable").drop(columns=["_sortTotal"])

        st.session_state["ppt_state"] = {
            "df_moc": df_moc,
            "df_neg": df_neg_otros,
            "df_neg992": df_neg_992,
            "df_asig": df_asig,
            "df_sliq": df_sliq,
        }

        for msg, kind in tags:
            if kind == "success":
                st.success(msg)
            elif kind == "warning":
                st.warning(msg)
            else:
                st.error(msg) if kind == "error" else st.info(msg)

        st.divider()
        st.subheader("Preview")
        st.write("**MOC**")
        st.dataframe(df_moc.head(30), use_container_width=True, hide_index=True)
        st.write("**SALDOS NEGATIVOS**")
        st.dataframe(df_neg_otros.head(30), use_container_width=True, hide_index=True)
        st.write("**SALDOS NEGATIVOS 992**")
        st.dataframe(df_neg_992.head(30), use_container_width=True, hide_index=True)

    # Export definitivo
    if do_export:
        state = st.session_state.get("ppt_state", {})
        if not state:
            st.warning("Primero tocá **Validar**.")
            st.stop()

        hoy = datetime.now().strftime("%d-%m-%Y")
        filename = f"MOC {hoy} DEFINITIVO.xlsx"

        xbytes = export_definitivo(
            state["df_moc"],
            state["df_neg"],
            state["df_neg992"],
            state.get("df_asig"),
            state.get("df_sliq"),
        )

        st.download_button(
            "Descargar MOC DEFINITIVO.xlsx",
            data=xbytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ppt_dl_def",
        )

    # Export ventas
    if do_export_ventas:
        state = st.session_state.get("ppt_state", {})
        if not state:
            st.warning("Primero tocá **Validar**.")
            st.stop()

        df_ventas = generate_ventas(state["df_moc"])
        if df_ventas.empty:
            st.warning("No hay ventas para exportar (todas son 0).")
            st.stop()

        hoy = datetime.now().strftime("%d-%m-%Y")
        filename = f"VENTAS {hoy}.xlsx"

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_ventas.to_excel(writer, sheet_name="VENTAS", index=False)

        st.download_button(
            "Descargar VENTAS.xlsx",
            data=out.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ppt_dl_ventas",
        )
