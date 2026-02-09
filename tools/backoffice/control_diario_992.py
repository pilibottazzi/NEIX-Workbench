# tools/backoffice/control_diario_992.py
from __future__ import annotations

import io
import re
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import streamlit as st


# =========================================================
# UI (NEIX Premium, minimal)
# =========================================================
NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "rgba(255,255,255,0.92)"


def _inject_ui_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1180px;
            padding-top: 1.35rem;
            padding-bottom: 2.2rem;
          }}
          h2 {{
            margin-bottom: .15rem !important;
            color: {TEXT} !important;
            letter-spacing: -0.02em;
          }}
          .stCaption {{ color: {MUTED} !important; }}

          .neix-card {{
            border: 1px solid {BORDER};
            background: {CARD_BG};
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 10px 30px rgba(17,24,39,0.06);
            margin-top: 14px;
          }}

          label {{
            font-weight: 650 !important;
            color: {TEXT} !important;
            font-size: 0.92rem !important;
          }}

          .help {{
            color: {MUTED};
            font-size: .88rem;
            margin-top: .25rem;
          }}

          div.stButton > button {{
            background: {NEIX_RED} !important;
            color: #fff !important;
            border: 1px solid rgba(0,0,0,0.04) !important;
            border-radius: 14px !important;
            padding: 0.72rem 1.0rem !important;
            font-weight: 800 !important;
            width: 100% !important;
            box-shadow: 0 10px 22px rgba(255,59,48,0.18) !important;
            transition: transform .08s ease, box-shadow .08s ease;
          }}
          div.stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(255,59,48,0.22) !important;
          }}

          section[data-testid="stFileUploader"] {{
            border-radius: 16px !important;
            border: 1px dashed {BORDER} !important;
            padding: 10px !important;
          }}

          div[data-testid="stAlert"] {{ border-radius: 14px !important; }}

          div[data-testid="stDataFrame"] {{
            border-radius: 14px !important;
            overflow: hidden;
            border: 1px solid {BORDER};
          }}

          .neix-footer {{
            margin-top: 18px;
            display:flex;
            justify-content:space-between;
            color: {MUTED};
            font-size: 0.88rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers (equivalentes al JS)
# =========================================================
SHEET_ENT = "Transferencia de posiciones ent"
SHEET_SAL = "Transferencia de posiciones sal"


def _safe_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def _norm_header(s: str) -> str:
    s = _safe_str(s).strip().lower()
    try:
        s = s.normalize("NFD")  # type: ignore[attr-defined]
    except Exception:
        pass
    s = re.sub(r"[\u0300-\u036f]", "", s)
    s = s.replace("ï¿½", "o")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _find_col_idx(headers: List[str], wanted: str) -> int:
    w = _norm_header(wanted)
    H = [_norm_header(h) for h in headers]
    try:
        return H.index(w)
    except ValueError:
        return -1


def _digits_key(v) -> str:
    m = re.search(r"(\d+)", _safe_str(v))
    return m.group(1) if m else ""


def _parse_cantidad(activos_text) -> Optional[float]:
    s = _safe_str(activos_text).strip()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        n = float(m.group(1))
        return n if pd.notna(n) else None
    except Exception:
        return None


def _parse_id_contract(activos_text) -> Optional[int]:
    s = _safe_str(activos_text)
    m = re.search(r"Id\.?\s*del\s*contrato\s+(\d+)", s, flags=re.I)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n
    except Exception:
        return None


def _parse_floor_number(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return ""
    if isinstance(v, (int, float)) and pd.notna(v):
        return int(float(v) // 1)

    s = _safe_str(v).strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        n = float(s)
        return int(n // 1) if pd.notna(n) else ""
    except Exception:
        return ""


def _normalize_ticker_key(v) -> str:
    return re.sub(r"[^A-Z0-9]", "", _safe_str(v).strip().upper())


def _parse_codigo_number(v) -> Optional[int]:
    s = _safe_str(v).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_approx_match_key(ticker_key: str, norm_keys: List[str]) -> str:
    key = _safe_str(ticker_key).strip()
    if len(key) < 2:
        return ""

    def filter_by_prefix(n: int) -> List[str]:
        p = key[:n]
        cands = [nk for nk in norm_keys if nk.startswith(p)]
        return cands[:2]  # early cap for ambiguity

    c2 = filter_by_prefix(2)
    if len(c2) == 1:
        return c2[0]
    if len(c2) == 0:
        return ""

    if len(key) >= 3:
        c3 = filter_by_prefix(3)
        if len(c3) == 1:
            return c3[0]
        if len(c3) == 0:
            return "__AMBIG__"
        c2 = c3

    if len(key) >= 4:
        c4 = filter_by_prefix(4)
        if len(c4) == 1:
            return c4[0]
        if len(c4) > 1:
            return "__AMBIG__"

    return "__AMBIG__"


# =========================================================
# Readers
# =========================================================
def _read_xls_status(file) -> pd.ExcelFile:
    # xls (BIFF). En cloud puede necesitar xlrd instalado.
    # Si no está, vas a ver error claro.
    return pd.ExcelFile(file, engine="xlrd")


def _read_excel_any(file) -> pd.ExcelFile:
    return pd.ExcelFile(file)


def _read_csv_any(file) -> pd.DataFrame:
    raw = file.getvalue()
    text = raw.decode("latin1", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str)


# =========================================================
# Core logic (Control ASAP)
# =========================================================
def _sheet_to_unified_rows(df: pd.DataFrame, tipo: str) -> List[Dict[str, Any]]:
    # en JS se leía AOA con headers en fila 1.
    # acá leemos por nombre exacto.
    # columnas esperadas:
    # "Fecha de la solicitud" / "Activos" / "Estado"
    cols = {c: _safe_str(c) for c in df.columns}
    want_fecha = None
    want_activos = None
    want_estado = None

    for c in cols:
        nc = _safe_str(c).strip().lower()
        if nc == "fecha de la solicitud":
            want_fecha = c
        elif nc == "activos":
            want_activos = c
        elif nc == "estado":
            want_estado = c

    if want_fecha is None or want_activos is None or want_estado is None:
        raise ValueError(
            "No se encontraron encabezados esperados (Fecha de la solicitud / Activos / Estado)."
        )

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        fecha = r.get(want_fecha, "")
        activos = r.get(want_activos, "")
        estado = r.get(want_estado, "")

        if _safe_str(fecha).strip() == "" and _safe_str(activos).strip() == "" and _safe_str(estado).strip() == "":
            continue

        cantidad = _parse_cantidad(activos)
        id_contract = _parse_id_contract(activos)

        out.append(
            {
                "Tipo": tipo,
                "Fecha de la solicitud": fecha,
                "Activos": activos,
                "Estado": estado,
                "Cantidad": cantidad if cantidad is not None else "",
                "Id contract": id_contract if id_contract is not None else "",
                "Ticker": "",
                "Código CVSA": "",
            }
        )
    return out


def _build_conid_to_ticker_map(con_df: pd.DataFrame) -> Dict[str, str]:
    # JS: default conid=B, ticker=C si no hay headers.
    headers = list(con_df.columns)

    idx_conid = _find_col_idx(headers, "conid")
    idx_ticker = _find_col_idx(headers, "ticker")

    if idx_conid == -1:
        idx_conid = 1
    if idx_ticker == -1:
        idx_ticker = 2

    map_: Dict[str, str] = {}
    for _, row in con_df.iterrows():
        k = _digits_key(row.iloc[idx_conid] if idx_conid < len(row) else "")
        t = _safe_str(row.iloc[idx_ticker] if idx_ticker < len(row) else "").strip()
        if k and t and k not in map_:
            map_[k] = t
    return map_


def _build_norm_to_codigo_map(gallo_df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    headers = list(gallo_df.columns)

    idx_codigo = _find_col_idx(headers, "codigo")
    idx_norm = _find_col_idx(headers, "norm.")
    if idx_norm == -1:
        idx_norm = _find_col_idx(headers, "norm")

    if idx_codigo == -1:
        idx_codigo = 0
    if idx_norm == -1:
        idx_norm = 9

    norm_to_codigo: Dict[str, int] = {}

    for _, row in gallo_df.iterrows():
        raw_norm = _safe_str(row.iloc[idx_norm] if idx_norm < len(row) else "").strip().upper()

        if "_US" in raw_norm or "_USE" in raw_norm or "_CL" in raw_norm:
            continue
        if re.search(r"\d", raw_norm) or re.search(r"\s", raw_norm):
            continue

        norm_key = _normalize_ticker_key(raw_norm)
        cod_num = _parse_codigo_number(row.iloc[idx_codigo] if idx_codigo < len(row) else None)
        if norm_key and cod_num is not None:
            if norm_key not in norm_to_codigo:
                norm_to_codigo[norm_key] = cod_num

    keys = list(norm_to_codigo.keys())
    return norm_to_codigo, keys


def _build_resumen_sheet(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # En Python lo armamos como DF (más simple). Luego escribimos a Excel en una hoja RESUMEN.
    # Armamos:
    # - tabla 1: resumen por tipo/estado (dinámica)
    df = pd.DataFrame(rows).copy()

    df["Tipo"] = df["Tipo"].astype(str).str.strip().str.upper()
    df["Estado"] = df["Estado"].astype(str).str.strip().str.upper()

    estados = sorted(df["Estado"].dropna().unique().tolist())
    if "PENDIENTE" in estados:
        estados = ["PENDIENTE"] + [e for e in estados if e != "PENDIENTE"]

    tipos = sorted(df["Tipo"].dropna().unique().tolist())

    # pivot tipo x estado
    pivot = pd.pivot_table(df, index="Tipo", columns="Estado", values="Activos", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=estados, fill_value=0)
    pivot.insert(0, "TOTAL", pivot.sum(axis=1))
    pivot = pivot.reset_index()

    # fila total general
    total_row = {"Tipo": "CANTIDAD DE CONVERSIONES", "TOTAL": int(len(df))}
    for e in estados:
        total_row[e] = int((df["Estado"] == e).sum())
    pivot = pd.concat([pd.DataFrame([total_row]), pivot], ignore_index=True)

    # Las tablas detalle las dejamos en la misma hoja con separadores como filas "título"
    # (Excel writer las escribe tal cual)
    # Pendiente
    det_cols = ["Tipo", "Cantidad", "Ticker", "Código CVSA"]
    df_det = df.copy()
    df_det["Cantidad"] = pd.to_numeric(df_det["Cantidad"], errors="coerce")
    df_det["Código CVSA"] = pd.to_numeric(df_det["Código CVSA"], errors="coerce")

    def _detail_block(title: str, mask: pd.Series) -> pd.DataFrame:
        block = df_det.loc[mask, ["Tipo", "Cantidad", "Ticker", "Código CVSA"]].copy()
        block = block.sort_values(["Tipo", "Ticker"], kind="stable")
        if block.empty:
            block = pd.DataFrame([{"Tipo": "(Sin filas)", "Cantidad": "", "Ticker": "", "Código CVSA": ""}])
        title_df = pd.DataFrame([{"Tipo": title, "Cantidad": "", "Ticker": "", "Código CVSA": ""}])
        header_df = pd.DataFrame([{"Tipo": "Tipo", "Cantidad": "Cantidad", "Ticker": "Ticker", "Código CVSA": "Código CVSA"}])
        return pd.concat([title_df, header_df, block], ignore_index=True)

    pendiente_block = _detail_block("Estado Pendiente", df_det["Estado"] == "PENDIENTE")
    cancelrech_block = _detail_block(
        "Estado Cancelada-Rechazada",
        df_det["Estado"].str.contains("CANCEL", na=False) | df_det["Estado"].str.contains("RECHAZ", na=False),
    )

    spacer = pd.DataFrame([{"Tipo": "", "Cantidad": "", "Ticker": "", "Código CVSA": ""} for _ in range(2)])

    # "pivot" ocupa columnas (Tipo, TOTAL, estados...). Para pegar detalle, unificamos con columnas del pivot.
    # Convertimos detalle a columnas del pivot rellenando.
    def _as_pivot_cols(detail_df: pd.DataFrame, pivot_cols: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(columns=pivot_cols)
        for c in pivot_cols:
            out[c] = ""
        # volcamos en primeras 4 columnas del pivot si existen
        # pivot: ["Tipo","TOTAL", estados...]
        out["Tipo"] = detail_df["Tipo"]
        if "TOTAL" in out.columns:
            out["TOTAL"] = detail_df["Cantidad"]
        if len(estados) >= 1:
            out[estados[0]] = detail_df["Ticker"]
        if len(estados) >= 2:
            out[estados[1]] = detail_df["Código CVSA"]
        return out

    pivot_cols = list(pivot.columns)
    combined = pd.concat(
        [
            pivot,
            spacer.reindex(columns=pivot_cols, fill_value=""),
            _as_pivot_cols(pendiente_block, pivot_cols),
            spacer.reindex(columns=pivot_cols, fill_value=""),
            _as_pivot_cols(cancelrech_block, pivot_cols),
        ],
        ignore_index=True,
    )
    return combined


def _write_xlsx_control_asap(rows: List[Dict[str, Any]]) -> bytes:
    df_unif = pd.DataFrame(rows)

    # Tipos numéricos (similar a JS)
    for c in ["Cantidad", "Id contract", "Código CVSA"]:
        if c in df_unif.columns:
            df_unif[c] = pd.to_numeric(df_unif[c], errors="ignore")

    df_resumen = _build_resumen_sheet(rows)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_unif.to_excel(writer, sheet_name="UNIFICADO", index=False)
        df_resumen.to_excel(writer, sheet_name="RESUMEN", index=False)
    out.seek(0)
    return out.read()


# =========================================================
# Core logic (Seguimiento)
# =========================================================
def _build_status_index(status_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    # conid -> {estado -> cantidad}
    idx: Dict[int, Dict[str, float]] = {}
    for _, r in status_df.iterrows():
        activos = r.get("Activos", "")
        estado = _safe_str(r.get("Estado", "")).strip().upper()
        conid = _parse_id_contract(activos)
        qty = _parse_cantidad(activos)
        if not conid or not qty or not estado:
            continue
        idx.setdefault(conid, {})
        idx[conid][estado] = idx[conid].get(estado, 0.0) + float(qty)
    return idx


def _read_trading_992(file) -> pd.DataFrame:
    # JS: headers en fila 2 (aoa[1]) y data desde fila 3 (i=2)
    # En pandas: header=1
    df = pd.read_excel(file, header=1, dtype=object)
    # dropear filas totalmente vacías
    df = df.dropna(how="all").copy()
    return df


def _read_conid_xlsx(file) -> pd.DataFrame:
    return pd.read_excel(file, dtype=object)


def _read_instr_998_csv(file) -> pd.DataFrame:
    df = _read_csv_any(file)
    return df


def _build_nasdaq_index(df_csv: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # JS:
    # instrumento = 2da columna (index 1)
    # cantidad/nominal: header detectado o fallback 3
    # estado: header detectado o fallback 4
    headers = list(df_csv.columns)
    idx_cant = _find_col_idx(headers, "Cantidad/nominal")
    idx_estado = _find_col_idx(headers, "Estado de Instrucción")

    if idx_cant == -1:
        idx_cant = 3
    if idx_estado == -1:
        idx_estado = 4

    # instrumento: columna 2 (index 1)
    idx_instr = 1

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df_csv.iterrows():
        instrumento = _safe_str(row.iloc[idx_instr] if idx_instr < len(row) else "").strip()
        if not instrumento:
            continue

        estado = _safe_str(row.iloc[idx_estado] if idx_estado < len(row) else "").strip().upper()
        if not estado:
            continue

        try:
            cantidad = float(str(row.iloc[idx_cant]).replace(",", "."))
        except Exception:
            continue

        if not pd.notna(cantidad):
            continue

        out.setdefault(instrumento, {})
        out[instrumento][estado] = out[instrumento].get(estado, 0.0) + float(cantidad)
    return out


def _write_xlsx_seguimiento(df_out: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="SEGUIMIENTO", index=False)
    out.seek(0)
    return out.read()


# =========================================================
# UI
# =========================================================
def render(back_to_home=None):
    _inject_ui_css()

    st.markdown("## Control Diario 992")
    st.caption("Back Office · NEIX")

    # Uploaders
    st.markdown('<div class="neix-card">', unsafe_allow_html=True)
    colA, colB = st.columns([0.5, 0.5])

    with colA:
        f_status_ibkr = st.file_uploader("1. Status IBKR (.xls)", type=["xls"], key="cd992_status")
        f_instr_998 = st.file_uploader("2. Instrucciones de Liquidación NASDAQ (998) (.csv)", type=["csv"], key="cd992_998")
        f_trading_992 = st.file_uploader("3. Trading - 992 (.xlsx)", type=["xlsx"], key="cd992_trading")
    with colB:
        f_especies_gallo = st.file_uploader("4. Especies GALLO (.xlsx)", type=["xlsx"], key="cd992_gallo")
        f_conid_ibkr = st.file_uploader("5. Con ID (IBKR) (.xlsx)", type=["xlsx"], key="cd992_conid")

    cbtn1, cbtn2 = st.columns([0.5, 0.5])
    with cbtn1:
        run_asap = st.button("Control ASAP", use_container_width=True)
    with cbtn2:
        run_seg = st.button("Seguimiento", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # CONTROL ASAP
    # -------------------------
    if run_asap:
        if not f_status_ibkr:
            st.error("✗ Falta adjuntar el archivo Status IBKR (.xls).")
            return
        if not f_conid_ibkr:
            st.error("✗ Falta adjuntar el archivo Con ID (IBKR) (.xlsx).")
            return
        if not f_especies_gallo:
            st.error("✗ Falta adjuntar el archivo Especies GALLO (.xlsx).")
            return

        try:
            with st.spinner("Procesando Control ASAP (Status IBKR)..."):
                xls = _read_xls_status(f_status_ibkr)

                used_sheets = []
                rows: List[Dict[str, Any]] = []

                ws_ent = None
                ws_sal = None

                if SHEET_ENT in xls.sheet_names:
                    ws_ent = pd.read_excel(xls, sheet_name=SHEET_ENT, dtype=object)
                if SHEET_SAL in xls.sheet_names:
                    ws_sal = pd.read_excel(xls, sheet_name=SHEET_SAL, dtype=object)

                if ws_ent is None and ws_sal is None:
                    st.error(f'✗ No se encontraron las hojas "{SHEET_ENT}" ni "{SHEET_SAL}".')
                    return

                if ws_ent is not None:
                    rows.extend(_sheet_to_unified_rows(ws_ent, "ENTRANTE"))
                    used_sheets.append(SHEET_ENT)

                if ws_sal is not None:
                    rows.extend(_sheet_to_unified_rows(ws_sal, "SALIENTE"))
                    used_sheets.append(SHEET_SAL)

                # ConId: conid -> ticker
                con_df = _read_conid_xlsx(f_conid_ibkr)
                con_map = _build_conid_to_ticker_map(con_df)

                filled_ticker = 0
                for r in rows:
                    k = _digits_key(r.get("Id contract", ""))
                    if k and k in con_map:
                        r["Ticker"] = con_map[k]
                        filled_ticker += 1

                # Gallo: ticker -> codigo
                gallo_df = pd.read_excel(f_especies_gallo, dtype=object)
                norm_to_codigo, norm_keys = _build_norm_to_codigo_map(gallo_df)

                filled_exact = 0
                filled_approx = 0
                ambiguous = 0

                for r in rows:
                    tkey = _normalize_ticker_key(r.get("Ticker", ""))
                    if not tkey:
                        continue

                    if tkey in norm_to_codigo:
                        r["Código CVSA"] = norm_to_codigo[tkey]
                        filled_exact += 1
                        continue

                    mk = _find_approx_match_key(tkey, norm_keys)
                    if mk == "__AMBIG__":
                        ambiguous += 1
                        continue
                    if mk and mk in norm_to_codigo:
                        r["Código CVSA"] = norm_to_codigo[mk]
                        filled_approx += 1

                # Orden: pendientes primero
                def _is_pend(x: Dict[str, Any]) -> int:
                    e = _safe_str(x.get("Estado", "")).strip().lower()
                    return 0 if e == "pendiente" else 1

                rows.sort(key=_is_pend)

                xlsx_bytes = _write_xlsx_control_asap(rows)

            ts = dt.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            fname = f"Control ASAP {ts}.xlsx"

            st.success(f"✓ Listo. Se generó el output ({len(rows)} filas).")
            st.info(
                " · ".join(
                    [
                        f"Hojas usadas: {' + '.join(used_sheets)}",
                        f"Tickers completados: {filled_ticker} de {len(rows)}",
                        f"CVSA exactos: {filled_exact}",
                        f"CVSA aproximados: {filled_approx}",
                        f"CVSA ambiguos (sin asignar): {ambiguous}",
                    ]
                )
            )

            st.download_button(
                "📥 Descargar Excel",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"✗ Error en Control ASAP: {e}")
            st.exception(e)
            return

    # -------------------------
    # SEGUIMIENTO
    # -------------------------
    if run_seg:
        # requeridos JS: 1,3,5 y además NASDAQ 998
        if not f_status_ibkr or not f_trading_992 or not f_conid_ibkr:
            st.error("✗ Falta alguno de los archivos requeridos (1, 3 o 5).")
            return
        if not f_instr_998:
            st.error("✗ Falta el archivo Instrucciones NASDAQ (998).")
            return

        try:
            with st.spinner("Procesando Seguimiento..."):
                # A) Trading 992 (base)
                df_trading = _read_trading_992(f_trading_992).copy()

                # limpiamos columnas excluidas en JS: Estado y Qty 992
                cols = list(df_trading.columns)
                idx_estado = _find_col_idx(cols, "Estado")
                idx_qty992 = _find_col_idx(cols, "Qty 992")

                drop_cols = []
                if idx_estado != -1:
                    drop_cols.append(cols[idx_estado])
                if idx_qty992 != -1:
                    drop_cols.append(cols[idx_qty992])

                df_base = df_trading.drop(columns=drop_cols, errors="ignore").copy()

                # normalizamos qty ibkr/byma con floor
                if "Qty IBKR" in df_trading.columns:
                    df_base["Qty IBKR"] = df_trading["Qty IBKR"].apply(_parse_floor_number)
                if "Qty BYMA" in df_trading.columns:
                    df_base["Qty BYMA"] = df_trading["Qty BYMA"].apply(_parse_floor_number)

                # B) Con ID (ticker → conid)
                con_df = _read_conid_xlsx(f_conid_ibkr)
                t2c: Dict[str, str] = {}
                con_headers = list(con_df.columns)
                idx_ticker = _find_col_idx(con_headers, "ticker")
                idx_conid = _find_col_idx(con_headers, "conid")
                if idx_ticker == -1:
                    idx_ticker = 2
                if idx_conid == -1:
                    idx_conid = 1

                for _, r in con_df.iterrows():
                    t = _normalize_ticker_key(r.iloc[idx_ticker] if idx_ticker < len(r) else "")
                    c = _digits_key(r.iloc[idx_conid] if idx_conid < len(r) else "")
                    if t and c and t not in t2c:
                        t2c[t] = c

                t2c_keys = list(t2c.keys())

                # C) Merge Trading → Con ID (exact + approx)
                sin_conid = 0
                id_contract_vals: List[Optional[int]] = []

                for _, r in df_base.iterrows():
                    tkey = _normalize_ticker_key(r.get("Symbol", ""))
                    if not tkey:
                        id_contract_vals.append(None)
                        sin_conid += 1
                        continue

                    if tkey in t2c:
                        id_contract_vals.append(int(t2c[tkey]))
                        continue

                    mk = _find_approx_match_key(tkey, t2c_keys)
                    if mk and mk != "__AMBIG__" and mk in t2c:
                        id_contract_vals.append(int(t2c[mk]))
                    else:
                        id_contract_vals.append(None)
                        sin_conid += 1

                df_base["Id contract"] = id_contract_vals

                # D) Merge Trading → Status IBKR (index por hoja)
                xls = _read_xls_status(f_status_ibkr)

                status_ent = pd.read_excel(xls, sheet_name=SHEET_ENT, dtype=object) if SHEET_ENT in xls.sheet_names else None
                status_sal = pd.read_excel(xls, sheet_name=SHEET_SAL, dtype=object) if SHEET_SAL in xls.sheet_names else None

                statusEntIdx = _build_status_index(status_ent) if status_ent is not None else {}
                statusSalIdx = _build_status_index(status_sal) if status_sal is not None else {}

                def _ibkr_cell(tipo: str, op: str, conid: Optional[int]) -> str:
                    if not conid:
                        return ""
                    tipo_u = _safe_str(tipo).upper()
                    op_u = _safe_str(op).upper()

                    status_map: Optional[Dict[int, Dict[str, float]]] = None
                    if (tipo_u == "CEDEAR" and op_u == "ISSUE") or (tipo_u == "ADR" and op_u == "CXL"):
                        status_map = statusSalIdx
                    elif (tipo_u == "CEDEAR" and op_u == "CXL") or (tipo_u == "ADR" and op_u == "ISSUE"):
                        status_map = statusEntIdx
                    else:
                        return ""

                    if conid not in status_map:
                        return ""

                    parts = [f"{v:g} {k}" for k, v in status_map[conid].items()]
                    return " - ".join(parts)

                df_base["IBKR"] = [
                    _ibkr_cell(df_base.iloc[i].get("Tipo", ""), df_base.iloc[i].get("Operación", ""), df_base.iloc[i].get("Id contract", None))
                    for i in range(len(df_base))
                ]

                # E) Merge Trading → NASDAQ (998)
                df_998 = _read_instr_998_csv(f_instr_998)
                nasdaq_idx = _build_nasdaq_index(df_998)

                def _nasdaq_cell(csva_code: str) -> str:
                    cvsa = _safe_str(csva_code).strip()
                    if not cvsa or cvsa not in nasdaq_idx:
                        return ""
                    parts = [f"{v:g} {k}" for k, v in nasdaq_idx[cvsa].items()]
                    return " - ".join(parts)

                if "Csva code" in df_base.columns:
                    df_base["NASDAQ"] = df_base["Csva code"].apply(_nasdaq_cell)
                else:
                    df_base["NASDAQ"] = ""

                # Export
                xlsx_bytes = _write_xlsx_seguimiento(df_base)

            ts = dt.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            fname = f"Seguimiento {ts}.xlsx"

            st.success("✓ Seguimiento generado.")
            st.info(f"Filas exportadas: **{len(df_base)}** · Filas sin conid: **{sin_conid}**")

            st.download_button(
                "📥 Descargar Excel",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"✗ Error en Seguimiento: {e}")
            st.exception(e)
            return

    st.markdown(
        '<div class="neix-footer"><span>NEIX · Back Office</span><span>Control Diario 992</span></div>',
        unsafe_allow_html=True,
    )
