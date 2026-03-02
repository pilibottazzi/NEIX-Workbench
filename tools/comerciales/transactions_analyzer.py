# tools/comerciales/transactions_analyzer.py
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# UI config (simple, prolijo, sin ruido)
# =========================================================
TITLE = "Movimientos CV — Transactions Analyzer"
SUBTITLE = (
    "Subí el Excel exportado (Transactions). "
    "Arrancamos simple: CASH_MOVEMENT (solo Federal Funds) y TRADE."
)

# Columnas objetivo (las que vos pediste)
COL_PROCESS_DATE = "Process Date"
COL_SETTLEMENT_DATE = "Settlement Date"
COL_NET_BASE = "Net Amount (Base Currency)"
COL_TX_TYPE = "Transaction Type"
COL_SEC_DESC = "Security Description"

# Para TRADE (mínimo útil)
COL_SYMBOL = "SYMBOL"
COL_BUYSELL = "Buy/Sell"
COL_QTY = "Quantity"
COL_PRICE = "Price (Transaction Currency)"

# Reglas iniciales (ordenadas para ir ampliando)
CASH_TYPE_ALLOWLIST = {
    "FEDERAL FUNDS RECEIVED",
}
CASH_TYPE_EXCLUDE = {
    "ACTIVITY WITHIN YOUR ACCT",  # lo sacamos del análisis de cash (por ahora)
    # fees/impuestos/etc se tratarán después en otras categorías
}


# =========================================================
# Helpers: parsing robusto
# =========================================================
def _norm(s: object) -> str:
    if s is None:
        return ""
    return str(s).strip()

def _norm_upper(s: object) -> str:
    return _norm(s).upper()

def _to_float(x: object) -> float:
    """
    Convierte números estilo US/EU/mixto.
    - Soporta "189,525.00" (US) y "189.525,00" (EU) y números puros.
    - Si no puede, devuelve np.nan.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)) and not pd.isna(x):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    # dejar solo dígitos, separadores y signo
    s = re.sub(r"[^0-9\.,\-]", "", s)

    if s.count(",") > 0 and s.count(".") > 0:
        # Si viene "189,525.00" => commas miles, dot decimal
        # Si viene "189.525,00" => dot miles, comma decimal
        # Decidimos por el último separador: el último suele ser decimal
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_dot > last_comma:
            # decimal = dot
            s = s.replace(",", "")
        else:
            # decimal = comma
            s = s.replace(".", "")
            s = s.replace(",", ".")
    else:
        # Solo uno de los dos
        # Si hay una sola coma y parece decimal => reemplazar por punto
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        # Si hay puntos, asumimos que el último punto es decimal si tiene 2 dígitos,
        # si no, igual dejamos que float() intente.
        # (si fueran miles con puntos, float lo toma mal, pero suele venir con coma+dot mezclado)

    try:
        return float(s)
    except Exception:
        return np.nan

def _to_datetime(x: object) -> pd.Timestamp:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT
    # pandas suele parsear bien
    return pd.to_datetime(x, errors="coerce")

def _find_header_row(df_raw: pd.DataFrame, needle: str = "Process Date") -> Optional[int]:
    """
    Busca la fila donde aparece el header real.
    Escanea todas las celdas como texto y encuentra la primera fila que contenga 'Process Date'.
    """
    target = needle.strip().lower()
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].astype(str).str.strip().str.lower()
        if (row == target).any():
            return i
    return None

def _build_table_from_header(df_raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    headers = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = [str(h).strip() for h in headers]
    df = df.dropna(how="all")
    # limpia columnas "Unnamed"
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df

def _pick_existing_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    existing = []
    for c in wanted:
        if c in df.columns:
            existing.append(c)
    return existing


# =========================================================
# Loader
# =========================================================
@st.cache_data(show_spinner=False)
def load_transactions_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, dict]:
    """
    Devuelve:
      - df_full: tabla ya con headers correctos
      - meta: info (header_row, sheet_name, etc.)
    """
    meta: dict = {}

    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    # Priorizamos una hoja que contenga "Transactions" en el nombre, si existe
    sheet_name = None
    for s in xls.sheet_names:
        if "trans" in s.lower():
            sheet_name = s
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    meta["sheet_name"] = sheet_name

    df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=object)
    header_row = _find_header_row(df_raw, needle=COL_PROCESS_DATE)
    if header_row is None:
        # fallback: si no encuentra, intenta con "Settlement Date"
        header_row = _find_header_row(df_raw, needle=COL_SETTLEMENT_DATE)

    if header_row is None:
        raise ValueError(
            "No pude encontrar la fila de encabezados (no encontré 'Process Date' ni 'Settlement Date'). "
            "Revisá que el Excel sea el export de Transactions."
        )

    meta["header_row"] = int(header_row)
    df = _build_table_from_header(df_raw, header_row=header_row)

    return df, meta


# =========================================================
# Normalización mínima (para empezar ordenadas)
# =========================================================
def normalize_minimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fechas
    if COL_PROCESS_DATE in df.columns:
        df[COL_PROCESS_DATE] = df[COL_PROCESS_DATE].apply(_to_datetime)
    if COL_SETTLEMENT_DATE in df.columns:
        df[COL_SETTLEMENT_DATE] = df[COL_SETTLEMENT_DATE].apply(_to_datetime)

    # Monto base
    if COL_NET_BASE in df.columns:
        df[COL_NET_BASE] = df[COL_NET_BASE].apply(_to_float)

    # Normalizaciones de texto
    if COL_TX_TYPE in df.columns:
        df[COL_TX_TYPE] = df[COL_TX_TYPE].astype(str).str.strip()
    if COL_SEC_DESC in df.columns:
        df[COL_SEC_DESC] = df[COL_SEC_DESC].astype(str).str.strip()

    # Campos TRADE (si están)
    if COL_SYMBOL in df.columns:
        df[COL_SYMBOL] = df[COL_SYMBOL].astype(str).str.strip().replace({"nan": "", "NaN": ""})
    if COL_BUYSELL in df.columns:
        df[COL_BUYSELL] = df[COL_BUYSELL].astype(str).str.strip().replace({"nan": "", "NaN": ""})
    if COL_QTY in df.columns:
        df[COL_QTY] = df[COL_QTY].apply(_to_float)
    if COL_PRICE in df.columns:
        df[COL_PRICE] = df[COL_PRICE].apply(_to_float)

    return df


# =========================================================
# Segmentación inicial
# =========================================================
def build_cash_movements(df: pd.DataFrame) -> pd.DataFrame:
    """
    CASH_MOVEMENT inicial:
    - Solo Transaction Type in CASH_TYPE_ALLOWLIST (por ahora: FEDERAL FUNDS RECEIVED)
    - Excluye ACTIVITY WITHIN YOUR ACCT
    - Clasifica IN/OUT por signo del Net Amount (Base Currency)
    """
    need = [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para CASH_MOVEMENT: {missing}")

    d = df.copy()

    d["_tx_type_u"] = d[COL_TX_TYPE].map(_norm_upper)
    d = d[~d["_tx_type_u"].isin({t.upper() for t in CASH_TYPE_EXCLUDE})]

    d = d[d["_tx_type_u"].isin({t.upper() for t in CASH_TYPE_ALLOWLIST})].copy()

    # Dirección por signo
    d["direction"] = np.where(d[COL_NET_BASE] >= 0, "IN", "OUT")

    # Orden y columnas finales (las que pediste)
    out = d[
        [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC, "direction"]
    ].copy()

    out = out.sort_values(by=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE], ascending=True, na_position="last")
    out = out.reset_index(drop=True)
    return out


def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    TRADE inicial (simple, sin sobre-interpretar):
    - Toma filas donde haya Buy/Sell o SYMBOL o Quantity > 0
    - Muestra columnas básicas
    """
    cols = _pick_existing_cols(
        df,
        [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_SYMBOL, COL_BUYSELL, COL_QTY, COL_PRICE, COL_NET_BASE, COL_SEC_DESC, COL_TX_TYPE],
    )
    if not cols:
        raise ValueError("No encuentro columnas para armar TRADE (SYMBOL / Buy/Sell / Quantity).")

    d = df.copy()

    # Heurística de trade
    has_bs = (COL_BUYSELL in d.columns) & (d[COL_BUYSELL].astype(str).str.strip() != "")
    has_sym = (COL_SYMBOL in d.columns) & (d[COL_SYMBOL].astype(str).str.strip() != "") & (~d[COL_SYMBOL].astype(str).str.lower().eq("nan"))
    has_qty = (COL_QTY in d.columns) & (pd.to_numeric(d[COL_QTY], errors="coerce").fillna(0).abs() > 0)

    m = has_bs | has_sym | has_qty
    d = d.loc[m, cols].copy()

    # Orden
    if COL_SETTLEMENT_DATE in d.columns:
        d = d.sort_values(by=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE], ascending=True, na_position="last")
    d = d.reset_index(drop=True)
    return d


# =========================================================
# Render
# =========================================================
def render(_=None):
    st.markdown(f"## 🧾 {TITLE}")
    st.caption(SUBTITLE)

    uploaded = st.file_uploader(
        "Subí el Excel exportado (Transactions)",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.info("Subí un Excel para empezar.")
        return

    try:
        file_bytes = uploaded.getvalue()
        df_full, meta = load_transactions_excel(file_bytes)
        df_full = normalize_minimal(df_full)

    except Exception as e:
        st.error("No pude leer el archivo. Revisá que sea el export de Pershing (Transactions).")
        st.exception(e)
        return

    # Top meta (chiquito)
    with st.expander("Ver detalles de lectura (debug)", expanded=False):
        st.write(meta)
        st.write("Columnas detectadas:", list(df_full.columns))

    tab_cash, tab_trade = st.tabs(["CASH_MOVEMENT", "TRADE"])

    # -------------------------
    # CASH_MOVEMENT
    # -------------------------
    with tab_cash:
        st.markdown("### CASH_MOVEMENT (in/out) — *solo Federal Funds*")
        st.caption(
            "Por ahora: solo `FEDERAL FUNDS RECEIVED`. "
            "`ACTIVITY WITHIN YOUR ACCT` queda afuera. Fees/impuestos los vemos después."
        )

        try:
            cash = build_cash_movements(df_full)
        except Exception as e:
            st.error("No pude construir CASH_MOVEMENT.")
            st.exception(e)
            return

        total = float(np.nansum(cash[COL_NET_BASE].values)) if len(cash) else 0.0
        n = int(len(cash))

        c1, c2, c3 = st.columns([1.2, 1, 1])
        c1.metric("Total (Net Amount Base)", f"{total:,.2f}")
        c2.metric("Movimientos", f"{n:,}")
        c3.metric("Rango fechas", f"{cash[COL_SETTLEMENT_DATE].min().date() if n else '-'} → {cash[COL_SETTLEMENT_DATE].max().date() if n else '-'}")

        st.divider()

        # Tabla (solo columnas clave, ordenadas)
        show_cols = [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC, "direction"]
        st.dataframe(
            cash[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------
    # TRADE
    # -------------------------
    with tab_trade:
        st.markdown("### TRADE (operaciones)")
        st.caption("Vista inicial simple para validar que estamos leyendo bien. Luego lo refinamos.")

        try:
            trades = build_trades(df_full)
        except Exception as e:
            st.error("No pude construir TRADE.")
            st.exception(e)
            return

        n = int(len(trades))
        total = float(np.nansum(trades[COL_NET_BASE].values)) if (COL_NET_BASE in trades.columns and n) else 0.0

        c1, c2 = st.columns([1.2, 1])
        c1.metric("Movimientos", f"{n:,}")
        if COL_NET_BASE in trades.columns:
            c2.metric("Total (Net Amount Base)", f"{total:,.2f}")
        else:
            c2.metric("Total (Net Amount Base)", "—")

        st.divider()
        st.dataframe(trades, use_container_width=True, hide_index=True)
