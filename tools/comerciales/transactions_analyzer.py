# tools/comerciales/transactions_analyzer.py
# =========================================================
# Movimientos CV — Transactions Analyzer (NEIX Workbench)
#
# Fase 1 (simple y controlado):
#   TAB 1) CASH_MOVEMENT
#       - SOLO: FEDERAL FUNDS RECEIVED / FEDERAL FUNDS SENT
#       - (excluye todo lo demás: fees, activity within acct, taxes, dividends, etc.)
#
#   TAB 2) ETFs
#       - SOLO: Security Type == "EXCHANGE TRADED FUNDS"
#       - Solo trades reales: Buy/Sell = BUY o SELL
#
#   TAB 3) STOCKS
#       - SOLO: Security Type == "COMMON STOCK"
#       - Solo trades reales: Buy/Sell = BUY o SELL
#
# ROBUSTEZ:
#   - Detecta el header real buscando "Process Date" (aunque haya metadata arriba)
#   - Si read_excel devuelve dict (múltiples sheets), usa "Transactions" o la primera
#   - Normaliza números con coma/punto y fechas
#
# UI:
#   - Presentación limpia, sin features “de más”
#   - Filtros básicos: fecha (Settlement/Process), rango, symbol (para trades)
# =========================================================

from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


NEIX_RED = "#ef4444"

# =========================
# Pershing columns (nombres EXACTOS del excel)
# =========================
COL_PROCESS_DATE = "Process Date"
COL_SETTLEMENT_DATE = "Settlement Date"
COL_NET_BASE = "Net Amount (Base Currency)"
COL_TX_TYPE = "Transaction Type"
COL_SEC_DESC = "Security Description"
COL_TX_DESC = "Transaction Description"

COL_NET_TX = "Net Amount (Transaction Currency)"
COL_BUYSELL = "Buy/Sell"
COL_QTY = "Quantity"
COL_PRICE = "Price (Transaction Currency)"
COL_TX_CCY = "Transaction Currency"
COL_SECURITY_TYPE = "Security Type"
COL_PAYEE = "Payee"
COL_PAID_FOR = "Paid For (Name)"
COL_REQ_REASON = "Request Reason"
COL_CUSIP = "CUSIP"
COL_FX = "FX Rate (To Base)"
COL_ISIN = "ISIN"
COL_SEDOL = "SEDOL"
COL_SYMBOL = "SYMBOL"
COL_TRADE_DATE = "Trade Date"

# =========================
# Phase 1 rules
# =========================
CASH_IN = "FEDERAL FUNDS RECEIVED"
CASH_OUT = "FEDERAL FUNDS SENT"

SEC_ETF = "EXCHANGE TRADED FUNDS"
SEC_STOCK = "COMMON STOCK"

# tabla "larga" que pediste para trades
TRADE_LONG_COLS = [
    COL_NET_BASE,
    COL_TX_DESC,
    COL_TX_TYPE,
    COL_SEC_DESC,
    COL_NET_TX,
    COL_BUYSELL,
    COL_QTY,
    COL_PRICE,
    COL_TX_CCY,
    COL_SECURITY_TYPE,
    COL_PAYEE,
    COL_PAID_FOR,
    COL_REQ_REASON,
    COL_CUSIP,
    COL_FX,
    COL_ISIN,
    COL_SEDOL,
    COL_SYMBOL,
    COL_TRADE_DATE,
]


# =========================
# UI helpers
# =========================
def _ui_css() -> None:
    st.markdown(
        f"""
        <style>
          .ta-title {{
            font-weight: 900;
            letter-spacing: .01em;
            font-size: 2.0rem;
            margin: 0 0 .25rem 0;
          }}
          .ta-sub {{
            color: #6b7280;
            margin: 0 0 1.0rem 0;
            font-size: .98rem;
          }}
          .ta-card {{
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 16px;
            padding: 12px 14px;
            background: #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,.04);
          }}
          .ta-kpis {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-top: 10px;
            margin-bottom: 8px;
          }}
          @media (max-width: 950px) {{
            .ta-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
          }}
          .ta-kpi {{
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 16px;
            padding: 12px 14px;
            background: #fff;
          }}
          .ta-kpi .label {{
            color:#6b7280;
            font-size:.85rem;
            font-weight: 900;
            margin-bottom: 6px;
          }}
          .ta-kpi .value {{
            font-size: 1.85rem;
            font-weight: 900;
            letter-spacing: .01em;
          }}
          .ta-kpi .hint {{
            color:#6b7280;
            font-size:.82rem;
            margin-top: 4px;
          }}
          .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid rgba(0,0,0,0.08);
            padding-left: 2px;
            margin-top: 6px;
          }}
          .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border: none;
            font-weight: 900;
            color: #6b7280;
            padding: 10px 14px;
            font-size: .95rem;
          }}
          .stTabs [aria-selected="true"] {{
            color:#111827;
            border-bottom: 3px solid {NEIX_RED};
          }}
          div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 14px;
            overflow: hidden;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _upper(x: object) -> str:
    return _safe_str(x).strip().upper()


def _money_fmt(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def _pick_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _require_cols(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para {label}: {missing}")


# =========================
# Excel reading (robusto)
# =========================
def _sheet_to_df(obj) -> pd.DataFrame:
    if isinstance(obj, dict):
        # preferimos sheet "Transactions"
        for k in obj.keys():
            if str(k).strip().lower() == "transactions":
                return obj[k]
        # sino: primer sheet
        return next(iter(obj.values()))
    return obj


def _find_header_row(raw: pd.DataFrame) -> int:
    target = _upper(COL_PROCESS_DATE)
    limit = min(len(raw), 200)

    for r in range(limit):
        row = raw.iloc[r].astype(str).map(_upper)
        if (row == target).any():
            return r

    raise ValueError("No pude encontrar la fila de encabezados (no aparece 'Process Date').")


def _read_pershing_transactions_excel(file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)

    raw_obj = pd.read_excel(bio, sheet_name=sheet_name, header=None, engine="openpyxl")
    raw = _sheet_to_df(raw_obj)

    header_row = _find_header_row(raw)

    bio2 = io.BytesIO(file_bytes)
    df_obj = pd.read_excel(bio2, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    df = _sheet_to_df(df_obj)

    # clean
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df = df.dropna(how="all").copy()
    return df


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _coerce_number(s: pd.Series) -> pd.Series:
    def to_float(v: object) -> float:
        if v is None:
            return np.nan
        try:
            if pd.isna(v):
                return np.nan
        except Exception:
            pass

        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)

        t = str(v).strip()
        if not t:
            return np.nan

        # soporta "14488,79" y "14,488.79"
        if "," in t and "." in t:
            t = t.replace(",", "")
        else:
            t = t.replace(".", "").replace(",", ".")

        t = t.replace("$", "").replace("USD", "").replace("ARS", "").strip()
        try:
            return float(t)
        except Exception:
            return np.nan

    return s.map(to_float)


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # dates
    for c in [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_TRADE_DATE]:
        if c in d.columns:
            d[c] = _coerce_date(d[c])

    # numbers
    for c in [COL_NET_BASE, COL_NET_TX, COL_QTY, COL_PRICE, COL_FX]:
        if c in d.columns:
            d[c] = _coerce_number(d[c])

    # text
    for c in [
        COL_TX_TYPE,
        COL_SEC_DESC,
        COL_TX_DESC,
        COL_BUYSELL,
        COL_SYMBOL,
        COL_SECURITY_TYPE,
        COL_TX_CCY,
        COL_PAYEE,
        COL_PAID_FOR,
        COL_REQ_REASON,
        COL_CUSIP,
        COL_ISIN,
        COL_SEDOL,
    ]:
        if c in d.columns:
            d[c] = d[c].map(_safe_str)

    return d


def _filter_by_date(df: pd.DataFrame, date_col: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if date_col not in df.columns:
        return df.copy()

    s = pd.to_datetime(df[date_col], errors="coerce")
    m = pd.Series(True, index=df.index)

    if start is not None:
        m &= s >= pd.to_datetime(start)
    if end is not None:
        m &= s <= pd.to_datetime(end)

    return df.loc[m].copy()


def _norm_buysell(x: object) -> str:
    s = _upper(x)
    if s in {"BUY", "B"}:
        return "BUY"
    if s in {"SELL", "S"}:
        return "SELL"
    return ""


# =========================
# Builders (Fase 1)
# =========================
def build_cash_movements(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC], "CASH_MOVEMENT")

    d = df.copy()
    d["_tx_u"] = d[COL_TX_TYPE].map(_upper)

    # SOLO federal funds (sin fees ni activity)
    d = d[d["_tx_u"].isin([CASH_IN, CASH_OUT])].copy()
    d["direction"] = np.where(d["_tx_u"] == _upper(CASH_IN), "IN", "OUT")

    out = d[[COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC, "direction"]].copy()
    out = out.sort_values(by=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE], ascending=True, na_position="last")
    return out.reset_index(drop=True)


def build_trades_by_security_type(df: pd.DataFrame, security_type_exact: str) -> pd.DataFrame:
    _require_cols(df, [COL_SECURITY_TYPE, COL_BUYSELL], f"TRADES ({security_type_exact})")

    d = df.copy()
    d["_sec_u"] = d[COL_SECURITY_TYPE].map(_upper)
    d = d[d["_sec_u"] == _upper(security_type_exact)].copy()

    d["buy_sell_norm"] = d[COL_BUYSELL].map(_norm_buysell)
    d = d[d["buy_sell_norm"].isin(["BUY", "SELL"])].copy()

    cols = _pick_existing_cols(d, TRADE_LONG_COLS)
    out = d[cols].copy()

    sort_cols = [c for c in [COL_TRADE_DATE, COL_SETTLEMENT_DATE, COL_PROCESS_DATE] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=True, na_position="last")

    return out.reset_index(drop=True)


# =========================
# Render
# =========================
def render(_ctx=None) -> None:
    _ui_css()

    st.markdown("<div class='ta-title'>🧾 Movimientos CV — Transactions Analyzer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ta-sub'>Fase 1: cash (solo <b>FEDERAL FUNDS</b>) + trades separados en <b>ETFs</b> y <b>STOCKS</b> (solo <b>BUY/SELL</b>).</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.4, 1.0, 0.9])

    with c1:
        up = st.file_uploader("Subí el Excel exportado (Transactions)", type=["xlsx", "xls"])

    with c2:
        date_col = st.selectbox(
            "Fecha para análisis",
            options=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE],
            index=0,
            help="Usamos esta fecha para el rango 'Desde/Hasta'.",
        )

    with c3:
        st.markdown(
            """
            <div class="ta-card">
              <b>Reglas (Fase 1)</b><br/>
              • CASH: FEDERAL FUNDS (IN/OUT)<br/>
              • ETFs: EXCHANGE TRADED FUNDS (BUY/SELL)<br/>
              • STOCKS: COMMON STOCK (BUY/SELL)
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not up:
        st.info("Subí un Excel para empezar.")
        return

    try:
        df_raw = _read_pershing_transactions_excel(up.getvalue(), sheet_name=None)
        df = standardize_df(df_raw)
    except Exception as e:
        st.error("No pude leer el Excel (o detectar la fila de encabezados).")
        st.exception(e)
        return

    # Rango fechas
    if date_col in df.columns:
        ds = pd.to_datetime(df[date_col], errors="coerce").dropna()
    else:
        ds = pd.Series([], dtype="datetime64[ns]")

    if len(ds) > 0:
        dmin = ds.min().date()
        dmax = ds.max().date()
    else:
        dmin, dmax = None, None

    f1, f2, f3 = st.columns([0.55, 0.55, 0.9])
    with f1:
        start = st.date_input("Desde", value=dmin, disabled=(dmin is None))
    with f2:
        end = st.date_input("Hasta", value=dmax, disabled=(dmax is None))
    with f3:
        symbol_filter = st.text_input("Filtrar SYMBOL (solo para ETFs/STOCKS)", value="", placeholder="Ej: SPY, META, MELI")

    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None

    df_f = _filter_by_date(df, date_col, start_ts, end_ts)

    if symbol_filter.strip():
        if COL_SYMBOL in df_f.columns:
            s = df_f[COL_SYMBOL].map(_upper)
            df_f = df_f[s.str.contains(_upper(symbol_filter), na=False)].copy()

    # Build tabs
    try:
        df_cash = build_cash_movements(df_f)
    except Exception as e:
        df_cash = pd.DataFrame()
        st.warning("No pude construir CASH_MOVEMENT con este archivo/rango.")
        st.exception(e)

    try:
        df_etf = build_trades_by_security_type(df_f, SEC_ETF)
    except Exception as e:
        df_etf = pd.DataFrame()
        st.warning("No pude construir ETFs con este archivo/rango.")
        st.exception(e)

    try:
        df_stocks = build_trades_by_security_type(df_f, SEC_STOCK)
    except Exception as e:
        df_stocks = pd.DataFrame()
        st.warning("No pude construir STOCKS con este archivo/rango.")
        st.exception(e)

    # KPIs por tab (mantener simple)
    tabs = st.tabs(["CASH_MOVEMENT", "ETFs", "STOCKS"])

    # -------------------------
    # TAB CASH_MOVEMENT
    # -------------------------
    with tabs[0]:
        total = float(df_cash[COL_NET_BASE].sum()) if (not df_cash.empty and COL_NET_BASE in df_cash.columns) else 0.0
        cnt = int(len(df_cash)) if not df_cash.empty else 0
        in_sum = float(df_cash.loc[df_cash["direction"] == "IN", COL_NET_BASE].sum()) if (not df_cash.empty) else 0.0
        out_sum = float(df_cash.loc[df_cash["direction"] == "OUT", COL_NET_BASE].sum()) if (not df_cash.empty) else 0.0

        st.markdown(
            f"""
            <div class="ta-kpis">
              <div class="ta-kpi">
                <div class="label">CASH — Neto (Base)</div>
                <div class="value">{_money_fmt(total)}</div>
                <div class="hint">IN + OUT (solo Federal Funds)</div>
              </div>
              <div class="ta-kpi">
                <div class="label">CASH — Movimientos</div>
                <div class="value">{cnt}</div>
                <div class="hint">cantidad de filas</div>
              </div>
              <div class="ta-kpi">
                <div class="label">CASH — Ingresos</div>
                <div class="value">{_money_fmt(in_sum)}</div>
                <div class="hint">FEDERAL FUNDS RECEIVED</div>
              </div>
              <div class="ta-kpi">
                <div class="label">CASH — Egresos</div>
                <div class="value">{_money_fmt(out_sum)}</div>
                <div class="hint">FEDERAL FUNDS SENT</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_cols = [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC, "direction"]
        show_cols = _pick_existing_cols(df_cash, show_cols)
        st.dataframe(df_cash[show_cols], use_container_width=True, hide_index=True)

    # -------------------------
    # TAB ETFs
    # -------------------------
    with tabs[1]:
        total = float(df_etf[COL_NET_BASE].sum()) if (not df_etf.empty and COL_NET_BASE in df_etf.columns) else 0.0
        cnt = int(len(df_etf)) if not df_etf.empty else 0
        buys = int((df_etf[COL_BUYSELL].map(_upper) == "BUY").sum()) if (not df_etf.empty and COL_BUYSELL in df_etf.columns) else 0
        sells = int((df_etf[COL_BUYSELL].map(_upper) == "SELL").sum()) if (not df_etf.empty and COL_BUYSELL in df_etf.columns) else 0

        st.markdown(
            f"""
            <div class="ta-kpis">
              <div class="ta-kpi">
                <div class="label">ETFs — Total (Base)</div>
                <div class="value">{_money_fmt(total)}</div>
                <div class="hint">suma neta del período</div>
              </div>
              <div class="ta-kpi">
                <div class="label">ETFs — Movimientos</div>
                <div class="value">{cnt}</div>
                <div class="hint">solo BUY/SELL</div>
              </div>
              <div class="ta-kpi">
                <div class="label">ETFs — BUY</div>
                <div class="value">{buys}</div>
                <div class="hint">cantidad</div>
              </div>
              <div class="ta-kpi">
                <div class="label">ETFs — SELL</div>
                <div class="value">{sells}</div>
                <div class="hint">cantidad</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(df_etf[_pick_existing_cols(df_etf, TRADE_LONG_COLS)], use_container_width=True, hide_index=True)

    # -------------------------
    # TAB STOCKS
    # -------------------------
    with tabs[2]:
        total = float(df_stocks[COL_NET_BASE].sum()) if (not df_stocks.empty and COL_NET_BASE in df_stocks.columns) else 0.0
        cnt = int(len(df_stocks)) if not df_stocks.empty else 0
        buys = int((df_stocks[COL_BUYSELL].map(_upper) == "BUY").sum()) if (not df_stocks.empty and COL_BUYSELL in df_stocks.columns) else 0
        sells = int((df_stocks[COL_BUYSELL].map(_upper) == "SELL").sum()) if (not df_stocks.empty and COL_BUYSELL in df_stocks.columns) else 0

        st.markdown(
            f"""
            <div class="ta-kpis">
              <div class="ta-kpi">
                <div class="label">STOCKS — Total (Base)</div>
                <div class="value">{_money_fmt(total)}</div>
                <div class="hint">suma neta del período</div>
              </div>
              <div class="ta-kpi">
                <div class="label">STOCKS — Movimientos</div>
                <div class="value">{cnt}</div>
                <div class="hint">solo BUY/SELL</div>
              </div>
              <div class="ta-kpi">
                <div class="label">STOCKS — BUY</div>
                <div class="value">{buys}</div>
                <div class="hint">cantidad</div>
              </div>
              <div class="ta-kpi">
                <div class="label">STOCKS — SELL</div>
                <div class="value">{sells}</div>
                <div class="hint">cantidad</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(df_stocks[_pick_existing_cols(df_stocks, TRADE_LONG_COLS)], use_container_width=True, hide_index=True)
