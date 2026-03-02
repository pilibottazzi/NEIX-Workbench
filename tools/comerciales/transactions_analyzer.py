# tools/comerciales/transactions_analyzer.py
# =========================================================
# Movimientos CV — Transactions Analyzer (NEIX Workbench)
# Objetivo (fase 1):
#   - Leer Excel "Transactions" (Pershing)
#   - Detectar automáticamente la fila header (donde aparece "Process Date")
#   - Mostrar SOLO 2 pestañas: CASH_MOVEMENT y TRADE
#   - CASH_MOVEMENT: ingresos/egresos SOLO por FEDERAL FUNDS RECEIVED / FEDERAL FUNDS SENT
#     (sin fees ni Activity Within Your Acct en esta fase)
#   - TRADE: SOLO ETFs (Security Type = EXCHANGE TRADED FUNDS) y Buy/Sell normalizado a BUY/SELL
#
# Nota: está pensado para integrarse a tu router:
#   from tools.comerciales import transactions_analyzer
#   transactions_analyzer.render(None)
# =========================================================

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# UI Helpers
# =========================
NEIX_RED = "#ef4444"


def _ui_css():
    st.markdown(
        f"""
        <style>
          .ta-title {{
            font-weight: 900;
            letter-spacing: .02em;
            font-size: 2.0rem;
            margin: 0 0 .25rem 0;
          }}
          .ta-sub {{
            color: #6b7280;
            margin: 0 0 1.15rem 0;
            font-size: .98rem;
          }}

          .ta-card {{
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 16px;
            padding: 14px 16px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,.04);
          }}

          .ta-kpis {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-top: 10px;
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
            font-weight: 700;
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

          /* Tabs: minimal con underline rojo */
          .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid rgba(0,0,0,0.08);
            padding-left: 2px;
            margin-top: 6px;
          }}
          .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border: none;
            font-weight: 800;
            color: #6b7280;
            padding: 10px 14px;
            font-size: .95rem;
          }}
          .stTabs [aria-selected="true"] {{
            color:#111827;
            border-bottom: 3px solid {NEIX_RED};
          }}

          /* dataframe */
          div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 14px;
            overflow: hidden;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _money_fmt(x: float) -> str:
    # simple: miles con coma y 2 decimales (como screenshot)
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def _norm_upper(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().upper()


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _pick_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# =========================
# Column names (Pershing)
# =========================
COL_PROCESS_DATE = "Process Date"
COL_SETTLEMENT_DATE = "Settlement Date"
COL_NET_BASE = "Net Amount (Base Currency)"
COL_TX_TYPE = "Transaction Type"
COL_SEC_DESC = "Security Description"
COL_TX_DESC = "Transaction Description"

# Trade-related
COL_BUYSELL = "Buy/Sell"
COL_QTY = "Quantity"
COL_PRICE = "Price (Transaction Currency)"
COL_SYMBOL = "SYMBOL"
COL_SECURITY_TYPE = "Security Type"


# =========================
# Phase-1 category rules
# =========================
CASH_IN = "FEDERAL FUNDS RECEIVED"
CASH_OUT = "FEDERAL FUNDS SENT"
SEC_TYPE_ETF = "EXCHANGE TRADED FUNDS"


# =========================
# Excel parsing
# =========================
def _find_header_row(raw: pd.DataFrame) -> int:
    """
    Pershing trae metadatos arriba. Detectamos la fila donde está el header real
    buscando "Process Date" en alguna celda.
    """
    target = _norm_upper(COL_PROCESS_DATE)
    # raw es sin header (header=None), columnas 0..N
    for r in range(min(len(raw), 80)):  # con 80 alcanza para el bloque de metadata
        row = raw.iloc[r].astype(str).map(_norm_upper)
        if (row == target).any():
            return r
    raise ValueError("No pude encontrar la fila de encabezados (no aparece 'Process Date').")


def _read_pershing_transactions_excel(file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Lee el Excel y devuelve el DataFrame ya con headers correctos.
    """
    bio = io.BytesIO(file_bytes)

    # 1) leer crudo sin headers para detectar fila
    raw = pd.read_excel(bio, sheet_name=sheet_name, header=None, engine="openpyxl")
    if isinstance(raw, dict):
        # si vino dict por múltiples sheets, agarrar el primero
        raw = list(raw.values())[0]

    header_row = _find_header_row(raw)

    # 2) volver a leer usando esa fila como header
    bio2 = io.BytesIO(file_bytes)
    df = pd.read_excel(bio2, sheet_name=sheet_name, header=header_row, engine="openpyxl")

    # limpiar columnas "Unnamed"
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()

    # drop filas totalmente vacías
    df = df.dropna(how="all").copy()

    return df


def _coerce_date(s: pd.Series) -> pd.Series:
    # Pershing a veces viene como "feb 6, 2025" (texto).
    # Pandas suele parsear; si no, forzamos.
    out = pd.to_datetime(s, errors="coerce")
    return out


def _coerce_number(s: pd.Series) -> pd.Series:
    """
    Pershing viene en formato US (189525.00) pero también puede venir como texto con comas.
    """
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
        # sacar $ y comas
        t = t.replace("$", "").replace(",", "")
        try:
            return float(t)
        except Exception:
            return np.nan

    return s.map(to_float)


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # fechas
    for c in [COL_PROCESS_DATE, COL_SETTLEMENT_DATE]:
        if c in d.columns:
            d[c] = _coerce_date(d[c])

    # números
    if COL_NET_BASE in d.columns:
        d[COL_NET_BASE] = _coerce_number(d[COL_NET_BASE])

    if COL_QTY in d.columns:
        d[COL_QTY] = _coerce_number(d[COL_QTY])

    if COL_PRICE in d.columns:
        d[COL_PRICE] = _coerce_number(d[COL_PRICE])

    # strings
    for c in [COL_TX_TYPE, COL_SEC_DESC, COL_TX_DESC, COL_BUYSELL, COL_SYMBOL, COL_SECURITY_TYPE]:
        if c in d.columns:
            d[c] = d[c].map(_safe_str)

    return d


# =========================
# CASH_MOVEMENT (fase 1)
# =========================
def build_cash_movements(df: pd.DataFrame) -> pd.DataFrame:
    req = [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para CASH_MOVEMENT: {missing}")

    d = df.copy()

    t = d[COL_TX_TYPE].map(_norm_upper)
    d["_tx_u"] = t

    # SOLO federal funds received/sent (sin fees ni activity)
    d = d[d["_tx_u"].isin([CASH_IN, CASH_OUT])].copy()

    # dirección (signo esperado por tipo)
    d["direction"] = np.where(d["_tx_u"] == CASH_IN, "IN", "OUT")

    # en base currency ya viene con signo muchas veces, pero mantenemos net_amount_base tal cual
    # y también un "amount_abs" para sumar sin confundir
    d["amount_abs"] = d[COL_NET_BASE].abs()

    # columnas a mostrar (las que pediste)
    out = d[[COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC, "direction"]].copy()

    # ordenar por settlement date (más útil para cash)
    out = out.sort_values(by=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE], ascending=True, na_position="last")
    out = out.reset_index(drop=True)

    return out


# =========================
# TRADE (fase 1: ETFs + BUY/SELL)
# =========================
def _norm_buysell(x: object) -> str:
    s = _norm_upper(x)
    if s in {"BUY", "B"}:
        return "BUY"
    if s in {"SELL", "S"}:
        return "SELL"
    return ""


def build_trades_etf(df: pd.DataFrame) -> pd.DataFrame:
    req = [COL_SECURITY_TYPE, COL_BUYSELL, COL_SYMBOL, COL_SETTLEMENT_DATE]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para TRADE (ETF): {missing}")

    d = df.copy()

    # 1) Filtrar por Security Type = ETF
    d["_sec_type_u"] = d[COL_SECURITY_TYPE].map(_norm_upper)
    d = d[d["_sec_type_u"] == SEC_TYPE_ETF].copy()

    # 2) Normalizar Buy/Sell y filtrar lo demás
    d["buy_sell_norm"] = d[COL_BUYSELL].map(_norm_buysell)
    d = d[d["buy_sell_norm"].isin(["BUY", "SELL"])].copy()

    cols = _pick_existing_cols(
        d,
        [
            COL_PROCESS_DATE,
            COL_SETTLEMENT_DATE,
            COL_SYMBOL,
            "buy_sell_norm",
            COL_QTY,
            COL_PRICE,
            COL_NET_BASE,
            COL_SEC_DESC,
            COL_TX_TYPE,
            COL_SECURITY_TYPE,
        ],
    )
    out = d[cols].copy()
    out = out.sort_values(by=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE], ascending=True, na_position="last").reset_index(drop=True)
    return out


# =========================
# Date filtering (safe)
# =========================
def _filter_by_date(df: pd.DataFrame, date_col: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if date_col not in df.columns:
        return df.copy()

    s = pd.to_datetime(df[date_col], errors="coerce")
    m = pd.Series(True, index=df.index)

    if start is not None:
        m &= (s >= pd.to_datetime(start))
    if end is not None:
        m &= (s <= pd.to_datetime(end))

    return df.loc[m].copy()


# =========================
# Render
# =========================
def render(_ctx=None) -> None:
    _ui_css()

    st.markdown("<div class='ta-title'>🧾 Movimientos CV — Transactions Analyzer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ta-sub'>Subí el Excel exportado (Transactions). Empezamos simple: cash (ingresos/egresos) y trades (ETFs) para ordenar la lectura.</div>",
        unsafe_allow_html=True,
    )

    top = st.container()
    with top:
        c1, c2, c3 = st.columns([1.35, 1.0, 0.9])

        with c1:
            up = st.file_uploader("Subí el Excel exportado (Transactions)", type=["xlsx", "xls"])

        with c2:
            date_col = st.selectbox(
                "Fecha para análisis",
                options=[COL_SETTLEMENT_DATE, COL_PROCESS_DATE],
                index=0,
                help="En cash suele tener más sentido Settlement Date.",
            )

        with c3:
            st.markdown(
                "<div class='ta-card'><b>Fase 1</b><br/>Solo <span style='color:#111827;font-weight:900'>FEDERAL FUNDS</span> en cash y <span style='color:#111827;font-weight:900'>ETFs BUY/SELL</span> en trades.</div>",
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

    # rango de fechas (seguro)
    date_series = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([], dtype="datetime64[ns]")
    date_series = date_series.dropna()
    if len(date_series) > 0:
        min_d = date_series.min().date()
        max_d = date_series.max().date()
    else:
        min_d = None
        max_d = None

    f1, f2, _ = st.columns([1, 1, 1])
    with f1:
        start = st.date_input("Desde", value=min_d, disabled=min_d is None)
    with f2:
        end = st.date_input("Hasta", value=max_d, disabled=max_d is None)

    start_ts = pd.Timestamp(start) if min_d is not None else None
    end_ts = pd.Timestamp(end) if max_d is not None else None

    df_f = _filter_by_date(df, date_col, start_ts, end_ts)

    # =========================
    # Build datasets
    # =========================
    try:
        cash = build_cash_movements(df_f)
    except Exception as e:
        cash = pd.DataFrame()
        st.warning("No pude armar CASH_MOVEMENT con este archivo.")
        st.exception(e)

    try:
        trades = build_trades_etf(df_f)
    except Exception as e:
        trades = pd.DataFrame()
        st.warning("No pude armar TRADE (ETF) con este archivo.")
        st.exception(e)

    # =========================
    # KPIs (cash)
    # =========================
    cash_in = float(cash.loc[cash["direction"] == "IN", COL_NET_BASE].sum()) if (not cash.empty and COL_NET_BASE in cash.columns) else 0.0
    cash_out = float(cash.loc[cash["direction"] == "OUT", COL_NET_BASE].sum()) if (not cash.empty and COL_NET_BASE in cash.columns) else 0.0
    cash_net = cash_in + cash_out  # out suele venir negativo; así queda consistente
    cash_n = int(len(cash)) if not cash.empty else 0

    kpi = st.container()
    with kpi:
        st.markdown(
            f"""
            <div class="ta-kpis">
              <div class="ta-kpi">
                <div class="label">CASH — Ingresos (FEDERAL FUNDS RECEIVED)</div>
                <div class="value">{_money_fmt(cash_in)}</div>
                <div class="hint">Suma de Net Amount (Base Currency)</div>
              </div>
              <div class="ta-kpi">
                <div class="label">CASH — Egresos (FEDERAL FUNDS SENT)</div>
                <div class="value">{_money_fmt(cash_out)}</div>
                <div class="hint">Suma de Net Amount (Base Currency)</div>
              </div>
              <div class="ta-kpi">
                <div class="label">CASH — Neto</div>
                <div class="value">{_money_fmt(cash_net)}</div>
                <div class="hint">Ingresos + Egresos (con signo)</div>
              </div>
              <div class="ta-kpi">
                <div class="label">CASH — Movimientos</div>
                <div class="value">{cash_n}</div>
                <div class="hint">Cantidad de filas (solo FEDERAL FUNDS)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================
    # Tabs (solo 2)
    # =========================
    tab_cash, tab_trade = st.tabs(["CASH_MOVEMENT", "TRADE"])

    with tab_cash:
        st.markdown("### CASH_MOVEMENT")
        st.caption("Solo FEDERAL FUNDS RECEIVED / FEDERAL FUNDS SENT. (Sin fees ni Activity Within Your Acct en esta fase.)")

        show_cols = _pick_existing_cols(
            cash,
            [COL_PROCESS_DATE, COL_SETTLEMENT_DATE, COL_NET_BASE, COL_TX_TYPE, COL_SEC_DESC, "direction"],
        )
        if cash.empty:
            st.info("No hay movimientos cash para el filtro actual.")
        else:
            st.dataframe(cash[show_cols], use_container_width=True, hide_index=True)

            # descarga
            out = io.BytesIO()
            cash[show_cols].to_excel(out, index=False, sheet_name="CASH_MOVEMENT")
            st.download_button(
                "Descargar CASH_MOVEMENT (Excel)",
                data=out.getvalue(),
                file_name="cash_movement.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with tab_trade:
        st.markdown("### TRADE")
        st.caption("Solo ETFs (Security Type = EXCHANGE TRADED FUNDS) y Buy/Sell normalizado a BUY/SELL.")

        if trades.empty:
            st.info("No hay trades ETF BUY/SELL para el filtro actual.")
        else:
            # KPI quick
            n_tr = len(trades)
            buy_n = int((trades["buy_sell_norm"] == "BUY").sum()) if "buy_sell_norm" in trades.columns else 0
            sell_n = int((trades["buy_sell_norm"] == "SELL").sum()) if "buy_sell_norm" in trades.columns else 0

            a, b, c = st.columns([1, 1, 2])
            a.metric("Trades", n_tr)
            b.metric("BUY / SELL", f"{buy_n} / {sell_n}")
            c.metric("Symbols", int(trades[COL_SYMBOL].nunique()) if COL_SYMBOL in trades.columns else 0)

            show_cols = _pick_existing_cols(
                trades,
                [COL_SETTLEMENT_DATE, COL_SYMBOL, "buy_sell_norm", COL_QTY, COL_PRICE, COL_NET_BASE, COL_SEC_DESC],
            )
            st.dataframe(trades[show_cols], use_container_width=True, hide_index=True)

            out = io.BytesIO()
            trades[show_cols].to_excel(out, index=False, sheet_name="TRADE_ETF")
            st.download_button(
                "Descargar TRADE (Excel)",
                data=out.getvalue(),
                file_name="trade_etf.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
