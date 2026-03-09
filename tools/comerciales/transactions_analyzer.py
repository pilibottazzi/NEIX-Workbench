# tools/comerciales/transactions_analyzer.py
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# UI (minimal, pro)
# =========================
def _inject_css() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.2rem; max-width: 1220px; }
    h1 { margin-bottom: 0.2rem; }
    .subtle { color: rgba(0,0,0,0.55); font-size: 0.95rem; margin-top: 0.1rem; }
    .kpi {
        padding: 12px 14px;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        background: #fff;
    }
    .kpi .label {
        color: rgba(0,0,0,0.55);
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .kpi .value {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .pill {
        display:inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.12);
        font-size: 0.82rem;
        color: rgba(0,0,0,0.7);
    }
    .hr {
        height:1px;
        background: rgba(0,0,0,0.08);
        margin: 10px 0 12px;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Parsing helpers
# =========================
CANON_COLS = [
    "Process Date",
    "Security Identifier",
    "Settlement Date",
    "Net Amount (Base Currency)",
    "Transaction Description",
    "Transaction Type",
    "Security Description",
    "Net Amount (Transaction Currency)",
    "Buy/Sell",
    "Quantity",
    "Price (Transaction Currency)",
    "Transaction Currency",
    "Security Type",
    "Payee",
    "Paid For (Name)",
    "Request Reason",
    "CUSIP",
    "FX Rate (To Base)",
    "ISIN",
    "SEDOL",
    "SYMBOL",
    "Trade Date",
    "Transaction code",
    "Withdrawal/Deposit Type",
    "Request ID #",
    "Commission",
]


def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = (
        s.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    s = re.sub(r"\s+", " ", s)
    return s


def _find_header_row(df0: pd.DataFrame) -> int:
    """
    Pershing export: arriba hay metadata (Account, Client, etc).
    Buscamos la fila donde aparezca 'Process Date' y otros headers clave.
    """
    key = _norm("Process Date")
    candidates = []

    for i in range(min(len(df0), 80)):
        row = df0.iloc[i].astype(str).map(_norm).tolist()
        if key in row:
            score = 0
            for must in [
                "settlement date",
                "transaction type",
                "security type",
                "net amount (base currency)",
            ]:
                if _norm(must) in row:
                    score += 1
            candidates.append((score, i))

    if not candidates:
        return -1

    candidates.sort(reverse=True)
    return candidates[0][1]


def _to_float(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if s == "" or s == "-":
        return None

    s = s.replace(" ", "")

    # soporta "22.733,58" y "22733.58"
    if re.search(r"\d+,\d+$", s) and s.count(",") == 1 and s.count(".") >= 1:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return None


def _to_date(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def _read_pershing_excel(file_bytes: bytes) -> pd.DataFrame:
    """
    Lee el Excel y devuelve un DF con columnas canon.
    IMPORTANTE: evitamos sheet_name=None.
    """
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet0 = xls.sheet_names[0]
    df0 = pd.read_excel(xls, sheet_name=sheet0, header=None, dtype=object)

    hdr = _find_header_row(df0)
    if hdr < 0:
        raise ValueError("No pude detectar la fila de encabezados (Process Date).")

    headers = df0.iloc[hdr].astype(str).tolist()
    df = df0.iloc[hdr + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all")

    col_map: Dict[str, str] = {}
    for c in df.columns:
        cn = _norm(c)
        for canon in CANON_COLS:
            if cn == _norm(canon):
                col_map[c] = canon
                break

    df = df.rename(columns=col_map)

    keep = [c for c in CANON_COLS if c in df.columns]
    df = df[keep].copy()

    for date_col in ["Settlement Date", "Process Date", "Trade Date"]:
        if date_col in df.columns:
            df[date_col] = df[date_col].apply(_to_date)

    for num_col in [
        "Net Amount (Base Currency)",
        "Net Amount (Transaction Currency)",
        "Quantity",
        "Price (Transaction Currency)",
        "Commission",
        "FX Rate (To Base)",
    ]:
        if num_col in df.columns:
            df[num_col] = df[num_col].apply(_to_float)

    if "SYMBOL" in df.columns:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
        df.loc[df["SYMBOL"].isin(["nan", "None"]), "SYMBOL"] = ""

    if "Security Identifier" in df.columns:
        df["Security Identifier"] = df["Security Identifier"].astype(str).str.strip()
        df.loc[df["Security Identifier"].isin(["nan", "None"]), "Security Identifier"] = ""

    if "Security Type" in df.columns:
        df["Security Type"] = df["Security Type"].astype(str).str.strip().str.upper()

    if "Transaction Type" in df.columns:
        df["Transaction Type"] = df["Transaction Type"].astype(str).str.strip()

    if "Buy/Sell" in df.columns:
        df["Buy/Sell"] = df["Buy/Sell"].astype(str).str.strip().str.upper()
        df.loc[~df["Buy/Sell"].isin(["BUY", "SELL"]), "Buy/Sell"] = ""

    return df


# =========================
# Classification (fase actual)
# =========================
CASH_TX = {"FEDERAL FUNDS RECEIVED", "FEDERAL FUNDS SENT"}
INTERNAL_TX = {"ACTIVITY WITHIN YOUR ACCT"}

DIV_TX_MARKERS = [
    "CASH DIVIDEND RECEIVED",
    "FOREIGN SECURITY DIVIDEND RECEIVED",
]

TAX_MARKERS = [
    "NON-RESIDENT ALIEN TAX",
    "FOREIGN TAX WITHHELD",
]

FEE_MARKERS = [
    "FEE",
    "ADVISORY",
    "CUSTODY",
    "SUBSCRIPTION",
    "INT.",
    "INTEREST",
    "ASSET BASED FEE",
]


def _is_cash_movement(row: pd.Series) -> bool:
    tx = str(row.get("Transaction Type", "")).upper()
    if tx in INTERNAL_TX:
        return False
    return tx in CASH_TX


def _is_dividend(row: pd.Series) -> bool:
    tx = str(row.get("Transaction Type", "")).upper()
    return any(m in tx for m in DIV_TX_MARKERS)


def _is_tax(row: pd.Series) -> bool:
    tx = str(row.get("Transaction Type", "")).upper()
    code = str(row.get("Transaction code", "")).upper()
    desc = str(row.get("Transaction Description", "")).upper()

    if any(m in tx for m in TAX_MARKERS):
        return True
    if code in {"NRA", "FGN", "FGF"}:
        return True
    if "TAX" in tx:
        return True
    if "TAX WITHHELD" in desc:
        return True
    return False


def _is_fee(row: pd.Series) -> bool:
    tx = str(row.get("Transaction Type", "")).upper()
    desc = str(row.get("Transaction Description", "")).upper()
    code = str(row.get("Transaction code", "")).upper()
    comm = row.get("Commission", None)

    if isinstance(comm, (int, float)) and pd.notna(comm) and comm > 0:
        return True
    if "FEE" in tx or "FEE" in desc:
        return True
    if code in {"PDS", "NTF", "INM", "PCT", "/FG"}:
        return True
    if any(m in tx for m in FEE_MARKERS):
        return True
    return False


def _is_trade_real(row: pd.Series) -> bool:
    bs = str(row.get("Buy/Sell", "")).upper().strip()
    return bs in {"BUY", "SELL"}


ETF_TYPES = {"EXCHANGE TRADED FUNDS"}
STOCK_TYPES = {
    "COMMON STOCK",
    "COMMON STOCK ADR",
    "OPEN END TAXABLE LOAD FUND",
    "INDEX LINKED CORP BOND",
}


# =========================
# Enrichment / monthly base
# =========================
def _month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def _first_non_empty(series: pd.Series) -> str:
    vals = [str(x).strip() for x in series.dropna().tolist() if str(x).strip() not in {"", "-", "nan", "None"}]
    return vals[0] if vals else ""


def _prepare_analysis_df(df: pd.DataFrame, date_col: str = "Settlement Date") -> pd.DataFrame:
    out = df.copy()

    out = out[out[date_col].notna()].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()

    out["month_end"] = out[date_col].dt.to_period("M").dt.to_timestamp("M")
    out["month"] = out["month_end"].dt.strftime("%Y-%m")

    out["signed_qty"] = 0.0
    out.loc[out["Buy/Sell"].eq("BUY"), "signed_qty"] = out.loc[out["Buy/Sell"].eq("BUY"), "Quantity"].fillna(0.0)
    out.loc[out["Buy/Sell"].eq("SELL"), "signed_qty"] = -out.loc[out["Buy/Sell"].eq("SELL"), "Quantity"].fillna(0.0)

    out["analysis_bucket"] = "Other"

    mask_cash_in = out.apply(_is_cash_movement, axis=1) & out["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS RECEIVED")
    mask_cash_out = out.apply(_is_cash_movement, axis=1) & out["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS SENT")
    mask_div = out.apply(_is_dividend, axis=1)
    mask_tax = out.apply(_is_tax, axis=1)
    mask_fee = out.apply(_is_fee, axis=1)
    mask_buy = out.apply(_is_trade_real, axis=1) & out["Buy/Sell"].eq("BUY")
    mask_sell = out.apply(_is_trade_real, axis=1) & out["Buy/Sell"].eq("SELL")

    out.loc[mask_cash_in, "analysis_bucket"] = "Cash In"
    out.loc[mask_cash_out, "analysis_bucket"] = "Cash Out"
    out.loc[mask_div, "analysis_bucket"] = "Dividend"
    out.loc[mask_tax, "analysis_bucket"] = "Tax"
    out.loc[mask_fee, "analysis_bucket"] = "Fee"
    out.loc[mask_buy, "analysis_bucket"] = "Buy"
    out.loc[mask_sell, "analysis_bucket"] = "Sell"

    out["symbol_key"] = out["SYMBOL"].astype(str).str.strip()
    out.loc[out["symbol_key"].eq(""), "symbol_key"] = out["Security Identifier"].astype(str).str.strip()

    return out


def _build_monthly_consolidated(df: pd.DataFrame, date_col: str = "Settlement Date") -> pd.DataFrame:
    dfa = _prepare_analysis_df(df, date_col=date_col)

    if dfa.empty:
        return pd.DataFrame()

    month_index = pd.date_range(
        dfa["month_end"].min(),
        dfa["month_end"].max(),
        freq="ME",
    )
    base = pd.DataFrame({"month_end": month_index}).set_index("month_end")

    def _group_sum(mask: pd.Series, label: str, absolute: bool = False) -> pd.Series:
        temp = dfa.loc[mask, ["month_end", "Net Amount (Base Currency)"]].copy()
        if temp.empty:
            return pd.Series(dtype=float, name=label)
        if absolute:
            temp["Net Amount (Base Currency)"] = temp["Net Amount (Base Currency)"].abs()
        return temp.groupby("month_end")["Net Amount (Base Currency)"].sum().rename(label)

    monthly = base.copy()

    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Cash In"), "cash_in_base", absolute=False))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Cash Out"), "cash_out_base", absolute=True))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Buy"), "buys_base", absolute=True))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Sell"), "sells_base", absolute=False))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Dividend"), "dividends_base", absolute=False))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Tax"), "taxes_base", absolute=False))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Fee"), "fees_base", absolute=False))

    monthly = monthly.fillna(0.0)

    rows = dfa.groupby("month_end").size().rename("rows")
    unique_symbols = (
        dfa.loc[dfa["symbol_key"].astype(str).str.strip().ne("")]
        .groupby("month_end")["symbol_key"]
        .nunique()
        .rename("unique_symbols")
    )
    trade_count = dfa.loc[dfa["analysis_bucket"].isin(["Buy", "Sell"])].groupby("month_end").size().rename("trade_count")

    monthly = monthly.join(rows).join(unique_symbols).join(trade_count)
    monthly[["rows", "unique_symbols", "trade_count"]] = monthly[["rows", "unique_symbols", "trade_count"]].fillna(0)

    monthly["net_external_flow_base"] = monthly["cash_in_base"] - monthly["cash_out_base"]
    monthly["net_trading_flow_base"] = monthly["sells_base"] - monthly["buys_base"]
    monthly["net_income_costs_base"] = monthly["dividends_base"] + monthly["taxes_base"] + monthly["fees_base"]
    monthly["net_total_flow_base"] = (
        monthly["net_external_flow_base"]
        + monthly["net_trading_flow_base"]
        + monthly["net_income_costs_base"]
    )

    monthly = monthly.reset_index()
    monthly["month"] = monthly["month_end"].dt.strftime("%Y-%m")

    ordered_cols = [
        "month",
        "month_end",
        "cash_in_base",
        "cash_out_base",
        "buys_base",
        "sells_base",
        "dividends_base",
        "taxes_base",
        "fees_base",
        "net_external_flow_base",
        "net_trading_flow_base",
        "net_income_costs_base",
        "net_total_flow_base",
        "trade_count",
        "unique_symbols",
        "rows",
    ]
    return monthly[ordered_cols].copy()


def _build_monthly_positions(df: pd.DataFrame, date_col: str = "Settlement Date") -> pd.DataFrame:
    dfa = _prepare_analysis_df(df, date_col=date_col)
    trades = dfa[dfa.apply(_is_trade_real, axis=1)].copy()

    if trades.empty:
        return pd.DataFrame()

    trades = trades[trades["symbol_key"].astype(str).str.strip().ne("")].copy()

    trades["buy_qty"] = np.where(trades["Buy/Sell"].eq("BUY"), trades["Quantity"].fillna(0.0), 0.0)
    trades["sell_qty"] = np.where(trades["Buy/Sell"].eq("SELL"), trades["Quantity"].fillna(0.0), 0.0)
    trades["buy_amount_base"] = np.where(
        trades["Buy/Sell"].eq("BUY"),
        trades["Net Amount (Base Currency)"].abs().fillna(0.0),
        0.0,
    )
    trades["sell_amount_base"] = np.where(
        trades["Buy/Sell"].eq("SELL"),
        trades["Net Amount (Base Currency)"].fillna(0.0),
        0.0,
    )

    monthly_symbol = (
        trades.groupby(["month_end", "symbol_key"], dropna=False)
        .agg(
            qty_delta=("signed_qty", "sum"),
            buy_qty=("buy_qty", "sum"),
            sell_qty=("sell_qty", "sum"),
            buy_amount_base=("buy_amount_base", "sum"),
            sell_amount_base=("sell_amount_base", "sum"),
            trade_count=("symbol_key", "size"),
            security_description=("Security Description", _first_non_empty),
            security_type=("Security Type", _first_non_empty),
            isin=("ISIN", _first_non_empty),
            cusip=("CUSIP", _first_non_empty),
        )
        .reset_index()
    )

    symbol_master = (
        trades.groupby("symbol_key", dropna=False)
        .agg(
            security_description=("Security Description", _first_non_empty),
            security_type=("Security Type", _first_non_empty),
            isin=("ISIN", _first_non_empty),
            cusip=("CUSIP", _first_non_empty),
        )
        .reset_index()
    )

    all_months = pd.date_range(
        dfa["month_end"].min(),
        dfa["month_end"].max(),
        freq="ME",
    )

    grid = (
        symbol_master.assign(_k=1)
        .merge(pd.DataFrame({"month_end": all_months, "_k": 1}), on="_k", how="inner")
        .drop(columns="_k")
    )

    out = grid.merge(
        monthly_symbol[
            [
                "month_end",
                "symbol_key",
                "qty_delta",
                "buy_qty",
                "sell_qty",
                "buy_amount_base",
                "sell_amount_base",
                "trade_count",
            ]
        ],
        on=["month_end", "symbol_key"],
        how="left",
    )

    fill_cols = [
        "qty_delta",
        "buy_qty",
        "sell_qty",
        "buy_amount_base",
        "sell_amount_base",
        "trade_count",
    ]
    out[fill_cols] = out[fill_cols].fillna(0.0)

    out = out.sort_values(["symbol_key", "month_end"]).copy()
    out["closing_qty"] = out.groupby("symbol_key")["qty_delta"].cumsum()
    out["avg_buy_cost_month"] = np.where(
        out["buy_qty"] > 0,
        out["buy_amount_base"] / out["buy_qty"],
        np.nan,
    )

    # mostramos meses con posición abierta o con actividad en el mes
    out = out[(out["closing_qty"].round(10) != 0) | (out["trade_count"] > 0)].copy()

    out["month"] = out["month_end"].dt.strftime("%Y-%m")

    ordered_cols = [
        "month",
        "month_end",
        "symbol_key",
        "security_description",
        "security_type",
        "isin",
        "cusip",
        "buy_qty",
        "sell_qty",
        "qty_delta",
        "closing_qty",
        "buy_amount_base",
        "sell_amount_base",
        "avg_buy_cost_month",
        "trade_count",
    ]
    return out[ordered_cols].sort_values(["month_end", "symbol_key"]).copy()


# =========================
# Display helpers
# =========================
def _fmt_money(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.2f}"


def _kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="kpi">
    <div class="label">{label}</div>
    <div class="value">{value}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _select_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols_ok = [c for c in cols if c in df.columns]
    return df[cols_ok].copy()


def _prepare_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# Main render
# =========================
def render(_ctx=None) -> None:
    _inject_css()

    st.title("Movimientos CV — Transactions Analyzer")
    st.markdown(
        '<div class="subtle">Lectura, clasificación y análisis mensual base del comitente. Todo en Base Currency (USD) y filtrado por Settlement Date.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    up = st.file_uploader("Subí el Excel exportado (Transactions)", type=["xlsx", "xls"])

    ctrl_col1, ctrl_col2 = st.columns([1.0, 1.4])

    with ctrl_col1:
        st.caption("Fecha para análisis (global)")
        date_col = "Settlement Date"

    df_raw: Optional[pd.DataFrame] = None
    if up is not None:
        try:
            df_raw = _read_pershing_excel(up.getvalue())
        except Exception as e:
            st.error("No pude leer el Excel o detectar correctamente la fila de encabezados.")
            st.exception(e)
            return
    else:
        st.info("Subí el Excel para empezar.")
        return

    if date_col not in df_raw.columns:
        st.error("No encontré 'Settlement Date' en el archivo.")
        return

    df = df_raw.copy()
    df = df[df[date_col].notna()].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()

    if df.empty:
        st.warning("El archivo no tiene filas válidas con Settlement Date.")
        return

    min_d = df[date_col].min().date()
    max_d = df[date_col].max().date()

    with ctrl_col2:
        st.caption("Desde / Hasta (global)")
        from_d, to_d = st.date_input(
            "Rango",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            label_visibility="collapsed",
        )

    df = df[(df[date_col].dt.date >= from_d) & (df[date_col].dt.date <= to_d)].copy()

    # =========================
    # Clasificación actual
    # =========================
    df_cash = df[df.apply(_is_cash_movement, axis=1)].copy()

    df_etf = df[
        (df.get("Security Type", "").astype(str).str.upper().isin(ETF_TYPES))
        & (df.apply(_is_trade_real, axis=1))
    ].copy()

    df_stock = df[
        (df.get("Security Type", "").astype(str).str.upper().isin(STOCK_TYPES))
        & (df.apply(_is_trade_real, axis=1))
    ].copy()

    df_div_gross = df[df.apply(_is_dividend, axis=1)].copy()
    df_tax_all = df[df.apply(_is_tax, axis=1)].copy()

    if "SYMBOL" in df.columns:
        div_symbols = df_div_gross.get("SYMBOL", pd.Series(dtype=str)).astype(str).tolist()
        df_tax_div = df_tax_all[df_tax_all["SYMBOL"].astype(str).isin(div_symbols)].copy()
    else:
        df_tax_div = df_tax_all.copy()

    agg_keys = [date_col]
    if "SYMBOL" in df.columns:
        agg_keys.append("SYMBOL")

    def _sumcol(dfx: pd.DataFrame, col: str) -> pd.Series:
        if col not in dfx.columns or dfx.empty:
            return pd.Series(dtype=float)
        return dfx.groupby(agg_keys)[col].sum()

    gross = _sumcol(df_div_gross, "Net Amount (Base Currency)").rename("Dividend (Gross, Base)")
    tax = _sumcol(df_tax_div, "Net Amount (Base Currency)").rename("Dividend Tax (Base)")
    div_table = pd.concat([gross, tax], axis=1).fillna(0.0)
    div_table["Dividend (Net, Base)"] = div_table["Dividend (Gross, Base)"] + div_table["Dividend Tax (Base)"]
    div_table = div_table.reset_index()

    df_fee = df[df.apply(_is_fee, axis=1)].copy()
    df_fee = df_fee[~df_fee.apply(_is_cash_movement, axis=1)]
    df_fee = df_fee[~df_fee.apply(_is_tax, axis=1)]
    df_fee = df_fee[~df_fee.apply(_is_dividend, axis=1)]

    df_tax = df_tax_all.copy()

    # =========================
    # Nuevas tablas mensuales
    # =========================
    monthly_cons = _build_monthly_consolidated(df, date_col=date_col)
    monthly_pos = _build_monthly_positions(df, date_col=date_col)

    # =========================
    # Tabs
    # =========================
    tab_over, tab_cash, tab_etf, tab_stock, tab_div, tab_fee, tab_tax = st.tabs(
        ["Overview", "Cash Movement", "ETFs", "Stocks", "Dividends", "Fees", "Taxes"]
    )

    # =========================
    # Overview
    # =========================
    with tab_over:
        c1, c2, c3, c4 = st.columns(4)

        cash_in = (
            df_cash[df_cash["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS RECEIVED")]["Net Amount (Base Currency)"].sum()
            if "Net Amount (Base Currency)" in df_cash.columns
            else 0.0
        )
        cash_out = (
            df_cash[df_cash["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS SENT")]["Net Amount (Base Currency)"].sum()
            if "Net Amount (Base Currency)" in df_cash.columns
            else 0.0
        )

        etf_trades = len(df_etf)
        stock_trades = len(df_stock)

        with c1:
            _kpi("Cash In (Base)", _fmt_money(cash_in))
        with c2:
            _kpi("Cash Out (Base)", _fmt_money(abs(cash_out)))
        with c3:
            _kpi("ETF Trades", f"{etf_trades:,}")
        with c4:
            _kpi("Stock Trades", f"{stock_trades:,}")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="pill">Rango: {from_d} → {to_d}</span> &nbsp; '
            f'<span class="pill">Filas: {len(df):,}</span>',
            unsafe_allow_html=True,
        )

        st.caption("Fase actual: lectura, clasificación y orden mensual base. Todavía sin valuation ni rentabilidad.")
        st.dataframe(df.head(25), use_container_width=True, height=420)

    # =========================
    # Cash Movement
    # =========================
    with tab_cash:
        st.subheader("Cash Movement")
        st.caption("Solo FEDERAL FUNDS RECEIVED / FEDERAL FUNDS SENT (excluye ACTIVITY WITHIN YOUR ACCT).")

        cols_cash = [
            "Process Date",
            "Settlement Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Security Description",
            "Transaction Description",
        ]
        view = _select_columns(df_cash, cols_cash).sort_values("Settlement Date")
        st.dataframe(view, use_container_width=True, height=520)

    # =========================
    # ETFs
    # =========================
    with tab_etf:
        st.subheader("ETFs")
        st.caption("Security Type = EXCHANGE TRADED FUNDS y solo BUY/SELL reales.")

        cols_long = [
            "Settlement Date",
            "Process Date",
            "Net Amount (Base Currency)",
            "Transaction Description",
            "Transaction Type",
            "Security Description",
            "Net Amount (Transaction Currency)",
            "Buy/Sell",
            "Quantity",
            "Price (Transaction Currency)",
            "Transaction Currency",
            "Security Type",
            "Payee",
            "Paid For (Name)",
            "Request Reason",
            "CUSIP",
            "FX Rate (To Base)",
            "ISIN",
            "SEDOL",
            "SYMBOL",
            "Trade Date",
        ]
        view = _select_columns(df_etf, cols_long).sort_values("Settlement Date")
        st.dataframe(view, use_container_width=True, height=520)

    # =========================
    # Stocks
    # =========================
    with tab_stock:
        st.subheader("Stocks")
        st.caption("Incluye COMMON STOCK + COMMON STOCK ADR. Por ahora también: OPEN END TAXABLE LOAD FUND + INDEX LINKED CORP BOND.")

        cols_long = [
            "Settlement Date",
            "Process Date",
            "Net Amount (Base Currency)",
            "Transaction Description",
            "Transaction Type",
            "Security Description",
            "Net Amount (Transaction Currency)",
            "Buy/Sell",
            "Quantity",
            "Price (Transaction Currency)",
            "Transaction Currency",
            "Security Type",
            "Payee",
            "Paid For (Name)",
            "Request Reason",
            "CUSIP",
            "FX Rate (To Base)",
            "ISIN",
            "SEDOL",
            "SYMBOL",
            "Trade Date",
        ]
        view = _select_columns(df_stock, cols_long).sort_values("Settlement Date")
        st.dataframe(view, use_container_width=True, height=520)

    # =========================
    # Dividends
    # =========================
    with tab_div:
        st.subheader("Dividends")
        st.caption("Neto = Dividend (gross) + Dividend tax (negativo). Agregado por Settlement Date y Symbol.")

        show_cols = agg_keys + ["Dividend (Gross, Base)", "Dividend Tax (Base)", "Dividend (Net, Base)"]
        st.dataframe(div_table[show_cols].sort_values(agg_keys), use_container_width=True, height=520)

        with st.expander("Ver filas raw (dividendos)"):
            st.dataframe(
                _select_columns(
                    df_div_gross,
                    [
                        "Settlement Date",
                        "SYMBOL",
                        "Net Amount (Base Currency)",
                        "Transaction Type",
                        "Security Description",
                        "Transaction Description",
                        "Transaction code",
                    ],
                ).sort_values(["Settlement Date", "SYMBOL"]),
                use_container_width=True,
                height=420,
            )

        with st.expander("Ver filas raw (taxes asociados / candidatos)"):
            st.dataframe(
                _select_columns(
                    df_tax_div,
                    [
                        "Settlement Date",
                        "SYMBOL",
                        "Net Amount (Base Currency)",
                        "Transaction Type",
                        "Security Description",
                        "Transaction Description",
                        "Transaction code",
                    ],
                ).sort_values(["Settlement Date", "SYMBOL"]),
                use_container_width=True,
                height=420,
            )

    # =========================
    # Fees
    # =========================
    with tab_fee:
        st.subheader("Fees")
        st.caption("Fees/costs (clasificación simple). No incluye Cash Movement, Taxes ni Dividends.")

        cols_fee = [
            "Settlement Date",
            "Process Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Transaction Description",
            "Security Description",
            "Transaction code",
            "Commission",
        ]
        view = _select_columns(df_fee, cols_fee).sort_values("Settlement Date")
        st.dataframe(view, use_container_width=True, height=520)

    # =========================
    # Taxes
    # =========================
    with tab_tax:
        st.subheader("Taxes")
        st.caption("Taxes (incluye NRA / foreign tax withheld / etc).")

        cols_tax = [
            "Settlement Date",
            "Process Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Transaction Description",
            "Security Description",
            "Transaction code",
        ]
        view = _select_columns(df_tax, cols_tax).sort_values("Settlement Date")
        st.dataframe(view, use_container_width=True, height=520)

    # =========================
    # BLOQUE NUEVO — MONTHLY ANALYSIS
    # =========================
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.header("Monthly Analysis")
    st.caption("Base mensual consolidada y posición teórica a cierre de cada mes. Todavía sin valuation ni rentabilidad.")

    if monthly_cons.empty:
        st.warning("No pude construir la tabla mensual consolidada.")
        return

    months_available = monthly_cons["month"].tolist()
    default_month = months_available[-1] if months_available else None

    f1, f2 = st.columns([1.0, 1.2])

    with f1:
        selected_month = st.selectbox(
            "Mes a analizar",
            options=months_available,
            index=len(months_available) - 1 if months_available else 0,
        )

    with f2:
        if not monthly_pos.empty:
            security_types = ["Todos"] + sorted(
                [x for x in monthly_pos["security_type"].dropna().astype(str).unique().tolist() if x.strip() != ""]
            )
        else:
            security_types = ["Todos"]

        selected_sec_type = st.selectbox(
            "Security Type",
            options=security_types,
            index=0,
        )

    tab_m1, tab_m2, tab_m3 = st.tabs(
        ["Resumen mensual", "Posición a cierre", "Detalle del mes"]
    )

    # -------------------------
    # Resumen mensual
    # -------------------------
    with tab_m1:
        last_row = monthly_cons[monthly_cons["month"].eq(selected_month)].copy()

        if not last_row.empty:
            r = last_row.iloc[0]

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                _kpi("Cash In", _fmt_money(r["cash_in_base"]))
            with k2:
                _kpi("Cash Out", _fmt_money(r["cash_out_base"]))
            with k3:
                _kpi("Net Trading Flow", _fmt_money(r["net_trading_flow_base"]))
            with k4:
                _kpi("Net Total Flow", _fmt_money(r["net_total_flow_base"]))

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.dataframe(monthly_cons, use_container_width=True, height=420)

        st.download_button(
            "Descargar tabla mensual consolidada (CSV)",
            data=_prepare_download(monthly_cons),
            file_name="monthly_consolidated.csv",
            mime="text/csv",
        )

    # -------------------------
    # Posición a cierre
    # -------------------------
    with tab_m2:
        st.subheader("Posición teórica al cierre del mes")
        st.caption("Se reconstruye acumulando BUY y SELL por símbolo. No incluye todavía precio de cierre ni market value.")

        if monthly_pos.empty:
            st.info("No encontré trades reales para construir posición mensual.")
        else:
            pos_view = monthly_pos[monthly_pos["month"].eq(selected_month)].copy()

            if selected_sec_type != "Todos":
                pos_view = pos_view[pos_view["security_type"].astype(str).eq(selected_sec_type)].copy()

            pos_view = pos_view.sort_values(["closing_qty", "symbol_key"], ascending=[False, True])

            c1, c2, c3 = st.columns(3)
            with c1:
                _kpi("Símbolos abiertos", f"{pos_view['symbol_key'].nunique():,}")
            with c2:
                _kpi("Trades del mes", f"{int(pos_view['trade_count'].sum()):,}")
            with c3:
                _kpi("Qty neta (suma)", f"{pos_view['closing_qty'].sum():,.4f}")

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.dataframe(pos_view, use_container_width=True, height=460)

            st.download_button(
                f"Descargar posición cierre {selected_month} (CSV)",
                data=_prepare_download(pos_view),
                file_name=f"monthly_position_{selected_month}.csv",
                mime="text/csv",
            )

    # -------------------------
    # Detalle del mes
    # -------------------------
    with tab_m3:
        st.subheader("Detalle del mes seleccionado")

        dfa = _prepare_analysis_df(df, date_col=date_col)
        detail_month = dfa[dfa["month"].eq(selected_month)].copy()

        if selected_sec_type != "Todos" and not detail_month.empty:
            detail_month = detail_month[detail_month["Security Type"].astype(str).eq(selected_sec_type)].copy()

        cols_detail = [
            "Settlement Date",
            "Process Date",
            "analysis_bucket",
            "symbol_key",
            "Security Description",
            "Security Type",
            "Transaction Type",
            "Transaction Description",
            "Buy/Sell",
            "Quantity",
            "Net Amount (Base Currency)",
            "Price (Transaction Currency)",
            "Transaction Currency",
            "Commission",
            "ISIN",
            "CUSIP",
        ]
        detail_view = _select_columns(detail_month, cols_detail).sort_values(["Settlement Date", "symbol_key"])

        st.dataframe(detail_view, use_container_width=True, height=520)

        st.download_button(
            f"Descargar detalle {selected_month} (CSV)",
            data=_prepare_download(detail_view),
            file_name=f"monthly_detail_{selected_month}.csv",
            mime="text/csv",
        )
