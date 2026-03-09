# tools/comerciales/transactions_analyzer.py
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# UI (minimal)
# =========================
def _inject_css() -> None:
    st.markdown(
        """
<style>
    .block-container {
        padding-top: 1.1rem;
        max-width: 1120px;
    }

    h1 {
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    .subtle {
        color: rgba(0,0,0,0.55);
        font-size: 0.96rem;
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
    }

    .kpi {
        padding: 14px 16px;
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 16px;
        background: #fff;
    }

    .kpi .label {
        color: rgba(0,0,0,0.52);
        font-size: 0.85rem;
        margin-bottom: 6px;
    }

    .kpi .value {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.10);
        font-size: 0.82rem;
        color: rgba(0,0,0,0.68);
        background: #fff;
    }

    .hr {
        height: 1px;
        background: rgba(0,0,0,0.08);
        margin: 12px 0 14px;
    }

    .section-space {
        height: 12px;
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
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet0 = xls.sheet_names[0]
    df0 = pd.read_excel(xls, sheet_name=sheet0, header=None, dtype=object)

    hdr = _find_header_row(df0)
    if hdr < 0:
        raise ValueError("No pude detectar la fila de encabezados (Process Date).")

    headers = df0.iloc[hdr].astype(str).tolist()
    df = df0.iloc[hdr + 1:].copy()
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
# Classification
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
# Analysis helpers
# =========================
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

    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Cash In"), "cash_in_base"))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Cash Out"), "cash_out_base", absolute=True))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Buy"), "buys_base", absolute=True))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Sell"), "sells_base"))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Dividend"), "dividends_base"))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Tax"), "taxes_base"))
    monthly = monthly.join(_group_sum(dfa["analysis_bucket"].eq("Fee"), "fees_base"))

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

    return monthly[
        [
            "month",
            "cash_in_base",
            "cash_out_base",
            "buys_base",
            "sells_base",
            "dividends_base",
            "taxes_base",
            "fees_base",
            "net_total_flow_base",
        ]
    ].copy()


def _build_monthly_positions(df: pd.DataFrame, date_col: str = "Settlement Date") -> pd.DataFrame:
    dfa = _prepare_analysis_df(df, date_col=date_col)
    trades = dfa[dfa.apply(_is_trade_real, axis=1)].copy()

    if trades.empty:
        return pd.DataFrame()

    trades = trades[trades["symbol_key"].astype(str).str.strip().ne("")].copy()

    monthly_symbol = (
        trades.groupby(["month_end", "symbol_key"], dropna=False)
        .agg(
            qty_delta=("signed_qty", "sum"),
            security_description=("Security Description", _first_non_empty),
            security_type=("Security Type", _first_non_empty),
        )
        .reset_index()
    )

    symbol_master = (
        trades.groupby("symbol_key", dropna=False)
        .agg(
            security_description=("Security Description", _first_non_empty),
            security_type=("Security Type", _first_non_empty),
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
        monthly_symbol[["month_end", "symbol_key", "qty_delta"]],
        on=["month_end", "symbol_key"],
        how="left",
    )

    out["qty_delta"] = out["qty_delta"].fillna(0.0)
    out = out.sort_values(["symbol_key", "month_end"]).copy()
    out["closing_qty"] = out.groupby("symbol_key")["qty_delta"].cumsum()

    out = out[out["closing_qty"].round(10) != 0].copy()
    out["month"] = out["month_end"].dt.strftime("%Y-%m")

    return out[
        [
            "month",
            "symbol_key",
            "security_description",
            "security_type",
            "closing_qty",
        ]
    ].sort_values(["month", "symbol_key"]).copy()


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


def _show_df(df: pd.DataFrame, height: int = 420) -> None:
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)


# =========================
# Main
# =========================
def render(_ctx=None) -> None:
    _inject_css()

    st.title("Movimientos CV — Transactions Analyzer")
    st.markdown(
        '<div class="subtle">Lectura, clasificación y análisis mensual base del comitente. Todo en Base Currency (USD).</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    up = st.file_uploader("Subí el Excel exportado (Transactions)", type=["xlsx", "xls"])

    if up is None:
        st.info("Subí el Excel para empezar.")
        return

    try:
        df_raw = _read_pershing_excel(up.getvalue())
    except Exception as e:
        st.error("No pude leer el Excel o detectar correctamente la fila de encabezados.")
        st.exception(e)
        return

    date_col = "Settlement Date"

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

    # =========================
    # Clasificación base
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

    monthly_cons = _build_monthly_consolidated(df, date_col=date_col)
    monthly_pos = _build_monthly_positions(df, date_col=date_col)

    # =========================
    # TOP: Overview tabs
    # =========================
    tab_over, tab_cash, tab_etf, tab_stock, tab_div, tab_fee, tab_tax = st.tabs(
        ["Overview", "Cash", "ETFs", "Stocks", "Dividends", "Fees", "Taxes"]
    )

    with tab_over:
        cash_in = (
            df_cash[df_cash["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS RECEIVED")]["Net Amount (Base Currency)"].sum()
            if "Net Amount (Base Currency)" in df_cash.columns
            else 0.0
        )
        cash_out = (
            df_cash[df_cash["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS SENT")]["Net Amount (Base Currency)"].abs().sum()
            if "Net Amount (Base Currency)" in df_cash.columns
            else 0.0
        )
        etf_trades = len(df_etf)
        stock_trades = len(df_stock)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _kpi("Cash In", _fmt_money(cash_in))
        with c2:
            _kpi("Cash Out", _fmt_money(cash_out))
        with c3:
            _kpi("ETF Trades", f"{etf_trades:,}")
        with c4:
            _kpi("Stock Trades", f"{stock_trades:,}")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="pill">Filas: {len(df):,}</span>',
            unsafe_allow_html=True,
        )

    with tab_cash:
        cols_cash = [
            "Settlement Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Transaction Description",
        ]
        view = _select_columns(df_cash, cols_cash).sort_values("Settlement Date")
        _show_df(view, height=440)

    with tab_etf:
        cols_etf = [
            "Settlement Date",
            "SYMBOL",
            "Security Description",
            "Buy/Sell",
            "Quantity",
            "Net Amount (Base Currency)",
        ]
        view = _select_columns(df_etf, cols_etf).sort_values("Settlement Date")
        _show_df(view, height=440)

    with tab_stock:
        cols_stock = [
            "Settlement Date",
            "SYMBOL",
            "Security Description",
            "Buy/Sell",
            "Quantity",
            "Net Amount (Base Currency)",
        ]
        view = _select_columns(df_stock, cols_stock).sort_values("Settlement Date")
        _show_df(view, height=440)

    with tab_div:
        show_cols = agg_keys + ["Dividend (Gross, Base)", "Dividend Tax (Base)", "Dividend (Net, Base)"]
        view = div_table[show_cols].sort_values(agg_keys) if not div_table.empty else div_table
        _show_df(view, height=440)

    with tab_fee:
        cols_fee = [
            "Settlement Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Transaction Description",
        ]
        view = _select_columns(df_fee, cols_fee).sort_values("Settlement Date")
        _show_df(view, height=440)

    with tab_tax:
        cols_tax = [
            "Settlement Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Transaction Description",
        ]
        view = _select_columns(df_tax, cols_tax).sort_values("Settlement Date")
        _show_df(view, height=440)

    # =========================
    # MONTHLY ANALYSIS
    # =========================
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.header("Monthly Analysis")
    st.markdown(
        '<div class="subtle">Vista mensual simple y ordenada del comitente.</div>',
        unsafe_allow_html=True,
    )

    if monthly_cons.empty:
        st.warning("No pude construir la tabla mensual.")
        return

    filter_cols = st.columns([1.0, 1.0, 3.0])

    with filter_cols[0]:
        months_available = monthly_cons["month"].tolist()
        selected_month = st.selectbox(
            "Mes",
            options=months_available,
            index=len(months_available) - 1 if months_available else 0,
        )

    with filter_cols[1]:
        if not monthly_pos.empty:
            security_types = ["Todos"] + sorted(
                [x for x in monthly_pos["security_type"].dropna().astype(str).unique().tolist() if x.strip() != ""]
            )
        else:
            security_types = ["Todos"]

        selected_sec_type = st.selectbox(
            "Tipo",
            options=security_types,
            index=0,
        )

    tab_m1, tab_m2, tab_m3 = st.tabs(["Resumen mensual", "Posición a cierre", "Detalle del mes"])

    with tab_m1:
        row = monthly_cons[monthly_cons["month"].eq(selected_month)].copy()

        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                _kpi("Cash In", _fmt_money(r["cash_in_base"]))
            with c2:
                _kpi("Cash Out", _fmt_money(r["cash_out_base"]))
            with c3:
                _kpi("Net Flow", _fmt_money(r["net_total_flow_base"]))

        st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

        summary_view = monthly_cons.rename(
            columns={
                "month": "Month",
                "cash_in_base": "Cash In",
                "cash_out_base": "Cash Out",
                "buys_base": "Buys",
                "sells_base": "Sells",
                "dividends_base": "Dividends",
                "taxes_base": "Taxes",
                "fees_base": "Fees",
                "net_total_flow_base": "Net Flow",
            }
        )

        _show_df(summary_view, height=360)

    with tab_m2:
        st.subheader("Posición a cierre")
        st.caption("Solo cantidad acumulada al cierre por símbolo.")

        if monthly_pos.empty:
            st.info("No encontré trades reales para construir la posición.")
        else:
            pos_view = monthly_pos[monthly_pos["month"].eq(selected_month)].copy()

            if selected_sec_type != "Todos":
                pos_view = pos_view[pos_view["security_type"].astype(str).eq(selected_sec_type)].copy()

            pos_view = pos_view.rename(
                columns={
                    "month": "Month",
                    "symbol_key": "Symbol",
                    "security_description": "Description",
                    "security_type": "Type",
                    "closing_qty": "Closing Qty",
                }
            )

            pos_view = pos_view[["Symbol", "Description", "Type", "Closing Qty"]].sort_values(
                ["Closing Qty", "Symbol"],
                ascending=[False, True],
            )

            c1, c2 = st.columns(2)
            with c1:
                _kpi("Símbolos", f"{pos_view['Symbol'].nunique():,}")
            with c2:
                _kpi("Closing Qty Total", f"{pos_view['Closing Qty'].sum():,.4f}")

            st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
            _show_df(pos_view, height=460)

    with tab_m3:
        detail_month = _prepare_analysis_df(df, date_col=date_col)
        detail_month = detail_month[detail_month["month"].eq(selected_month)].copy()

        if selected_sec_type != "Todos" and not detail_month.empty:
            detail_month = detail_month[detail_month["Security Type"].astype(str).eq(selected_sec_type)].copy()

        cols_detail = [
            "Settlement Date",
            "analysis_bucket",
            "symbol_key",
            "Security Description",
            "Buy/Sell",
            "Quantity",
            "Net Amount (Base Currency)",
        ]
        detail_view = _select_columns(detail_month, cols_detail).sort_values(["Settlement Date", "symbol_key"])

        detail_view = detail_view.rename(
            columns={
                "Settlement Date": "Settlement Date",
                "analysis_bucket": "Type",
                "symbol_key": "Symbol",
                "Security Description": "Description",
                "Buy/Sell": "Buy/Sell",
                "Quantity": "Qty",
                "Net Amount (Base Currency)": "Base Amount",
            }
        )

        _show_df(detail_view, height=460)
