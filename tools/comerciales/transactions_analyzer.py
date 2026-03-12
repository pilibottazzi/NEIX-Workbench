# tools/comerciales/transactions_analyzer.py
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================================================
# UI (minimal, limpio, ejecutivo)
# =========================================================
NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17, 24, 39, 0.08)"
POS_BG = "rgba(16, 185, 129, 0.10)"
POS_TXT = "#047857"
NEG_BG = "rgba(239, 68, 68, 0.10)"
NEG_TXT = "#b91c1c"
NEU_BG = "rgba(107, 114, 128, 0.10)"
NEU_TXT = "#4b5563"


def _inject_css() -> None:
    st.markdown(
        f"""
<style>
    .block-container {{
        padding-top: 1.05rem;
        max-width: 1180px;
        padding-bottom: 2.2rem;
    }}

    h1, h2, h3 {{
        letter-spacing: -0.02em;
        color: {TEXT};
    }}

    .ta-subtle {{
        color: rgba(17, 24, 39, 0.62);
        font-size: 0.95rem;
        margin-top: 0.1rem;
        margin-bottom: 0.25rem;
    }}

    .ta-hr {{
        height: 1px;
        background: rgba(17, 24, 39, 0.08);
        margin: 14px 0 18px 0;
    }}

    .ta-kpi {{
        position: relative;
        padding: 16px 17px;
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 18px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(250,250,250,0.98) 100%);
        box-shadow:
            0 8px 24px rgba(17, 24, 39, 0.05),
            0 1px 0 rgba(255,255,255,0.75) inset;
        min-height: 108px;
        overflow: hidden;
    }}

    .ta-kpi::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 3px;
        background: rgba(17, 24, 39, 0.06);
    }}

    .ta-kpi.pos::before {{
        background: linear-gradient(90deg, rgba(16,185,129,0.95), rgba(16,185,129,0.25));
    }}

    .ta-kpi.neg::before {{
        background: linear-gradient(90deg, rgba(239,68,68,0.95), rgba(239,68,68,0.25));
    }}

    .ta-kpi.neu::before {{
        background: linear-gradient(90deg, rgba(107,114,128,0.85), rgba(107,114,128,0.20));
    }}

    .ta-kpi-label {{
        color: rgba(17, 24, 39, 0.56);
        font-size: 0.80rem;
        font-weight: 600;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .ta-kpi-value {{
        font-size: 1.68rem;
        font-weight: 750;
        line-height: 1.05;
        letter-spacing: -0.04em;
        color: {TEXT};
        margin-bottom: 10px;
    }}

    .ta-kpi-delta {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        width: fit-content;
    }}

    .ta-kpi-delta.pos {{
        background: {POS_BG};
        color: {POS_TXT};
    }}

    .ta-kpi-delta.neg {{
        background: {NEG_BG};
        color: {NEG_TXT};
    }}

    .ta-kpi-delta.neu {{
        background: {NEU_BG};
        color: {NEU_TXT};
    }}

    .ta-pill {{
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        border: 1px solid rgba(17, 24, 39, 0.10);
        background: #ffffff;
        color: rgba(17, 24, 39, 0.68);
        font-size: 0.80rem;
        margin-right: 8px;
        margin-bottom: 6px;
    }}

    .ta-section-gap {{
        height: 10px;
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Parsing helpers
# =========================================================
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


ETF_TYPES = {"EXCHANGE TRADED FUNDS"}

STOCK_TYPES = {
    "COMMON STOCK",
    "COMMON STOCK ADR",
}

BOND_TYPES = {
    "CORP BOND",
    "CORPORATE BOND",
    "INDEX LINKED CORP BOND",
    "GOVERNMENT BOND",
    "MUNICIPAL BOND",
}

FUND_TYPES = {
    "OPEN END TAXABLE LOAD FUND",
    "OPEN END FUND",
    "MUTUAL FUND",
    "MONEY MARKET FUND",
}

CASH_TX = {
    "FEDERAL FUNDS RECEIVED",
    "FEDERAL FUNDS SENT",
}

INTERNAL_TX = {
    "ACTIVITY WITHIN YOUR ACCT",
}

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

    if "Transaction Description" in df.columns:
        df["Transaction Description"] = df["Transaction Description"].astype(str).str.strip()

    if "Buy/Sell" in df.columns:
        df["Buy/Sell"] = df["Buy/Sell"].astype(str).str.strip().str.upper()
        df.loc[~df["Buy/Sell"].isin(["BUY", "SELL"]), "Buy/Sell"] = ""

    return df


# =========================================================
# Classification helpers
# =========================================================
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


def _first_non_empty(series: pd.Series) -> str:
    vals = [
        str(x).strip()
        for x in series.dropna().tolist()
        if str(x).strip() not in {"", "-", "nan", "None"}
    ]
    return vals[0] if vals else ""


def _classify_asset_bucket(row: pd.Series) -> str:
    sec_type = str(row.get("Security Type", "")).upper().strip()

    if sec_type in ETF_TYPES:
        return "ETF"
    if sec_type in STOCK_TYPES:
        return "Stock"
    if sec_type in BOND_TYPES:
        return "Bond"
    if sec_type in FUND_TYPES:
        return "Fund"
    if sec_type in {"CURRENCY", "FOREIGN CURRENCY"}:
        return "Currency"
    if sec_type == "":
        return "Other"
    return "Other"


def _prepare_analysis_df(df: pd.DataFrame, date_col: str = "Settlement Date") -> pd.DataFrame:
    out = df.copy()

    out = out[out[date_col].notna()].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()

    out["month_end"] = out[date_col].dt.to_period("M").dt.to_timestamp("M")
    out["month"] = out["month_end"].dt.strftime("%Y-%m")

    out["signed_qty"] = 0.0
    out.loc[out["Buy/Sell"].eq("BUY"), "signed_qty"] = out.loc[
        out["Buy/Sell"].eq("BUY"), "Quantity"
    ].fillna(0.0)
    out.loc[out["Buy/Sell"].eq("SELL"), "signed_qty"] = -out.loc[
        out["Buy/Sell"].eq("SELL"), "Quantity"
    ].fillna(0.0)

    out["flow_bucket"] = "Other"

    mask_internal = out["Transaction Type"].astype(str).str.upper().isin(INTERNAL_TX)
    mask_cash_in = (
        out.apply(_is_cash_movement, axis=1)
        & out["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS RECEIVED")
    )
    mask_cash_out = (
        out.apply(_is_cash_movement, axis=1)
        & out["Transaction Type"].astype(str).str.upper().eq("FEDERAL FUNDS SENT")
    )
    mask_div = out.apply(_is_dividend, axis=1)
    mask_tax = out.apply(_is_tax, axis=1)
    mask_fee = out.apply(_is_fee, axis=1)
    mask_buy = out.apply(_is_trade_real, axis=1) & out["Buy/Sell"].eq("BUY")
    mask_sell = out.apply(_is_trade_real, axis=1) & out["Buy/Sell"].eq("SELL")

    out.loc[mask_internal, "flow_bucket"] = "Internal"
    out.loc[mask_cash_in, "flow_bucket"] = "Cash In"
    out.loc[mask_cash_out, "flow_bucket"] = "Cash Out"
    out.loc[mask_div, "flow_bucket"] = "Dividend"
    out.loc[mask_tax, "flow_bucket"] = "Tax"
    out.loc[mask_fee, "flow_bucket"] = "Fee"
    out.loc[mask_buy, "flow_bucket"] = "Buy"
    out.loc[mask_sell, "flow_bucket"] = "Sell"

    out["symbol_key"] = out["SYMBOL"].astype(str).str.strip()
    out.loc[out["symbol_key"].eq(""), "symbol_key"] = out["Security Identifier"].astype(str).str.strip()

    out["asset_bucket"] = out.apply(_classify_asset_bucket, axis=1)

    return out


# =========================================================
# Builds
# =========================================================
def _build_monthly_consolidated(dfa: pd.DataFrame) -> pd.DataFrame:
    if dfa.empty:
        return pd.DataFrame()

    month_index = pd.date_range(
        dfa["month_end"].min(),
        dfa["month_end"].max(),
        freq="ME",
    )
    monthly = pd.DataFrame({"month_end": month_index}).set_index("month_end")

    def _group_sum(mask: pd.Series, label: str, absolute: bool = False) -> pd.Series:
        temp = dfa.loc[mask, ["month_end", "Net Amount (Base Currency)"]].copy()
        if temp.empty:
            return pd.Series(dtype=float, name=label)

        if absolute:
            temp["Net Amount (Base Currency)"] = temp["Net Amount (Base Currency)"].abs()

        return temp.groupby("month_end")["Net Amount (Base Currency)"].sum().rename(label)

    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Cash In"), "cash_in_base"))
    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Cash Out"), "cash_out_base", absolute=True))
    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Buy"), "buys_base", absolute=True))
    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Sell"), "sells_base"))
    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Dividend"), "dividends_base"))
    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Tax"), "taxes_base"))
    monthly = monthly.join(_group_sum(dfa["flow_bucket"].eq("Fee"), "fees_base"))

    monthly = monthly.fillna(0.0)

    monthly["net_external_flow_base"] = monthly["cash_in_base"] - monthly["cash_out_base"]
    monthly["net_trading_flow_base"] = monthly["sells_base"] - monthly["buys_base"]
    monthly["net_income_costs_base"] = (
        monthly["dividends_base"] + monthly["taxes_base"] + monthly["fees_base"]
    )
    monthly["net_total_flow_base"] = (
        monthly["net_external_flow_base"]
        + monthly["net_trading_flow_base"]
        + monthly["net_income_costs_base"]
    )

    monthly["trade_count"] = (
        dfa.loc[dfa["flow_bucket"].isin(["Buy", "Sell"])]
        .groupby("month_end")
        .size()
        .reindex(monthly.index)
        .fillna(0)
    )

    monthly["rows"] = (
        dfa.groupby("month_end")
        .size()
        .reindex(monthly.index)
        .fillna(0)
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
            "trade_count",
            "rows",
        ]
    ].copy()


def _build_monthly_positions(dfa: pd.DataFrame) -> pd.DataFrame:
    trades = dfa[dfa["flow_bucket"].isin(["Buy", "Sell"])].copy()

    if trades.empty:
        return pd.DataFrame()

    trades = trades[trades["symbol_key"].astype(str).str.strip().ne("")].copy()

    monthly_symbol = (
        trades.groupby(["month_end", "symbol_key"], dropna=False)
        .agg(
            qty_delta=("signed_qty", "sum"),
            security_description=("Security Description", _first_non_empty),
            security_type=("Security Type", _first_non_empty),
            asset_bucket=("asset_bucket", _first_non_empty),
        )
        .reset_index()
    )

    symbol_master = (
        trades.groupby("symbol_key", dropna=False)
        .agg(
            security_description=("Security Description", _first_non_empty),
            security_type=("Security Type", _first_non_empty),
            asset_bucket=("asset_bucket", _first_non_empty),
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
            "asset_bucket",
            "closing_qty",
        ]
    ].sort_values(["month", "symbol_key"]).copy()


def _build_dividend_table(dfa: pd.DataFrame) -> pd.DataFrame:
    df_div_gross = dfa[dfa["flow_bucket"].eq("Dividend")].copy()
    df_tax_all = dfa[dfa["flow_bucket"].eq("Tax")].copy()

    if df_div_gross.empty and df_tax_all.empty:
        return pd.DataFrame()

    agg_keys = ["Settlement Date", "symbol_key"]

    gross = (
        df_div_gross.groupby(agg_keys)["Net Amount (Base Currency)"]
        .sum()
        .rename("Dividend Gross")
    )

    tax = (
        df_tax_all.groupby(agg_keys)["Net Amount (Base Currency)"]
        .sum()
        .rename("Dividend Tax")
    )

    out = pd.concat([gross, tax], axis=1).fillna(0.0).reset_index()
    out["Dividend Net"] = out["Dividend Gross"] + out["Dividend Tax"]

    return out.sort_values(["Settlement Date", "symbol_key"]).copy()


# =========================================================
# Formatting helpers
# =========================================================
def _fmt_money(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.2f}"


def _fmt_qty(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.4f}"


def _delta_class(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "neu"
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "neu"


def _delta_text(value: Optional[float], kind: str = "money") -> str:
    if value is None or pd.isna(value):
        return "Sin variación"

    if kind == "money":
        txt = _fmt_money(value)
    else:
        txt = f"{value:,.0f}"

    if value > 0:
        return f"▲ {txt}"
    if value < 0:
        return f"▼ {txt}"
    return f"• {txt}"


def _kpi(label: str, value: str, delta: Optional[float] = None, delta_kind: str = "money") -> None:
    klass = _delta_class(delta)
    delta_label = _delta_text(delta, kind=delta_kind)

    st.markdown(
        f"""
<div class="ta-kpi {klass}">
    <div class="ta-kpi-label">{label}</div>
    <div class="ta-kpi-value">{value}</div>
    <div class="ta-kpi-delta {klass}">{delta_label}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _apply_money_style(df: pd.DataFrame, money_cols: List[str], qty_cols: Optional[List[str]] = None):
    qty_cols = qty_cols or []

    styled = (
        df.style
        .format(
            {
                **{c: "{:,.2f}" for c in money_cols if c in df.columns},
                **{c: "{:,.4f}" for c in qty_cols if c in df.columns},
            },
            na_rep="-",
        )
    )

    for col in money_cols:
        if col in df.columns:
            styled = styled.map(
                lambda v: (
                    f"color: {POS_TXT}; background-color: {POS_BG}; font-weight: 600;"
                    if pd.notna(v) and isinstance(v, (int, float)) and v > 0
                    else (
                        f"color: {NEG_TXT}; background-color: {NEG_BG}; font-weight: 600;"
                        if pd.notna(v) and isinstance(v, (int, float)) and v < 0
                        else ""
                    )
                ),
                subset=pd.IndexSlice[:, [col]],
            )

    return styled


def _show_df(
    df: pd.DataFrame,
    height: int = 420,
    money_cols: Optional[List[str]] = None,
    qty_cols: Optional[List[str]] = None,
) -> None:
    money_cols = money_cols or []
    qty_cols = qty_cols or []

    if df.empty:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)
        return

    styled = _apply_money_style(df, money_cols=money_cols, qty_cols=qty_cols)
    st.dataframe(styled, use_container_width=True, height=height, hide_index=True)


def _prepare_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _prepare_download_excel(
    monthly_cons: pd.DataFrame,
    monthly_pos: pd.DataFrame,
    detail_month: pd.DataFrame,
    base_audit: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        monthly_cons.to_excel(writer, sheet_name="Resumen_Mensual", index=False)
        monthly_pos.to_excel(writer, sheet_name="Posicion_Cierre", index=False)
        detail_month.to_excel(writer, sheet_name="Detalle_Mes", index=False)
        base_audit.to_excel(writer, sheet_name="Base_Audit", index=False)
    return output.getvalue()


# =========================================================
# Main
# =========================================================
def render(_ctx=None) -> None:
    _inject_css()

    st.title("Movimientos CV — Transactions Analyzer")
    st.markdown(
        '<div class="ta-subtle">Lectura ordenada de transacciones, flujos, costos y posición del comitente en Base Currency.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ta-hr"></div>', unsafe_allow_html=True)

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

    dfa = _prepare_analysis_df(df, date_col=date_col)

    if dfa.empty:
        st.warning("No pude preparar una base válida de análisis.")
        return

    monthly_cons = _build_monthly_consolidated(dfa)
    monthly_pos = _build_monthly_positions(dfa)
    div_table = _build_dividend_table(dfa)

    months_available = monthly_cons["month"].tolist()
    if not months_available:
        st.warning("No pude construir la tabla mensual.")
        return

    # =========================================================
    # Filtros
    # =========================================================
    st.subheader("Filtros")

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])

    with c1:
        selected_month = st.selectbox(
            "Mes",
            options=months_available,
            index=len(months_available) - 1 if months_available else 0,
        )

    with c2:
        asset_options = ["Todos"] + sorted(
            [x for x in dfa["asset_bucket"].dropna().astype(str).unique().tolist() if x.strip() != ""]
        )
        selected_asset_bucket = st.selectbox(
            "Clase de activo",
            options=asset_options,
            index=0,
        )

    with c3:
        symbol_options = ["Todos"] + sorted(
            [
                x
                for x in dfa["symbol_key"].dropna().astype(str).unique().tolist()
                if x.strip() != ""
            ]
        )
        selected_symbol = st.selectbox(
            "Símbolo",
            options=symbol_options,
            index=0,
        )

    st.markdown(
        f"""
<span class="ta-pill">Filas base: {len(dfa):,}</span>
<span class="ta-pill">Mes seleccionado: {selected_month}</span>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # Filtros aplicados a detalle del mes
    # =========================================================
    detail_month = dfa[dfa["month"].eq(selected_month)].copy()

    if selected_asset_bucket != "Todos":
        detail_month = detail_month[detail_month["asset_bucket"].eq(selected_asset_bucket)].copy()

    if selected_symbol != "Todos":
        detail_month = detail_month[detail_month["symbol_key"].eq(selected_symbol)].copy()

    prev_month_row = monthly_cons[monthly_cons["month"] < selected_month].sort_values("month").tail(1)
    curr_month_row = monthly_cons[monthly_cons["month"].eq(selected_month)].copy()

    prev_map = prev_month_row.iloc[0].to_dict() if not prev_month_row.empty else {}
    curr_map = curr_month_row.iloc[0].to_dict() if not curr_month_row.empty else {}

    def _mom_delta(col: str) -> Optional[float]:
        if not curr_map:
            return None
        curr_val = curr_map.get(col, None)
        prev_val = prev_map.get(col, 0.0) if prev_map else 0.0
        if curr_val is None or pd.isna(curr_val):
            return None
        prev_val = 0.0 if prev_val is None or pd.isna(prev_val) else prev_val
        return float(curr_val) - float(prev_val)

    # =========================================================
    # KPIs del mes
    # =========================================================
    st.markdown('<div class="ta-hr"></div>', unsafe_allow_html=True)
    st.subheader("Resumen ejecutivo del mes")

    if detail_month.empty:
        st.info("No hay movimientos para los filtros seleccionados.")
    else:
        cash_in = detail_month.loc[
            detail_month["flow_bucket"].eq("Cash In"),
            "Net Amount (Base Currency)",
        ].sum()

        cash_out = detail_month.loc[
            detail_month["flow_bucket"].eq("Cash Out"),
            "Net Amount (Base Currency)",
        ].abs().sum()

        buys = detail_month.loc[
            detail_month["flow_bucket"].eq("Buy"),
            "Net Amount (Base Currency)",
        ].abs().sum()

        sells = detail_month.loc[
            detail_month["flow_bucket"].eq("Sell"),
            "Net Amount (Base Currency)",
        ].sum()

        income_net = (
            detail_month.loc[detail_month["flow_bucket"].eq("Dividend"), "Net Amount (Base Currency)"].sum()
            + detail_month.loc[detail_month["flow_bucket"].eq("Tax"), "Net Amount (Base Currency)"].sum()
            + detail_month.loc[detail_month["flow_bucket"].eq("Fee"), "Net Amount (Base Currency)"].sum()
        )

        net_flow = cash_in - cash_out + sells - buys + income_net
        trade_count = int(detail_month["flow_bucket"].isin(["Buy", "Sell"]).sum())

        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)

        with k1:
            _kpi("Cash In", _fmt_money(cash_in), delta=_mom_delta("cash_in_base"))
        with k2:
            _kpi("Cash Out", _fmt_money(cash_out), delta=-_mom_delta("cash_out_base") if _mom_delta("cash_out_base") is not None else None)
        with k3:
            _kpi("Buys", _fmt_money(buys), delta=-_mom_delta("buys_base") if _mom_delta("buys_base") is not None else None)
        with k4:
            _kpi("Sells", _fmt_money(sells), delta=_mom_delta("sells_base"))
        with k5:
            _kpi(
                "Income neto",
                _fmt_money(income_net),
                delta=(
                    (_mom_delta("dividends_base") or 0.0)
                    + (_mom_delta("taxes_base") or 0.0)
                    + (_mom_delta("fees_base") or 0.0)
                ),
            )
        with k6:
            _kpi("Net Flow", _fmt_money(net_flow), delta=_mom_delta("net_total_flow_base"))

        st.markdown('<div class="ta-section-gap"></div>', unsafe_allow_html=True)

        kk1, kk2 = st.columns(2)
        with kk1:
            _kpi("Trades", f"{trade_count:,}", delta=_mom_delta("trade_count"), delta_kind="count")
        with kk2:
            _kpi("Filas del mes", f"{len(detail_month):,}", delta=_mom_delta("rows"), delta_kind="count")

    # =========================================================
    # Resumen mensual
    # =========================================================
    st.markdown('<div class="ta-hr"></div>', unsafe_allow_html=True)
    st.subheader("Resumen mensual consolidado")
    st.markdown(
        '<div class="ta-subtle">Tabla central del módulo. Resume flujos, actividad operativa, ingresos y costos por mes.</div>',
        unsafe_allow_html=True,
    )

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
            "trade_count": "Trades",
            "rows": "Rows",
        }
    )

    _show_df(
        summary_view,
        height=330,
        money_cols=["Cash In", "Cash Out", "Buys", "Sells", "Dividends", "Taxes", "Fees", "Net Flow"],
    )

    # =========================================================
    # Detalle del mes
    # =========================================================
    st.markdown('<div class="ta-hr"></div>', unsafe_allow_html=True)
    st.subheader("Detalle del mes")

    tab_flujos, tab_trades, tab_income, tab_pos, tab_audit = st.tabs(
        [
            "Flujos",
            "Trades",
            "Income & Costs",
            "Posición a cierre",
            "Base audit",
        ]
    )

    with tab_flujos:
        flow_view = detail_month[
            detail_month["flow_bucket"].isin(["Cash In", "Cash Out", "Internal"])
        ].copy()

        flow_view = flow_view.rename(
            columns={
                "Settlement Date": "Settlement Date",
                "flow_bucket": "Flow",
                "Transaction Type": "Transaction Type",
                "Transaction Description": "Description",
                "Net Amount (Base Currency)": "Base Amount",
            }
        )

        flow_view = flow_view[
            [
                "Settlement Date",
                "Flow",
                "Transaction Type",
                "Description",
                "Base Amount",
            ]
        ].sort_values(["Settlement Date", "Flow"])

        if flow_view.empty:
            st.info("No hay flujos de caja para los filtros seleccionados.")
        else:
            _show_df(
                flow_view,
                height=430,
                money_cols=["Base Amount"],
            )

    with tab_trades:
        trades_view = detail_month[
            detail_month["flow_bucket"].isin(["Buy", "Sell"])
        ].copy()

        trades_view = trades_view.rename(
            columns={
                "Settlement Date": "Settlement Date",
                "flow_bucket": "Flow",
                "asset_bucket": "Asset Class",
                "symbol_key": "Symbol",
                "Security Description": "Description",
                "Security Type": "Security Type",
                "Buy/Sell": "Buy/Sell",
                "Quantity": "Qty",
                "Net Amount (Base Currency)": "Base Amount",
            }
        )

        trades_view = trades_view[
            [
                "Settlement Date",
                "Flow",
                "Asset Class",
                "Symbol",
                "Description",
                "Security Type",
                "Buy/Sell",
                "Qty",
                "Base Amount",
            ]
        ].sort_values(["Settlement Date", "Symbol"])

        if trades_view.empty:
            st.info("No hay trades para los filtros seleccionados.")
        else:
            _show_df(
                trades_view,
                height=430,
                money_cols=["Base Amount"],
                qty_cols=["Qty"],
            )

    with tab_income:
        income_view = detail_month[
            detail_month["flow_bucket"].isin(["Dividend", "Tax", "Fee"])
        ].copy()

        income_view = income_view.rename(
            columns={
                "Settlement Date": "Settlement Date",
                "flow_bucket": "Type",
                "symbol_key": "Symbol",
                "Security Description": "Description",
                "Transaction Type": "Transaction Type",
                "Transaction Description": "Transaction Description",
                "Net Amount (Base Currency)": "Base Amount",
            }
        )

        income_view = income_view[
            [
                "Settlement Date",
                "Type",
                "Symbol",
                "Description",
                "Transaction Type",
                "Transaction Description",
                "Base Amount",
            ]
        ].sort_values(["Settlement Date", "Symbol", "Type"])

        c1, c2, c3 = st.columns(3)

        div_total = detail_month.loc[
            detail_month["flow_bucket"].eq("Dividend"),
            "Net Amount (Base Currency)",
        ].sum()

        tax_total = detail_month.loc[
            detail_month["flow_bucket"].eq("Tax"),
            "Net Amount (Base Currency)",
        ].sum()

        fee_total = detail_month.loc[
            detail_month["flow_bucket"].eq("Fee"),
            "Net Amount (Base Currency)",
        ].sum()

        div_delta = _mom_delta("dividends_base")
        tax_delta = _mom_delta("taxes_base")
        fee_delta = _mom_delta("fees_base")

        with c1:
            _kpi("Dividendos", _fmt_money(div_total), delta=div_delta)
        with c2:
            _kpi("Taxes", _fmt_money(tax_total), delta=tax_delta)
        with c3:
            _kpi("Fees", _fmt_money(fee_total), delta=fee_delta)

        st.markdown('<div class="ta-section-gap"></div>', unsafe_allow_html=True)

        if income_view.empty:
            st.info("No hay dividendos, taxes ni fees para los filtros seleccionados.")
        else:
            _show_df(
                income_view,
                height=380,
                money_cols=["Base Amount"],
            )

        if not div_table.empty:
            st.markdown('<div class="ta-section-gap"></div>', unsafe_allow_html=True)
            st.caption("Vista agrupada de dividendos y retenciones")

            div_month = div_table.copy()
            div_month["month"] = pd.to_datetime(div_month["Settlement Date"]).dt.strftime("%Y-%m")
            div_month = div_month[div_month["month"].eq(selected_month)].copy()

            if selected_symbol != "Todos":
                div_month = div_month[div_month["symbol_key"].eq(selected_symbol)].copy()

            div_month = div_month.drop(columns=["month"], errors="ignore")
            div_month = div_month.rename(
                columns={
                    "Settlement Date": "Settlement Date",
                    "symbol_key": "Symbol",
                }
            )

            if not div_month.empty:
                _show_df(
                    div_month,
                    height=240,
                    money_cols=["Dividend Gross", "Dividend Tax", "Dividend Net"],
                )

    with tab_pos:
        if monthly_pos.empty:
            st.info("No encontré trades reales para construir la posición.")
        else:
            pos_view = monthly_pos[monthly_pos["month"].eq(selected_month)].copy()

            if selected_asset_bucket != "Todos":
                pos_view = pos_view[pos_view["asset_bucket"].eq(selected_asset_bucket)].copy()

            if selected_symbol != "Todos":
                pos_view = pos_view[pos_view["symbol_key"].eq(selected_symbol)].copy()

            pos_view = pos_view.rename(
                columns={
                    "symbol_key": "Symbol",
                    "security_description": "Description",
                    "security_type": "Security Type",
                    "asset_bucket": "Asset Class",
                    "closing_qty": "Closing Qty",
                }
            )

            pos_view = pos_view[
                [
                    "Symbol",
                    "Description",
                    "Asset Class",
                    "Security Type",
                    "Closing Qty",
                ]
            ].sort_values(
                ["Closing Qty", "Symbol"],
                ascending=[False, True],
            )

            if pos_view.empty:
                st.info("No hay posición a cierre para los filtros seleccionados.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    _kpi("Símbolos", f"{pos_view['Symbol'].nunique():,}", delta=None, delta_kind="count")
                with c2:
                    _kpi("Closing Qty Total", _fmt_qty(pos_view["Closing Qty"].sum()), delta=None)

                st.markdown('<div class="ta-section-gap"></div>', unsafe_allow_html=True)
                _show_df(
                    pos_view,
                    height=430,
                    qty_cols=["Closing Qty"],
                )

    with tab_audit:
        audit_view = detail_month.copy()

        audit_view = audit_view.rename(
            columns={
                "Settlement Date": "Settlement Date",
                "flow_bucket": "Flow",
                "asset_bucket": "Asset Class",
                "symbol_key": "Symbol",
                "Security Description": "Description",
                "Security Type": "Security Type",
                "Buy/Sell": "Buy/Sell",
                "Quantity": "Qty",
                "Net Amount (Base Currency)": "Base Amount",
                "Transaction Type": "Transaction Type",
                "Transaction Description": "Transaction Description",
            }
        )

        audit_cols = [
            "Settlement Date",
            "Flow",
            "Asset Class",
            "Symbol",
            "Description",
            "Security Type",
            "Transaction Type",
            "Transaction Description",
            "Buy/Sell",
            "Qty",
            "Base Amount",
        ]
        audit_cols = [c for c in audit_cols if c in audit_view.columns]
        audit_view = audit_view[audit_cols].sort_values(["Settlement Date", "Symbol"])

        if audit_view.empty:
            st.info("No hay base audit para los filtros seleccionados.")
        else:
            _show_df(
                audit_view,
                height=460,
                money_cols=["Base Amount"],
                qty_cols=["Qty"],
            )

    # =========================================================
    # Descargas
    # =========================================================
    st.markdown('<div class="ta-hr"></div>', unsafe_allow_html=True)
    st.subheader("Descargas")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Descargar detalle del mes en CSV",
            data=_prepare_download_csv(detail_month),
            file_name=f"transactions_detail_{selected_month}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            label="Descargar paquete Excel",
            data=_prepare_download_excel(
                monthly_cons=summary_view,
                monthly_pos=monthly_pos,
                detail_month=detail_month,
                base_audit=dfa,
            ),
            file_name=f"transactions_analyzer_{selected_month}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
