# tools/comerciales/twr_engine.py
# ─────────────────────────────────────────────────────────────────────────────
#  TWR Engine — Time-Weighted Return
#  Calcula rendimiento de cartera desde el Excel de Pershing + precios yfinance
#
#  Método: Modified Dietz mensual encadenado (aprox. TWR diario)
#    R_mes = (MV_fin - MV_ini - CF) / (MV_ini + W * CF)
#    W     = peso temporal del flujo (asumimos mitad de mes → W = 0.5)
#    TWR   = ∏(1 + R_i) - 1  sobre los sub-períodos seleccionados
#
#  Fuente de precios: Yahoo Finance vía yfinance (cache 1 h en session_state)
#
#  Dependencias: pip install yfinance altair streamlit pandas numpy openpyxl
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ── paleta ────────────────────────────────────────────────────────────────────
POS_C   = "#0e7a52"
NEG_C   = "#b12e2e"
ACC_C   = "#1a56db"
INK     = "#0f1117"
MUTED   = "#7a818f"

POS_BG  = "rgba(14,122,82,.10)"
NEG_BG  = "rgba(177,46,46,.10)"
ACC_BG  = "rgba(26,86,219,.10)"
NEU_BG  = "rgba(122,129,143,.10)"

ASSET_COLORS = {
    "ETF":      "#6366f1",
    "Stock":    "#f59e0b",
    "Bond":     "#10b981",
    "Fund":     "#3b82f6",
    "Currency": "#8b5cf6",
    "Other":    "#9ca3af",
    "Total":    ACC_C,
}

# ── constantes del formato Pershing ──────────────────────────────────────────
CANON_COLS = [
    "Process Date","Security Identifier","Settlement Date",
    "Net Amount (Base Currency)","Transaction Description","Transaction Type",
    "Security Description","Net Amount (Transaction Currency)","Buy/Sell","Quantity",
    "Price (Transaction Currency)","Transaction Currency","Security Type","Payee",
    "Paid For (Name)","Request Reason","CUSIP","FX Rate (To Base)","ISIN","SEDOL",
    "SYMBOL","Trade Date","Transaction code","Withdrawal/Deposit Type","Request ID #","Commission",
]
ETF_TYPES   = {"EXCHANGE TRADED FUNDS"}
STOCK_TYPES = {"COMMON STOCK","COMMON STOCK ADR"}
BOND_TYPES  = {"CORP BOND","CORPORATE BOND","INDEX LINKED CORP BOND","GOVERNMENT BOND","MUNICIPAL BOND"}
FUND_TYPES  = {"OPEN END TAXABLE LOAD FUND","OPEN END FUND","MUTUAL FUND","MONEY MARKET FUND"}

CATEGORIA_MAP = {
    "ETF":      "Renta Variable",
    "Stock":    "Renta Variable",
    "Bond":     "Renta Fija",
    "Fund":     "Renta Variable",
    "Currency": "MM, T Bills y Simis",
    "Other":    "Otros",
}

CASH_TX     = {"FEDERAL FUNDS RECEIVED","FEDERAL FUNDS SENT"}
INTERNAL_TX = {"ACTIVITY WITHIN YOUR ACCT"}
DIV_MARKERS = ["CASH DIVIDEND RECEIVED","FOREIGN SECURITY DIVIDEND RECEIVED"]
TAX_MARKERS = ["NON-RESIDENT ALIEN TAX","FOREIGN TAX WITHHELD AT   THE SOURCE",
               "FOREIGN TAX WITHHELD"]
FEE_EXACT   = {"PES BILLING FEE","FOREIGN CUSTODY FEE","ASSET BASED FEE",
               "PAPER DELIVERY            SUBSCRIPTION",
               "ASSET MANAGEMENT ACCOUNT  SPECIAL HANDLING FEE",
               "INT. CHARGED ON DEBIT     BALANCES",
               "FEE ON FOREIGN DIVIDEND   WITHHELD AT THE SOURCE"}
FEE_MARKERS = ["ADVISORY","CUSTODY","SUBSCRIPTION","ASSET BASED FEE"]
DIV_FEE_MARKERS = ["FEE ON FOREIGN DIVIDEND"]


# ═════════════════════════════════════════════════════════════════════════════
#  CSS — rediseño minimalista profesional
# ═════════════════════════════════════════════════════════════════════════════
def _css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400&family=Sora:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
.block-container { padding-top: 1.5rem; max-width: 1160px; padding-bottom: 3rem; }
h1, h2, h3 { font-family: 'Sora', system-ui, sans-serif; letter-spacing: -.02em; }

/* ── Header ── */
.twr-header { display: flex; align-items: flex-start; justify-content: space-between;
  padding-bottom: 18px; border-bottom: 1px solid rgba(15,17,23,.09); margin-bottom: 28px; }
.twr-header h1 { font-size: 20px; font-weight: 600; color: #0f1117; margin: 0 0 4px; }
.twr-header p  { font-size: 11px; color: #7a818f; font-family: 'DM Mono', monospace;
  letter-spacing: .03em; margin: 0; }
.twr-badge { display: inline-flex; align-items: center; gap: 5px; font-family: 'DM Mono', monospace;
  font-size: 10px; padding: 3px 9px; border-radius: 4px;
  background: rgba(26,86,219,.08); color: #1a56db;
  border: 1px solid rgba(26,86,219,.15); white-space: nowrap; }
.twr-badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: #1a56db; opacity: .8; }

/* ── Section label ── */
.sec-label { font-size: 10px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: #7a818f; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.sec-label::after { content: ''; flex: 1; height: 1px; background: rgba(15,17,23,.08); }

/* ── KPI cards ── */
.kpi-grid { display: grid; gap: 10px; margin-bottom: 28px; }
.kpi { background: #fff; border: 1px solid rgba(15,17,23,.08); border-radius: 10px;
  padding: 14px 16px; position: relative; overflow: hidden; }
.kpi::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: rgba(15,17,23,.06); }
.kpi.pos::after { background: linear-gradient(90deg, #0e7a52, transparent); }
.kpi.neg::after { background: linear-gradient(90deg, #b12e2e, transparent); }
.kpi.acc::after { background: linear-gradient(90deg, #1a56db, transparent); }
.kpi-lbl { font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: #7a818f; margin-bottom: 8px; font-family: 'Sora', sans-serif; }
.kpi-val { font-size: 22px; font-weight: 600; letter-spacing: -.04em; color: #0f1117;
  font-family: 'DM Mono', monospace; line-height: 1; margin-bottom: 6px; }
.kpi-val.sm { font-size: 16px; }
.kpi-sub { font-size: 11px; font-family: 'DM Mono', monospace; }
.kpi-sub.pos { color: #0e7a52; } .kpi-sub.neg { color: #b12e2e; } .kpi-sub.neu { color: #7a818f; }

/* ── Capital summary ── */
.cap-grid { display: grid; gap: 1px; background: rgba(15,17,23,.08);
  border-radius: 10px; overflow: hidden; margin-bottom: 28px; }
.cap-cell { background: #fff; padding: 14px 14px 12px; }
.cap-lbl { font-size: 10px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  color: #7a818f; margin-bottom: 6px; font-family: 'Sora', sans-serif;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cap-amt { font-family: 'DM Mono', monospace; font-size: 15px; color: #0f1117; }
.cap-tag { display: inline-block; margin-top: 5px; font-size: 9px; font-weight: 600;
  letter-spacing: .05em; text-transform: uppercase; padding: 2px 7px; border-radius: 3px; }
.cap-tag.pos { background: rgba(14,122,82,.10); color: #0e7a52; }
.cap-tag.neg { background: rgba(177,46,46,.10); color: #b12e2e; }
.cap-tag.neu { background: rgba(122,129,143,.10); color: #7a818f; }

/* ── Pills ── */
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.pill { display: inline-flex; align-items: center; gap: 5px; padding: 4px 11px;
  border-radius: 100px; border: 1px solid rgba(15,17,23,.10); font-size: 11px;
  color: #7a818f; background: #fff; font-family: 'Sora', sans-serif; }
.pill.ok { color: #0e7a52; border-color: rgba(14,122,82,.2); background: rgba(14,122,82,.05); }

/* ── Category headers ── */
.cat-hdr { display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px; background: #0f1117; color: #fff; border-radius: 7px;
  margin: 14px 0 6px; font-size: 11px; font-weight: 600; letter-spacing: .03em;
  font-family: 'Sora', sans-serif; }
.cat-hdr span:last-child { font-family: 'DM Mono', monospace; font-weight: 400; opacity: .7; }

/* ── Tables ── */
.tbl-outer { overflow-x: auto; border-radius: 8px; border: 1px solid rgba(15,17,23,.08);
  margin-bottom: 4px; }
.tbl-outer table { width: 100%; border-collapse: collapse; }
.tbl-outer th { font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: #7a818f; padding: 10px 14px; text-align: right; background: #f7f8fa;
  border-bottom: 1px solid rgba(15,17,23,.08); font-family: 'Sora', sans-serif; }
.tbl-outer th:first-child { text-align: left; }
.tbl-outer td { font-family: 'DM Mono', monospace; font-size: 12px; padding: 9px 14px;
  text-align: right; border-bottom: 1px solid rgba(15,17,23,.06); color: #3a3f4a; }
.tbl-outer td:first-child { font-family: 'Sora', sans-serif; font-size: 12px; font-weight: 500;
  text-align: left; color: #0f1117; }
.tbl-outer tr:last-child td { border-bottom: 0; }
.tbl-outer tr:hover td { background: #f7f8fa; }
.num-pos { color: #0e7a52 !important; font-weight: 500 !important; }
.num-neg { color: #b12e2e !important; font-weight: 500 !important; }
.tbl-foot td { font-weight: 600 !important; background: #f0f2f5 !important;
  border-top: 2px solid rgba(15,17,23,.12) !important; }

/* ── Divider ── */
.hr { height: 1px; background: rgba(15,17,23,.08); margin: 22px 0 26px; }

/* ── Totalizador ── */
.total-bar { text-align: right; padding: 10px 14px;
  border-top: 2px solid rgba(15,17,23,.12); margin-top: 6px;
  font-family: 'DM Mono', monospace; font-size: 14px; font-weight: 600; color: #0f1117; }

/* ── Nota al pie de fórmula ── */
.formula-note { margin-top: 10px; font-size: 10px; color: #7a818f;
  font-family: 'DM Mono', monospace; }

/* ── Movimientos 2 columnas ── */
.movs-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.mov-col-hdr { display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px; color: #fff; border-radius: 7px; margin: 0 0 6px;
  font-size: 11px; font-weight: 600; letter-spacing: .03em; font-family: 'Sora', sans-serif; }
.mov-col-hdr span:last-child { font-family: 'DM Mono', monospace; font-weight: 400; opacity: .8; }

/* ── Missing NAV panel ── */
.missing-panel { border: 1px solid rgba(15,17,23,.10); border-radius: 9px;
  padding: 16px 18px; background: #fff; margin-bottom: 24px; }
.missing-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
  font-size: 12px; font-weight: 600; color: #0f1117; font-family: 'Sora', sans-serif; }
.missing-icon { width: 18px; height: 18px; border-radius: 4px; background: #fef3c7;
  display: flex; align-items: center; justify-content: center; font-size: 11px; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers de parseo
# ═════════════════════════════════════════════════════════════════════════════
def _norm(s: str) -> str:
    return re.sub(r"\s+"," ",
        str(s or "").strip().lower()
        .replace("á","a").replace("é","e").replace("í","i")
        .replace("ó","o").replace("ú","u").replace("ñ","n"))

def _to_float(x):
    if pd.isna(x): return None
    if isinstance(x,(int,float)): return float(x)
    s = str(x).strip().replace(" ","")
    if s in ("","-"): return None
    if re.search(r"\d+,\d+$",s) and s.count(",")==1 and s.count(".")>=1:
        s = s.replace(".","").replace(",",".")
    elif s.count(",")==1 and s.count(".")==0:
        s = s.replace(",",".")
    else:
        s = s.replace(",","")
    try: return float(s)
    except: return None

def _to_date(x):
    try: return pd.to_datetime(x, errors="coerce")
    except: return pd.NaT

def _first(series: pd.Series) -> str:
    vals = [str(x).strip() for x in series.dropna() if str(x).strip() not in {"","-","nan","None"}]
    return vals[0] if vals else ""

def _find_header_row(df0: pd.DataFrame) -> int:
    key = _norm("Process Date")
    best = (-1,-1)
    for i in range(min(len(df0),80)):
        row = df0.iloc[i].astype(str).map(_norm).tolist()
        if key not in row: continue
        score = sum(1 for m in ["settlement date","transaction type","security type",
                                "net amount (base currency)"] if m in row)
        if score > best[0]: best = (score,i)
    return best[1]

def _read_pershing_excel(data: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(data))
    df0 = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None, dtype=object)
    hdr = _find_header_row(df0)
    if hdr < 0:
        raise ValueError("No pude detectar fila de encabezados (Process Date).")
    headers = df0.iloc[hdr].astype(str).tolist()
    df = df0.iloc[hdr+1:].copy()
    df.columns = headers
    df = df.dropna(how="all")
    col_map = {c: canon for c in df.columns for canon in CANON_COLS if _norm(c)==_norm(canon)}
    df = df.rename(columns=col_map)[[c for c in CANON_COLS if c in df.columns]].copy()
    for dc in ["Settlement Date","Process Date","Trade Date"]:
        if dc in df.columns: df[dc] = df[dc].apply(_to_date)
    for nc in ["Net Amount (Base Currency)","Net Amount (Transaction Currency)",
               "Quantity","Price (Transaction Currency)","Commission","FX Rate (To Base)"]:
        if nc in df.columns: df[nc] = df[nc].apply(_to_float)
    for col in ["SYMBOL","Security Identifier"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan","None"]),col] = ""
    for col in ["Security Type","Transaction Type","Buy/Sell"]:
        if col in df.columns: df[col] = df[col].astype(str).str.strip().str.upper()
    if "Buy/Sell" in df.columns:
        df.loc[~df["Buy/Sell"].isin(["BUY","SELL"]),"Buy/Sell"] = ""
    if "Transaction Description" in df.columns:
        df["Transaction Description"] = df["Transaction Description"].astype(str).str.strip()
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Clasificadores de fila
# ═════════════════════════════════════════════════════════════════════════════
def _bucket(row) -> str:
    tx   = str(row.get("Transaction Type","")).upper().strip()
    code = str(row.get("Transaction code","")).upper().strip()
    comm = row.get("Commission",None)
    bs   = str(row.get("Buy/Sell","")).upper().strip()

    if bs == "BUY":  return "Buy"
    if bs == "SELL": return "Sell"
    if tx in INTERNAL_TX:              return "Internal"
    if tx == "FEDERAL FUNDS RECEIVED": return "Cash In"
    if tx == "FEDERAL FUNDS SENT":     return "Cash Out"
    if any(m in tx for m in DIV_MARKERS): return "Dividend"
    if (any(m in tx for m in TAX_MARKERS)
            or code in {"NRA","FGN","FGF"}
            or tx == "NON-RESIDENT ALIEN TAX"):
        return "Tax"
    if tx in FEE_EXACT:                              return "Fee"
    if any(m in tx for m in DIV_FEE_MARKERS):        return "Fee"
    if any(m in tx for m in FEE_MARKERS):            return "Fee"
    if code in {"PDS","NTF","INM","PCT","/FG"}:      return "Fee"
    if isinstance(comm,(int,float)) and pd.notna(comm) and comm > 0: return "Fee"
    return "Other"

def _asset_class(row) -> str:
    st_ = str(row.get("Security Type","")).upper().strip()
    if st_ in ETF_TYPES:   return "ETF"
    if st_ in STOCK_TYPES: return "Stock"
    if st_ in BOND_TYPES:  return "Bond"
    if st_ in FUND_TYPES:  return "Fund"
    if st_ in {"CURRENCY","FOREIGN CURRENCY"}: return "Currency"
    return "Other"


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline de análisis
# ═════════════════════════════════════════════════════════════════════════════
def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["Settlement Date"].notna()].copy()
    out["Settlement Date"] = pd.to_datetime(out["Settlement Date"], errors="coerce")
    out = out[out["Settlement Date"].notna()].copy()
    out["month_end"] = out["Settlement Date"].dt.to_period("M").dt.to_timestamp("M")
    out["month"]     = out["month_end"].dt.strftime("%Y-%m")
    out["flow_bucket"]  = out.apply(_bucket, axis=1)
    out["asset_bucket"] = out.apply(_asset_class, axis=1)
    out["symbol_key"]   = out["SYMBOL"].astype(str).str.strip()
    out.loc[out["symbol_key"].eq(""), "symbol_key"] = out["Security Identifier"].astype(str).str.strip()
    out["signed_qty"] = 0.0
    out.loc[out["Buy/Sell"].eq("BUY"),  "signed_qty"] =  out.loc[out["Buy/Sell"].eq("BUY"),  "Quantity"].fillna(0.0)
    out.loc[out["Buy/Sell"].eq("SELL"), "signed_qty"] = -out.loc[out["Buy/Sell"].eq("SELL"), "Quantity"].fillna(0.0)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Posición acumulada
# ═════════════════════════════════════════════════════════════════════════════
def _build_positions(dfa: pd.DataFrame) -> pd.DataFrame:
    trades = dfa[dfa["flow_bucket"].isin(["Buy","Sell"])].copy()
    trades = trades[trades["symbol_key"].ne("") & trades["symbol_key"].notna()].copy()
    if trades.empty:
        return pd.DataFrame(columns=["month_end","symbol_key","asset_bucket","closing_qty"])

    sym_meta = (trades.groupby("symbol_key", sort=False)
                .agg(asset_bucket=("asset_bucket", _first))
                .reset_index())

    all_months = pd.date_range(dfa["month_end"].min(), dfa["month_end"].max(), freq="ME")
    delta = (trades.groupby(["month_end","symbol_key"], sort=False)["signed_qty"]
             .sum().reset_index())

    grid = (sym_meta.assign(_k=1)
            .merge(pd.DataFrame({"month_end": all_months, "_k":1}), on="_k")
            .drop(columns="_k"))
    grid = grid.merge(delta, on=["month_end","symbol_key"], how="left")
    grid["signed_qty"] = grid["signed_qty"].fillna(0.0)
    grid = grid.sort_values(["symbol_key","month_end"])
    grid["closing_qty"] = grid.groupby("symbol_key")["signed_qty"].cumsum()

    pos = grid[grid["closing_qty"].round(10).ne(0)].copy()
    pos["month"] = pos["month_end"].dt.strftime("%Y-%m")
    return pos[["month_end","month","symbol_key","asset_bucket","closing_qty"]].copy()


# ═════════════════════════════════════════════════════════════════════════════
#  Precios yfinance
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_yf(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers), start=start, end=end,
            progress=False, auto_adjust=True, threads=True
        )
        if raw.empty:
            return pd.DataFrame()
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=tickers[0])
        monthly = closes.resample("ME").last()
        monthly.index = monthly.index.normalize()
        return monthly
    except Exception as e:
        st.warning(f"yfinance: error al descargar precios — {e}")
        return pd.DataFrame()

def _get_prices(dfa: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    trades = dfa[dfa["flow_bucket"].isin(["Buy","Sell"])].copy()
    trades = trades[trades["symbol_key"].ne("") & trades["symbol_key"].notna()].copy()
    if trades.empty:
        st.session_state[cache_key] = pd.DataFrame()
        return pd.DataFrame()
    tickers = tuple(sorted(trades["symbol_key"].unique().tolist()))
    start   = (dfa["Settlement Date"].min() - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    end     = (dfa["Settlement Date"].max() + pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    prices  = _fetch_yf(tickers, start, end)
    st.session_state[cache_key] = prices
    return prices


# ═════════════════════════════════════════════════════════════════════════════
#  Market Value mensual
# ═════════════════════════════════════════════════════════════════════════════
def _market_value_series(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    if positions.empty or prices.empty:
        return pd.DataFrame()

    rows = []
    all_months = sorted(positions["month_end"].unique())

    for me in all_months:
        pos_m = positions[positions["month_end"].eq(me)].copy()
        for _, row in pos_m.iterrows():
            sym = row["symbol_key"]
            qty = row["closing_qty"]
            price = np.nan
            if sym in prices.columns:
                subset = prices.loc[prices.index <= me, sym].dropna()
                if not subset.empty:
                    price = float(subset.iloc[-1])
            rows.append({
                "month_end":    me,
                "symbol_key":   sym,
                "asset_bucket": row["asset_bucket"],
                "closing_qty":  qty,
                "price":        price,
                "mv":           qty * price if not np.isnan(price) else np.nan,
                "price_ok":     not np.isnan(price),
            })

    if not rows:
        return pd.DataFrame()

    detail = pd.DataFrame(rows)

    if group_col is None:
        agg = (detail.groupby("month_end")
               .agg(mv_end=("mv","sum"), price_ok=("price_ok","all"))
               .reset_index())
        agg["group"] = "Total"
    elif group_col == "symbol_key":
        agg = (detail.groupby(["month_end","symbol_key"])
               .agg(mv_end=("mv","sum"), price_ok=("price_ok","all"))
               .reset_index()
               .rename(columns={"symbol_key":"group"}))
    else:
        agg = (detail.groupby(["month_end","asset_bucket"])
               .agg(mv_end=("mv","sum"), price_ok=("price_ok","all"))
               .reset_index()
               .rename(columns={"asset_bucket":"group"}))

    agg = agg.sort_values(["group","month_end"])
    agg["mv_start"] = agg.groupby("group")["mv_end"].shift(1)
    return agg


# ═════════════════════════════════════════════════════════════════════════════
#  Flujos externos
# ═════════════════════════════════════════════════════════════════════════════
def _cash_flows(dfa: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    ci = dfa[dfa["flow_bucket"].eq("Cash In")].copy()
    co = dfa[dfa["flow_bucket"].eq("Cash Out")].copy()
    ci_m = ci.groupby("month_end")["Net Amount (Base Currency)"].sum().rename("ci")
    co["_abs_amt"] = co["Net Amount (Base Currency)"].abs()
    co_m = co.groupby("month_end")["_abs_amt"].sum().rename("co")
    cf = pd.concat([ci_m, co_m], axis=1).fillna(0.0)
    cf["net_cf"] = cf["ci"] - cf["co"]
    cf = cf.reset_index()[["month_end","net_cf"]]
    return cf


# ═════════════════════════════════════════════════════════════════════════════
#  Modified Dietz
# ═════════════════════════════════════════════════════════════════════════════
def _modified_dietz(mv_start: float, mv_end: float, net_cf: float, w: float = 0.5) -> Optional[float]:
    if any(np.isnan(v) for v in [mv_start, mv_end]):
        return None
    denom = mv_start + w * net_cf
    if abs(denom) < 1e-9:
        return None
    return (mv_end - mv_start - net_cf) / denom


# ═════════════════════════════════════════════════════════════════════════════
#  Motor TWR
# ═════════════════════════════════════════════════════════════════════════════
def _compute_twr(
    mv_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    if mv_df.empty:
        return pd.DataFrame()

    mv = mv_df.copy()
    mv = mv.merge(cf_df[["month_end","net_cf"]], on="month_end", how="left")
    mv["net_cf"] = mv["net_cf"].fillna(0.0)
    if group_col == "symbol_key":
        mv["net_cf"] = 0.0

    mv["r_monthly"] = mv.apply(
        lambda r: _modified_dietz(r["mv_start"], r["mv_end"], r["net_cf"]),
        axis=1,
    )

    records = []
    for grp, g in mv.groupby("group", sort=False):
        g = g.sort_values("month_end").copy()
        g["month"] = g["month_end"].dt.strftime("%Y-%m")

        factors = []
        cum = 1.0
        for r in g["r_monthly"]:
            if r is None or np.isnan(r):
                factors.append(np.nan)
                cum = np.nan
            else:
                cum = (cum if not np.isnan(cum) else 1.0) * (1 + r)
                factors.append(cum)
        g["cum_factor"] = factors

        def _sub_twr(g, n_months=None, ytd=False, inception=False):
            vals = []
            for i, row in enumerate(g.itertuples()):
                if inception:
                    cf = row.cum_factor
                    vals.append((cf - 1) * 100 if not np.isnan(cf) else np.nan)
                elif ytd:
                    cur_year  = pd.Period(row.month, "M").year
                    dec_prev  = f"{cur_year-1}-12"
                    base_rows = g[g["month"].eq(dec_prev)]
                    if base_rows.empty:
                        base_rows = g[g["month"].str.startswith(str(cur_year))].head(1)
                        base_cf   = 1.0
                    else:
                        base_cf   = base_rows.iloc[0]["cum_factor"]
                    cur_cf = row.cum_factor
                    if np.isnan(cur_cf) or np.isnan(base_cf) or base_cf == 0:
                        vals.append(np.nan)
                    else:
                        vals.append((cur_cf / base_cf - 1) * 100)
                else:
                    if i < n_months:
                        vals.append(np.nan)
                    else:
                        base_cf = g.iloc[i - n_months]["cum_factor"]
                        cur_cf  = row.cum_factor
                        if np.isnan(cur_cf) or np.isnan(base_cf) or base_cf == 0:
                            vals.append(np.nan)
                        else:
                            vals.append((cur_cf / base_cf - 1) * 100)
            return vals

        g["r_cum_twr"]     = g["cum_factor"].apply(lambda x: (x-1)*100 if not np.isnan(x) else np.nan)
        g["r_3m"]          = _sub_twr(g, n_months=3)
        g["r_12m"]         = _sub_twr(g, n_months=12)
        g["r_ytd"]         = _sub_twr(g, ytd=True)
        g["r_inception"]   = _sub_twr(g, inception=True)
        g["r_monthly_pct"] = g["r_monthly"].apply(lambda x: x*100 if x is not None and not np.isnan(x) else np.nan)
        records.append(g)

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, ignore_index=True)
    return out[[
        "month","group","mv_start","mv_end","net_cf",
        "r_monthly_pct","r_cum_twr","r_3m","r_12m","r_ytd","r_inception",
        "price_ok",
    ]].copy()


# ═════════════════════════════════════════════════════════════════════════════
#  Estado de posición mensual
# ═════════════════════════════════════════════════════════════════════════════
def _build_estado_posicion(positions, prices_df, dfa, month):
    pos_mes = positions[positions["month"].eq(month)].copy()
    if pos_mes.empty:
        return pd.DataFrame()

    me = pd.Period(month, "M").to_timestamp("M")

    desc_map, isin_map = {}, {}
    for sym in pos_mes["symbol_key"].unique():
        rows = dfa[dfa["symbol_key"].eq(sym)]
        desc_map[sym] = (_first(rows["Security Description"].dropna()) or sym).strip()
        isin = _first(rows["ISIN"].dropna())
        isin_map[sym] = str(isin).strip() if isin and str(isin) not in {"nan","None",""} else ""

    rows_out = []
    for _, row in pos_mes.iterrows():
        sym = row["symbol_key"]; qty = row["closing_qty"]; pr = np.nan
        if not prices_df.empty and sym in prices_df.columns:
            sub = prices_df.loc[prices_df.index <= me, sym].dropna()
            if not sub.empty: pr = float(sub.iloc[-1])
        is_bond = row["asset_bucket"] in {"Bond"}
        if not np.isnan(pr):
            mv = qty * (pr / 100.0) if is_bond else qty * pr
        else:
            mv = np.nan
        cat = CATEGORIA_MAP.get(row["asset_bucket"], "Otros")
        rows_out.append({"categoria": cat, "Instrumento": desc_map.get(sym, sym),
                         "ISIN": isin_map.get(sym,""), "Cantidad": qty,
                         "Precio": pr, "Valuación": mv})

    all_flows = dfa[dfa["month"] <= month].copy()
    cash_residual = float(all_flows.loc[
        all_flows["flow_bucket"].isin(["Cash In","Cash Out","Dividend","Tax","Fee","Sell"]),
        "Net Amount (Base Currency)"].sum()) + float(all_flows.loc[
        all_flows["flow_bucket"].eq("Buy"), "Net Amount (Base Currency)"].sum())
    if cash_residual > 0.5:
        rows_out.append({"categoria": "MM, T Bills y Simis",
                         "Instrumento": "FEDERATED HERMES STD USD RET",
                         "ISIN": "", "Cantidad": cash_residual,
                         "Precio": 1.0, "Valuación": cash_residual})

    df_out = pd.DataFrame(rows_out)
    if df_out.empty: return df_out
    total_mv = df_out["Valuación"].sum(skipna=True)
    df_out["% Cartera"] = (df_out["Valuación"] / total_mv * 100).round(2)

    cat_order = ["MM, T Bills y Simis","Renta Fija","Renta Variable","Otros"]
    df_out["_cat_ord"] = df_out["categoria"].map({c:i for i,c in enumerate(cat_order)})
    df_out = df_out.sort_values(["_cat_ord","Valuación"], ascending=[True,False]).drop(columns="_cat_ord")
    return df_out


# ═════════════════════════════════════════════════════════════════════════════
#  Movimientos del mes
# ═════════════════════════════════════════════════════════════════════════════
def _build_movimientos_mes(dfa, month):
    mes = dfa[dfa["month"].eq(month)].copy()

    def _meta(sym):
        rows = dfa[dfa["symbol_key"].eq(sym)]
        desc = _first(rows["Security Description"].dropna()) or sym
        isin = _first(rows["ISIN"].dropna())
        cat  = CATEGORIA_MAP.get(_first(rows["asset_bucket"].dropna()) or "", "Otros")
        isin = str(isin).strip() if isin and str(isin) not in {"nan","None",""} else ""
        return desc.strip(), isin, cat

    def _build(bucket):
        rows = mes[mes["flow_bucket"].eq(bucket)].copy()
        if rows.empty: return pd.DataFrame()
        out = []
        for _, r in rows.iterrows():
            sym = r["symbol_key"]; qty = abs(float(r.get("Quantity") or 0))
            amt = abs(float(r["Net Amount (Base Currency)"]))
            pr_col = r.get("Price (Transaction Currency)")
            pr = float(pr_col) if pr_col and str(pr_col) not in {"nan","None",""} else (amt/qty if qty else 0)
            desc, isin, cat = _meta(sym)
            out.append({"Categoría": cat, "Instrumento": desc, "ISIN": isin,
                        "Cantidad": qty, "Precio unit.": pr, "Total USD": amt})
        df = pd.DataFrame(out)
        cat_order = ["MM, T Bills y Simis","Renta Fija","Renta Variable","Otros"]
        df["_o"] = df["Categoría"].map({c:i for i,c in enumerate(cat_order)})
        return df.sort_values(["_o","Total USD"], ascending=[True,False]).drop(columns="_o")

    return {"compras": _build("Buy"), "ventas": _build("Sell")}


# ═════════════════════════════════════════════════════════════════════════════
#  Rentabilidad mes a mes
# ═════════════════════════════════════════════════════════════════════════════
def _build_rentabilidad_mensual(dfa, twr_total):
    rows = twr_total[twr_total["group"].eq("Total")].sort_values("month").copy()
    if rows.empty: return pd.DataFrame()

    ci_mes = (dfa[dfa["flow_bucket"].eq("Cash In")]
              .groupby("month")["Net Amount (Base Currency)"].sum().rename("ingresos"))

    MESES_ES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

    result = []
    for _, r in rows.iterrows():
        m = r["month"]; inicio = r["mv_start"]; fin = r["mv_end"]
        ingresos = float(ci_mes.get(m, 0.0))
        fin_neto = (fin - ingresos) if (fin is not None and not pd.isna(fin)) else None
        rent_usd = ((fin_neto - inicio) if (fin_neto is not None and inicio is not None
                    and not pd.isna(inicio)) else None)
        try:
            mes_dt = pd.Period(m, "M").to_timestamp()
            mes_label = f"{MESES_ES[mes_dt.month-1]} {mes_dt.year}"
        except Exception:
            mes_label = m
        result.append({"Mes": mes_label, "_month": m, "Inicio": inicio,
                       "Ingresos": ingresos if ingresos else None, "Fin": fin,
                       "Fin (neto ingr.)": fin_neto, "Rent. USD": rent_usd,
                       "Rent. %": r["r_monthly_pct"], "TWR mensual %": r["r_monthly_pct"],
                       "TWR acumulado %": r["r_cum_twr"]})
    return pd.DataFrame(result)


# ═════════════════════════════════════════════════════════════════════════════
#  Resumen de capital
# ═════════════════════════════════════════════════════════════════════════════
def _capital_summary(dfa, mv_ultima):
    cash_in_rows = dfa[dfa["flow_bucket"].eq("Cash In")].sort_values("Settlement Date")
    cash_out_rows = dfa[dfa["flow_bucket"].eq("Cash Out")]
    if cash_in_rows.empty:
        pos_inicial = ingresos_adic = 0.0
    else:
        pos_inicial   = float(cash_in_rows.iloc[0]["Net Amount (Base Currency)"])
        ingresos_adic = float(cash_in_rows.iloc[1:]["Net Amount (Base Currency)"].sum())
    fondeo_total = pos_inicial + ingresos_adic
    egresos = abs(float(cash_out_rows["Net Amount (Base Currency)"].sum())) if not cash_out_rows.empty else 0.0
    fondeo_neto = fondeo_total - egresos
    resultado = (mv_ultima - fondeo_neto) if mv_ultima is not None else None
    return {"pos_ini": pos_inicial, "ingresos": ingresos_adic, "fondeo_total": fondeo_total,
            "egresos": egresos, "fondeo_neto": fondeo_neto, "pos_fin": mv_ultima,
            "resultado": resultado}


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers de UI
# ═════════════════════════════════════════════════════════════════════════════
def _fmt_m(x):  return f"{x:,.2f}" if (x is not None and not pd.isna(x)) else "—"
def _fmt_p(x):  return f"{x:+.2f}%" if (x is not None and not pd.isna(x)) else "—"
def _dc(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "neu"
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")

def _sym_description(dfa, sym):
    rows = dfa[dfa["symbol_key"].eq(sym)]["Security Description"].dropna()
    return _first(rows) or sym

def _kpi(label, val_str, sub_str, cls):
    return (f'<div class="kpi {cls}">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{val_str}</div>'
            f'<div class="kpi-sub {cls}">{sub_str}</div>'
            f'</div>')

def _cap_cell(label, amt_str, tag_str, tag_cls):
    return (f'<div class="cap-cell">'
            f'<div class="cap-lbl">{label}</div>'
            f'<div class="cap-amt">{amt_str}</div>'
            f'<span class="cap-tag {tag_cls}">{tag_str}</span>'
            f'</div>')

def _summary_periods(twr, selected_month):
    row_sel = twr[twr["month"].eq(selected_month)].copy()
    if row_sel.empty: return pd.DataFrame()
    out = row_sel[["group","r_3m","r_12m","r_ytd","r_inception","mv_end"]].copy()
    out.columns = ["Grupo","3 meses","12 meses","YTD","Desde inception","MV cierre"]
    return out.sort_values("MV cierre", ascending=False)

def _to_excel(twr_total, twr_asset, twr_sym, positions):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        twr_total.to_excel(w, sheet_name="TWR_Cartera_Total", index=False)
        twr_asset.to_excel(w, sheet_name="TWR_Asset_Class",   index=False)
        twr_sym.to_excel(  w, sheet_name="TWR_Por_Simbolo",   index=False)
        positions.to_excel(w, sheet_name="Posiciones_Cierre", index=False)
    return buf.getvalue()

def _show_df(df, height=420, money=None, pct=None, qty=None):
    money = money or []; pct = pct or []; qty = qty or []
    if df.empty:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)
        return
    fmt = {**{c:"{:,.2f}" for c in money if c in df.columns},
           **{c:"{:+.2f}%" for c in pct   if c in df.columns},
           **{c:"{:,.4f}"  for c in qty   if c in df.columns}}
    styled = df.style.format(fmt, na_rep="—")
    for col in pct:
        if col in df.columns:
            styled = styled.map(
                lambda v: (f"color:{POS_C};font-weight:500" if isinstance(v,(int,float)) and not np.isnan(v) and v>0
                     else f"color:{NEG_C};font-weight:500"  if isinstance(v,(int,float)) and not np.isnan(v) and v<0
                     else ""), subset=pd.IndexSlice[:,[col]])
    for col in money:
        if col in df.columns:
            styled = styled.map(
                lambda v: (f"color:{POS_C};font-weight:500" if isinstance(v,(int,float)) and not np.isnan(v) and v>0
                     else f"color:{NEG_C};font-weight:500"  if isinstance(v,(int,float)) and not np.isnan(v) and v<0
                     else ""), subset=pd.IndexSlice[:,[col]])
    st.dataframe(styled, use_container_width=True, height=height, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Gráficos Altair — estética renovada
# ═════════════════════════════════════════════════════════════════════════════
def _chart_total_twr(twr):
    df = twr[twr["group"].eq("Total")].copy()
    df = df[df["r_cum_twr"].notna() & df["r_monthly_pct"].notna()].copy()
    if df.empty:
        st.info("Sin datos suficientes para graficar el TWR total.")
        return
    df["month_dt"] = pd.to_datetime(df["month"])
    df["color_bar"] = df["r_monthly_pct"].apply(lambda x: "positivo" if x >= 0 else "negativo")

    area = (alt.Chart(df)
            .mark_area(opacity=.12, color=ACC_C,
                       line={"color": ACC_C, "strokeWidth": 1.5})
            .encode(
                x=alt.X("month_dt:T", title="", axis=alt.Axis(format="%b %Y", labelFontSize=10,
                          tickColor="transparent", domainColor="transparent", labelColor="#7a818f")),
                y=alt.Y("r_cum_twr:Q", title="TWR acumulado (%)",
                        axis=alt.Axis(labelFontSize=10, gridColor="rgba(15,17,23,.06)",
                                      tickColor="transparent", domainColor="transparent",
                                      labelColor="#7a818f", format="+.1f")),
                tooltip=[alt.Tooltip("month:N", title="Mes"),
                         alt.Tooltip("r_cum_twr:Q", title="TWR Acum %", format="+.2f"),
                         alt.Tooltip("mv_end:Q", title="MV cierre", format=",.0f")]
            ))
    zero_l = (alt.Chart(pd.DataFrame({"y":[0]}))
              .mark_rule(color="#7a818f", strokeDash=[4,4], opacity=.4)
              .encode(y="y:Q"))
    bars = (alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("month_dt:T", title="",
                         axis=alt.Axis(format="%b %Y", labelFontSize=10,
                                       tickColor="transparent", domainColor="transparent",
                                       labelColor="#7a818f")),
                y=alt.Y("r_monthly_pct:Q", title="Rend. mensual (%)",
                        axis=alt.Axis(labelFontSize=10, gridColor="rgba(15,17,23,.06)",
                                      tickColor="transparent", domainColor="transparent",
                                      labelColor="#7a818f", format="+.1f")),
                color=alt.Color("color_bar:N",
                    scale=alt.Scale(domain=["positivo","negativo"], range=[POS_C, NEG_C]),
                    legend=None),
                tooltip=[alt.Tooltip("month:N", title="Mes"),
                         alt.Tooltip("r_monthly_pct:Q", title="Rend. mensual %", format="+.2f")]
            ))
    zero_b = (alt.Chart(pd.DataFrame({"y":[0]}))
              .mark_rule(color="#7a818f", strokeDash=[4,4], opacity=.4)
              .encode(y="y:Q"))

    chart = (alt.vconcat(
                (area + zero_l).properties(height=180, title=alt.TitleParams(
                    "TWR acumulado", fontSize=11, fontWeight=500, color="#7a818f")),
                (bars + zero_b).properties(height=90, title=alt.TitleParams(
                    "Rendimiento mensual", fontSize=11, fontWeight=500, color="#7a818f")),
             )
             .configure_view(strokeWidth=0)
             .configure_axis(labelFont="'DM Mono', monospace", titleFont="'Sora', sans-serif",
                             titleFontSize=10, titleColor="#7a818f")
             .configure_concat(spacing=16))
    st.altair_chart(chart, use_container_width=True)


def _chart_by_group(twr, title, palette):
    df = twr[twr["r_cum_twr"].notna()].copy()
    if df.empty:
        st.info("Sin datos suficientes.")
        return
    df["month_dt"] = pd.to_datetime(df["month"])
    groups = df["group"].unique().tolist()
    colors = [palette.get(g, MUTED) for g in groups]

    chart = (alt.Chart(df)
             .mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=30, filled=True))
             .encode(
                 x=alt.X("month_dt:T", title="",
                         axis=alt.Axis(format="%b %Y", labelFontSize=10,
                                       tickColor="transparent", domainColor="transparent",
                                       labelColor="#7a818f")),
                 y=alt.Y("r_cum_twr:Q", title="TWR acumulado (%)",
                         axis=alt.Axis(labelFontSize=10, gridColor="rgba(15,17,23,.06)",
                                       tickColor="transparent", domainColor="transparent",
                                       labelColor="#7a818f", format="+.1f")),
                 color=alt.Color("group:N",
                     scale=alt.Scale(domain=groups, range=colors),
                     legend=alt.Legend(title="", labelFontSize=11, symbolSize=60,
                                       orient="bottom", columns=4)),
                 tooltip=[alt.Tooltip("group:N", title="Grupo"),
                          alt.Tooltip("month:N", title="Mes"),
                          alt.Tooltip("r_cum_twr:Q", title="TWR Acum %", format="+.2f"),
                          alt.Tooltip("mv_end:Q", title="MV cierre", format=",.0f")]
             )
             .properties(height=240, title=alt.TitleParams(
                 title, fontSize=11, fontWeight=500, color="#7a818f"))
             .configure_view(strokeWidth=0))
    st.altair_chart(chart, use_container_width=True)


def _chart_symbol_heatmap(twr):
    df = twr[twr["r_monthly_pct"].notna()].copy()
    if df.empty:
        st.info("Sin datos para el heatmap.")
        return
    df["month_dt"] = pd.to_datetime(df["month"])
    top = (df.groupby("group")["mv_end"].mean()
           .sort_values(ascending=False).head(30).index.tolist())
    df = df[df["group"].isin(top)].copy()

    chart = (alt.Chart(df)
             .mark_rect(cornerRadius=2)
             .encode(
                 x=alt.X("month_dt:T", title="",
                         axis=alt.Axis(format="%b %Y", labelFontSize=9,
                                       tickColor="transparent", domainColor="transparent",
                                       labelColor="#7a818f")),
                 y=alt.Y("group:N", title="", sort=top,
                         axis=alt.Axis(labelFontSize=10, labelColor="#3a3f4a",
                                       tickColor="transparent", domainColor="transparent")),
                 color=alt.Color("r_monthly_pct:Q", title="Rend. mensual %",
                     scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
                 tooltip=[alt.Tooltip("group:N", title="Símbolo"),
                          alt.Tooltip("month:N", title="Mes"),
                          alt.Tooltip("r_monthly_pct:Q", title="Rend. mensual %", format="+.2f"),
                          alt.Tooltip("mv_end:Q", title="MV cierre", format=",.0f")]
             )
             .properties(height=max(28*len(top), 200),
                         title=alt.TitleParams("Rendimiento mensual por símbolo",
                                               fontSize=11, fontWeight=500, color="#7a818f"))
             .configure_view(strokeWidth=0))
    st.altair_chart(chart, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ▶  RENDER PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
def render(_ctx=None) -> None:
    _css()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="twr-header">'
        '<div><h1>TWR Engine</h1>'
        '<p>time-weighted return &nbsp;·&nbsp; modified dietz mensual &nbsp;·&nbsp; yahoo finance</p></div>'
        '<span class="twr-badge">pershing xlsx</span>'
        '</div>',
        unsafe_allow_html=True)

    up = st.file_uploader("", type=["xlsx","xls"],
                          label_visibility="collapsed")
    if up is None:
        st.markdown('<p style="font-size:12px;color:#7a818f;margin-top:8px">'
                    'Subí el Excel de transacciones Pershing para comenzar.</p>',
                    unsafe_allow_html=True)
        return

    # ── Parseo ───────────────────────────────────────────────────────────────
    try:
        df_raw = _read_pershing_excel(up.getvalue())
    except Exception as e:
        st.error("No pude leer el Excel.")
        st.exception(e)
        return

    if "Settlement Date" not in df_raw.columns:
        st.error("No encontré 'Settlement Date'.")
        return

    dfa = _enrich(df_raw)
    if dfa.empty:
        st.warning("Sin filas válidas.")
        return

    # ── Precios ──────────────────────────────────────────────────────────────
    cache_key = f"prices_{up.name}"
    if cache_key not in st.session_state:
        with st.spinner("Descargando precios desde Yahoo Finance…"):
            prices_df = _get_prices(dfa, cache_key)
    else:
        prices_df = st.session_state[cache_key]

    # ── Instrumentos sin precio ──────────────────────────────────────────────
    all_syms   = set(dfa.loc[dfa["flow_bucket"].isin(["Buy","Sell"]) & dfa["symbol_key"].ne(""), "symbol_key"])
    found_syms = set(prices_df.columns.tolist()) if not prices_df.empty else set()
    missing_sym = sorted(all_syms - found_syms)

    if missing_sym:
        with st.expander(f"⚠ {len(missing_sym)} instrumento(s) sin precio en Yahoo Finance — cargá NAV manual",
                         expanded=True):
            st.markdown(
                '<p style="font-size:11px;color:#7a818f;margin-bottom:12px">'
                'Pegá el valor de cuotaparte (NAV) al cierre de cada mes. '
                'Dejá en blanco los meses sin posición. Formato: número con punto decimal (ej: 42.54)</p>',
                unsafe_allow_html=True)
            all_months_nav = sorted(dfa["month"].dropna().unique().tolist())
            manual_key = f"manual_prices_{up.name}"
            if manual_key not in st.session_state:
                st.session_state[manual_key] = {}

            for sym in missing_sym:
                st.markdown(f'<p style="font-size:12px;font-weight:600;margin:10px 0 6px;color:#0f1117">'
                            f'{sym} — {_sym_description(dfa, sym)}</p>', unsafe_allow_html=True)
                cols = st.columns(min(len(all_months_nav), 7))
                for i, m in enumerate(all_months_nav):
                    col_idx = i % len(cols)
                    key_widget = f"nav_{sym}_{m}"
                    val = cols[col_idx].text_input(
                        m, value=st.session_state[manual_key].get(key_widget, ""),
                        key=key_widget, label_visibility="visible")
                    if val.strip():
                        try:
                            st.session_state[manual_key][key_widget] = val.strip()
                        except Exception:
                            pass

            manual_rows = {}
            for sym in missing_sym:
                for m in all_months_nav:
                    key_widget = f"nav_{sym}_{m}"
                    raw = st.session_state[manual_key].get(key_widget, "")
                    if raw:
                        try:
                            me = pd.Period(m, "M").to_timestamp("M")
                            manual_rows.setdefault(me, {})[sym] = float(raw.replace(",","."))
                        except Exception:
                            pass
            if manual_rows:
                manual_df = pd.DataFrame(manual_rows).T
                manual_df.index = pd.to_datetime(manual_df.index).normalize()
                if prices_df.empty:
                    prices_df = manual_df
                else:
                    prices_df = prices_df.join(manual_df, how="outer")
                st.session_state[cache_key] = prices_df
                still_missing = [s for s in missing_sym
                                 if s not in prices_df.columns or prices_df[s].isna().all()]
                if still_missing:
                    st.caption(f"Todavía sin precio: {', '.join(still_missing)}")

    # ── Posiciones y cómputo ─────────────────────────────────────────────────
    positions = _build_positions(dfa)
    cf_df     = _cash_flows(dfa)

    with st.spinner("Calculando TWR…"):
        mv_total = _market_value_series(positions, prices_df, group_col=None)
        mv_asset = _market_value_series(positions, prices_df, group_col="asset_bucket")
        mv_sym   = _market_value_series(positions, prices_df, group_col="symbol_key")
        twr_total = _compute_twr(mv_total, cf_df, group_col=None)
        twr_asset = _compute_twr(mv_asset, cf_df, group_col="asset_bucket")
        twr_sym   = _compute_twr(mv_sym,   cf_df, group_col="symbol_key")

    all_months = sorted(twr_total["month"].unique().tolist()) if not twr_total.empty else []
    if not all_months:
        st.warning("No hay datos suficientes para calcular el TWR.")
        return

    # ── Filtro de mes ─────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    col_f, _ = st.columns([1,3])
    with col_f:
        selected_month = st.selectbox("Mes de referencia", all_months,
                                      index=len(all_months)-1,
                                      label_visibility="visible")

    n_syms   = positions[positions["month"].eq(selected_month)]["symbol_key"].nunique() if not positions.empty else 0
    mv_end_t = twr_total.loc[twr_total["month"].eq(selected_month), "mv_end"].sum()
    has_price = not prices_df.empty

    ok_cls = "ok" if has_price else ""
    st.markdown(
        f'<div class="pill-row">'
        f'<span class="pill">Símbolos en cartera: {n_syms}</span>'
        f'<span class="pill">MV estimado: {_fmt_m(mv_end_t)}</span>'
        f'<span class="pill {ok_cls}">Precios: {"✓ OK" if has_price else "⚠ sin datos"}</span>'
        f'</div>',
        unsafe_allow_html=True)

    # ── Resumen de capital ───────────────────────────────────────────────────
    total_rows = twr_total[twr_total["group"].eq("Total")].sort_values("month")
    mv_ultima  = total_rows["mv_end"].dropna().iloc[-1] if not total_rows.empty else None
    cap = _capital_summary(dfa, mv_ultima)

    st.markdown('<div class="sec-label">Resumen de capital</div>', unsafe_allow_html=True)
    res = cap["resultado"]
    res_cls = "pos" if res and res > 0 else ("neg" if res and res < 0 else "neu")
    res_tag = "▲ ganancia" if res and res > 0 else ("▼ pérdida" if res and res < 0 else "—")

    cols_html = "".join([
        _cap_cell("Posición inicial",      _fmt_m(cap["pos_ini"]),      "primer ingreso", "neu"),
        _cap_cell("Ingresos adicionales",  _fmt_m(cap["ingresos"]),     "post-apertura",  "neu"),
        _cap_cell("Fondeo total",          _fmt_m(cap["fondeo_total"]), "aportado",       "neu"),
        _cap_cell("Egresos",               _fmt_m(cap["egresos"]),      "retirado",       "neg" if cap["egresos"] > 0 else "neu"),
        _cap_cell("Fondeo neto",           _fmt_m(cap["fondeo_neto"]),  "neto",           "neu"),
        _cap_cell("Posición final",        _fmt_m(cap["pos_fin"]),      "MV mercado",     "neu"),
        _cap_cell("Resultado $",           _fmt_m(res),                 res_tag,          res_cls),
    ])
    st.markdown(
        f'<div class="cap-grid" style="grid-template-columns:repeat(7,1fr)">{cols_html}</div>',
        unsafe_allow_html=True)

    # ── KPIs del mes ──────────────────────────────────────────────────────────
    def _get_r(df, grp, col):
        row = df[(df["group"].eq(grp)) & (df["month"].eq(selected_month))]
        if row.empty: return None
        v = row.iloc[0][col]
        return None if pd.isna(v) else float(v)

    r_m   = _get_r(twr_total, "Total", "r_monthly_pct")
    r_3m  = _get_r(twr_total, "Total", "r_3m")
    r_ytd = _get_r(twr_total, "Total", "r_ytd")
    r_12m = _get_r(twr_total, "Total", "r_12m")
    r_si  = _get_r(twr_total, "Total", "r_inception")
    mv_end = _get_r(twr_total, "Total", "mv_end")
    mv_ini = _get_r(twr_total, "Total", "mv_start")

    st.markdown('<div class="sec-label" style="margin-top:24px">Rendimientos</div>',
                unsafe_allow_html=True)
    kpis_html = "".join([
        _kpi("Rend. mensual",   _fmt_p(r_m),        f"dic {selected_month[:4]}",  _dc(r_m)),
        _kpi("3 meses",         _fmt_p(r_3m),       "rolling 3M",                 _dc(r_3m)),
        _kpi("YTD",             _fmt_p(r_ytd),      f"ene–{selected_month[5:7]}",  _dc(r_ytd)),
        _kpi("12 meses",        _fmt_p(r_12m),      "rolling 12M",                _dc(r_12m)),
        _kpi("Desde inception", _fmt_p(r_si),       "TWR encadenado",             _dc(r_si)),
        _kpi("MV cierre",
             f'<span class="kpi-val sm">{_fmt_m(mv_end)}</span>',
             "USD base",  "acc"),
    ])
    st.markdown(
        f'<div class="kpi-grid" style="grid-template-columns:repeat(6,1fr)">{kpis_html}</div>',
        unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Cartera total", "Asset class", "Por símbolo",
        "Períodos", "Posición", "Movimientos", "Mes a mes",
    ])

    # ─ Tab 1 ─────────────────────────────────────────────────────────────────
    with t1:
        _chart_total_twr(twr_total)
        view = twr_total[twr_total["group"].eq("Total")].copy()
        view = view.rename(columns={
            "month":"Mes","mv_start":"MV inicio","mv_end":"MV cierre","net_cf":"Flujo neto",
            "r_monthly_pct":"Rend. mensual %","r_cum_twr":"TWR acum %",
            "r_3m":"3M %","r_12m":"12M %","r_ytd":"YTD %","r_inception":"Inception %"})
        _show_df(view[["Mes","MV inicio","MV cierre","Flujo neto",
                       "Rend. mensual %","TWR acum %","3M %","12M %","YTD %","Inception %"]],
                 height=380,
                 money=["MV inicio","MV cierre","Flujo neto"],
                 pct=["Rend. mensual %","TWR acum %","3M %","12M %","YTD %","Inception %"])
        st.markdown(
            '<div class="formula-note">R = (MV_fin − MV_ini − CF) / (MV_ini + 0.5×CF) &nbsp;·&nbsp; TWR = ∏(1+Rᵢ)−1</div>',
            unsafe_allow_html=True)

    # ─ Tab 2 ─────────────────────────────────────────────────────────────────
    with t2:
        sym_palette = {k: ASSET_COLORS.get(k, MUTED) for k in twr_asset["group"].unique()}
        _chart_by_group(twr_asset, "TWR acumulado por asset class", sym_palette)
        view_a = twr_asset.rename(columns={
            "group":"Asset Class","month":"Mes","mv_end":"MV cierre",
            "r_monthly_pct":"Rend. mensual %","r_cum_twr":"TWR acum %",
            "r_3m":"3M %","r_12m":"12M %","r_ytd":"YTD %","r_inception":"Inception %"})
        _show_df(
            view_a[["Mes","Asset Class","MV cierre","Rend. mensual %",
                    "TWR acum %","3M %","12M %","YTD %","Inception %"]].sort_values(["Mes","Asset Class"]),
            height=420, money=["MV cierre"],
            pct=["Rend. mensual %","TWR acum %","3M %","12M %","YTD %","Inception %"])

    # ─ Tab 3 ─────────────────────────────────────────────────────────────────
    with t3:
        sym_opts = ["Todos"] + sorted(twr_sym["group"].unique().tolist())
        sel_sym  = st.selectbox("Símbolo", sym_opts, key="sym_filter")
        sym_twr  = twr_sym if sel_sym == "Todos" else twr_sym[twr_sym["group"].eq(sel_sym)]

        if sel_sym == "Todos":
            _chart_symbol_heatmap(sym_twr)
        else:
            _chart_by_group(sym_twr, f"TWR acumulado — {sel_sym}", {sel_sym: ACC_C})

        view_s = sym_twr.rename(columns={
            "group":"Símbolo","month":"Mes","mv_end":"MV cierre",
            "r_monthly_pct":"Rend. mensual %","r_cum_twr":"TWR acum %",
            "r_3m":"3M %","r_12m":"12M %","r_ytd":"YTD %","r_inception":"Inception %"})
        _show_df(
            view_s[["Mes","Símbolo","MV cierre","Rend. mensual %",
                    "TWR acum %","3M %","12M %","YTD %","Inception %"]].sort_values(["Mes","Símbolo"]),
            height=440, money=["MV cierre"],
            pct=["Rend. mensual %","TWR acum %","3M %","12M %","YTD %","Inception %"])

    # ─ Tab 4 ─────────────────────────────────────────────────────────────────
    with t4:
        st.markdown(f'<p class="formula-note" style="margin-bottom:12px">'
                    f'Retornos acumulados al cierre de <b>{selected_month}</b></p>',
                    unsafe_allow_html=True)
        for label, twr_df in [("Cartera total", twr_total),
                               ("Por asset class", twr_asset),
                               ("Por símbolo", twr_sym)]:
            st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)
            s = _summary_periods(twr_df, selected_month)
            if not s.empty:
                h = 80 if label == "Cartera total" else (180 if "asset" in label else 460)
                _show_df(s, height=h, money=["MV cierre"],
                         pct=["3 meses","12 meses","YTD","Desde inception"])

    # ─ Tab 5 ─────────────────────────────────────────────────────────────────
    with t5:
        st.markdown(f'<p class="formula-note" style="margin-bottom:4px">'
                    f'Posición al cierre de <b>{selected_month}</b></p>',
                    unsafe_allow_html=True)
        estado = _build_estado_posicion(positions, prices_df, dfa, selected_month)
        if estado.empty:
            st.info("Sin posiciones para el mes seleccionado.")
        else:
            total_mv = estado["Valuación"].sum(skipna=True)
            for cat in ["MM, T Bills y Simis","Renta Fija","Renta Variable","Otros"]:
                bloque = estado[estado["categoria"].eq(cat)].copy()
                if bloque.empty: continue
                subtotal  = bloque["Valuación"].sum(skipna=True)
                pct_cat   = subtotal / total_mv * 100 if total_mv else 0
                st.markdown(
                    f'<div class="cat-hdr"><span>{cat}</span>'
                    f'<span>{_fmt_m(subtotal)} · {pct_cat:.2f}%</span></div>',
                    unsafe_allow_html=True)
                view = bloque[["Instrumento","ISIN","Cantidad","Precio","Valuación","% Cartera"]].copy()
                _show_df(view, height=min(50+len(view)*38, 400),
                         money=["Precio","Valuación"])

            st.markdown(f'<div class="total-bar">Total cartera &nbsp; {_fmt_m(total_mv)}</div>',
                        unsafe_allow_html=True)
            if missing_sym:
                st.caption(f"⚠ Sin precio en Yahoo Finance: {', '.join(sorted(missing_sym))}")

    # ─ Tab 6 ─────────────────────────────────────────────────────────────────
    with t6:
        movs = _build_movimientos_mes(dfa, selected_month)
        col_c, col_v = st.columns(2)

        def _render_mov(col, df, tipo, bg_color):
            with col:
                if df.empty:
                    st.caption(f"Sin {tipo.lower()} en {selected_month}.")
                    return
                total = df["Total USD"].sum()
                st.markdown(
                    f'<div class="mov-col-hdr" style="background:{bg_color}">'
                    f'<span>{tipo}</span><span>Total: {_fmt_m(total)} USD</span></div>',
                    unsafe_allow_html=True)
                for cat in ["MM, T Bills y Simis","Renta Fija","Renta Variable","Otros"]:
                    bloque = df[df["Categoría"].eq(cat)]
                    if bloque.empty: continue
                    subtotal = bloque["Total USD"].sum()
                    st.markdown(
                        f'<div class="cat-hdr">'
                        f'<span>{cat}</span><span>{_fmt_m(subtotal)} USD</span></div>',
                        unsafe_allow_html=True)
                    view = bloque[["Instrumento","ISIN","Cantidad","Precio unit.","Total USD"]].copy()
                    _show_df(view, height=min(50+len(view)*38, 360),
                             money=["Precio unit.","Total USD"])

        _render_mov(col_c, movs["compras"], "Compras", POS_C)
        _render_mov(col_v, movs["ventas"],  "Ventas",  NEG_C)

    # ─ Tab 7 ─────────────────────────────────────────────────────────────────
    with t7:
        rent_df = _build_rentabilidad_mensual(dfa, twr_total)
        if rent_df.empty:
            st.info("Sin datos suficientes.")
        else:
            twr_final = rent_df["TWR acumulado %"].dropna().iloc[-1] if not rent_df.empty else None
            view = rent_df[[
                "Mes","Inicio","Ingresos","Fin",
                "Fin (neto ingr.)","Rent. USD","Rent. %","TWR acumulado %"
            ]].copy()

            st.markdown(
                '<div class="cat-hdr" style="margin-bottom:8px">'
                '<span>Rentabilidad mes a mes</span></div>',
                unsafe_allow_html=True)

            fmt = {"Inicio":"{:,.3f}", "Ingresos":"{:,.3f}", "Fin":"{:,.3f}",
                   "Fin (neto ingr.)":"{:,.3f}", "Rent. USD":"{:,.3f}",
                   "Rent. %":"{:+.2f}%", "TWR acumulado %":"{:+.2f}%"}
            styled = (view.style
                      .format(fmt, na_rep="—")
                      .map(lambda v: (f"color:{POS_C};font-weight:500"
                                     if isinstance(v,(int,float)) and not pd.isna(v) and v > 0
                                     else f"color:{NEG_C};font-weight:500"
                                     if isinstance(v,(int,float)) and not pd.isna(v) and v < 0
                                     else ""),
                           subset=pd.IndexSlice[:,["Rent. USD","Rent. %","TWR acumulado %"]]))
            st.dataframe(styled, use_container_width=True,
                         height=min(80+len(view)*36, 600), hide_index=True)

            if twr_final is not None:
                color = POS_C if twr_final > 0 else NEG_C
                bg    = POS_BG if twr_final > 0 else NEG_BG
                st.markdown(
                    f'<div style="text-align:right;padding:8px 0;margin-top:4px;">'
                    f'TWR total &nbsp;'
                    f'<span style="background:{bg};color:{color};padding:5px 14px;'
                    f'border-radius:6px;font-weight:600;font-size:15px;font-family:\'DM Mono\',monospace">'
                    f'{twr_final:+.2f}%</span></div>',
                    unsafe_allow_html=True)

            st.markdown(
                '<div class="formula-note">Rent. % = (Fin neto ingresos − Inicio) / Inicio &nbsp;·&nbsp; TWR = ∏(1+Rᵢ)−1</div>',
                unsafe_allow_html=True)

    # ── Descarga ──────────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Descarga</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "⬇ Descargar Excel completo (TWR + posiciones)",
            data=_to_excel(twr_total, twr_asset, twr_sym, positions),
            file_name=f"twr_{selected_month}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_d2:
        st.download_button(
            "⬇ Tabla de períodos (CSV)",
            data=(_summary_periods(twr_total, selected_month)
                  .to_csv(index=False).encode("utf-8-sig")) if not twr_total.empty else b"",
            file_name=f"twr_periodos_{selected_month}.csv",
            mime="text/csv",
            use_container_width=True)
