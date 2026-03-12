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
ACCENT  = "#ff3b30"
TEXT    = "#111827"
MUTED   = "#6b7280"
POS_BG  = "rgba(16,185,129,.10)";  POS_TXT = "#047857"
NEG_BG  = "rgba(239,68,68,.10)";   NEG_TXT = "#b91c1c"
NEU_BG  = "rgba(107,114,128,.10)"; NEU_TXT = "#4b5563"

ASSET_COLORS = {
    "ETF":      "#6366f1",
    "Stock":    "#f59e0b",
    "Bond":     "#10b981",
    "Fund":     "#3b82f6",
    "Currency": "#8b5cf6",
    "Other":    "#9ca3af",
    "Total":    ACCENT,
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
CASH_TX     = {"FEDERAL FUNDS RECEIVED","FEDERAL FUNDS SENT"}
INTERNAL_TX = {"ACTIVITY WITHIN YOUR ACCT"}
DIV_MARKERS = ["CASH DIVIDEND RECEIVED","FOREIGN SECURITY DIVIDEND RECEIVED"]
TAX_MARKERS = ["NON-RESIDENT ALIEN TAX","FOREIGN TAX WITHHELD AT   THE SOURCE",
               "FOREIGN TAX WITHHELD"]
# Fees exactos de Pershing (match exacto para evitar falsos positivos con texto libre de trades)
FEE_EXACT   = {"PES BILLING FEE","FOREIGN CUSTODY FEE","ASSET BASED FEE",
               "PAPER DELIVERY            SUBSCRIPTION",
               "ASSET MANAGEMENT ACCOUNT  SPECIAL HANDLING FEE",
               "INT. CHARGED ON DEBIT     BALANCES",
               "FEE ON FOREIGN DIVIDEND   WITHHELD AT THE SOURCE"}
FEE_MARKERS = ["ADVISORY","CUSTODY","SUBSCRIPTION","ASSET BASED FEE"]
# El dividendo de PBR trae un fee separado — lo sumamos a Fee, no a Dividend
DIV_FEE_MARKERS = ["FEE ON FOREIGN DIVIDEND"]


# ═════════════════════════════════════════════════════════════════════════════
#  CSS
# ═════════════════════════════════════════════════════════════════════════════
def _css() -> None:
    st.markdown(f"""<style>
.block-container{{padding-top:1rem;max-width:1200px;padding-bottom:2rem}}
h1,h2,h3{{letter-spacing:-.02em;color:{TEXT}}}
.subtle{{color:rgba(17,24,39,.6);font-size:.93rem;margin-bottom:.3rem}}
.hr{{height:1px;background:rgba(17,24,39,.08);margin:14px 0 18px}}
.kpi{{position:relative;padding:16px 18px;border:1px solid rgba(17,24,39,.08);
      border-radius:18px;background:#fafafa;min-height:106px;overflow:hidden}}
.kpi::before{{content:"";position:absolute;left:0;top:0;width:100%;height:3px;background:rgba(17,24,39,.06)}}
.kpi.pos::before{{background:linear-gradient(90deg,rgba(16,185,129,.9),rgba(16,185,129,.2))}}
.kpi.neg::before{{background:linear-gradient(90deg,rgba(239,68,68,.9),rgba(239,68,68,.2))}}
.kpi.neu::before{{background:linear-gradient(90deg,rgba(107,114,128,.8),rgba(107,114,128,.15))}}
.kpi-lbl{{color:rgba(17,24,39,.54);font-size:.78rem;font-weight:600;text-transform:uppercase;
           letter-spacing:.05em;margin-bottom:8px}}
.kpi-val{{font-size:1.65rem;font-weight:750;letter-spacing:-.04em;color:{TEXT};margin-bottom:8px}}
.kpi-dlt{{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
           border-radius:999px;font-size:.77rem;font-weight:600}}
.kpi-dlt.pos{{background:{POS_BG};color:{POS_TXT}}}
.kpi-dlt.neg{{background:{NEG_BG};color:{NEG_TXT}}}
.kpi-dlt.neu{{background:{NEU_BG};color:{NEU_TXT}}}
.pill{{display:inline-block;padding:4px 10px;border-radius:999px;
       border:1px solid rgba(17,24,39,.10);background:#fff;
       color:rgba(17,24,39,.65);font-size:.79rem;margin-right:6px;margin-bottom:5px}}
.gap{{height:10px}}
</style>""", unsafe_allow_html=True)


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

    # Trades: Buy/Sell col es la fuente de verdad — va PRIMERO para evitar
    # que el texto libre "Buy 711 share(s) of EWZ..." matchee como fee/dividend
    if bs == "BUY":  return "Buy"
    if bs == "SELL": return "Sell"

    # Flujos internos
    if tx in INTERNAL_TX:              return "Internal"
    if tx == "FEDERAL FUNDS RECEIVED": return "Cash In"
    if tx == "FEDERAL FUNDS SENT":     return "Cash Out"

    # Dividendos (antes que tax, porque algunos tx tienen ambas palabras)
    if any(m in tx for m in DIV_MARKERS): return "Dividend"

    # Tax / retenciones
    if (any(m in tx for m in TAX_MARKERS)
            or code in {"NRA","FGN","FGF"}
            or tx == "NON-RESIDENT ALIEN TAX"):
        return "Tax"

    # Fee — primero match exacto, luego substrings seguros
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
#  Posición acumulada — qty por símbolo a cierre de cada mes
# ═════════════════════════════════════════════════════════════════════════════
def _build_positions(dfa: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve DataFrame: month_end | symbol_key | asset_bucket | closing_qty
    Una fila por (mes, símbolo) con posición no-cero.
    """
    trades = dfa[dfa["flow_bucket"].isin(["Buy","Sell"])].copy()
    trades = trades[trades["symbol_key"].ne("") & trades["symbol_key"].notna()].copy()
    if trades.empty:
        return pd.DataFrame(columns=["month_end","symbol_key","asset_bucket","closing_qty"])

    sym_meta = (trades.groupby("symbol_key", sort=False)
                .agg(asset_bucket=("asset_bucket", _first))
                .reset_index())

    all_months = pd.date_range(dfa["month_end"].min(), dfa["month_end"].max(), freq="ME")

    # delta de cantidad por (mes, símbolo)
    delta = (trades.groupby(["month_end","symbol_key"], sort=False)["signed_qty"]
             .sum().reset_index())

    # grid completo símbolos × meses
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
#  Precios yfinance — caché por sesión + st.cache_data (1 h)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_yf(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados y los resamplea al último día hábil de cada mes.
    Retorna DataFrame: índice=month_end, columnas=tickers.
    Tickers que yfinance no reconoce quedan con NaN (se reportan al usuario).
    """
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
        monthly.index = monthly.index.normalize()   # quita el componente de hora
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
#  MV(mes, nivel) = Σ qty_i × price_i  agrupado por el nivel deseado
# ═════════════════════════════════════════════════════════════════════════════
def _market_value_series(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    group_col: Optional[str] = None,   # None → cartera total, "asset_bucket" → por clase, "symbol_key" → por símbolo
) -> pd.DataFrame:
    """
    Devuelve DataFrame:
      month_end | [group_col] | mv_end | mv_start | price_ok
    donde mv_start = mv_end del mes anterior para ese grupo.
    price_ok indica si TODOS los símbolos del grupo tienen precio.
    """
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
        # Cartera total
        agg = (detail.groupby("month_end")
               .agg(mv_end=("mv","sum"), price_ok=("price_ok","all"))
               .reset_index())
        agg["group"] = "Total"
    elif group_col == "symbol_key":
        agg = (detail.groupby(["month_end","symbol_key"])
               .agg(mv_end=("mv","sum"), price_ok=("price_ok","all"))
               .reset_index()
               .rename(columns={"symbol_key":"group"}))
    else:  # asset_bucket
        agg = (detail.groupby(["month_end","asset_bucket"])
               .agg(mv_end=("mv","sum"), price_ok=("price_ok","all"))
               .reset_index()
               .rename(columns={"asset_bucket":"group"}))

    agg = agg.sort_values(["group","month_end"])
    agg["mv_start"] = agg.groupby("group")["mv_end"].shift(1)
    return agg


# ═════════════════════════════════════════════════════════════════════════════
#  Flujos externos netos por mes (y por grupo si aplica)
# ═════════════════════════════════════════════════════════════════════════════
def _cash_flows(dfa: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    CF neto = Cash In - |Cash Out|.
    Para grupo "Total" o "asset_bucket": el flujo externo no se puede asignar
    a un activo, así que lo asignamos todo al nivel Total / usamos el flujo
    total repartido igual para asset_bucket (aproximación razonable).
    Para symbol_key: no hay CF externo por símbolo (los flujos van a cash primero),
    así que CF=0 → Modified Dietz colapsa a retorno puro de precio.
    """
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
#  Modified Dietz por sub-período
# ═════════════════════════════════════════════════════════════════════════════
def _modified_dietz(mv_start: float, mv_end: float, net_cf: float, w: float = 0.5) -> Optional[float]:
    """
    R = (EV - BV - CF) / (BV + W×CF)
    W = 0.5 → asume flujos a mitad de mes (conservador).
    Retorna None si el denominador es 0 o si hay datos faltantes.
    """
    if any(np.isnan(v) for v in [mv_start, mv_end]):
        return None
    denom = mv_start + w * net_cf
    if abs(denom) < 1e-9:
        return None
    return (mv_end - mv_start - net_cf) / denom


# ═════════════════════════════════════════════════════════════════════════════
#  Motor TWR — encadenado
# ═════════════════════════════════════════════════════════════════════════════
def _compute_twr(
    mv_df: pd.DataFrame,     # output de _market_value_series
    cf_df: pd.DataFrame,     # output de _cash_flows
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retorna DataFrame con columnas:
      month | group | mv_start | mv_end | net_cf | r_monthly | r_cum_twr
      r_3m  | r_ytd | r_12m    | r_inception
    """
    if mv_df.empty:
        return pd.DataFrame()

    # Alinear CF con MV
    mv = mv_df.copy()
    mv = mv.merge(cf_df[["month_end","net_cf"]], on="month_end", how="left")
    mv["net_cf"] = mv["net_cf"].fillna(0.0)

    # Para símbolo individual CF=0 (no hay flujo externo por ticker)
    if group_col == "symbol_key":
        mv["net_cf"] = 0.0

    # Calcular retorno mensual por grupo
    mv["r_monthly"] = mv.apply(
        lambda r: _modified_dietz(r["mv_start"], r["mv_end"], r["net_cf"]),
        axis=1,
    )

    # Encadenar TWR por grupo
    records = []
    for grp, g in mv.groupby("group", sort=False):
        g = g.sort_values("month_end").copy()
        g["month"] = g["month_end"].dt.strftime("%Y-%m")

        # Factor acumulado (producto de (1+r_i))
        factors = []
        cum = 1.0
        for r in g["r_monthly"]:
            if r is None or np.isnan(r):
                factors.append(np.nan)
                cum = np.nan      # si perdemos un período, el acumulado se quiebra
            else:
                cum = (cum if not np.isnan(cum) else 1.0) * (1 + r)
                factors.append(cum)
        g["cum_factor"] = factors

        # TWR desde inception (factor final)
        # TWR 3m, 12m, YTD: cociente de factores acumulados
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
                        # si no hay diciembre anterior, usamos el primer mes del año en cartera
                        base_rows = g[g["month"].str.startswith(str(cur_year))].head(1)
                        base_cf   = 1.0
                    else:
                        base_cf   = base_rows.iloc[0]["cum_factor"]
                    cur_cf = row.cum_factor
                    if np.isnan(cur_cf) or np.isnan(base_cf) or base_cf == 0:
                        vals.append(np.nan)
                    else:
                        vals.append((cur_cf / base_cf - 1) * 100)
                else:  # rolling n_months
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

        g["r_cum_twr"]   = g["cum_factor"].apply(lambda x: (x-1)*100 if not np.isnan(x) else np.nan)
        g["r_3m"]        = _sub_twr(g, n_months=3)
        g["r_12m"]       = _sub_twr(g, n_months=12)
        g["r_ytd"]       = _sub_twr(g, ytd=True)
        g["r_inception"] = _sub_twr(g, inception=True)
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
#  UI helpers
# ═════════════════════════════════════════════════════════════════════════════
def _fmt_m(x):   return f"{x:,.2f}"  if (x is not None and not pd.isna(x)) else "—"
def _fmt_p(x):   return f"{x:+.2f}%" if (x is not None and not pd.isna(x)) else "—"
def _fmt_p0(x):  return f"{x:+.1f}%" if (x is not None and not pd.isna(x)) else "—"
def _dc(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "neu"
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")

def _kpi_html(label, value, delta=None, delta_label=None):
    klass = _dc(delta)
    dl    = delta_label if delta_label is not None else ("—" if delta is None or (isinstance(delta,float) and np.isnan(delta)) else (f"▲ {delta:+.2f}%" if delta>0 else f"▼ {delta:+.2f}%"))
    return (f'<div class="kpi {klass}">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{value}</div>'
            f'<div class="kpi-dlt {klass}">{dl}</div>'
            f'</div>')

def _show_df(df, height=420, money=None, pct=None, qty=None):
    money = money or []; pct = pct or []; qty = qty or []
    if df.empty:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)
        return
    fmt = {**{c:"{:,.2f}" for c in money if c in df.columns},
           **{c:"{:+.2f}%" for c in pct   if c in df.columns},
           **{c:"{:,.4f}"  for c in qty   if c in df.columns}}
    styled = df.style.format(fmt, na_rep="—")
    for col in money:
        if col in df.columns:
            styled = styled.map(
                lambda v: (f"color:{POS_TXT};background:{POS_BG};font-weight:600" if isinstance(v,(int,float)) and not np.isnan(v) and v>0
                     else f"color:{NEG_TXT};background:{NEG_BG};font-weight:600"  if isinstance(v,(int,float)) and not np.isnan(v) and v<0
                     else ""), subset=pd.IndexSlice[:,[col]])
    for col in pct:
        if col in df.columns:
            styled = styled.map(
                lambda v: (f"color:{POS_TXT};font-weight:600" if isinstance(v,(int,float)) and not np.isnan(v) and v>0
                     else f"color:{NEG_TXT};font-weight:600"  if isinstance(v,(int,float)) and not np.isnan(v) and v<0
                     else ""), subset=pd.IndexSlice[:,[col]])
    st.dataframe(styled, use_container_width=True, height=height, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Gráficos Altair
# ═════════════════════════════════════════════════════════════════════════════
def _chart_total_twr(twr: pd.DataFrame) -> None:
    """Área + barras mensuales para la cartera total."""
    df = twr[twr["group"].eq("Total")].copy()
    df = df[df["r_cum_twr"].notna() & df["r_monthly_pct"].notna()].copy()
    if df.empty:
        st.info("Sin datos suficientes para graficar el TWR total.")
        return
    df["month_dt"] = pd.to_datetime(df["month"])
    df["color_bar"] = df["r_monthly_pct"].apply(lambda x: "positivo" if x >= 0 else "negativo")

    area = (alt.Chart(df)
            .mark_area(opacity=.15, color=ACCENT,
                       line={"color":ACCENT,"strokeWidth":2})
            .encode(
                x=alt.X("month_dt:T", title=""),
                y=alt.Y("r_cum_twr:Q", title="TWR acumulado (%)"),
                tooltip=[alt.Tooltip("month:N",title="Mes"),
                         alt.Tooltip("r_cum_twr:Q",title="TWR Acum %",format="+.2f"),
                         alt.Tooltip("mv_end:Q",title="MV cierre",format=",.0f")]
            ))
    zero_l = (alt.Chart(pd.DataFrame({"y":[0]}))
              .mark_rule(color=MUTED, strokeDash=[4,4], opacity=.5)
              .encode(y="y:Q"))
    bars = (alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("month_dt:T", title=""),
                y=alt.Y("r_monthly_pct:Q", title="Rend. mensual (%)"),
                color=alt.Color("color_bar:N",
                    scale=alt.Scale(domain=["positivo","negativo"],range=["#10b981","#ef4444"]),
                    legend=None),
                tooltip=[alt.Tooltip("month:N",title="Mes"),
                         alt.Tooltip("r_monthly_pct:Q",title="Rend. mensual %",format="+.2f")]
            ))
    zero_b = (alt.Chart(pd.DataFrame({"y":[0]}))
              .mark_rule(color=MUTED, strokeDash=[4,4], opacity=.5)
              .encode(y="y:Q"))

    chart = (alt.vconcat(
                (area + zero_l).properties(height=210, title="TWR acumulado — cartera total"),
                (bars + zero_b).properties(height=110, title="Rendimiento mensual"),
             )
             .configure_view(strokeWidth=0)
             .configure_axis(labelFontSize=11, titleFontSize=11)
             .configure_title(fontSize=12, fontWeight=500, color=MUTED))
    st.altair_chart(chart, use_container_width=True)


def _chart_by_group(twr: pd.DataFrame, title: str, palette: dict) -> None:
    """Líneas por grupo para r_cum_twr."""
    df = twr[twr["r_cum_twr"].notna()].copy()
    if df.empty:
        st.info("Sin datos suficientes.")
        return
    df["month_dt"] = pd.to_datetime(df["month"])
    groups = df["group"].unique().tolist()
    colors = [palette.get(g, MUTED) for g in groups]

    chart = (alt.Chart(df)
             .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40))
             .encode(
                 x=alt.X("month_dt:T", title=""),
                 y=alt.Y("r_cum_twr:Q", title="TWR acumulado (%)"),
                 color=alt.Color("group:N",
                     scale=alt.Scale(domain=groups, range=colors),
                     legend=alt.Legend(title="")),
                 tooltip=[alt.Tooltip("group:N",title="Grupo"),
                          alt.Tooltip("month:N",title="Mes"),
                          alt.Tooltip("r_cum_twr:Q",title="TWR Acum %",format="+.2f"),
                          alt.Tooltip("mv_end:Q",title="MV cierre",format=",.0f")]
             )
             .properties(height=280, title=title)
             .configure_view(strokeWidth=0)
             .configure_axis(labelFontSize=11, titleFontSize=11)
             .configure_title(fontSize=12, fontWeight=500, color=MUTED))
    st.altair_chart(chart, use_container_width=True)


def _chart_symbol_heatmap(twr: pd.DataFrame) -> None:
    """Heatmap: símbolos × meses coloreado por r_monthly_pct."""
    df = twr[twr["r_monthly_pct"].notna()].copy()
    if df.empty:
        st.info("Sin datos para el heatmap.")
        return
    df["month_dt"] = pd.to_datetime(df["month"])

    # Limitar a top 30 símbolos por valor de mercado promedio
    top = (df.groupby("group")["mv_end"].mean()
           .sort_values(ascending=False).head(30).index.tolist())
    df = df[df["group"].isin(top)].copy()

    chart = (alt.Chart(df)
             .mark_rect()
             .encode(
                 x=alt.X("month_dt:T", title=""),
                 y=alt.Y("group:N", title="", sort=top),
                 color=alt.Color("r_monthly_pct:Q",
                     title="Rend. mensual %",
                     scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
                 tooltip=[alt.Tooltip("group:N",title="Símbolo"),
                          alt.Tooltip("month:N",title="Mes"),
                          alt.Tooltip("r_monthly_pct:Q",title="Rend. mensual %",format="+.2f"),
                          alt.Tooltip("mv_end:Q",title="MV cierre",format=",.0f")]
             )
             .properties(height=max(30*len(top),200), title="Rendimiento mensual por símbolo (heatmap)")
             .configure_view(strokeWidth=0)
             .configure_axis(labelFontSize=10)
             .configure_title(fontSize=12, fontWeight=500, color=MUTED))
    st.altair_chart(chart, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Tabla resumen de períodos
# ═════════════════════════════════════════════════════════════════════════════
def _summary_periods(twr: pd.DataFrame, selected_month: str) -> pd.DataFrame:
    """Fila por grupo con los 4 retornos al mes seleccionado."""
    row_sel = twr[twr["month"].eq(selected_month)].copy()
    if row_sel.empty:
        return pd.DataFrame()
    out = row_sel[["group","r_3m","r_12m","r_ytd","r_inception","mv_end"]].copy()
    out.columns = ["Grupo","3 meses","12 meses","YTD","Desde inception","MV cierre"]
    return out.sort_values("MV cierre", ascending=False)


# ═════════════════════════════════════════════════════════════════════════════
#  Excel de descarga
# ═════════════════════════════════════════════════════════════════════════════
def _to_excel(twr_total: pd.DataFrame, twr_asset: pd.DataFrame,
              twr_sym: pd.DataFrame, positions: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        twr_total.to_excel(w, sheet_name="TWR_Cartera_Total", index=False)
        twr_asset.to_excel(w, sheet_name="TWR_Asset_Class",   index=False)
        twr_sym.to_excel(  w, sheet_name="TWR_Por_Simbolo",   index=False)
        positions.to_excel(w, sheet_name="Posiciones_Cierre", index=False)
    return buf.getvalue()


def _capital_summary(dfa: pd.DataFrame, mv_primera: float, mv_ultima: float) -> dict:
    """
    Calcula los bloques de capital para el resumen superior:
      - Posición inicial : MV del primer mes con posición valorizada
      - Posición final   : MV del último mes valorizado
      - Ingresos         : suma de FEDERAL FUNDS RECEIVED (Cash In)
      - Egresos          : suma de FEDERAL FUNDS SENT     (Cash Out)
      - Fondeo total     : Ingresos + Egresos (neto absoluto aportado)
      - Resultado $      : Posición final − Posición inicial − Fondeo neto
    """
    ingresos = dfa.loc[dfa["flow_bucket"].eq("Cash In"),
                       "Net Amount (Base Currency)"].sum()
    egresos  = abs(dfa.loc[dfa["flow_bucket"].eq("Cash Out"),
                           "Net Amount (Base Currency)"].sum())
    fondeo   = ingresos - egresos          # neto aportado
    resultado = mv_ultima - mv_primera - fondeo  if (mv_primera and mv_ultima) else None
    return {
        "pos_ini":   mv_primera,
        "pos_fin":   mv_ultima,
        "ingresos":  ingresos,
        "egresos":   egresos,
        "fondeo":    fondeo,
        "resultado": resultado,
    }



    """Devuelve la descripción del instrumento para mostrar en el expander."""
    rows = dfa[dfa["symbol_key"].eq(sym)]["Security Description"].dropna()
    return _first(rows) or sym


# ═════════════════════════════════════════════════════════════════════════════
#  ▶  RENDER PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
def render(_ctx=None) -> None:
    _css()

    st.title("TWR Engine — Rendimiento de cartera")
    st.markdown(
        '<div class="subtle">Time-Weighted Return mensual encadenado · '
        'Precios de cierre vía Yahoo Finance · Modified Dietz intra-mes</div>',
        unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    up = st.file_uploader("Subí el Excel de transacciones (Pershing)", type=["xlsx","xls"])
    if up is None:
        st.info("Subí el Excel para empezar.")
        return

    # ── parseo ──────────────────────────────────────────────────────────────
    try:
        df_raw = _read_pershing_excel(up.getvalue())
    except Exception as e:
        st.error("No pude leer el Excel.")
        st.exception(e)
        return

    date_col = "Settlement Date"
    if date_col not in df_raw.columns:
        st.error("No encontré 'Settlement Date'.")
        return

    dfa = _enrich(df_raw)
    if dfa.empty:
        st.warning("Sin filas válidas.")
        return

    # ── precios yfinance ─────────────────────────────────────────────────────
    cache_key = f"prices_{up.name}"
    if cache_key not in st.session_state:
        with st.spinner("Descargando precios de cierre desde Yahoo Finance…"):
            prices_df = _get_prices(dfa, cache_key)
    else:
        prices_df = st.session_state[cache_key]

    # ── detectar tickers sin precio ─────────────────────────────────────────
    all_syms   = set(dfa.loc[dfa["flow_bucket"].isin(["Buy","Sell"]) & dfa["symbol_key"].ne(""), "symbol_key"])
    found_syms = set(prices_df.columns.tolist()) if not prices_df.empty else set()
    missing_sym = sorted(all_syms - found_syms)

    # ── input de precios manuales para instrumentos sin ticker ───────────────
    if missing_sym:
        with st.expander(
            f"⚠ {len(missing_sym)} instrumento(s) sin precio en Yahoo Finance — cargá NAV manual",
            expanded=True,
        ):
            st.markdown(
                '<div class="subtle">Pegá el valor de cuotaparte (NAV) al cierre de cada mes. '
                'Podés dejar en blanco los meses donde no tenías posición. '
                'Formato: número con punto decimal (ej: 42.54)</div>',
                unsafe_allow_html=True,
            )
            all_months_nav = sorted(dfa["month"].dropna().unique().tolist())
            manual_key = f"manual_prices_{up.name}"
            if manual_key not in st.session_state:
                st.session_state[manual_key] = {}

            for sym in missing_sym:
                st.markdown(f"**{sym}** — {_sym_description(dfa, sym)}")
                cols = st.columns(min(len(all_months_nav), 7))
                for i, m in enumerate(all_months_nav):
                    col_idx = i % len(cols)
                    key_widget = f"nav_{sym}_{m}"
                    val = cols[col_idx].text_input(
                        m, value=st.session_state[manual_key].get(key_widget, ""),
                        key=key_widget, label_visibility="visible"
                    )
                    if val.strip():
                        try:
                            st.session_state[manual_key][key_widget] = val.strip()
                        except Exception:
                            pass

            # Construir DataFrame de precios manuales y fusionar con prices_df
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
                still_missing = [s for s in missing_sym if s not in prices_df.columns or prices_df[s].isna().all()]
                if still_missing:
                    st.caption(f"Todavía sin precio: {', '.join(still_missing)}")

    # ── posiciones y flujos ──────────────────────────────────────────────────
    positions  = _build_positions(dfa)
    cf_df      = _cash_flows(dfa)

    # ── cómputo TWR (3 niveles) ──────────────────────────────────────────────
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

    # ── filtro de mes ─────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([1,3])
    with col_f1:
        selected_month = st.selectbox("Mes de referencia", all_months, index=len(all_months)-1)

    n_syms    = positions[positions["month"].eq(selected_month)]["symbol_key"].nunique() if not positions.empty else 0
    mv_end_t  = twr_total.loc[twr_total["month"].eq(selected_month),"mv_end"].sum()
    has_price = not prices_df.empty

    st.markdown(
        f'<span class="pill">Símbolos en cartera: {n_syms}</span>'
        f'<span class="pill">MV estimado: {_fmt_m(mv_end_t)}</span>'
        f'<span class="pill">Precios: {"✓ OK" if has_price else "⚠ sin datos"}</span>',
        unsafe_allow_html=True)

    # ── Resumen de capital ───────────────────────────────────────────────────
    # MV del primer y último mes disponible en twr_total
    total_rows = twr_total[twr_total["group"].eq("Total")].sort_values("month")
    mv_primera = total_rows["mv_end"].dropna().iloc[0]  if not total_rows.empty else None
    mv_ultima  = total_rows["mv_end"].dropna().iloc[-1] if not total_rows.empty else None
    cap = _capital_summary(dfa, mv_primera, mv_ultima)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Resumen de capital")

    def _kpi_neu(label, value):
        """KPI sin color de delta — para valores de stock/flujo."""
        return (f'<div class="kpi neu">'
                f'<div class="kpi-lbl">{label}</div>'
                f'<div class="kpi-val">{value}</div>'
                f'<div class="kpi-dlt neu">USD</div>'
                f'</div>')

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(_kpi_neu("Posición inicial",  _fmt_m(cap["pos_ini"])),  unsafe_allow_html=True)
    with c2: st.markdown(_kpi_neu("Posición final",    _fmt_m(cap["pos_fin"])),  unsafe_allow_html=True)
    with c3: st.markdown(_kpi_neu("Ingresos",          _fmt_m(cap["ingresos"])), unsafe_allow_html=True)
    with c4: st.markdown(_kpi_neu("Egresos",           _fmt_m(cap["egresos"])),  unsafe_allow_html=True)
    with c5: st.markdown(_kpi_neu("Fondeo neto",       _fmt_m(cap["fondeo"])),   unsafe_allow_html=True)
    with c6:
        res = cap["resultado"]
        st.markdown(
            _kpi_html("Resultado $", _fmt_m(res), delta=res,
                      delta_label=("▲ ganancia" if res and res > 0 else ("▼ pérdida" if res and res < 0 else "—"))),
            unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # ── KPIs del mes seleccionado ─────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader(f"Rendimientos al {selected_month}")

    def _get_r(df, grp, col):
        row = df[(df["group"].eq(grp)) & (df["month"].eq(selected_month))]
        if row.empty: return None
        v = row.iloc[0][col]
        return None if pd.isna(v) else float(v)

    r_m     = _get_r(twr_total, "Total", "r_monthly_pct")
    r_3m    = _get_r(twr_total, "Total", "r_3m")
    r_ytd   = _get_r(twr_total, "Total", "r_ytd")
    r_12m   = _get_r(twr_total, "Total", "r_12m")
    r_si    = _get_r(twr_total, "Total", "r_inception")
    mv_end  = _get_r(twr_total, "Total", "mv_end")
    mv_ini  = _get_r(twr_total, "Total", "mv_start")

    k = st.columns(6)
    kpis = [
        ("Rend. mensual",    _fmt_p(r_m),   r_m),
        ("3 meses",          _fmt_p(r_3m),  r_3m),
        ("YTD",              _fmt_p(r_ytd), r_ytd),
        ("12 meses",         _fmt_p(r_12m), r_12m),
        ("Desde inception",  _fmt_p(r_si),  r_si),
        ("MV cierre (base)", _fmt_m(mv_end), (mv_end - mv_ini) if (mv_end and mv_ini) else None),
    ]
    for col, (label, val, delta) in zip(k, kpis):
        with col:
            st.markdown(_kpi_html(label, val, delta), unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # ── tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "Cartera total",
        "Por asset class",
        "Por símbolo",
        "Tabla de períodos",
        "Posiciones & precios",
    ])

    # ─ Tab 1: Cartera total ──────────────────────────────────────────────────
    with t1:
        _chart_total_twr(twr_total)
        st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
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
        st.caption("Modified Dietz: R = (MV_fin − MV_ini − CF) / (MV_ini + 0.5×CF) · TWR = ∏(1+Rᵢ)−1")

    # ─ Tab 2: Por asset class ────────────────────────────────────────────────
    with t2:
        sym_palette = {k: ASSET_COLORS.get(k, MUTED) for k in twr_asset["group"].unique()}
        _chart_by_group(twr_asset, "TWR acumulado por asset class", sym_palette)
        st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
        view_a = twr_asset.rename(columns={
            "group":"Asset Class","month":"Mes",
            "mv_end":"MV cierre","r_monthly_pct":"Rend. mensual %",
            "r_cum_twr":"TWR acum %","r_3m":"3M %","r_12m":"12M %",
            "r_ytd":"YTD %","r_inception":"Inception %"})
        _show_df(
            view_a[["Mes","Asset Class","MV cierre","Rend. mensual %",
                    "TWR acum %","3M %","12M %","YTD %","Inception %"]].sort_values(["Mes","Asset Class"]),
            height=420,
            money=["MV cierre"],
            pct=["Rend. mensual %","TWR acum %","3M %","12M %","YTD %","Inception %"])

    # ─ Tab 3: Por símbolo ───────────────────────────────────────────────────
    with t3:
        # Filtro de símbolo
        sym_opts = ["Todos"] + sorted(twr_sym["group"].unique().tolist())
        sel_sym  = st.selectbox("Símbolo", sym_opts, key="sym_filter")

        sym_twr = twr_sym if sel_sym == "Todos" else twr_sym[twr_sym["group"].eq(sel_sym)]

        if sel_sym == "Todos":
            _chart_symbol_heatmap(sym_twr)
        else:
            # Para un solo símbolo, graficamos línea de TWR acumulado
            _chart_by_group(sym_twr, f"TWR acumulado — {sel_sym}",
                            {sel_sym: ACCENT})

        st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
        view_s = sym_twr.rename(columns={
            "group":"Símbolo","month":"Mes","mv_end":"MV cierre",
            "r_monthly_pct":"Rend. mensual %","r_cum_twr":"TWR acum %",
            "r_3m":"3M %","r_12m":"12M %","r_ytd":"YTD %","r_inception":"Inception %"})
        _show_df(
            view_s[["Mes","Símbolo","MV cierre","Rend. mensual %",
                    "TWR acum %","3M %","12M %","YTD %","Inception %"]].sort_values(["Mes","Símbolo"]),
            height=440,
            money=["MV cierre"],
            pct=["Rend. mensual %","TWR acum %","3M %","12M %","YTD %","Inception %"])

    # ─ Tab 4: Tabla resumen de períodos ─────────────────────────────────────
    with t4:
        st.markdown(f'<div class="subtle">Retornos acumulados al cierre de <b>{selected_month}</b> — todos los niveles</div>', unsafe_allow_html=True)

        st.markdown("**Cartera total**")
        s_total = _summary_periods(twr_total, selected_month)
        if not s_total.empty:
            _show_df(s_total, height=100, money=["MV cierre"],
                     pct=["3 meses","12 meses","YTD","Desde inception"])

        st.markdown("**Por asset class**")
        s_asset = _summary_periods(twr_asset, selected_month)
        if not s_asset.empty:
            _show_df(s_asset, height=200, money=["MV cierre"],
                     pct=["3 meses","12 meses","YTD","Desde inception"])

        st.markdown("**Por símbolo**")
        s_sym = _summary_periods(twr_sym, selected_month)
        if not s_sym.empty:
            _show_df(s_sym, height=460, money=["MV cierre"],
                     pct=["3 meses","12 meses","YTD","Desde inception"])

    # ─ Tab 5: Posiciones & precios ──────────────────────────────────────────
    with t5:
        if positions.empty:
            st.info("Sin posiciones calculadas.")
        else:
            pos_mes = positions[positions["month"].eq(selected_month)].copy()

            # Agregar precio y MV
            price_rows = []
            for _, row in pos_mes.iterrows():
                sym  = row["symbol_key"]
                me   = pd.Period(selected_month,"M").to_timestamp("M")
                pr   = np.nan
                if not prices_df.empty and sym in prices_df.columns:
                    sub = prices_df.loc[prices_df.index <= me, sym].dropna()
                    if not sub.empty: pr = float(sub.iloc[-1])
                price_rows.append({"Símbolo": sym, "Asset Class": row["asset_bucket"],
                                   "Qty cierre": row["closing_qty"],
                                   "Precio cierre": pr,
                                   "MV (base)": row["closing_qty"]*pr if not np.isnan(pr) else np.nan})
            df_prices = pd.DataFrame(price_rows).sort_values("MV (base)", ascending=False, na_position="last")
            st.markdown(f'<div class="subtle">Posición al cierre de <b>{selected_month}</b></div>', unsafe_allow_html=True)
            _show_df(df_prices, height=440,
                     money=["Precio cierre","MV (base)"],
                     qty=["Qty cierre"])
            if missing_sym:
                st.caption(f"⚠ Sin precio Yahoo Finance: {', '.join(sorted(missing_sym))}")

    # ── descarga ─────────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Descarga")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "Descargar Excel completo (TWR + posiciones)",
            data=_to_excel(twr_total, twr_asset, twr_sym, positions),
            file_name=f"twr_{selected_month}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_d2:
        st.download_button(
            "Descargar tabla de períodos (CSV)",
            data=(_summary_periods(twr_total, selected_month)
                  .to_csv(index=False).encode("utf-8-sig")) if not twr_total.empty else b"",
            file_name=f"twr_periodos_{selected_month}.csv",
            mime="text/csv",
            use_container_width=True)
