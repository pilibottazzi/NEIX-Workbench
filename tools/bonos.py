from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

# =========================
# Config
# =========================
CASHFLOW_PATH = os.path.join("data", "cashflows_completos.xlsx")
PRICE_SUFFIX = "D"

# =========================
# Utils precios (CRÍTICO)
# =========================
def parse_ar_number(x) -> float:
    """
    Convierte formato AR:
    89.190,00 -> 89190.0
    6323      -> 6323.0
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"-", "nan", "none"}:
        return np.nan
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def usd_fix_if_needed(ticker: str, raw_last: str, value: float) -> float:
    """
    IOL publica tickers D (USD) en centavos:
    '6097' -> 60.97
    '6323' -> 63.23
    """
    if not np.isfinite(value):
        return value

    t = (ticker or "").upper()
    raw = (raw_last or "").strip()

    if not t.endswith("D"):
        return value

    # si ya tiene separador decimal, no tocar
    if "," in raw or "." in raw:
        return value

    # precios USD típicos vienen *100
    return value / 100.0


# =========================
# IRR / NPV
# =========================
def xnpv(rate: float, cashflows: list[tuple[dt.datetime, float]]) -> float:
    chron = sorted(cashflows, key=lambda x: x[0])
    t0 = chron[0][0]
    if rate <= -0.999999:
        return np.nan

    out = 0.0
    for t, cf in chron:
        years = (t - t0).days / 365.0
        out += cf / (1.0 + rate) ** years
    return out


def xirr(cashflows: list[tuple[dt.datetime, float]], guess: float = 0.10) -> float:
    try:
        r = optimize.newton(lambda rr: xnpv(rr, cashflows), guess, maxiter=200)
        return float(r) * 100.0
    except Exception:
        return np.nan


# =========================
# Cashflows helpers
# =========================
def _settlement(plazo_dias: int) -> dt.datetime:
    base = pd.Timestamp.today().normalize().to_pydatetime()
    return base + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])
    return df[df["date"] > settlement].sort_values("date")


# =========================
# Normalizaciones
# =========================
def normalize_law(x: str) -> str:
    s = (x or "").strip().upper().replace(".", "").replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if s in {"ARG", "AR", "LOCAL", "LEY LOCAL", "ARGENTINA"}:
        return "ARG"
    if s in {"NY", "NYC", "NEW YORK", "LEY NY"}:
        return "NY"
    return "NA"


def law_cell_label(norm: str) -> str:
    if norm == "ARG":
        return "ARG (Ley local)"
    if norm == "NY":
        return "NY (Ley NY)"
    return "Sin ley"


def normalize_text(x: str) -> str:
    s = (x or "").strip().upper().replace("_", " ").replace("-", " ")
    return " ".join(s.split()) if s else "NA"


# =========================
# Load cashflows
# =========================
def load_cashflows_bonos(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    req = {"date", "species", "law", "issuer", "description", "flujo_total"}
    if not req.issubset(df.columns):
        raise ValueError(f"Faltan columnas requeridas: {req}")

    df["species"] = df["species"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")

    df["law_norm"] = df["law"].apply(normalize_law)
    df["issuer_norm"] = df["issuer"].apply(normalize_text)
    df["desc_norm"] = df["description"].apply(normalize_text)

    return df.dropna(subset=["species", "date", "flujo_total"])


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        k: g[["date", "flujo_total"]].sort_values("date")
        for k, g in df.groupby("species")
    }


def build_species_meta(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("species")
        .agg(
            law_norm=("law_norm", "first"),
            issuer_norm=("issuer_norm", "first"),
            desc_norm=("desc_norm", "first"),
            vencimiento=("date", "max"),
        )
        .reset_index()
        .set_index("species")
    )


# =========================
# Precios IOL (FIX USD)
# =========================
def fetch_iol_bonos_prices() -> pd.DataFrame:
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    bonos = pd.read_html(url, flavor="html5lib")[0]

    df = pd.DataFrame()
    df["Ticker"] = bonos["Símbolo"].astype(str).str.upper().str.strip()
    df["RawPrecio"] = bonos["Último Operado"].astype(str)
    df["Precio"] = bonos["Último Operado"].apply(parse_ar_number)
    df["Precio"] = [
        usd_fix_if_needed(t, r, p)
        for t, r, p in zip(df["Ticker"], df["RawPrecio"], df["Precio"])
    ]

    if "Monto Operado" in bonos.columns:
        df["Volumen"] = bonos["Monto Operado"].apply(parse_ar_number).fillna(0)
    else:
        df["Volumen"] = 0

    return (
        df.dropna(subset=["Precio"])
        .set_index("Ticker")
        .sort_values("Volumen", ascending=False)
        .loc[lambda x: ~x.index.duplicated(keep="first")]
        .drop(columns=["RawPrecio"])
    )


# =========================
# Métricas
# =========================
def tir(cashflow: pd.DataFrame, precio: float, plazo: int) -> float:
    settlement = _settlement(plazo)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty or precio <= 0:
        return np.nan

    flows = [(settlement, -precio)] + [
        (r["date"].to_pydatetime(), r["flujo_total"]) for _, r in cf.iterrows()
    ]
    v = xirr(flows)
    return round(v, 2) if np.isfinite(v) else np.nan


def duration(cashflow: pd.DataFrame, precio: float, plazo: int) -> float:
    ytm = tir(cashflow, precio, plazo)
    if not np.isfinite(ytm):
        return np.nan

    settlement = _settlement(plazo)
    cf = _future_cashflows(cashflow, settlement)

    pv, t_pv = 0.0, 0.0
    for _, r in cf.iterrows():
        t = (r["date"].to_pydatetime() - settlement).days / 365
        d = r["flujo_total"] / (1 + ytm / 100) ** t
        pv += d
        t_pv += t * d

    return round(t_pv / pv, 2) if pv else np.nan


def modified_duration(cashflow, precio, plazo):
    dur = duration(cashflow, precio, plazo)
    ytm = tir(cashflow, precio, plazo)
    return round(dur / (1 + ytm / 100), 2) if np.isfinite(dur) else np.nan


# =========================
# Render
# =========================
def render(back_to_home=None):
    st.title("NEIX · Bonos USD")

    df_cf = load_cashflows_bonos(CASHFLOW_PATH)
    prices = fetch_iol_bonos_prices()

    plazo = st.selectbox("Plazo liquidación", [1, 0], format_func=lambda x: f"T{x}")
    calcular = st.button("Calcular", type="primary")

    # filtros
    st.subheader("Filtros")
    law_sel = st.multiselect("Ley", sorted(df_cf["law_norm"].unique()), default=list(df_cf["law_norm"].unique()))
    issuer_sel = st.multiselect("Issuer", sorted(df_cf["issuer_norm"].unique()), default=list(df_cf["issuer_norm"].unique()))
    ticker_sel = st.multiselect("Ticker", sorted(df_cf["species"].unique()), default=list(df_cf["species"].unique()))

    cols_all = ["Ticker", "Ley", "Issuer", "Descripción", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]
    cols_pick = st.multiselect("Columnas a mostrar", cols_all, default=cols_all)

    if not calcular:
        return

    df_use = df_cf[
        df_cf["law_norm"].isin(law_sel)
        & df_cf["issuer_norm"].isin(issuer_sel)
        & df_cf["species"].isin(ticker_sel)
    ]

    cashflows = build_cashflow_dict(df_use)
    meta = build_species_meta(df_use)

    rows = []
    for sp, m in meta.iterrows():
        px_ticker = f"{sp}{PRICE_SUFFIX}"
        if px_ticker not in prices.index:
            continue

        px = prices.loc[px_ticker, "Precio"]
        cf = cashflows.get(sp)
        if cf is None:
            continue

        rows.append({
            "Ticker": sp,
            "Ley": law_cell_label(m["law_norm"]),
            "Issuer": m["issuer_norm"],
            "Descripción": m["desc_norm"],
            "Precio": px,
            "TIR (%)": tir(cf, px, plazo),
            "MD": modified_duration(cf, px, plazo),
            "Duration": duration(cf, px, plazo),
            "Vencimiento": m["vencimiento"].date(),
            "Volumen": prices.loc[px_ticker, "Volumen"],
        })

    out = pd.DataFrame(rows)
    st.dataframe(out[cols_pick], use_container_width=True)
