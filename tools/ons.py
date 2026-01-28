# tools/ons.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

CASHFLOW_PATH = os.path.join("data", "cashflows_ON.xlsx")


# ======================================================
# 1) XNPV / XIRR
# ======================================================
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


# ======================================================
# 2) Cashflows
# ======================================================
def _settlement(plazo_dias: int) -> dt.datetime:
    return dt.datetime.today() + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Cupon"])
    df = df[df["Fecha"] > settlement].sort_values("Fecha")
    return df


def load_cashflows_from_repo(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_ON.xlsx)."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    req = {"ticker_original", "root_key", "Fecha", "Cupon"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    df["ticker_original"] = df["ticker_original"].astype(str).str.strip().str.upper()
    df["root_key"] = df["root_key"].astype(str).str.strip().str.upper()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")

    df = df.dropna(subset=["ticker_original", "root_key", "Fecha", "Cupon"]).sort_values(
        ["ticker_original", "Fecha"]
    )
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("ticker_original", sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


def build_root_map(df: pd.DataFrame) -> dict[str, str]:
    mp = (
        df.groupby("ticker_original")["root_key"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return {str(k).upper(): str(v).upper() for k, v in mp.items()}


# ======================================================
# 3) Precios IOL (USD por root_key: rootD o rootC)
# ======================================================
def fetch_iol_on_prices() -> pd.DataFrame:
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"
    on = pd.read_html(url)[0]

    def to_float_ar(s):
        if pd.isna(s):
            return np.nan
        s = str(s).replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return np.nan

    df = pd.DataFrame(
        {
            "Ticker": on["Símbolo"].astype(str).str.strip().str.upper(),
            "UltimoOperado": on["Último Operado"].apply(to_float_ar),
            "MontoOperado": on.get("Monto Operado", pd.Series([0] * len(on))).apply(to_float_ar).fillna(0),
        }
    ).dropna(subset=["UltimoOperado"])

    return df.set_index("Ticker")


def pick_usd_price_by_root(prices: pd.DataFrame, root_key: str) -> tuple[float, float, str]:
    rk = str(root_key).strip().upper()

    candidates: list[tuple[str, str]] = []
    if rk.endswith(("D", "C")):
        candidates.append((rk, rk[-1]))
    else:
        candidates.extend([(f"{rk}D", "D"), (f"{rk}C", "C")])

    for sym, src in candidates:
        if sym in prices.index:
            px = float(prices.loc[sym, "UltimoOperado"])
            vol = float(prices.loc[sym, "MontoOperado"])
            return px, vol, src

    return np.nan, np.nan, ""


# ======================================================
# 4) Métricas
# ======================================================
def tir(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
    if not np.isfinite(precio) or precio <= 0:
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty:
        return np.nan

    flujos = [(settlement, -float(precio))]
    for _, r in cf.iterrows():
        flujos.append((r["Fecha"].to_pydatetime(), float(r["Cupon"])))

    return round(xirr(flujos, guess=0.10), 2)


def duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(ytm):
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty:
        return np.nan

    denom, numer = [], []
    for _, row in cf.iterrows():
        t = row["Fecha"].to_pydatetime()
        tiempo = (t - settlement).days / 365.0
        cupon = float(row["Cupon"])
        pv = cupon / (1 + ytm / 100.0) ** tiempo
        denom.append(pv)
        numer.append(tiempo * pv)

    if np.sum(denom) == 0:
        return np.nan
    return round(float(np.sum(numer) / np.sum(denom)), 2)


def modified_duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
    dur = duration(cashflow, precio, plazo_dias=plazo_dias)
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return round(dur / (1 + ytm / 100.0), 2)


# ======================================================
# 5) UI (minimal)
# ======================================================
def _ui_css():
    st.markdown(
        """
    <style>
      .wrap{ max-width: 1200px; margin: 0 auto; }
      .head{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:10px; }
      .title{ font-size:22px; font-weight:800; letter-spacing:.04em; color:#111827; }
      .sub{ color:rgba(17,24,39,.60); font-size:13px; margin-top:2px; }
      .card{
        border:1px solid rgba(17,24,39,0.08);
        border-radius:14px;
        padding:14px 14px;
        background:#fff;
        box-shadow: 0 8px 26px rgba(17,24,39,0.05);
      }
      .muted{ color:rgba(17,24,39,.55); font-size:12px; }
      /* Reduce padding in multiselect tags a bit */
      div[data-baseweb="tag"]{
        border-radius:999px !important;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render(back_to_home=None):
    _ui_css()

    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Header
    left, right = st.columns([0.75, 0.25])
    with left:
        st.markdown(
            """
            <div class="head">
              <div>
                <div class="title">ONs · Rendimientos</div>
                <div class="sub">Cashflow en USD. Precio desde IOL por <b>root_key</b>: rootD (MEP) o rootC (Cable). Si no hay precio USD, no se muestra.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if back_to_home is not None:
            st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Cargar cashflows
    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
        cashflows = build_cashflow_dict(df_cf)
        root_map = build_root_map(df_cf)
    except Exception as e:
        st.error(str(e))
        st.info("Solución: columnas requeridas: ticker_original, root_key, Fecha, Cupon.")
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    tickers_all = sorted(cashflows.keys())

    # Controls (solo plazo + ticker)
    c1, c2, c3, c4 = st.columns([0.18, 0.52, 0.17, 0.13])
    with c1:
        plazo = st.selectbox("Plazo", [0, 1], index=0, format_func=lambda x: f"T{x}")
    with c2:
        tickers_sel = st.multiselect("Ticker", tickers_all, default=tickers_all)
    with c3:
        traer_precios = st.button("Actualizar IOL")
    with c4:
        calcular = st.button("Calcular", type="primary")

    st.caption(f"Fuente cashflows: `{CASHFLOW_PATH}`")

    # Precios cacheados
    if traer_precios or "ons_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            try:
                st.session_state["ons_iol_prices"] = fetch_iol_on_prices()
            except Exception as e:
                st.error(f"No pude leer IOL: {e}")
                st.session_state["ons_iol_prices"] = None

    prices = st.session_state.get("ons_iol_prices")

    if calcular:
        if not tickers_sel:
            st.warning("Elegí al menos 1 ticker.")
            st.markdown("</div></div>", unsafe_allow_html=True)
            return
        if prices is None:
            st.warning("No hay precios cargados. Probá 'Actualizar IOL'.")
            st.markdown("</div></div>", unsafe_allow_html=True)
            return

        settlement = _settlement(plazo)

        rows = []
        for t in tickers_sel:
            cf = cashflows[t]
            cf_future = _future_cashflows(cf, settlement)

            rk = root_map.get(t, "")
            px, vol, src = (np.nan, np.nan, "")
            if rk:
                px, vol, src = pick_usd_price_by_root(prices, rk)

            rows.append(
                {
                    "Ticker": t,
                    "USD": src,  # D o C
                    "Precio USD": px,
                    "TIR (%)": tir(cf, px, plazo_dias=plazo),
                    "MD": modified_duration(cf, px, plazo_dias=plazo),
                    "Duration": duration(cf, px, plazo_dias=plazo),
                    "Vencimiento": cf_future["Fecha"].max() if not cf_future.empty else pd.NaT,
                    "Volumen": vol,
                }
            )

        out = pd.DataFrame(rows)
        out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")

        # ✅ SOLO con precio USD (si no hay rootD/rootC => afuera)
        out = out[
            out["Precio USD"].notna()
            & np.isfinite(out["Precio USD"])
            & (out["Precio USD"] > 0)
        ].copy()

        out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

        # Tabla
        st.markdown("<div class='muted' style='margin-top:8px;'>Resultados</div>", unsafe_allow_html=True)

        show = out.copy()
        show["Vencimiento"] = show["Vencimiento"].dt.date

        st.dataframe(
            show[["Ticker", "USD", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "USD": st.column_config.TextColumn("USD (D/C)"),
                "Precio USD": st.column_config.NumberColumn("Precio USD", format="%.2f"),
                "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
                "MD": st.column_config.NumberColumn("MD", format="%.2f"),
                "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
                "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
                "Volumen": st.column_config.NumberColumn("Volumen", format="%.0f"),
            },
        )

    st.markdown("</div></div>", unsafe_allow_html=True)
