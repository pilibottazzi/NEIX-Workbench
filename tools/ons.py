# tools/ons.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize  # ✅ faltaba

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
    """
    Retorna TIR anual en % (ej: 12.34).
    """
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

    req = {"ticker_original", "Fecha", "Cupon"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    df["ticker_original"] = df["ticker_original"].astype(str).str.strip().str.upper()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["ticker_original", "Fecha", "Cupon"]).sort_values(
        ["ticker_original", "Fecha"]
    )
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("ticker_original", sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


# ======================================================
# 3) Precios desde IOL
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


def detect_price_scale(df_cf_one: pd.DataFrame, precio_iol: float) -> str:
    """
    Control simple de escala:
    - Si cupones son VN100 (>0.2 aprox) y el precio viene <5, probablemente está mal escalado.
    """
    if df_cf_one.empty or not np.isfinite(precio_iol):
        return "unknown"

    median_cupon = float(pd.to_numeric(df_cf_one["Cupon"], errors="coerce").dropna().median())
    if median_cupon > 0.2 and precio_iol < 5:
        return "price_divided_wrong"
    return "ok"


# ======================================================
# 4) Métricas (TIR / Duration / Modified Duration)
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
    r = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(r):
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
        pv = cupon / (1 + r / 100.0) ** tiempo
        denom.append(pv)
        numer.append(tiempo * pv)

    return round(float(np.sum(numer) / np.sum(denom)), 2) if np.sum(denom) != 0 else np.nan


def modified_duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
    dur = duration(cashflow, precio, plazo_dias=plazo_dias)
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return round(dur / (1 + ytm / 100.0), 2)


# ======================================================
# 5) UI
# ======================================================
def _neix_css():
    st.markdown(
        """
    <style>
      .neix-header{
        display:flex; align-items:flex-end; justify-content:space-between; gap:16px;
        margin-bottom:10px;
      }
      .neix-title{ font-size:28px; font-weight:800; letter-spacing:0.06em; color:#111827; }
      .neix-sub{ margin-top:2px; color:rgba(17,24,39,.62); }
      .hr{ height:1px; background:rgba(17,24,39,0.08); margin:14px 0; }
      .pill{
        display:inline-block; padding:6px 10px; border-radius:999px;
        border:1px solid rgba(17,24,39,0.10);
        background: rgba(17,24,39,0.03);
        font-size:12px; color:#111827;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render(back_to_home=None):
    _neix_css()

    st.markdown(
        """
    <div class="neix-header">
      <div>
        <div class="neix-title">NEIX · ONs</div>
        <div class="neix-sub">Rendimientos y métricas (USD): TIR · Duration · Modified Duration · con precios de IOL.</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Controles (solo T0 / T1)
    c1, c2, c3 = st.columns([0.30, 0.25, 0.45])
    with c1:
        plazo = st.selectbox("Plazo", [0, 1], index=0, format_func=lambda x: f"T{x}")
    with c2:
        traer_precios = st.button("🔄 Actualizar precios IOL")
    with c3:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Cargar cashflows (solo ticker_original)
    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
        cashflows = build_cashflow_dict(df_cf)
    except Exception as e:
        st.error(str(e))
        st.info("Solución: subí el archivo al repo y respetá columnas: ticker_original, Fecha, Cupon.")
        return

    tickers_all = sorted(cashflows.keys())
    tickers_sel = st.multiselect("Tickers", tickers_all, default=tickers_all)

    # Precios IOL con cache en session_state
    if traer_precios or "ons_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            try:
                st.session_state["ons_iol_prices"] = fetch_iol_on_prices()
            except Exception as e:
                st.error(f"No pude leer IOL: {e}")
                st.session_state["ons_iol_prices"] = None

    prices = st.session_state.get("ons_iol_prices")

    if st.button("▶️ Calcular", type="primary"):
        if not tickers_sel:
            st.warning("Elegí al menos 1 ticker.")
            return
        if prices is None:
            st.warning("No hay precios cargados.")
            return

        settlement = _settlement(plazo)
        rows, alerts = [], []

        for t in tickers_sel:
            cf = cashflows[t]
            cf_future = _future_cashflows(cf, settlement)

            # Precio IOL
            if t in prices.index:
                px = float(prices.loc[t, "UltimoOperado"])
                vol = float(prices.loc[t, "MontoOperado"])
            else:
                px, vol = np.nan, np.nan
                alerts.append((t, "No hay precio IOL para este ticker."))

            # Control escala
            scale_flag = detect_price_scale(cf, px)
            if scale_flag == "price_divided_wrong":
                alerts.append((t, "Escala rara: parece que el precio está /100. No dividas IOL por 100."))

            if cf_future.empty:
                alerts.append((t, "Sin flujos futuros (cashflow vencido o incompleto)."))

            rows.append(
                {
                    "Ticker": t,
                    "Vencimiento": cf_future["Fecha"].max() if not cf_future.empty else pd.NaT,
                    "Precio": px,
                    "Volumen": vol,
                    "TIR": tir(cf, px, plazo_dias=plazo),
                    "Duration": duration(cf, px, plazo_dias=plazo),
                    "MD": modified_duration(cf, px, plazo_dias=plazo),
                }
            )

        out = pd.DataFrame(rows)
        out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
        out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

        # Alerts
        if alerts:
            with st.expander("⚠️ Alertas / chequeos", expanded=False):
                for t, msg in alerts:
                    st.write(f"- **{t}**: {msg}")

        # Tabla comercial renombrada
        show = out.copy()
        show["Vencimiento"] = show["Vencimiento"].dt.strftime("%d/%m/%Y")
        st.markdown("### ON´s")
        st.dataframe(show, use_container_width=True)


