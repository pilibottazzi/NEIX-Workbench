# tools/ons.py
from __future__ import annotations
import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st


CASHFLOW_PATH = os.path.join("data", "cashflows_ON.xlsx")

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


def xirr(cashflows,guess=0.1):
    return optimize.newton(lambda r: xnpv(r,cashflows),guess)*100 #Con esto te itera


def _settlement(plazo_dias: int) -> dt.datetime:
    return dt.datetime.today() + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Cupon"])
    df = df[df["Fecha"] > settlement].sort_values("Fecha")
    return df


def tir(cashflow, precio, plazo=1):
    flujo_total=[(datetime.datetime.today()+ datetime.timedelta(days=plazo) , -precio)] #para que arranque como que compras el bono hoy
    for i in range (len(cashflow)):
      if cashflow.iloc[i,0].to_pydatetime()>datetime.datetime.today()+ datetime.timedelta(days=plazo):
        flujo_total.append((cashflow.iloc[i,0].to_pydatetime(),cashflow.iloc[i,1])) #lista de cashflows

    return round(xirr(flujo_total,guess=0.1),2)


def duration(cashflow,precio,plazo=2):
  r=tir(cashflow,precio,plazo=plazo)
  denom=[]
  numer=[]
  for i in range (len(cashflow)):
    if cashflow.iloc[i,0].to_pydatetime()>datetime.datetime.today()+ datetime.timedelta(days=plazo):
      tiempo=(cashflow.iloc[i,0]-datetime.datetime.today()).days/365 #tiempo al cupon en años
      cupon=cashflow.iloc[i,1]
      denom.append(cupon/(1+r/100)**tiempo) #sum (C(1+r)^t)
      numer.append(tiempo*(cupon/(1+r/100)**tiempo))

  return round(sum(numer)/sum(denom),2)


def modified_duration(cashflow,precio,plazo=2):
  dur=duration(cashflow,precio,plazo)
  return round(dur/(1+tir(cashflow,precio,plazo)/100),2)


def load_cashflows_from_repo(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}. Subilo al repo (ej: data/curva_on.xlsx).")

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
    df = df.dropna(subset=["ticker_original", "Fecha", "Cupon"]).sort_values(["ticker_original", "Fecha"])
    return df


def build_cashflow_dict(df: pd.DataFrame, key_col: str) -> dict[str, pd.DataFrame]:
    out = {}
    for k, g in df.groupby(key_col, sort=False):
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
        except:
            return np.nan

    df = pd.DataFrame({
        "Ticker": on["Símbolo"].astype(str).str.strip().str.upper(),
        "UltimoOperado": on["Último Operado"].apply(to_float_ar),
        "MontoOperado": on.get("Monto Operado", pd.Series([0]*len(on))).apply(to_float_ar).fillna(0),
    }).dropna(subset=["UltimoOperado"])

    return df.set_index("Ticker")



def detect_price_scale(df_cf_one: pd.DataFrame, precio_iol: float) -> str:
    """
    Si cupones parecen VN100 (ej 1.386) el precio debería estar ~80-110, NO 0.80-1.10.
    Si dividís precio por 100 cuando no corresponde, te da TIR absurda.
    """
    if df_cf_one.empty or not np.isfinite(precio_iol):
        return "unknown"

    median_cupon = float(pd.to_numeric(df_cf_one["Cupon"], errors="coerce").dropna().median())
    # Heurística simple:
    # - cupon mediano > 0.2 suele ser VN100 (1.3, 6.2, etc)
    # - si precio_iol < 5 y cupon VN100 => estás /100 mal
    if median_cupon > 0.2 and precio_iol < 5:
        return "price_divided_wrong"
    return "ok"



def _neix_css():
    st.markdown("""
    <style>
      .neix-header{
        display:flex; align-items:flex-end; justify-content:space-between; gap:16px;
        margin-bottom:10px;
      }
      .neix-title{ font-size:28px; font-weight:800; letter-spacing:0.06em; color:#111827; }
      .neix-sub{ margin-top:2px; color:rgba(17,24,39,.62); }
      .card{
        border:1px solid rgba(17,24,39,0.08);
        border-radius:16px;
        padding:14px 16px;
        background:white;
        box-shadow:0 6px 20px rgba(17,24,39,0.06);
      }
      .kpi-label{ color:rgba(17,24,39,.55); font-size:12px; letter-spacing:.06em; text-transform:uppercase; }
      .kpi-val{ font-size:22px; font-weight:800; color:#111827; }
      .hr{ height:1px; background:rgba(17,24,39,0.08); margin:14px 0; }
      .pill{
        display:inline-block; padding:6px 10px; border-radius:999px;
        border:1px solid rgba(17,24,39,0.10);
        background: rgba(17,24,39,0.03);
        font-size:12px; color:#111827;
      }
    </style>
    """, unsafe_allow_html=True)


def render(back_to_home=None):
    _neix_css()

    st.markdown("""
    <div class="neix-header">
      <div>
        <div class="neix-title">NEIX · ONs</div>
        <div class="neix-sub">Rendimientos y métricas (USD): TIR · Duration · Modified Duration · con precios de IOL.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if back_to_home is not None:
        st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Controles
    c1, c2, c3, c4 = st.columns([0.30, 0.18, 0.22, 0.30])
    with c1:
        key_col = st.selectbox("Agrupar por", ["ticker_original", "root_key"], index=0)
    with c2:
        plazo = st.selectbox("Plazo", [0, 1, 2], index=2)
    with c3:
        traer_precios = st.button("🔄 Actualizar precios IOL")
    with c4:
        st.write("")
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Cargar cashflows
    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
        cashflows = build_cashflow_dict(df_cf, key_col=key_col)
    except Exception as e:
        st.error(str(e))
        st.info("Solución: subí el archivo al repo y respetá columnas: ticker_original, root_key, Fecha, Cupon.")
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

        rows = []
        alerts = []

        for t in tickers_sel:
            cf = cashflows[t]
            cf_future = _future_cashflows(cf, settlement)

            # Precio IOL
            if t in prices.index:
                px = float(prices.loc[t, "UltimoOperado"])     # IOL suele venir 80-110 (par)
                vol = float(prices.loc[t, "MontoOperado"])
            else:
                px, vol = np.nan, np.nan

            # Control escala
            scale_flag = detect_price_scale(cf, px)
            if scale_flag != "ok":
                alerts.append((t, "Escala de precio vs cupón: parece que el precio está /100. No dividas IOL por 100."))

            if cf_future.empty:
                alerts.append((t, "Sin flujos futuros (cashflow vencido o incompleto)."))

            tir_v = tir(cf, px, plazo_dias=plazo)
            dur_v = duration(cf, px, plazo_dias=plazo)
            md_v  = modified_duration(cf, px, plazo_dias=plazo)

            rows.append({
                "Vencimiento": cf_future["Fecha"].max() if not cf_future.empty else pd.NaT,
                "TIR": tir_v,
                "MD": md_v,
                "Ticker": t,
                "Precio": px,
                "Volumen": vol  })

        out = pd.DataFrame(rows)
        out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
        out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

        # Tabla comercial
        show = out.copy()
        show["Vencimiento"] = show["Vencimiento"].dt.strftime("%d/%m/%Y")
        st.markdown("### Tabla comercial")
        st.dataframe(show, use_container_width=True)



