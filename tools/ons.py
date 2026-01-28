# tools/ons.py
from __future__ import annotations
import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st


CASHFLOW_PATH = os.path.join("data", "cashflows_ON.xlsx")


# ======================================================
# 1) Cálculo XIRR sin SciPy (robusto)
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


def xirr(cashflows: list[tuple[dt.datetime, float]]) -> float:
    """
    XIRR en % (sin SciPy).
    - busca bracket con cambio de signo
    - bisección
    """
    vals = [cf for _, cf in cashflows]
    if not (any(v < 0 for v in vals) and any(v > 0 for v in vals)):
        return np.nan

    def f(r):
        return xnpv(r, cashflows)

    a, b = -0.90, 5.0  # -90% a 500%
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return np.nan

    tries = 0
    while fa * fb > 0 and tries < 30:
        b *= 1.5
        fb = f(b)
        tries += 1
        if not np.isfinite(fb):
            return np.nan

    if fa * fb > 0:
        return np.nan

    for _ in range(250):
        m = (a + b) / 2
        fm = f(m)
        if not np.isfinite(fm):
            return np.nan
        if abs(fm) < 1e-9:
            return m * 100
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return ((a + b) / 2) * 100


def _settlement(plazo_dias: int) -> dt.datetime:
    return dt.datetime.today() + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Cupon"])
    df = df[df["Fecha"] > settlement].sort_values("Fecha")
    return df


def tir(df_cf: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    settlement = _settlement(plazo_dias)
    df_fut = _future_cashflows(df_cf, settlement)

    if df_fut.empty or not np.isfinite(precio) or precio <= 0:
        return np.nan

    flows = [(settlement, -float(precio))]
    flows += [(d.to_pydatetime(), float(c)) for d, c in zip(df_fut["Fecha"], df_fut["Cupon"])]

    out = xirr(flows)
    return round(out, 2) if np.isfinite(out) else np.nan


def duration(df_cf: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    r = tir(df_cf, precio, plazo_dias=plazo_dias)
    if not np.isfinite(r):
        return np.nan

    hoy = dt.datetime.today()
    settlement = _settlement(plazo_dias)
    df_fut = _future_cashflows(df_cf, settlement)
    if df_fut.empty:
        return np.nan

    denom, numer = 0.0, 0.0
    for _, row in df_fut.iterrows():
        fecha = row["Fecha"].to_pydatetime()
        cupon = float(row["Cupon"])
        t = (fecha - hoy).days / 365.0
        pv = cupon / (1 + r / 100) ** t
        denom += pv
        numer += t * pv

    if denom == 0:
        return np.nan
    return round(numer / denom, 2)


def modified_duration(df_cf: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    dur = duration(df_cf, precio, plazo_dias=plazo_dias)
    r = tir(df_cf, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(r):
        return np.nan
    return round(dur / (1 + r / 100), 2)


# ======================================================
# 2) Cashflows desde el repo (curva_on)
#    Formato esperado: “largo”
#    columnas: ticker_original, root_key, Fecha, Cupon
# ======================================================
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


# ======================================================
# 4) Escala precio vs cupones (CLAVE para que TIR no “explote”)
# ======================================================
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


# ======================================================
# 5) UI (Front NEIX)
# ======================================================
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
                "Volumen": vol,
                "Flujos_futuros": int(len(cf_future)),
            })

        out = pd.DataFrame(rows)
        out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
        out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

        # KPIs
        ok_tir = out["TIR"].dropna()
        ok_md = out["MD"].dropna()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="card"><div class="kpi-label">ONs</div><div class="kpi-val">{len(out)}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card"><div class="kpi-label">TIR Prom.</div><div class="kpi-val">{ok_tir.mean():.2f}%</div></div>' if len(ok_tir) else
                        '<div class="card"><div class="kpi-label">TIR Prom.</div><div class="kpi-val">—</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="card"><div class="kpi-label">TIR Mediana</div><div class="kpi-val">{ok_tir.median():.2f}%</div></div>' if len(ok_tir) else
                        '<div class="card"><div class="kpi-label">TIR Mediana</div><div class="kpi-val">—</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="card"><div class="kpi-label">MD Prom.</div><div class="kpi-val">{ok_md.mean():.2f}</div></div>' if len(ok_md) else
                        '<div class="card"><div class="kpi-label">MD Prom.</div><div class="kpi-val">—</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Tabla comercial
        show = out.copy()
        show["Vencimiento"] = show["Vencimiento"].dt.strftime("%d/%m/%Y")
        st.markdown("### Tabla comercial")
        st.dataframe(show, use_container_width=True)

        # Export
        st.markdown("### Exportar")
        csv = show.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name=f"ONs_{dt.datetime.now():%Y%m%d}.csv", mime="text/csv")

        # Alertas
        if alerts:
            with st.expander("⚠️ Alertas / Controles"):
                for t, msg in alerts:
                    st.markdown(f"- **{t}**: {msg}")

        # Texto para enviar
        st.markdown("### Texto listo para enviar")
        lines = [f"📌 ONs Ley NY – Rendimientos USD ({dt.datetime.now():%d/%m/%Y})"]
        for _, r in out.dropna(subset=["TIR","MD"]).head(12).iterrows():
            vto = r["Vencimiento"].strftime("%d/%m/%Y") if pd.notna(r["Vencimiento"]) else "-"
            lines.append(f"• {r['Ticker']}: TIR {r['TIR']:.2f}% | MD {r['MD']:.2f} | Vto {vto}")
        st.code("\n".join(lines), language="text")

