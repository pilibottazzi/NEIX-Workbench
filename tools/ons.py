# tools/ons.py
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# XNPV / XIRR (sin scipy)
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


def xirr(cashflows: list[tuple[dt.datetime, float]]) -> float:
    """
    XIRR robusto por bisección.
    Devuelve tasa en %.
    """
    vals = [cf for _, cf in cashflows]
    if not (any(v < 0 for v in vals) and any(v > 0 for v in vals)):
        return np.nan

    def f(r):
        return xnpv(r, cashflows)

    a, b = -0.90, 5.0
    fa, fb = f(a), f(b)

    if not (np.isfinite(fa) and np.isfinite(fb)):
        return np.nan

    tries = 0
    while fa * fb > 0 and tries < 25:
        b *= 1.5
        fb = f(b)
        tries += 1
        if not np.isfinite(fb):
            return np.nan

    if fa * fb > 0:
        return np.nan

    for _ in range(300):
        m = (a + b) / 2
        fm = f(m)
        if not np.isfinite(fm):
            return np.nan
        if abs(fm) < 1e-10:
            return m * 100
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return ((a + b) / 2) * 100


# =========================
# Helpers de fecha / PV
# =========================
def _settlement(plazo_dias: int) -> dt.datetime:
    hoy = dt.datetime.now()
    return hoy + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(cashflow_df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = cashflow_df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Cupon"])
    df = df.sort_values("Fecha")
    # IMPORTANTÍSIMO: excluir todo lo vencido o que liquida hoy/antes
    return df[df["Fecha"] > settlement].copy()


def tir(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    settlement = _settlement(plazo_dias)
    df_fut = _future_cashflows(cashflow_df, settlement)

    if df_fut.empty:
        return np.nan

    flujo_total = [(settlement, -float(precio))]
    for _, r in df_fut.iterrows():
        flujo_total.append((r["Fecha"].to_pydatetime(), float(r["Cupon"])))

    out = xirr(flujo_total)
    return round(out, 2) if np.isfinite(out) else np.nan


def duration(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    settlement = _settlement(plazo_dias)
    df_fut = _future_cashflows(cashflow_df, settlement)
    if df_fut.empty:
        return np.nan

    r = tir(cashflow_df, precio, plazo_dias=plazo_dias)
    if not np.isfinite(r):
        return np.nan

    denom = 0.0
    numer = 0.0

    # OJO: tiempo desde SETTLEMENT (consistente con la compra)
    for _, row in df_fut.iterrows():
        t = (row["Fecha"].to_pydatetime() - settlement).days / 365.0
        pv = float(row["Cupon"]) / (1 + r / 100) ** t
        denom += pv
        numer += t * pv

    if denom == 0:
        return np.nan
    return round(numer / denom, 4)


def modified_duration(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    dur = duration(cashflow_df, precio, plazo_dias=plazo_dias)
    r = tir(cashflow_df, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(r):
        return np.nan
    return round(dur / (1 + r / 100), 4)


# =========================
# Lectura cashflows (long)
# =========================
def load_cashflows_long_from_file(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = df.columns.astype(str).str.strip()

    required = {"ticker_original", "root_key", "Fecha", "Cupon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en cashflows: {sorted(missing)}")

    df["ticker_original"] = df["ticker_original"].astype(str).str.strip()
    df["root_key"] = df["root_key"].astype(str).str.strip()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")

    df = df.dropna(subset=["ticker_original", "Fecha", "Cupon"])
    df = df.sort_values(["ticker_original", "Fecha"]).reset_index(drop=True)
    return df


def build_cashflow_dict(df_cf: pd.DataFrame, key_col: str) -> dict[str, pd.DataFrame]:
    out = {}
    for k, g in df_cf.groupby(key_col, sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


# =========================
# Precios desde IOL
# =========================
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

    df = pd.DataFrame({
        "Ticker": on["Símbolo"].astype(str).str.strip(),
        "UltimoOperado": on["Último Operado"].apply(to_float_ar),
        "MontoOperado": on.get("Monto Operado", pd.Series([0] * len(on))).apply(to_float_ar).fillna(0),
    }).dropna(subset=["UltimoOperado"])

    return df.set_index("Ticker")


# =========================
# Render Workbench
# =========================
def render(back_to_home=None):
    st.markdown("## ONs")
    st.caption("Cálculo de TIR / Duration / Modified Duration con cashflows (Excel largo).")

    cf_file = st.file_uploader("📎 Cashflows ON (xlsx)", type=["xlsx"], key="ons_cf")
    if cf_file is None:
        st.info("Subí el Excel para continuar.")
        return

    try:
        df_cf = load_cashflows_long_from_file(cf_file)
    except Exception as e:
        st.error(f"No pude leer cashflows: {e}")
        return

    # Elegís la clave que querés usar
    key_col = st.radio("Clave de agrupación", ["ticker_original", "root_key"], horizontal=True)
    cashflows = build_cashflow_dict(df_cf, key_col=key_col)

    tickers_all = sorted(cashflows.keys())
    tickers_sel = st.multiselect("Tickers", options=tickers_all, default=tickers_all)

    plazo = st.selectbox("Plazo de liquidación (días)", options=[0, 1, 2], index=2)

    st.markdown("### Precios (IOL)")
    col1, col2 = st.columns([1, 2])

    with col1:
        btn = st.button("🔄 Traer precios IOL")
    with col2:
        iol_dividir_100 = st.checkbox(
            "IOL: dividir precio por 100",
            value=False,
            help="Si tus cupones están por VN100, normalmente NO querés dividir. Si tu precio te queda 0.8, está mal la escala."
        )

    precios_df = st.session_state.get("ons_precios_iol")

    if btn:
        with st.spinner("Leyendo IOL..."):
            try:
                precios_df = fetch_iol_on_prices()
                st.session_state["ons_precios_iol"] = precios_df
                st.success(f"Precios cargados: {len(precios_df)} tickers.")
            except Exception as e:
                st.error(f"No pude leer IOL: {e}")
                precios_df = None

    if st.button("▶️ Calcular", type="primary"):
        settlement = _settlement(plazo)
        rows = []
        diag = []

        for t in tickers_sel:
            cf = cashflows[t].copy()

            # Diagnóstico de flujos futuros
            df_fut = _future_cashflows(cf, settlement)
            diag.append({
                "Ticker": t,
                "Futuros": len(df_fut),
                "Primer flujo": df_fut["Fecha"].min() if not df_fut.empty else pd.NaT,
                "Ultimo flujo": df_fut["Fecha"].max() if not df_fut.empty else pd.NaT,
                "Cupon promedio": float(df_fut["Cupon"].mean()) if not df_fut.empty else np.nan,
            })

            # Precio
            precio = np.nan
            vol = np.nan
            if precios_df is not None and t in precios_df.index:
                precio = float(precios_df.loc[t, "UltimoOperado"])
                if iol_dividir_100:
                    precio = precio / 100.0
                vol = float(precios_df.loc[t, "MontoOperado"])

            # Sanity check escala
            escala_warn = ""
            if np.isfinite(precio) and precio > 0 and not df_fut.empty:
                cupon_med = float(df_fut["Cupon"].median())
                if precio < 5 and cupon_med > 0.05:
                    escala_warn = "⚠️ Precio muy chico vs cupón (probable escala /100 incorrecta)."

            if (not np.isfinite(precio)) or precio <= 0 or df_fut.empty:
                rows.append({"Ticker": t, "Precio": precio, "TIR": np.nan, "Duration": np.nan, "MD": np.nan, "Volumen": vol, "Obs": escala_warn})
                continue

            tir_v = tir(cf, precio, plazo_dias=plazo)
            dur_v = duration(cf, precio, plazo_dias=plazo)
            md_v  = modified_duration(cf, precio, plazo_dias=plazo)

            rows.append({
                "Ticker": t,
                "Precio": precio,
                "TIR": tir_v,
                "Duration": dur_v,
                "MD": md_v,
                "Volumen": vol,
                "Obs": escala_warn,
            })

        out = pd.DataFrame(rows).set_index("Ticker").sort_index()
        diag_df = pd.DataFrame(diag).set_index("Ticker").sort_index()

        st.markdown("### Diagnóstico cashflows (control de vencidos)")
        st.dataframe(diag_df, use_container_width=True)

        st.markdown("### Resultado")
        st.dataframe(out, use_container_width=True)

        st.info("Tip: si la TIR te da delirante, mirá la columna 'Obs' y probá cambiar el check 'IOL: dividir precio por 100'.")
