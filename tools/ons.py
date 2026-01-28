# tools/ons.py
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

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
    XIRR robusto sin scipy:
    - Busca un rango [a,b] con cambio de signo
    - Aplica bisección
    Devuelve tasa en %.
    """
    # Chequeo básico: debe haber al menos un negativo y un positivo
    vals = [cf for _, cf in cashflows]
    if not (any(v < 0 for v in vals) and any(v > 0 for v in vals)):
        return np.nan

    def f(r):
        return xnpv(r, cashflows)

    # Buscar bracket
    a, b = -0.90, 5.0  # -90% a 500%
    fa, fb = f(a), f(b)

    # Si no hay cambio de signo, expandimos b
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

    # Bisección
    for _ in range(200):
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


def tir(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    hoy = dt.datetime.today()
    settlement = hoy + dt.timedelta(days=plazo_dias)

    flujo_total = [(settlement, -float(precio))]

    for _, r in cashflow_df.iterrows():
        fecha = pd.to_datetime(r["Fecha"], errors="coerce")
        cupon = pd.to_numeric(r["Cupon"], errors="coerce")
        if pd.isna(fecha) or pd.isna(cupon):
            continue
        fecha = fecha.to_pydatetime()
        if fecha > settlement:
            flujo_total.append((fecha, float(cupon)))

    out = xirr(flujo_total)
    return round(out, 2) if np.isfinite(out) else np.nan


def duration(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    r = tir(cashflow_df, precio, plazo_dias=plazo_dias)
    if not np.isfinite(r):
        return np.nan

    hoy = dt.datetime.today()
    settlement = hoy + dt.timedelta(days=plazo_dias)

    denom = 0.0
    numer = 0.0

    for _, row in cashflow_df.iterrows():
        fecha = pd.to_datetime(row["Fecha"], errors="coerce")
        cupon = pd.to_numeric(row["Cupon"], errors="coerce")
        if pd.isna(fecha) or pd.isna(cupon):
            continue
        fecha = fecha.to_pydatetime()
        if fecha <= settlement:
            continue

        t = (fecha - hoy).days / 365.0
        pv = float(cupon) / (1 + r / 100) ** t
        denom += pv
        numer += t * pv

    if denom == 0:
        return np.nan
    return round(numer / denom, 2)


def modified_duration(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    dur = duration(cashflow_df, precio, plazo_dias=plazo_dias)
    r = tir(cashflow_df, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(r):
        return np.nan
    return round(dur / (1 + r / 100), 2)


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


def build_cashflow_dict(df_cf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for k, g in df_cf.groupby("ticker_original", sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


# =========================
# Precios (opcional)
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
        "MontoOperado": on.get("Monto Operado", pd.Series([0]*len(on))).apply(to_float_ar).fillna(0),
    }).dropna(subset=["UltimoOperado"])

    return df.set_index("Ticker")


def load_prices_from_excel(file) -> pd.DataFrame:
    """
    Espera columnas:
      - Ticker
      - Precio   (en % par o ya en precio unitario; lo ajustamos con un toggle)
      - Volumen  (opcional)
    """
    df = pd.read_excel(file)
    df.columns = df.columns.astype(str).str.strip()
    if "Ticker" not in df.columns or "Precio" not in df.columns:
        raise ValueError("El Excel de precios debe tener columnas: 'Ticker' y 'Precio'.")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")
    if "Volumen" in df.columns:
        df["Volumen"] = pd.to_numeric(df["Volumen"], errors="coerce")
    else:
        df["Volumen"] = np.nan
    return df.set_index("Ticker")


# =========================
# Render Workbench
# =========================
def render(back_to_home=None):
    st.markdown("## ONs")
    st.caption("Cálculo de TIR / Duration / Modified Duration a partir de cashflows en Excel (formato largo).")

    cf_file = st.file_uploader("📎 Subí el Excel de cashflows (cashflows_ON.xlsx)", type=["xlsx"], key="ons_cf")
    if cf_file is None:
        st.info("Subí el Excel para continuar.")
        return

    try:
        df_cf = load_cashflows_long_from_file(cf_file)
        cashflows = build_cashflow_dict(df_cf)
    except Exception as e:
        st.error(f"No pude leer cashflows: {e}")
        return

    tickers_all = sorted(cashflows.keys())
    tickers_sel = st.multiselect("Tickers a calcular", options=tickers_all, default=tickers_all, key="ons_tickers")

    plazo = st.selectbox("Plazo de liquidación (días)", options=[0, 1, 2], index=2, key="ons_plazo")

    st.markdown("### Precios")
    modo_precio = st.radio("Fuente de precios", ["IOL (web)", "Excel"], index=0, horizontal=True, key="ons_modo_precio")

    precios_df = None
    precio_en_porcentaje = True

    if modo_precio == "IOL (web)":
        if st.button("🔄 Traer precios de IOL", key="ons_btn_iol"):
            with st.spinner("Leyendo precios desde IOL..."):
                try:
                    precios_df = fetch_iol_on_prices()
                except Exception as e:
                    st.error(f"No pude leer IOL: {e}")
                    precios_df = None
        # cache simple en session_state
        if "ons_precios_iol" in st.session_state and precios_df is None:
            precios_df = st.session_state["ons_precios_iol"]
        if precios_df is not None:
            st.session_state["ons_precios_iol"] = precios_df
            st.success(f"Precios IOL cargados: {len(precios_df)} tickers.")
        else:
            st.warning("Traé precios con el botón (o elegí Excel).")
    else:
        px_file = st.file_uploader("📎 Subí Excel de precios (Ticker, Precio, Volumen opcional)", type=["xlsx"], key="ons_px")
        precio_en_porcentaje = st.checkbox("Precio está en % (par) -> dividir por 100", value=True, key="ons_px_pct")
        if px_file is not None:
            try:
                precios_df = load_prices_from_excel(px_file)
                st.success(f"Precios Excel cargados: {len(precios_df)} tickers.")
            except Exception as e:
                st.error(str(e))
                precios_df = None

    if st.button("▶️ Calcular", type="primary", key="ons_calc"):
        if not tickers_sel:
            st.warning("Elegí al menos 1 ticker.")
            return

        # Armar precios por ticker
        rows = []
        for t in tickers_sel:
            cf = cashflows[t]

            precio = np.nan
            vol = np.nan

            if precios_df is not None and t in precios_df.index:
                if modo_precio == "IOL (web)":
                    # IOL viene en % (último operado)
                    precio = float(precios_df.loc[t, "UltimoOperado"]) / 100.0
                    vol = float(precios_df.loc[t, "MontoOperado"])
                else:
                    precio = float(precios_df.loc[t, "Precio"])
                    if precio_en_porcentaje:
                        precio = precio / 100.0
                    vol = float(precios_df.loc[t, "Volumen"]) if "Volumen" in precios_df.columns else np.nan

            if not np.isfinite(precio) or precio <= 0:
                rows.append({"Ticker": t, "Precio": precio, "TIR": np.nan, "Duration": np.nan, "MD": np.nan, "Volumen": vol})
                continue

            tir_v = tir(cf, precio, plazo_dias=plazo)
            dur_v = duration(cf, precio, plazo_dias=plazo)
            md_v  = modified_duration(cf, precio, plazo_dias=plazo)

            rows.append({"Ticker": t, "Precio": precio, "TIR": tir_v, "Duration": dur_v, "MD": md_v, "Volumen": vol})

        out = pd.DataFrame(rows).set_index("Ticker").sort_index()

        st.markdown("### Resultado")
        st.dataframe(out, use_container_width=True)

        # Descarga Excel
        xls_bytes = None
        try:
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                out.to_excel(writer, sheet_name="ONs")
            buf.seek(0)
            xls_bytes = buf.read()
        except Exception as e:
            st.warning(f"No pude preparar descarga Excel: {e}")

        if xls_bytes:
            st.download_button(
                "⬇️ Descargar ON_metricas.xlsx",
                data=xls_bytes,
                file_name="ON_metricas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="ons_dl",
            )
