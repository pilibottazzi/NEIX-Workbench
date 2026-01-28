import datetime as dt
import numpy as np
import pandas as pd
from scipy import optimize
import streamlit as st

def render(back_to_home=None):
    st.markdown("## ONs")

def xnpv(rate: float, cashflows: list[tuple[dt.datetime, float]]) -> float:
    chron = sorted(cashflows, key=lambda x: x[0])
    t0 = chron[0][0]
    # Evitar problemas cuando (1+rate)<=0
    if rate <= -0.999999:
        return np.nan
    return sum(cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in chron)


def xirr(cashflows: list[tuple[dt.datetime, float]], guess: float = 0.10) -> float:
    # Newton con multi-guess (más robusto)
    def f(r): 
        return xnpv(r, cashflows)

    for g in [guess, 0.05, 0.20, 0.50, -0.10]:
        try:
            r = optimize.newton(f, g, maxiter=200)
            if np.isfinite(r):
                return r * 100
        except Exception:
            continue
    return np.nan


def tir(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    """
    cashflow_df: columnas ['Fecha','Cupon'] (Fecha datetime, Cupon float)
    precio: precio sucio/limpio según estés usando, en misma unidad que cupones
    plazo_dias: 0=CI, 1=24hs, 2=48hs (como venías usando)
    """
    hoy = dt.datetime.today()
    settlement = hoy + dt.timedelta(days=plazo_dias)

    flujo_total = [(settlement, -float(precio))]

    for _, r in cashflow_df.iterrows():
        fecha = pd.to_datetime(r["Fecha"], errors="coerce")
        cupon = r["Cupon"]
        if pd.isna(fecha) or pd.isna(cupon):
            continue
        fecha = fecha.to_pydatetime()
        if fecha > settlement:
            flujo_total.append((fecha, float(cupon)))

    out = xirr(flujo_total, guess=0.10)
    return np.round(out, 2) if np.isfinite(out) else np.nan


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
        cupon = row["Cupon"]
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
    return np.round(numer / denom, 2)


def modified_duration(cashflow_df: pd.DataFrame, precio: float, plazo_dias: int = 2) -> float:
    dur = duration(cashflow_df, precio, plazo_dias=plazo_dias)
    r = tir(cashflow_df, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(r):
        return np.nan
    return np.round(dur / (1 + r / 100), 2)


# ======================================================
# 2) Leer cashflows "largos" desde Excel
# ======================================================
def load_cashflows_long(path_xlsx: str, sheet_name: str | None = None) -> pd.DataFrame:
    df = pd.read_excel(path_xlsx, sheet_name=sheet_name)
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


def build_cashflow_dict(df_cf: pd.DataFrame, key_col: str = "ticker_original") -> dict[str, pd.DataFrame]:
    """
    Devuelve dict: {ticker: df[['Fecha','Cupon']]}
    """
    out = {}
    for k, g in df_cf.groupby(key_col, sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


# ======================================================
# 3) Precios desde IOL (opcional)
# ======================================================
def fetch_iol_on_prices() -> pd.DataFrame:
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"
    on = pd.read_html(url)[0]

    # Normalización de números AR
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
    })

    df = df.dropna(subset=["UltimoOperado"])
    df = df.set_index("Ticker")
    return df


# ======================================================
# 4) Cálculo final para una lista de tickers
# ======================================================
def calcular_metricas_on(
    cashflows_long_path: str,
    tickers: list[str] | None = None,
    sheet_name: str | None = None,
    usar_iol: bool = True,
    precio_col: str = "Precio_dolares",
    plazo_tir: int = 2,
) -> pd.DataFrame:

    df_cf = load_cashflows_long(cashflows_long_path, sheet_name=sheet_name)
    cashflows = build_cashflow_dict(df_cf, key_col="ticker_original")

    # Universo de tickers
    all_tickers = list(cashflows.keys())
    if tickers is None:
        tickers = all_tickers
    else:
        tickers = [t for t in tickers if t in cashflows]

    # Traer precios
    if usar_iol:
        precios = fetch_iol_on_prices()
        # tu lógica: precio en dólares = ultimo/100
        # (si tu precio viene en % par)
        df_prices = pd.DataFrame({
            "Ticker": tickers,
            "Precio": [ (precios.loc[t, "UltimoOperado"] / 100) if t in precios.index else np.nan for t in tickers ],
            "Volumen": [ precios.loc[t, "MontoOperado"] if t in precios.index else np.nan for t in tickers ],
        }).set_index("Ticker")
    else:
        # Si no usás IOL, acá podrías leer un Excel de precios.
        df_prices = pd.DataFrame(index=tickers, data={"Precio": np.nan, "Volumen": np.nan})

    # Calcular métricas
    rows = []
    for t in tickers:
        cf = cashflows[t]
        precio = df_prices.loc[t, "Precio"]
        if not np.isfinite(precio) or precio <= 0:
            rows.append({"Ticker": t, "Precio": precio, "TIR": np.nan, "Duration": np.nan, "MD": np.nan})
            continue

        tir_v = tir(cf, precio, plazo_dias=plazo_tir)
        dur_v = duration(cf, precio, plazo_dias=plazo_tir)
        md_v  = modified_duration(cf, precio, plazo_dias=plazo_tir)

        rows.append({
            "Ticker": t,
            "Precio": precio,
            "TIR": tir_v,
            "Duration": dur_v,
            "MD": md_v,
            "Volumen": df_prices.loc[t, "Volumen"],
        })

    out = pd.DataFrame(rows).set_index("Ticker").sort_index()
    return out


if __name__ == "__main__":
    # Ejemplo:
    path_cf = "cashflows_ON.xlsx"  # <-- tu archivo long
    df_out = calcular_metricas_on(path_cf, usar_iol=True, plazo_tir=2)
    print(df_out.head(20))
    df_out.to_excel("ON_metricas.xlsx")
