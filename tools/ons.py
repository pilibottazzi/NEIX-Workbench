import datetime as dt
from scipy import optimize
import pandas as pd
import numpy as np

def xnpv(rate, cashflows):
    chron_order = sorted(cashflows, key=lambda x: x[0])
    t0 = chron_order[0][0]
    # Evitar (1+rate)<=0 (explota con tasas muy negativas)
    if rate <= -0.999999:
        return np.nan
    return sum(cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in chron_order)

def xirr(cashflows, guess=0.1):
    # Newton puede fallar: dejamos que el caller capture excepción
    return optimize.newton(lambda r: xnpv(r, cashflows), guess) * 100

def tir(cashflow: pd.DataFrame, precio: float, plazo: int = 1) -> float:
    """
    cashflow: DF con [fecha, flujo]
    precio: precio en unidad monetaria (ej: 0.85 si es 85% en dólares)
    plazo: días que corrés la fecha de compra (CI=0, 24hs=1, 48hs=2)
    """
    hoy = dt.datetime.today()
    t_compra = hoy + dt.timedelta(days=plazo)

    flujo_total = [(t_compra, -float(precio))]

    # Asegurar tipos
    cf = cashflow.copy()
    cf.iloc[:, 0] = pd.to_datetime(cf.iloc[:, 0], errors="coerce", dayfirst=True)
    cf.iloc[:, 1] = pd.to_numeric(cf.iloc[:, 1], errors="coerce")

    for i in range(len(cf)):
        fecha = cf.iloc[i, 0]
        monto = cf.iloc[i, 1]
        if pd.isna(fecha) or pd.isna(monto):
            continue
        fecha_dt = fecha.to_pydatetime()
        if fecha_dt > t_compra:
            flujo_total.append((fecha_dt, float(monto)))

    if len(flujo_total) < 2:
        return np.nan

    return round(xirr(flujo_total, guess=0.1), 2)

def duration(cashflow: pd.DataFrame, precio: float, plazo: int = 2) -> float:
    r = tir(cashflow, precio, plazo=plazo)
    if pd.isna(r):
        return np.nan

    hoy = dt.datetime.today()
    t_compra = hoy + dt.timedelta(days=plazo)

    denom = []
    numer = []

    cf = cashflow.copy()
    cf.iloc[:, 0] = pd.to_datetime(cf.iloc[:, 0], errors="coerce", dayfirst=True)
    cf.iloc[:, 1] = pd.to_numeric(cf.iloc[:, 1], errors="coerce")

    for i in range(len(cf)):
        fecha = cf.iloc[i, 0]
        cupon = cf.iloc[i, 1]
        if pd.isna(fecha) or pd.isna(cupon):
            continue
        fecha_dt = fecha.to_pydatetime()
        if fecha_dt > t_compra:
            tiempo = (fecha_dt - hoy).days / 365.0
            pv = float(cupon) / (1 + r / 100) ** tiempo
            denom.append(pv)
            numer.append(tiempo * pv)

    if sum(denom) == 0:
        return np.nan

    return round(sum(numer) / sum(denom), 2)

def modified_duration(cashflow: pd.DataFrame, precio: float, plazo: int = 2) -> float:
    dur = duration(cashflow, precio, plazo=plazo)
    ytm = tir(cashflow, precio, plazo=plazo)
    if pd.isna(dur) or pd.isna(ytm):
        return np.nan
    return round(dur / (1 + ytm / 100), 2)


# ======================================================
# 2) Utilidades de mercado + lectura de Excel de CFs
# ======================================================
def parse_ar_number(series: pd.Series) -> pd.Series:
    """
    Convierte números estilo AR: '1.234.567,89' -> 1234567.89
    """
    s = series.astype(str)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s = s.str.replace("nan", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def get_precios_on_iol() -> pd.DataFrame:
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"
    on = pd.read_html(url)[0]

    # Normalizar columnas esperadas
    # (si IOL cambia nombres, este bloque es el primero a tocar)
    precios = pd.DataFrame({
        "Ticker": on["Símbolo"],
        "UltimoOperado": parse_ar_number(on["Último Operado"]),
    })

    if "Monto Operado" in on.columns:
        precios["MontoOperado"] = parse_ar_number(on["Monto Operado"])
    else:
        precios["MontoOperado"] = 0.0

    precios = precios.dropna(subset=["Ticker"])
    precios["Ticker"] = precios["Ticker"].astype(str).str.strip()
    precios = precios.set_index("Ticker")

    return precios[["UltimoOperado", "MontoOperado"]]

def load_cashflows_excel(path_excel: str, sheet_data: str = "Data_ON"):
    """
    Lee Data_ON y todas las hojas de cashflows en una sola pasada.
    Devuelve:
      - data_on (DF)
      - cashflows dict[ticker_pesos] = DF(cashflow)
    """
    xls = pd.ExcelFile(path_excel)

    data_on = pd.read_excel(xls, sheet_name=sheet_data)
    data_on.columns = data_on.columns.astype(str).str.strip()

    # Normalizar columnas tickers
    for col in ["ticker_dolares", "ticker_pesos"]:
        if col in data_on.columns:
            data_on[col] = data_on[col].astype(str).str.strip()
        else:
            data_on[col] = ""

    # Cargar cashflows por cada ticker_pesos (si existe la hoja)
    cashflows = {}
    tickers_pesos = (
        data_on["ticker_pesos"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    for tkr in tickers_pesos:
        if tkr in xls.sheet_names:
            cf = pd.read_excel(xls, sheet_name=tkr)
            if cf.shape[1] >= 2:
                cashflows[tkr] = cf.iloc[:, :2].copy()
        # si no está la hoja, lo dejamos fuera y luego quedará NaN

    return data_on, cashflows


# ======================================================
# 3) Pipeline principal
# ======================================================
def build_tabla_ons(
    excel_cashflows: str = "cashflows_ON.xlsx",
    sheet_data: str = "Data_ON",
    plazo_tir: int = 2,
    plazo_md: int = 2,
) -> pd.DataFrame:
    # 1) precios mercado
    precios = get_precios_on_iol()

    # 2) data base + cashflows
    data_on, cashflows = load_cashflows_excel(excel_cashflows, sheet_data=sheet_data)

    # 3) Intersección de tickers dólar con los que están en IOL
    comunes = sorted(list(set(precios.index) & set(data_on["ticker_dolares"].dropna().astype(str))))
    data_on = data_on.loc[data_on["ticker_dolares"].isin(comunes)].copy()

    # 4) Precio USD (IOL suele venir como % del VN => /100)
    data_on["Precio_dolares"] = (precios.loc[comunes, "UltimoOperado"].values / 100.0)
    data_on["Volumen"] = precios.loc[comunes, "MontoOperado"].values

    # 5) Precio pesos (si existe el ticker y está en precios)
    data_on["Precio_pesos"] = np.nan
    for idx, row in data_on.iterrows():
        tkr_p = str(row.get("ticker_pesos", "")).strip()
        if tkr_p and tkr_p in precios.index:
            data_on.at[idx, "Precio_pesos"] = precios.at[tkr_p, "UltimoOperado"]

    # Fix puntual tuyo
    if "ticker_pesos" in data_on.columns:
        data_on.loc[data_on["ticker_pesos"] == "CAC2O", "Precio_pesos"] = 8

    # 6) Limpieza de amortización
    if "Amortizacion" in data_on.columns:
        data_on["Amortizacion"] = data_on["Amortizacion"].fillna("No Bullet")

    # 7) Set index a ticker_pesos (como tu versión)
    if "ticker_pesos" in data_on.columns:
        data_on = data_on.set_index("ticker_pesos", drop=True)

    # 8) Calcular métricas por ticker_pesos usando Precio_dolares
    resultados = []
    for tkr_pesos in data_on.index:
        precio = data_on.loc[tkr_pesos, "Precio_dolares"]

        cf = cashflows.get(tkr_pesos)
        if cf is None or pd.isna(precio) or precio == 0:
            resultados.append((tkr_pesos, np.nan, np.nan, np.nan))
            continue

        try:
            ytm = tir(cf, precio, plazo=plazo_tir)
            dur = duration(cf, precio, plazo=plazo_tir)
            md  = modified_duration(cf, precio, plazo=plazo_md)
            resultados.append((tkr_pesos, md, dur, ytm))
        except Exception as e:
            print(f"[WARN] {tkr_pesos}: {e}")
            resultados.append((tkr_pesos, np.nan, np.nan, np.nan))

    tir_df = pd.DataFrame(resultados, columns=["ticker_pesos", "MD", "Duration", "TIR"]).set_index("ticker_pesos")

    out = pd.concat([data_on, tir_df], axis=1)
    return out


def main():
    df = build_tabla_ons(
        excel_cashflows="cashflows_ON.xlsx",
        sheet_data="Data_ON",
        plazo_tir=2,   # 48hs
        plazo_md=2,
    )
    print(df)
    # opcional: guardar salida
    df.to_excel("ONs_tir_duration.xlsx")


if __name__ == "__main__":
    main()


