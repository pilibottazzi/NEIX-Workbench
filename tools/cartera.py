# tools/cartera.py
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

LIQ_PLAZO_DIAS = 1  # ✅ T1 fijo
PRICE_SUFFIX = "D"

# ✅ TIR fija como pediste (sin slider)
TIR_MIN = -15.0
TIR_MAX = 20.0

# =========================
# Excepciones ticker pesos -> ticker USD (MEP)
# (cuando NO es simplemente "ticker + D")
# =========================
PESOS_TO_USD_OVERRIDES: dict[str, str] = {
    "BPOB7": "BPB7D",
    "BPOC7": "BPC7D",
    "BPOD7": "BPD7D",

    "BPY26": "BPY6D",

    "AL30": "AL30D",
    "AL35": "AL35D",
    "AE38": "AE38D",
    "AL41": "AL41D",

    "GD30": "GD30D",
    "GD35": "GD35D",
    "GD38": "GD38D",
    "GD41": "GD41D",

    # extras que querés contemplar (si aparecen en cashflow en pesos con esa sigla)
    "BPA7": "BPA7D",
    "BPB7": "BPB7D",
    "BPC7": "BPC7D",
    "BPA8": "BPA8D",
    "BPB8": "BPB8D",
}


# =========================
# Utils número AR + fix USD D
# =========================
def parse_ar_number(x) -> float:
    """
    Convierte:
      89.190,00 -> 89190.00
      22.733.580,97 -> 22733580.97
      6323 -> 6323.0
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
    Fix para tickers USD (terminan en D).
    A veces IOL devuelve '6097' (sin separadores) para D y en realidad es 60.97.
    Regla:
      - termina en D
      - raw NO tiene '.' ni ','
      => dividir por 100
    """
    if not np.isfinite(value):
        return value

    t = (ticker or "").strip().upper()
    raw = (raw_last or "").strip()

    if not t.endswith("D"):
        return value
    if ("," in raw) or ("." in raw):
        return value

    return value / 100.0


def resolve_usd_ticker(species: str) -> str:
    """
    Resuelve ticker USD (MEP) a buscar en IOL:
    - si viene ya con D => lo deja
    - si está en overrides => usa ese
    - si no => default agrega 'D'
    """
    sp = str(species).strip().upper()
    if sp.endswith("D"):
        return sp
    if sp in PESOS_TO_USD_OVERRIDES:
        return PESOS_TO_USD_OVERRIDES[sp]
    return f"{sp}{PRICE_SUFFIX}"


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
# Helpers cashflows
# =========================
def _settlement(plazo_dias: int) -> dt.datetime:
    base = pd.Timestamp.today().normalize().to_pydatetime()
    return base + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])
    df = df[df["date"] > settlement].sort_values("date")
    return df


# =========================
# Normalizaciones
# =========================
def normalize_law(x: str) -> str:
    s = (x or "").strip().upper()
    s = s.replace(".", "").replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if s in {"ARG", "AR", "LOCAL", "LEY LOCAL", "ARGENTINA"}:
        return "ARG"
    if s in {"NYC", "NY", "NEW YORK", "NEWYORK", "LEY NY", "LEY NEW YORK", "N Y", "N Y C"}:
        return "NY"
    if s in {"", "NA", "NONE", "NAN"}:
        return "NA"
    return s


def law_cell_label(norm: str) -> str:
    if norm == "ARG":
        return "ARG (Ley local)"
    if norm == "NY":
        return "NY (Ley NY)"
    if norm == "NA":
        return "Sin ley"
    return norm


def normalize_issuer(x: str) -> str:
    s = (x or "").strip().upper()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s if s else "NA"


def normalize_desc(x: str) -> str:
    s = (x or "").strip().upper()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s if s else "NA"


# =========================
# Load cashflows
# =========================
def load_cashflows(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_completos.xlsx)."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    req = {"date", "species", "law", "issuer", "description", "flujo_total"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)} (requeridas: {sorted(req)})")

    df["species"] = df["species"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")

    df["law_norm"] = df["law"].apply(normalize_law)
    df["issuer_norm"] = df["issuer"].apply(normalize_issuer)
    df["desc_norm"] = df["description"].apply(normalize_desc)

    df = df.dropna(subset=["species", "date", "flujo_total"]).sort_values(["species", "date"])
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("species", sort=False):
        out[str(k)] = g[["date", "flujo_total"]].copy().sort_values("date")
    return out


def build_species_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.groupby("species")
        .agg(
            law_norm=("law_norm", lambda s: s.value_counts().index[0]),
            issuer_norm=("issuer_norm", lambda s: s.value_counts().index[0]),
            desc_norm=("desc_norm", lambda s: s.value_counts().index[0]),
            vencimiento=("date", "max"),
        )
        .reset_index()
    )
    return meta


# =========================
# Precios IOL (bonos/ONs en una misma tabla "bonos/todos")
# =========================
def fetch_iol_prices() -> pd.DataFrame:
    """
    Index: Ticker (IOL)
    Columns: Precio, Volumen
    - parser fallback: lxml -> html5lib
    - fix /100 para tickers D si vienen enteros
    """
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"

    tables = None
    try:
        tables = pd.read_html(url, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(url, flavor="html5lib")
        except Exception:
            tables = pd.read_html(url)

    if not tables:
        return pd.DataFrame()

    t = tables[0]
    cols = set(t.columns.astype(str))
    if "Símbolo" not in cols or "Último Operado" not in cols:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["Ticker"] = t["Símbolo"].astype(str).str.strip().str.upper()

    df["RawPrecio"] = t["Último Operado"].astype(str).str.strip()
    df["Precio"] = t["Último Operado"].apply(parse_ar_number)
    df["Precio"] = [
        usd_fix_if_needed(tk, raw, val)
        for tk, raw, val in zip(df["Ticker"], df["RawPrecio"], df["Precio"])
    ]

    if "Monto Operado" in cols:
        df["Volumen"] = t["Monto Operado"].apply(parse_ar_number).fillna(0)
    else:
        df["Volumen"] = 0

    df = df.dropna(subset=["Precio"]).set_index("Ticker")
    df = df.sort_values("Volumen", ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    return df[["Precio", "Volumen"]].copy()


# =========================
# Métricas por instrumento
# =========================
def tir(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    if not np.isfinite(precio) or precio <= 0:
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty:
        return np.nan

    flujos = [(settlement, -float(precio))]
    for _, r in cf.iterrows():
        flujos.append((r["date"].to_pydatetime(), float(r["flujo_total"])))

    v = xirr(flujos, guess=0.10)
    return v if np.isfinite(v) else np.nan


def duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(ytm):
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty:
        return np.nan

    denom, numer = 0.0, 0.0
    for _, row in cf.iterrows():
        t = row["date"].to_pydatetime()
        tiempo = (t - settlement).days / 365.0
        monto = float(row["flujo_total"])
        pv = monto / (1 + ytm / 100.0) ** tiempo
        denom += pv
        numer += tiempo * pv

    if denom == 0:
        return np.nan
    return numer / denom


def modified_duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    dur = duration(cashflow, precio, plazo_dias=plazo_dias)
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return dur / (1 + ytm / 100.0)


# =========================
# UI style
# =========================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .top-title{ font-size: 26px; font-weight: 850; letter-spacing: .04em; color:#111827; margin-bottom: 2px;}
  .top-sub{ color: rgba(17,24,39,.60); font-size: 13px; margin-top: 0px; }
  .section-title{ font-size: 18px; font-weight: 800; color:#111827; margin: 14px 0 8px; }
  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  label { margin-bottom: .18rem !important; }

  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 700; }
  .stSelectbox div[data-baseweb="select"]{ border-radius: 14px; }
  .stMultiSelect div[data-baseweb="select"]{ border-radius: 14px; }

  /* Chips gris clarito */
  div[data-baseweb="tag"]{
    background: rgba(17,24,39,.06) !important;
    color:#111827 !important;
    border: 1px solid rgba(17,24,39,.10) !important;
    border-radius: 999px !important;
    font-weight: 650 !important;
  }
  div[data-baseweb="tag"] svg{ color: rgba(17,24,39,.55) !important; }

  div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# Universe: elegibles (precio + TIR disponible + TIR en rango)
# =========================
def compute_universe(df_cf: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        usd_tk = resolve_usd_ticker(species)
        if usd_tk not in prices.index:
            continue

        px = float(prices.loc[usd_tk, "Precio"])
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        t = tir(cf, px, plazo_dias=LIQ_PLAZO_DIAS)
        if not np.isfinite(t):
            continue
        if not (TIR_MIN <= t <= TIR_MAX):
            continue

        rows.append(
            {
                "Ticker": species,
                "Ticker precio": usd_tk,
                "Precio (USD, VN100)": px,
                "TIR (%)": t,
                "MD": modified_duration(cf, px, plazo_dias=LIQ_PLAZO_DIAS),
                "Duration": duration(cf, px, plazo_dias=LIQ_PLAZO_DIAS),
                "Vencimiento": meta.loc[species, "vencimiento"],
                "Ley": law_cell_label(meta.loc[species, "law_norm"]),
                "Issuer": meta.loc[species, "issuer_norm"],
                "Descripción": meta.loc[species, "desc_norm"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Vencimiento", "Ticker"]).reset_index(drop=True)
    return out


# =========================
# Portfolio: tabla + flujos
# =========================
def build_portfolio_table(universe: pd.DataFrame, selected: list[str], capital_usd: float, weights_pct: dict[str, float]) -> pd.DataFrame:
    u = universe.set_index("Ticker")
    rows = []
    for tk in selected:
        if tk not in u.index:
            continue
        pct = float(weights_pct.get(tk, 0.0))
        usd = float(capital_usd) * pct / 100.0

        price = float(u.loc[tk, "Precio (USD, VN100)"])
        # VN estimada = USD / (Precio/100) = USD*100/Precio
        vn = usd / (price / 100.0) if price > 0 else np.nan

        rows.append(
            {
                "Ticker": tk,
                "%": pct,
                "USD": usd,
                "Precio (USD, VN100)": price,
                "VN estimada": vn,
                "TIR (%)": float(u.loc[tk, "TIR (%)"]),
                "MD": float(u.loc[tk, "MD"]),
                "Duration": float(u.loc[tk, "Duration"]),
                "Vencimiento": u.loc[tk, "Vencimiento"],
                "Ley": u.loc[tk, "Ley"],
                "Issuer": u.loc[tk, "Issuer"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce").dt.date
    return out


def build_portfolio_cashflow_calendar(df_cf: pd.DataFrame, portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - tabla por ticker x mes (USD), con Total
      - serie total por mes (USD)
    """
    if portfolio.empty:
        return pd.DataFrame(), pd.DataFrame()

    cf_dict = build_cashflow_dict(df_cf)
    settlement = _settlement(LIQ_PLAZO_DIAS)

    frames = []
    for _, r in portfolio.iterrows():
        tk = str(r["Ticker"]).upper()
        vn = float(r["VN estimada"]) if np.isfinite(r["VN estimada"]) else np.nan
        if tk not in cf_dict or not np.isfinite(vn):
            continue

        cf = _future_cashflows(cf_dict[tk], settlement)
        if cf.empty:
            continue

        scale = vn / 100.0  # cashflows son por VN100
        x = cf.copy()
        x["usd"] = x["flujo_total"].astype(float) * float(scale)
        x["Ticker"] = tk
        frames.append(x[["Ticker", "date", "usd"]])

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    allcf = pd.concat(frames, ignore_index=True)
    allcf["date"] = pd.to_datetime(allcf["date"], errors="coerce")
    allcf = allcf.dropna(subset=["date", "usd"])

    allcf["Mes"] = allcf["date"].dt.to_period("M").dt.to_timestamp()

    pivot = (
        allcf.pivot_table(index="Ticker", columns="Mes", values="usd", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )
    pivot["Total"] = pivot.sum(axis=1)

    total_by_month = pivot.drop(columns=["Total"]).sum(axis=0).to_frame("USD").reset_index().rename(columns={"Mes": "Mes"})
    total_by_month = total_by_month.rename(columns={"index": "Mes"})
    total_by_month.columns = ["Mes", "USD"]
    total_by_month["Mes"] = pd.to_datetime(total_by_month["Mes"], errors="coerce")

    # ordenar columnas (meses)
    month_cols = [c for c in pivot.columns if c != "Total"]
    month_cols = sorted(month_cols)
    pivot = pivot[month_cols + ["Total"]]

    return pivot, total_by_month


# =========================
# Charts (matplotlib, no plotly)
# =========================
def pie_allocation(portfolio: pd.DataFrame):
    fig, ax = plt.subplots()
    data = portfolio[["Ticker", "%"]].copy()
    data = data[data["%"] > 0]
    ax.pie(data["%"], labels=data["Ticker"], autopct=lambda p: f"{p:.0f}%")
    ax.set_title("Asignación por activo")
    st.pyplot(fig, use_container_width=True)


def bar_monthly_cashflows(total_by_month: pd.DataFrame):
    if total_by_month.empty:
        return
    fig, ax = plt.subplots()
    x = total_by_month["Mes"]
    y = total_by_month["USD"]
    ax.bar(x, y)
    ax.set_title("Flujos estimados por mes (USD)")
    ax.set_xlabel("Mes")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig, use_container_width=True)


# =========================
# Render
# =========================
def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    st.markdown('<div class="top-title">NEIX · Cartera Comercial (USD MEP)</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="top-sub">Activos elegibles: con precio IOL + TIR disponible. TIR fija en [{TIR_MIN:.1f}, {TIR_MAX:.1f}].</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Load cashflows
    try:
        df_cf = load_cashflows(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Precios cache
    cbtn1, cbtn2 = st.columns([0.7, 0.3], vertical_alignment="center")
    with cbtn2:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_refresh_prices")

    if refresh or "cartera_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            st.session_state["cartera_prices"] = fetch_iol_prices()

    prices = st.session_state.get("cartera_prices")
    if prices is None or prices.empty:
        st.warning("No pude leer precios desde IOL (tabla vacía o cambió el formato).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Universe cache (depende de precios)
    if refresh or "cartera_universe" not in st.session_state:
        with st.spinner("Armando universo elegible (precio + TIR)..."):
            st.session_state["cartera_universe"] = compute_universe(df_cf, prices)

    universe = st.session_state.get("cartera_universe")
    if universe is None or universe.empty:
        st.warning("No encontré activos elegibles con TIR en rango y precio disponible.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # Selección + Capital + Asignación (sin sliders)
    # =========================
    st.markdown('<div class="section-title">Selección de activos</div>', unsafe_allow_html=True)

    elig = universe["Ticker"].tolist()
    selected = st.multiselect("Activos (bonos + ONs)", options=elig, default=elig[:6], key="cartera_sel")

    # Capital ENTRE selección y asignación (como pediste)
    st.markdown('<div class="section-title">Capital</div>', unsafe_allow_html=True)
    capital = st.number_input("Capital (USD)", min_value=0.0, value=100000.0, step=1000.0, format="%.0f", key="cartera_capital")

    st.markdown('<div class="section-title">Asignación por activo</div>', unsafe_allow_html=True)
    st.caption("Editá la columna % (ideal: que sume 100%).")

    # Persistencia de % por ticker en session_state
    if "cartera_weights" not in st.session_state:
        st.session_state["cartera_weights"] = {}

    weights: dict[str, float] = st.session_state["cartera_weights"]

    # armar df editable
    if selected:
        # inicializar nuevos con pesos iguales
        missing = [tk for tk in selected if tk not in weights]
        if missing:
            eq = 100.0 / len(selected) if len(selected) else 0.0
            for tk in missing:
                weights[tk] = eq

        # remover los que ya no están
        for tk in list(weights.keys()):
            if tk not in selected:
                weights.pop(tk, None)

        alloc_df = pd.DataFrame({"Ticker": selected, "%": [float(weights.get(tk, 0.0)) for tk in selected]})
    else:
        alloc_df = pd.DataFrame({"Ticker": [], "%": []})

    edited = st.data_editor(
        alloc_df,
        hide_index=True,
        use_container_width=True,
        disabled=["Ticker"],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "%": st.column_config.NumberColumn("%", min_value=0.0, max_value=100.0, format="%.0f"),
        },
        key="cartera_alloc_editor",
    )

    # actualizar weights desde editor
    try:
        for _, r in edited.iterrows():
            weights[str(r["Ticker"]).upper()] = float(r["%"]) if pd.notna(r["%"]) else 0.0
    except Exception:
        pass

    s = float(np.nansum(list(weights.values()))) if weights else 0.0
    if selected:
        if abs(s - 100.0) > 0.5:
            st.warning(f"La suma de % da {s:.0f}% (ideal 100%). Igual calculo con esa suma.")
        else:
            st.success(f"La suma de % da {s:.0f}%.")

    st.divider()

    # Botón calcular (T1 fijo, sin selector)
    calcular = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_calc")
    if not calcular:
        st.info("Elegí activos, asigná % y tocá **Calcular cartera**.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # Cálculo cartera
    # =========================
    port = build_portfolio_table(universe, selected, float(capital), weights)
    if port.empty:
        st.warning("No pude armar cartera con la selección actual.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ponderaciones por USD real (si suma % != 100)
    w_usd = port["USD"].astype(float)
    w_total = float(w_usd.sum()) if float(w_usd.sum()) != 0 else np.nan
    w = (w_usd / w_total) if np.isfinite(w_total) else np.nan

    tir_total = float((port["TIR (%)"].astype(float) * w).sum()) if np.isfinite(w_total) else np.nan
    md_total = float((port["MD"].astype(float) * w).sum()) if np.isfinite(w_total) else np.nan
    dur_total = float((port["Duration"].astype(float) * w).sum()) if np.isfinite(w_total) else np.nan

    # sin decimales
    port_show = port.copy()
    for col in ["%", "USD", "Precio (USD, VN100)", "VN estimada", "TIR (%)", "MD", "Duration"]:
        port_show[col] = pd.to_numeric(port_show[col], errors="coerce")

    # =========================
    # Resumen (estilo ejecutivo)
    # =========================
    st.markdown('<div class="section-title">Resumen</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capital", f"$ {float(capital):,.0f}".replace(",", "."))
    m2.metric("TIR total (pond.)", f"{tir_total:.0f}%" if np.isfinite(tir_total) else "—")
    m3.metric("MD total (pond.)", f"{md_total:.0f}" if np.isfinite(md_total) else "—")
    m4.metric("Duration total (pond.)", f"{dur_total:.0f}" if np.isfinite(dur_total) else "—")

    st.markdown('<div class="section-title">Cartera</div>', unsafe_allow_html=True)

    st.dataframe(
        port_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "%": st.column_config.NumberColumn("%", format="%.0f"),
            "USD": st.column_config.NumberColumn("USD", format="$ %.0f"),
            "Precio (USD, VN100)": st.column_config.NumberColumn("Precio (USD, VN100)", format="%.0f"),
            "VN estimada": st.column_config.NumberColumn("VN estimada", format="%.0f"),
            "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.0f"),
            "MD": st.column_config.NumberColumn("MD", format="%.0f"),
            "Duration": st.column_config.NumberColumn("Duration", format="%.0f"),
            "Vencimiento": st.column_config.TextColumn("Vencimiento"),
            "Ley": st.column_config.TextColumn("Ley"),
            "Issuer": st.column_config.TextColumn("Issuer"),
        },
    )

    # =========================
    # Flujo de fondos (calendario)
    # =========================
    st.markdown('<div class="section-title">Flujo de fondos</div>', unsafe_allow_html=True)

    cal, total_by_month = build_portfolio_cashflow_calendar(df_cf, port)
    if cal.empty:
        st.info("No hay flujos futuros para la selección (o faltan cashflows).")
    else:
        # formateo sin decimales y $ en todo
        cal_show = cal.copy()
        for c in cal_show.columns:
            cal_show[c] = pd.to_numeric(cal_show[c], errors="coerce").fillna(0.0)

        # renombrar columnas a "MMM-YYYY"
        cols = list(cal_show.columns)
        new_cols = []
        for c in cols:
            if c == "Total":
                new_cols.append("Total")
            else:
                # Timestamp
                try:
                    ts = pd.Timestamp(c)
                    new_cols.append(ts.strftime("%b-%Y"))
                except Exception:
                    new_cols.append(str(c))
        cal_show.columns = new_cols

        # total al final
        st.dataframe(
            cal_show,
            use_container_width=True,
            column_config={col: st.column_config.NumberColumn(col, format="$ %.0f") for col in cal_show.columns},
        )
