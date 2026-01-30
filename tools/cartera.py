# tools/cartera_comercial.py
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

# Filtro fijo TIR (sin slider)
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
    # Si tu cashflow trae estos "sin D", quedan contemplados:
    "BPA7": "BPA7D",
    "BPB7": "BPB7D",
    "BPC7": "BPC7D",
    "BPA8": "BPA8D",
    "BPB8": "BPB8D",
}

# =========================
# Utils num parsing (AR)
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
    Caso real: IOL a veces devuelve '6097' (sin separadores) y en realidad es 60.97.
    Regla:
      - ticker termina en D
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


# =========================
# IRR / NPV (XIRR)
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


def flows_by_month_for_year(cf: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas: month (1..12), flujo_total sumado (USD)
    """
    df = cf.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])
    df = df[df["date"].dt.year == int(year)]
    if df.empty:
        return pd.DataFrame({"month": list(range(1, 13)), "flujo_total": [0.0] * 13}).iloc[:12]

    g = df.groupby(df["date"].dt.month)["flujo_total"].sum().reset_index()
    g.columns = ["month", "flujo_total"]
    base = pd.DataFrame({"month": list(range(1, 13))})
    out = base.merge(g, on="month", how="left").fillna({"flujo_total": 0.0})
    return out


# =========================
# Normalizaciones (meta)
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
        raise ValueError(
            f"Faltan columnas en {path}: {sorted(missing)} (requeridas: {sorted(req)})"
        )

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
# Precios IOL (bonos todos)
# =========================
def fetch_iol_bonos_prices() -> pd.DataFrame:
    """
    Index: Ticker
    Columns: Precio, Volumen
    - No fijamos flavor (para no romper en distintos entornos).
    - Aplica fix /100 para tickers D cuando vienen enteros sin separadores.
    """
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    tables = pd.read_html(url)
    if not tables:
        return pd.DataFrame()

    bonos = tables[0]
    cols = set(bonos.columns.astype(str))

    if "Símbolo" not in cols or "Último Operado" not in cols:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["Ticker"] = bonos["Símbolo"].astype(str).str.strip().str.upper()

    df["RawPrecio"] = bonos["Último Operado"].astype(str).str.strip()
    df["Precio"] = bonos["Último Operado"].apply(parse_ar_number)
    df["Precio"] = [
        usd_fix_if_needed(tk, raw, val)
        for tk, raw, val in zip(df["Ticker"], df["RawPrecio"], df["Precio"])
    ]

    if "Monto Operado" in cols:
        df["Volumen"] = bonos["Monto Operado"].apply(parse_ar_number).fillna(0)
    else:
        df["Volumen"] = 0

    df = df.dropna(subset=["Precio"]).set_index("Ticker")
    df = df.sort_values("Volumen", ascending=False)
    df = df[~df.index.duplicated(keep="first")]

    return df[["Precio", "Volumen"]].copy()


# =========================
# Resolver ticker USD
# =========================
def resolve_usd_ticker(species: str) -> str:
    """
    Resuelve ticker USD (MEP) a buscar en IOL:
    - si viene ya con D => lo deja
    - si está en overrides (pesos->usd) => usa ese
    - si no => default agrega 'D'
    """
    sp = str(species).strip().upper()
    if sp.endswith("D"):
        return sp
    if sp in PESOS_TO_USD_OVERRIDES:
        return PESOS_TO_USD_OVERRIDES[sp]
    return f"{sp}{PRICE_SUFFIX}"


def _pick_price_usd(prices: pd.DataFrame, species: str) -> tuple[float, float, str]:
    usd_ticker = resolve_usd_ticker(species)
    if usd_ticker in prices.index:
        return float(prices.loc[usd_ticker, "Precio"]), float(prices.loc[usd_ticker, "Volumen"]), usd_ticker
    return np.nan, np.nan, ""


# =========================
# Métricas
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
    return round(v, 2) if np.isfinite(v) else np.nan


def duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(ytm):
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty:
        return np.nan

    denom, numer = [], []
    for _, row in cf.iterrows():
        t = row["date"].to_pydatetime()
        tiempo = (t - settlement).days / 365.0
        monto = float(row["flujo_total"])
        pv = monto / (1 + ytm / 100.0) ** tiempo
        denom.append(pv)
        numer.append(tiempo * pv)

    if np.sum(denom) == 0:
        return np.nan
    return round(float(np.sum(numer) / np.sum(denom)), 2)


def modified_duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    dur = duration(cashflow, precio, plazo_dias=plazo_dias)
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return round(dur / (1 + ytm / 100.0), 2)


# =========================
# UI aesthetics
# =========================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .top-title{ font-size: 26px; font-weight: 850; letter-spacing: .04em; color:#111827; margin-bottom: 2px;}
  .top-sub{ color: rgba(17,24,39,.60); font-size: 13px; margin-top: 0px; }
  .section-title{ font-size: 18px; font-weight: 800; color:#111827; margin: 12px 0 8px; }
  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  label { margin-bottom: .18rem !important; }

  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 700; }
  .stSelectbox div[data-baseweb="select"]{ border-radius: 14px; }
  .stMultiSelect div[data-baseweb="select"]{ border-radius: 14px; }

  /* Chips gris clarito (como pediste) */
  div[data-baseweb="tag"]{
    background: rgba(17,24,39,.06) !important;
    color:#111827 !important;
    border: 1px solid rgba(17,24,39,.10) !important;
    border-radius: 999px !important;
    font-weight: 650 !important;
  }
  div[data-baseweb="tag"] svg{
    color: rgba(17,24,39,.55) !important;
  }

  div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _multiselect_with_all(label: str, options: list[str], key: str, default_all: bool = True) -> list[str]:
    if not options:
        return []
    options_sorted = sorted(set([str(x) for x in options if str(x).strip() != ""]))
    all_token = "Seleccionar todo"
    opts = [all_token] + options_sorted
    default = [all_token] if default_all else []
    sel = st.multiselect(label, options=opts, default=default, key=key)
    if all_token in sel:
        return options_sorted
    return sel


# =========================
# Elegibles (precio USD + flujos futuros + TIR dentro de rango)
# =========================
def compute_eligibles(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        px, vol, px_ticker = _pick_price_usd(prices, species)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        settlement = _settlement(plazo)
        cf_fut = _future_cashflows(cf, settlement)
        if cf_fut.empty:
            continue

        t = tir(cf, px, plazo_dias=plazo)
        if not np.isfinite(t):
            continue

        # Filtro fijo TIR
        if not (TIR_MIN <= t <= TIR_MAX):
            continue

        rows.append(
            {
                "Ticker": species,
                "Ticker precio": px_ticker,
                "Ley": meta.loc[species, "law_norm"],
                "Issuer": meta.loc[species, "issuer_norm"],
                "Descripción": meta.loc[species, "desc_norm"],
                "Precio": px,
                "TIR (%)": t,
                "MD": modified_duration(cf, px, plazo_dias=plazo),
                "Duration": duration(cf, px, plazo_dias=plazo),
                "Vencimiento": meta.loc[species, "vencimiento"],
                "Volumen": vol,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Issuer", "Ley", "Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)
    out["Ley"] = out["Ley"].apply(law_cell_label)

    # Categoría (simple y útil)
    out["Categoría"] = out["Issuer"].fillna("NA").astype(str).str.title()
    return out


# =========================
# Cartera: asignación por %
# =========================
def compute_portfolio_table(elig: pd.DataFrame, selected: list[str], pct_map: dict[str, float], capital_usd: float) -> pd.DataFrame:
    df = elig[elig["Ticker"].isin(selected)].copy()
    if df.empty:
        return df

    df["%"] = df["Ticker"].map(pct_map).fillna(0.0)
    df["USD"] = (df["%"] / 100.0) * float(capital_usd)

    # VN estimada: USD / (Precio/100)  => USD * 100 / Precio
    df["VN Estimada"] = np.where(df["Precio"] > 0, df["USD"] * 100.0 / df["Precio"], np.nan)

    # Orden visual
    cols = [
        "Categoría",
        "Ticker",
        "%",
        "USD",
        "Precio",
        "VN Estimada",
        "TIR (%)",
        "MD",
        "Duration",
        "Vencimiento",
        "Ley",
        "Issuer",
        "Ticker precio",
    ]
    df = df[cols].copy()
    df["Vencimiento"] = pd.to_datetime(df["Vencimiento"], errors="coerce").dt.date
    return df


def weighted_metric(df: pd.DataFrame, col: str) -> float:
    w = pd.to_numeric(df["USD"], errors="coerce").fillna(0.0)
    x = pd.to_numeric(df[col], errors="coerce")
    mask = np.isfinite(x) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def build_portfolio_cashflow_calendar(df_cf: pd.DataFrame, portfolio: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Matriz meses (1..12) x tickers seleccionados + Total
    Usa los cashflows del excel (asumidos en USD coherentes con precio MEP)
    """
    if portfolio.empty:
        return pd.DataFrame()

    cf_dict = build_cashflow_dict(df_cf)

    tickers = portfolio["Ticker"].tolist()
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    mat = pd.DataFrame(index=tickers, columns=month_names, data=0.0)

    for tk in tickers:
        cf = cf_dict.get(tk)
        if cf is None or cf.empty:
            continue
        m = flows_by_month_for_year(cf, year)
        for _, r in m.iterrows():
            mat.loc[tk, month_names[int(r["month"]) - 1]] = float(r["flujo_total"])

    mat["Total Ticker"] = mat.sum(axis=1)
    total_row = mat.sum(axis=0).to_frame().T
    total_row.index = ["Totales"]
    out = pd.concat([mat, total_row], axis=0)
    return out


# =========================
# Render
# =========================
def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with c1:
        st.markdown('<div class="top-title">NEIX · Cartera Comercial (USD MEP)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="top-sub">Activos elegibles: precio USD MEP + flujos futuros + TIR en rango [{TIR_MIN}, {TIR_MAX}].</div>',
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

    # Top controls
    top = st.columns([0.18, 0.22, 0.20, 0.40], vertical_alignment="bottom")
    with top[0]:
        plazo = st.selectbox("Plazo de liquidación", [1, 0], index=0, format_func=lambda x: f"T{x}", key="cartera_plazo")
    with top[1]:
        capital = st.number_input("Capital (USD)", min_value=0.0, value=200_000.0, step=10_000.0, format="%.2f")
    with top[2]:
        traer_precios = st.button("Actualizar precios", use_container_width=True, key="cartera_refresh")
    with top[3]:
        calcular = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_calc")

    # Prices cache
    if traer_precios or "cartera_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            st.session_state["cartera_iol_prices"] = fetch_iol_bonos_prices()

    prices = st.session_state.get("cartera_iol_prices")
    if prices is None or prices.empty:
        st.warning("No pude leer precios desde IOL. Probá 'Actualizar precios'.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Filtros meta (opcional) para achicar universo
    st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)
    laws = sorted(df_cf["law_norm"].dropna().unique().tolist())
    issuers = sorted(df_cf["issuer_norm"].dropna().unique().tolist())
    descs = sorted(df_cf["desc_norm"].dropna().unique().tolist())
    tickers_all = sorted(df_cf["species"].dropna().unique().tolist())

    f1, f2, f3, f4 = st.columns([0.22, 0.26, 0.26, 0.26], vertical_alignment="top")
    with f1:
        law_sel = _multiselect_with_all("Ley", laws, key="cartera_law", default_all=True)
    with f2:
        issuer_sel = _multiselect_with_all("Issuer", issuers, key="cartera_issuer", default_all=True)
    with f3:
        desc_sel = _multiselect_with_all("Descripción", descs, key="cartera_desc", default_all=True)
    with f4:
        ticker_sel = _multiselect_with_all("Ticker", tickers_all, key="cartera_ticker", default_all=True)

    df_use = df_cf.copy()
    if law_sel:
        df_use = df_use[df_use["law_norm"].isin(law_sel)]
    if issuer_sel:
        df_use = df_use[df_use["issuer_norm"].isin(issuer_sel)]
    if desc_sel:
        df_use = df_use[df_use["desc_norm"].isin(desc_sel)]
    if ticker_sel:
        df_use = df_use[df_use["species"].isin(ticker_sel)]

    st.divider()

    if not calcular:
        st.info("Ajustá filtros (si querés) y tocá **Calcular cartera**.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.spinner("Armando universo elegible..."):
        elig = compute_eligibles(df_use, prices, plazo)

    if elig.empty:
        st.warning("No se encontraron activos elegibles con esta selección.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # Selección de activos
    # =========================
    st.markdown("## Selección de activos")
    st.caption("Solo aparecen instrumentos con Precio USD MEP + Cashflows futuros + TIR dentro de rango.")

    # Multiselect elegibles
    options = elig["Ticker"].tolist()
    default_sel = options[:6] if len(options) >= 6 else options
    selected = st.multiselect("Activos", options=options, default=default_sel, key="cartera_selected")

    if not selected:
        st.warning("Elegí al menos 1 activo.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # Asignación por %
    # =========================
    st.markdown("## Asignación por porcentaje")
    st.caption("Asigná % por activo. Ideal: que sumen 100% (si no, igualmente calcula con lo que pusiste).")

    pct_map: dict[str, float] = {}
    n = len(selected)
    cols = st.columns(2)
    for i, tk in enumerate(selected):
        with cols[i % 2]:
            pct_map[tk] = st.slider(
                f"{tk} (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(100.0 / n, 2) if n > 0 else 0.0,
                step=0.5,
                key=f"pct_{tk}",
            )

    pct_sum = float(np.sum(list(pct_map.values())))
    if abs(pct_sum - 100.0) > 0.01:
        st.warning(f"La suma de % da **{pct_sum:.2f}%** (ideal 100%). Igual calculo con esa suma.")

    # Tabla cartera
    port = compute_portfolio_table(elig, selected, pct_map, capital)

    # Totales ponderados
    tir_w = weighted_metric(port, "TIR (%)")
    md_w = weighted_metric(port, "MD")
    dur_w = weighted_metric(port, "Duration")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Capital", f"USD {capital:,.2f}")
    k2.metric("TIR total (pond.)", f"{tir_w:.2f}%" if np.isfinite(tir_w) else "—")
    k3.metric("MD total (pond.)", f"{md_w:.2f}" if np.isfinite(md_w) else "—")
    k4.metric("Duration (pond.)", f"{dur_w:.2f}" if np.isfinite(dur_w) else "—")

    st.markdown("## Cartera")
    st.dataframe(
        port,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Categoría": st.column_config.TextColumn("Categoría"),
            "Ticker": st.column_config.TextColumn("Activo"),
            "%": st.column_config.NumberColumn("%", format="%.2f"),
            "USD": st.column_config.NumberColumn("USD", format="%.2f"),
            "Precio": st.column_config.NumberColumn("Precio (USD)", format="%.2f"),
            "VN Estimada": st.column_config.NumberColumn("VN Estimada", format="%.2f"),
            "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
            "MD": st.column_config.NumberColumn("MD", format="%.2f"),
            "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
            "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
            "Ley": st.column_config.TextColumn("Ley"),
            "Issuer": st.column_config.TextColumn("Issuer"),
            "Ticker precio": st.column_config.TextColumn("Ticker (IOL)"),
        },
    )

    # =========================
    # Gráficos
    # =========================
    st.markdown("## Gráficos")

    pie_df = port[["Ticker", "USD"]].copy()
    pie_df = pie_df[pie_df["USD"] > 0].sort_values("USD", ascending=False)

    g1, g2 = st.columns([0.42, 0.58])
    with g1:
        st.caption("Distribución por USD")
        st.plotly_chart(
            {
                "data": [
                    {
                        "type": "pie",
                        "labels": pie_df["Ticker"].tolist(),
                        "values": pie_df["USD"].tolist(),
                        "hole": 0.55,
                    }
                ],
                "layout": {"margin": {"l": 10, "r": 10, "t": 10, "b": 10}},
            },
            use_container_width=True,
        )
    with g2:
        st.caption("Riesgo / retorno (TIR vs MD)")
        scat = port.copy()
        scat["TIR (%)"] = pd.to_numeric(scat["TIR (%)"], errors="coerce")
        scat["MD"] = pd.to_numeric(scat["MD"], errors="coerce")
        scat = scat[np.isfinite(scat["TIR (%)"]) & np.isfinite(scat["MD"])]
        st.plotly_chart(
            {
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers+text",
                        "x": scat["MD"].tolist(),
                        "y": scat["TIR (%)"].tolist(),
                        "text": scat["Ticker"].tolist(),
                        "textposition": "top center",
                        "marker": {"size": 10},
                    }
                ],
                "layout": {
                    "xaxis": {"title": "MD"},
                    "yaxis": {"title": "TIR (%)"},
                    "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
                },
            },
            use_container_width=True,
        )

    # =========================
    # Calendario de flujos
    # =========================
    st.markdown("## Calendario de flujos")
    years = sorted(pd.to_datetime(df_cf["date"], errors="coerce").dropna().dt.year.unique().tolist())
    default_year = int(pd.Timestamp.today().year)
    year = st.selectbox("Año", options=years, index=years.index(default_year) if default_year in years else 0)

    cal = build_portfolio_cashflow_calendar(df_cf, port, year)
    if cal.empty:
        st.info("No hay flujos para el año seleccionado.")
    else:
        st.dataframe(
            cal,
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

