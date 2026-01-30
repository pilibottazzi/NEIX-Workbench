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
PRICE_SUFFIX = "D"

DEFAULT_TIR_MIN = -15.0
DEFAULT_TIR_MAX = 20.0

# Excepciones pesos->USD (cuando no es simplemente +D)
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
    "BPA7": "BPA7D",
    "BPB7": "BPB7D",
    "BPC7": "BPC7D",
    "BPD7": "BPD7D",
    "BPA8": "BPA8D",
    "BPB8": "BPB8D",
}

# =========================
# Utils parse + tickers
# =========================
def parse_ar_number(x) -> float:
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
    # Si termina en D y viene entero (sin separadores), suele ser *100
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


def _settlement(plazo_dias: int) -> dt.datetime:
    base = pd.Timestamp.today().normalize().to_pydatetime()
    return base + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])
    return df[df["date"] > settlement].sort_values("date")


# =========================
# Normalizaciones meta
# =========================
def normalize_law(x: str) -> str:
    s = (x or "").strip().upper().replace(".", "").replace("-", " ").replace("_", " ")
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
    s = (x or "").strip().upper().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s if s else "NA"


def normalize_desc(x: str) -> str:
    s = (x or "").strip().upper().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s if s else "NA"


# =========================
# Load cashflows
# =========================
def load_cashflows(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    req = {"date", "species", "law", "issuer", "description", "flujo_total"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")

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
    return (
        df.groupby("species")
        .agg(
            law_norm=("law_norm", lambda s: s.value_counts().index[0]),
            issuer_norm=("issuer_norm", lambda s: s.value_counts().index[0]),
            desc_norm=("desc_norm", lambda s: s.value_counts().index[0]),
            vencimiento=("date", "max"),
        )
        .reset_index()
    )


# =========================
# IOL prices
# =========================
def fetch_iol_prices() -> pd.DataFrame:
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    tables = pd.read_html(url)  # requiere lxml localmente
    if not tables:
        return pd.DataFrame()
    t0 = tables[0]
    cols = set(t0.columns.astype(str))
    if "Símbolo" not in cols or "Último Operado" not in cols:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["Ticker"] = t0["Símbolo"].astype(str).str.strip().str.upper()
    df["RawPrecio"] = t0["Último Operado"].astype(str).str.strip()
    df["Precio"] = t0["Último Operado"].apply(parse_ar_number)
    df["Precio"] = [
        usd_fix_if_needed(tk, raw, val)
        for tk, raw, val in zip(df["Ticker"], df["RawPrecio"], df["Precio"])
    ]

    if "Monto Operado" in cols:
        df["Volumen"] = t0["Monto Operado"].apply(parse_ar_number).fillna(0)
    else:
        df["Volumen"] = 0

    df = df.dropna(subset=["Precio"]).set_index("Ticker")
    df = df[~df.index.duplicated(keep="first")]
    return df.drop(columns=["RawPrecio"], errors="ignore")


# =========================
# Metrics per asset
# =========================
def tir_asset(cashflow: pd.DataFrame, precio: float, plazo_dias: int) -> float:
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


def duration_asset(cashflow: pd.DataFrame, precio: float, plazo_dias: int) -> float:
    ytm = tir_asset(cashflow, precio, plazo_dias)
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
    return round(numer / denom, 2)


def md_asset(cashflow: pd.DataFrame, precio: float, plazo_dias: int) -> float:
    dur = duration_asset(cashflow, precio, plazo_dias)
    ytm = tir_asset(cashflow, precio, plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return round(dur / (1 + ytm / 100.0), 2)


# =========================
# Build eligible universe
# =========================
def build_universe(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
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

        t = tir_asset(cf, px, plazo)
        if not np.isfinite(t):
            continue
        if not (DEFAULT_TIR_MIN <= t <= DEFAULT_TIR_MAX):
            continue

        rows.append(
            {
                "Ticker": species,
                "Ticker precio": usd_tk,
                "Precio": px,
                "TIR (%)": t,
                "MD": md_asset(cf, px, plazo),
                "Duration": duration_asset(cf, px, plazo),
                "Vencimiento": meta.loc[species, "vencimiento"],
                "Ley": meta.loc[species, "law_norm"],
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
# Portfolio math
# =========================
def vn_estimada(usd: float, precio: float) -> float:
    # VN = USD / (Precio/100)
    if not np.isfinite(usd) or not np.isfinite(precio) or precio <= 0:
        return np.nan
    return usd / (precio / 100.0)


def build_flow_calendar(df_cf: pd.DataFrame, alloc: pd.DataFrame, plazo: int) -> pd.DataFrame:
    """
    alloc: columnas requeridas -> Ticker, VN
    Devuelve tabla mensual (YYYY-MM) con flujos USD por ticker y total.
    """
    cashflows = build_cashflow_dict(df_cf)
    settlement = _settlement(plazo)

    # universo de fechas = flujos futuros
    all_rows = []
    for _, r in alloc.iterrows():
        tk = str(r["Ticker"]).strip().upper()
        vn = float(r["VN"])
        if not np.isfinite(vn) or vn <= 0:
            continue
        cf = cashflows.get(tk)
        if cf is None or cf.empty:
            continue
        cf_fut = _future_cashflows(cf, settlement)
        if cf_fut.empty:
            continue

        # cashflow_df es por VN100 (flujo_total por 100)
        # Escalamos por VN/100
        scale = vn / 100.0
        tmp = cf_fut.copy()
        tmp["Monto_USD"] = tmp["flujo_total"].astype(float) * scale
        tmp["Ticker"] = tk
        all_rows.append(tmp[["date", "Ticker", "Monto_USD"]])

    if not all_rows:
        return pd.DataFrame()

    flows = pd.concat(all_rows, ignore_index=True)
    flows["Mes"] = pd.to_datetime(flows["date"]).dt.to_period("M").astype(str)

    piv = (
        flows.groupby(["Mes", "Ticker"])["Monto_USD"]
        .sum()
        .reset_index()
        .pivot(index="Ticker", columns="Mes", values="Monto_USD")
        .fillna(0.0)
    )

    piv["Total"] = piv.sum(axis=1)
    piv.loc["TOTAL"] = piv.sum(axis=0)
    return piv


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
  .section-title{ font-size: 18px; font-weight: 800; color:#111827; margin: 14px 0 10px; }
  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 750; }
  .stSelectbox div[data-baseweb="select"]{ border-radius: 14px; }
  .stMultiSelect div[data-baseweb="select"]{ border-radius: 14px; }

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
# Render
# =========================
def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    st.markdown('<div class="top-title">NEIX · Cartera Comercial (USD MEP)</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="top-sub">Activos elegibles: con precio IOL + TIR disponible. TIR fija en [{DEFAULT_TIR_MIN}, {DEFAULT_TIR_MAX}].</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Inputs top
    c1, c2, c3 = st.columns([0.30, 0.20, 0.50], vertical_alignment="bottom")
    with c1:
        capital = st.number_input("Capital (USD)", min_value=0.0, value=200000.0, step=5000.0, format="%.2f")
    with c2:
        plazo = st.selectbox("Liquidación", [1, 0], index=0, format_func=lambda x: f"T{x}", key="cartera_plazo")
    with c3:
        colA, colB = st.columns([0.5, 0.5])
        with colA:
            refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_refresh")
        with colB:
            calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_calc")

    # Load cashflows
    try:
        df_cf = load_cashflows(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Prices cache
    if refresh or "cartera_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            try:
                st.session_state["cartera_prices"] = fetch_iol_prices()
            except Exception as e:
                st.error(str(e))
                st.markdown("</div>", unsafe_allow_html=True)
                return

    prices = st.session_state.get("cartera_prices")
    if prices is None or prices.empty:
        st.warning("No pude leer precios desde IOL.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Universe
    universe = build_universe(df_cf, prices, plazo)
    if universe.empty:
        st.warning("No hay activos elegibles (precio+tir) dentro del rango.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Select assets
    st.markdown('<div class="section-title">Selección de activos</div>', unsafe_allow_html=True)
    options = universe["Ticker"].tolist()
    default_sel = options[:6] if len(options) >= 6 else options

    selected = st.multiselect(
        "Activos (bonos + ONs)",
        options=options,
        default=default_sel,
        help="Elegí los activos que querés incluir. Luego asignás % por activo.",
        key="cartera_assets",
    )

    if not selected:
        st.info("Seleccioná al menos 1 activo.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    uni_sel = universe[universe["Ticker"].isin(selected)].copy()

    # Allocation table (editable, sin sliders)
    st.markdown('<div class="section-title">Asignación por activo</div>', unsafe_allow_html=True)
    st.caption("Editá la columna **%**. Ideal: que sume 100%.")

    alloc = pd.DataFrame({"Ticker": uni_sel["Ticker"].tolist()})
    if "cartera_alloc_pct" not in st.session_state:
        # default: igual ponderado
        alloc["%"] = round(100.0 / len(alloc), 2)
        st.session_state["cartera_alloc_pct"] = alloc
    else:
        # mantener lo previo pero sincronizar tickers (agrega/saca)
        prev = st.session_state["cartera_alloc_pct"].copy()
        prev = prev[prev["Ticker"].isin(alloc["Ticker"])]
        missing = [t for t in alloc["Ticker"] if t not in prev["Ticker"].tolist()]
        if missing:
            add = pd.DataFrame({"Ticker": missing, "%": round(100.0 / len(alloc), 2)})
            prev = pd.concat([prev, add], ignore_index=True)
        alloc = prev[["Ticker", "%"]].copy()

    edited = st.data_editor(
        alloc,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "%": st.column_config.NumberColumn("%", min_value=0.0, max_value=100.0, step=0.25, format="%.2f"),
        },
        key="cartera_alloc_editor",
    )

    # guardar
    st.session_state["cartera_alloc_pct"] = edited.copy()

    pct_sum = float(pd.to_numeric(edited["%"], errors="coerce").fillna(0).sum())
    if abs(pct_sum - 100.0) > 0.25:
        st.warning(f"La suma de % es {pct_sum:.2f}% (ideal 100%). Igual calculo proporcional con esa suma.")

    # Compute portfolio (only if calc)
    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Merge metrics
    df = uni_sel.merge(edited, on="Ticker", how="left")
    df["%"] = pd.to_numeric(df["%"], errors="coerce").fillna(0.0)
    denom = df["%"].sum()
    if denom <= 0:
        st.error("La suma de % tiene que ser > 0.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # USD allocation
    df["USD"] = capital * (df["%"] / denom)

    # VN estimada (USD / (Precio/100))
    df["VN"] = df.apply(lambda r: vn_estimada(float(r["USD"]), float(r["Precio"])), axis=1)

    # Ponderados
    w = df["USD"] / df["USD"].sum() if df["USD"].sum() > 0 else 0
    tir_total = float((df["TIR (%)"] * w).sum())
    md_total = float((df["MD"] * w).sum())
    dur_total = float((df["Duration"] * w).sum())

    # Header metrics
    st.markdown('<div class="section-title">Resumen</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capital", f"USD {capital:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    m2.metric("TIR total (pond.)", f"{tir_total:.2f}%")
    m3.metric("MD total (pond.)", f"{md_total:.2f}")
    m4.metric("Duration total (pond.)", f"{dur_total:.2f}")

    # Portfolio table
    st.markdown('<div class="section-title">Cartera</div>', unsafe_allow_html=True)

    show = df.copy()
    show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date
    show["Ley"] = show["Ley"].apply(law_cell_label)

    # orden limpio
    show = show[
        ["Ticker", "%", "USD", "Precio", "VN", "TIR (%)", "MD", "Duration", "Vencimiento", "Ley", "Issuer"]
    ].copy()

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "%": st.column_config.NumberColumn("%", format="%.2f"),
            "USD": st.column_config.NumberColumn("USD", format="%.2f"),
            "Precio": st.column_config.NumberColumn("Precio (USD, VN100)", format="%.2f"),
            "VN": st.column_config.NumberColumn("VN estimada", format="%.2f"),
            "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
            "MD": st.column_config.NumberColumn("MD", format="%.2f"),
            "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
            "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
            "Ley": st.column_config.TextColumn("Ley"),
            "Issuer": st.column_config.TextColumn("Issuer"),
        },
    )

    # Pie chart (simple, prolijo)
    st.markdown('<div class="section-title">Asignación</div>', unsafe_allow_html=True)
    pie_df = show[["Ticker", "USD"]].copy()
    pie_df = pie_df[pie_df["USD"] > 0].sort_values("USD", ascending=False)

    st.plotly_chart(
        {
            "data": [
                {
                    "type": "pie",
                    "labels": pie_df["Ticker"],
                    "values": pie_df["USD"],
                    "hole": 0.55,
                }
            ],
            "layout": {
                "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
                "showlegend": True,
            },
        },
        use_container_width=True,
    )

    # Flow calendar
    st.markdown('<div class="section-title">Flujo de fondos</div>', unsafe_allow_html=True)
    alloc_for_flow = df[["Ticker", "VN"]].rename(columns={"VN": "VN"}).copy()
    cal = build_flow_calendar(df_cf, alloc_for_flow.rename(columns={"VN": "VN"}), plazo)

    if cal.empty:
        st.info("No pude construir el calendario de flujos con lo seleccionado.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.dataframe(cal, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
