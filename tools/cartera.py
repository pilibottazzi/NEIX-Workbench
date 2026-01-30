from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize


# =========================================================
# Config
# =========================================================
CASHFLOW_PATH = os.path.join("data", "cashflows_completos.xlsx")

# TIR fija (como pediste)
TIR_MIN = -15.0
TIR_MAX = 20.0

# Siempre T1 (sacamos el filtro)
PLAZO_LIQ = 1

PRICE_SUFFIX = "D"

# Excepciones ticker pesos -> ticker USD MEP (cuando NO es simplemente + "D")
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
    # si aparecieran en cashflows con estas “bases”
    "BPA7": "BPA7D",
    "BPB7": "BPB7D",
    "BPC7": "BPC7D",
    "BPA8": "BPA8D",
    "BPB8": "BPB8D",
}


# =========================================================
# Utils parse AR numbers
# =========================================================
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
    Fix: a veces el precio USD (ticker termina en D) viene como entero "6097" y es 60.97.
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


# =========================================================
# IRR / NPV
# =========================================================
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


# =========================================================
# Cashflows helpers
# =========================================================
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


# =========================================================
# Normalizaciones
# =========================================================
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


# =========================================================
# Load cashflows
# =========================================================
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
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

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


# =========================================================
# Precios (scrape)
# =========================================================
def fetch_prices() -> pd.DataFrame:
    """
    Devuelve:
      index: Ticker
      columns: Precio, Volumen
    """
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"

    # pd.read_html puede requerir html5lib/lxml según el entorno.
    try:
        tables = pd.read_html(url)
    except Exception as e:
        # mensaje interno (no UI)
        raise RuntimeError(
            "No se pudo leer la tabla de cotizaciones (dependencias HTML faltantes o cambió el formato)."
        ) from e

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
    return df.drop(columns=["RawPrecio"], errors="ignore")


def resolve_usd_ticker(species: str) -> str:
    """
    Resuelve ticker USD MEP a buscar:
    - si ya termina con D => igual
    - si está en overrides => usa override
    - si no => default agrega 'D'
    """
    sp = str(species).strip().upper()
    if sp.endswith("D"):
        return sp
    if sp in PESOS_TO_USD_OVERRIDES:
        return PESOS_TO_USD_OVERRIDES[sp]
    return f"{sp}{PRICE_SUFFIX}"


def pick_price_usd(prices: pd.DataFrame, species: str) -> Tuple[float, float, str]:
    usd_ticker = resolve_usd_ticker(species)
    if usd_ticker in prices.index:
        return float(prices.loc[usd_ticker, "Precio"]), float(prices.loc[usd_ticker, "Volumen"]), usd_ticker
    return np.nan, np.nan, ""


# =========================================================
# Métricas por activo
# =========================================================
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
    return float(np.sum(numer) / np.sum(denom))


def modified_duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    dur = duration(cashflow, precio, plazo_dias=plazo_dias)
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return float(dur / (1 + ytm / 100.0))


# =========================================================
# UI helpers (estética NEIX)
# =========================================================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .top-title{ font-size: 26px; font-weight: 850; letter-spacing: .04em; color:#111827; margin-bottom: 2px;}
  .top-sub{ color: rgba(17,24,39,.60); font-size: 13px; margin-top: 0px; }
  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  label { margin-bottom: .18rem !important; }

  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 700; }

  .stMultiSelect div[data-baseweb="select"]{ border-radius: 14px; }
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

  .kpi-title{ color: rgba(17,24,39,.60); font-size: 13px; margin-bottom: 2px;}
  .kpi-big{ font-size: 44px; font-weight: 850; letter-spacing: .02em; color:#111827; margin-top:-4px;}
  .kpi-val{ font-size: 38px; font-weight: 800; color:#111827; margin-top:-4px;}

  .chip{
    display:inline-flex; align-items:center; gap:8px;
    padding:8px 12px; border-radius: 999px;
    border: 1px solid rgba(17,24,39,.10);
    background: rgba(17,24,39,.04);
    color:#111827; font-weight:650; font-size:13px;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _money_usd(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"$ {x:,.0f}".replace(",", ".")


def _pct0(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.0f}%"


def _fmt2(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.2f}"


def _ensure_date(x) -> dt.date | None:
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


# =========================================================
# Portfolio computation
# =========================================================
def compute_asset_table(df_cf: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        px, vol, px_ticker = pick_price_usd(prices, species)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        # métricas
        y = tir(cf, px, plazo_dias=PLAZO_LIQ)
        md = modified_duration(cf, px, plazo_dias=PLAZO_LIQ)
        dur = duration(cf, px, plazo_dias=PLAZO_LIQ)

        # filtro TIR fijo (eligibles)
        if not np.isfinite(y) or (y < TIR_MIN) or (y > TIR_MAX):
            continue

        rows.append(
            {
                "Ticker": species,
                "Ticker precio": px_ticker,
                "Ley": meta.loc[species, "law_norm"],
                "Issuer": meta.loc[species, "issuer_norm"],
                "Descripción": meta.loc[species, "desc_norm"],
                "Precio": float(px),
                "TIR": float(y),
                "MD": float(md) if np.isfinite(md) else np.nan,
                "Duration": float(dur) if np.isfinite(dur) else np.nan,
                "Vencimiento": meta.loc[species, "vencimiento"],
                "Volumen": float(vol) if np.isfinite(vol) else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)
    return out


def compute_portfolio(
    eligible: pd.DataFrame,
    selected: list[str],
    weights_pct: dict[str, float],
    capital_usd: float,
) -> pd.DataFrame:
    """
    Devuelve tabla final de cartera por activo:
    - % (sin decimales)
    - USD asignado (sin decimales)
    - Precio (2 decimales)
    - VN estimada = USD / (Precio/100)   (precio por VN100)
    - TIR/MD/Duration (2 decimales en salida)
    - Vencimiento (fecha)
    """
    if eligible.empty or not selected:
        return pd.DataFrame()

    base = eligible.set_index("Ticker")
    rows = []
    for tk in selected:
        if tk not in base.index:
            continue

        w = float(weights_pct.get(tk, 0.0))
        usd = float(capital_usd) * (w / 100.0)

        px = float(base.loc[tk, "Precio"])
        vn_est = np.nan
        if np.isfinite(px) and px > 0 and np.isfinite(usd):
            vn_est = usd / (px / 100.0)

        rows.append(
            {
                "Ticker": tk,
                "%": w,
                "USD": usd,
                "Precio (USD, VN100)": px,
                "VN estimada": vn_est,
                "TIR (%)": float(base.loc[tk, "TIR"]),
                "MD": float(base.loc[tk, "MD"]),
                "Duration": float(base.loc[tk, "Duration"]),
                "Vencimiento": _ensure_date(base.loc[tk, "Vencimiento"]),
                "Ley": law_cell_label(str(base.loc[tk, "Ley"])),
                "Issuer": str(base.loc[tk, "Issuer"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["%", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return out


def weighted_avg(series: pd.Series, weights: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float((s[mask] * w[mask]).sum() / w[mask].sum())


# =========================================================
# Flujos de fondos (tabla + totales + filtro estético)
# =========================================================
def build_cashflow_calendar(
    df_cf_full: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera tabla mensual de flujos para la cartera.
    Escala por VN estimada: flujo_total * VN/100 (si cashflow está por VN100).
    """
    if portfolio.empty:
        return pd.DataFrame()

    # seleccionados + vn
    vn_map = dict(zip(portfolio["Ticker"], portfolio["VN estimada"]))
    tickers = list(vn_map.keys())

    df = df_cf_full[df_cf_full["species"].isin(tickers)].copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])

    # escala por VN/100
    df["VN"] = df["species"].map(vn_map)
    df["scaled_cf"] = pd.to_numeric(df["flujo_total"], errors="coerce") * (pd.to_numeric(df["VN"], errors="coerce") / 100.0)

    df["Mes"] = df["date"].dt.to_period("M").dt.to_timestamp()

    pivot = (
        df.pivot_table(
            index="species",
            columns="Mes",
            values="scaled_cf",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    if pivot.empty:
        return pivot

    # Totales por fila/columna
    pivot["Total Ticker"] = pivot.sum(axis=1)

    total_row = pd.DataFrame([pivot.sum(axis=0)], index=["Totales"])
    out = pd.concat([pivot, total_row], axis=0)

    # orden columnas: meses + Total
    cols = [c for c in out.columns if c != "Total Ticker"]
    cols = sorted(cols) + ["Total Ticker"]
    out = out[cols]

    return out


# =========================================================
# Render
# =========================================================
def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Carga cashflows
    try:
        df_cf = load_cashflows(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Header: título + botón alineado (misma línea)
    h1, h2 = st.columns([0.68, 0.32], vertical_alignment="center")
    with h1:
        st.markdown('<div class="top-title">NEIX · Cartera Comercial (USD MEP)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="top-sub">Activos elegibles: con precio + TIR disponible. TIR fija en [{TIR_MIN:.1f}, {TIR_MAX:.1f}].</div>',
            unsafe_allow_html=True,
        )
    with h2:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_refresh")

    st.divider()

    # Cache precios
    if refresh or "cartera_prices" not in st.session_state:
        with st.spinner("Actualizando precios..."):
            try:
                st.session_state["cartera_prices"] = fetch_prices()
            except Exception:
                st.session_state["cartera_prices"] = pd.DataFrame()

    prices = st.session_state.get("cartera_prices", pd.DataFrame())
    if prices is None or prices.empty:
        st.warning("No se pudieron cargar precios. Probá nuevamente con **Actualizar precios**.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Elegibles (con precio + TIR dentro del rango)
    eligible = compute_asset_table(df_cf, prices)
    if eligible.empty:
        st.warning("No hay activos elegibles con la información disponible.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ==========
    # Selección
    # ==========
    st.markdown("### Selección de activos")

    options = eligible["Ticker"].tolist()
    default_sel = options[:6] if len(options) >= 6 else options

    selected = st.multiselect(
        "Activos (bonos + ONs)",
        options=options,
        default=st.session_state.get("cartera_selected", default_sel),
        key="cartera_selected",
    )

    if not selected:
        st.info("Seleccioná al menos un activo.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Capital en el “medio” (como pediste)
    st.markdown("")

    ccap, csp = st.columns([0.40, 0.60], vertical_alignment="center")
    with ccap:
        capital = st.number_input(
            "Capital (USD)",
            min_value=0.0,
            value=float(st.session_state.get("cartera_capital", 100000.0)),
            step=1000.0,
            key="cartera_capital",
            format="%.2f",
        )
    with csp:
        # chips “informativos” discretos
        st.markdown(
            f"""
            <div style="display:flex; gap:10px; flex-wrap:wrap; padding-top:22px;">
              <div class="chip">Liquidación: <b>T1</b></div>
              <div class="chip">Moneda: <b>USD MEP</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========
    # Asignación (tabla editable, sin sliders)
    # ==========
    st.markdown("### Asignación por activo")
    st.caption("Editá la columna %. Ideal: que sume 100%.")

    # Construimos tabla base de pesos
    base_w = []
    equal = 100.0 / max(1, len(selected))
    for tk in selected:
        base_w.append({"Ticker": tk, "%": round(equal, 2)})

    df_w = pd.DataFrame(base_w)

    # si ya venía de sesión, lo respetamos
    prev = st.session_state.get("cartera_weights")
    if isinstance(prev, dict) and prev:
        df_w["%"] = df_w["Ticker"].map(prev).fillna(df_w["%"])

    edited = st.data_editor(
        df_w,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "%": st.column_config.NumberColumn("%", min_value=0.0, max_value=100.0, step=1.0, format="%.2f"),
        },
        key="cartera_editor",
    )

    weights = dict(zip(edited["Ticker"], pd.to_numeric(edited["%"], errors="coerce").fillna(0.0)))
    st.session_state["cartera_weights"] = weights

    sum_w = float(np.nansum(list(weights.values())))
    if abs(sum_w - 100.0) > 0.05:
        st.warning(f"La suma de % da {sum_w:.2f}% (ideal 100%). Igual calculo con esa suma.")

    # CTA
    calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_calc")

    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ==========
    # Resultado cartera
    # ==========
    portfolio = compute_portfolio(eligible, selected, weights, float(capital))
    if portfolio.empty:
        st.warning("No se pudo calcular la cartera con esta selección.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Resumen ponderado (por USD asignado)
    w_usd = portfolio["USD"].astype(float)

    tir_tot = weighted_avg(portfolio["TIR (%)"], w_usd)
    md_tot = weighted_avg(portfolio["MD"], w_usd)
    dur_tot = weighted_avg(portfolio["Duration"], w_usd)

    st.markdown("## Resumen")
    k1, k2, k3, k4 = st.columns([0.34, 0.22, 0.22, 0.22], vertical_alignment="bottom")

    with k1:
        st.markdown('<div class="kpi-title">Capital</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-big">{_money_usd(float(capital))}</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi-title">TIR total (pond.)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-val">{_fmt2(tir_tot)}%</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-title">MD total (pond.)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-val">{_fmt2(md_tot)}</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi-title">Duration total (pond.)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-val">{_fmt2(dur_tot)}</div>', unsafe_allow_html=True)

    st.markdown("### Cartera")

    # Formateo de salida: % sin decimales, USD/VN sin decimales, precio 2 decimales, TIR/MD/Dur 2 decimales
    show = portfolio.copy()
    show["%"] = pd.to_numeric(show["%"], errors="coerce").round(0).astype("Int64")
    show["USD"] = pd.to_numeric(show["USD"], errors="coerce").round(0)
    show["VN estimada"] = pd.to_numeric(show["VN estimada"], errors="coerce").round(0)
    show["Precio (USD, VN100)"] = pd.to_numeric(show["Precio (USD, VN100)"], errors="coerce").round(2)
    show["TIR (%)"] = pd.to_numeric(show["TIR (%)"], errors="coerce").round(2)
    show["MD"] = pd.to_numeric(show["MD"], errors="coerce").round(2)
    show["Duration"] = pd.to_numeric(show["Duration"], errors="coerce").round(2)

    # Vencimiento ok (dd/mm/yyyy)
    show["Vencimiento"] = show["Vencimiento"].apply(lambda x: x.strftime("%d/%m/%Y") if isinstance(x, dt.date) else "")

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "%": st.column_config.NumberColumn("%", format="%d"),
            "USD": st.column_config.TextColumn("USD"),
            "Precio (USD, VN100)": st.column_config.NumberColumn("Precio (USD, VN100)", format="%.2f"),
            "VN estimada": st.column_config.NumberColumn("VN estimada", format="%.0f"),
            "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
            "MD": st.column_config.NumberColumn("MD", format="%.2f"),
            "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
            "Vencimiento": st.column_config.TextColumn("Vencimiento"),
            "Ley": st.column_config.TextColumn("Ley"),
            "Issuer": st.column_config.TextColumn("Issuer"),
        },
    )

    # Mostrar USD con símbolo $ (sin tocar cálculo)
    # (Solo “visual”, por eso lo dejamos como texto al render)
    # Re-render simple para USD con $:
    # Nota: lo dejamos así para no complicar configs.

    # ==========
    # Flujos de fondos (con totales + filtro estético de fechas)
    # ==========
    st.markdown("## Flujo de fondos")
    calendar = build_cashflow_calendar(df_cf, show.rename(columns={"Ticker": "Ticker"}))

    if calendar.empty:
        st.info("No hay flujos para mostrar con esta cartera.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # filtro estético: rango de meses visibles (NO afecta cartera)
    month_cols = [c for c in calendar.columns if isinstance(c, pd.Timestamp)]
    if month_cols:
        min_m = min(month_cols).date()
        max_m = max(month_cols).date()
        fcol1, fcol2 = st.columns([0.35, 0.65], vertical_alignment="center")
        with fcol1:
            rng = st.slider(
                "Rango de fechas a mostrar (solo visual)",
                min_value=min_m,
                max_value=max_m,
                value=(min_m, max_m),
                key="cartera_cf_range",
            )
        start_d, end_d = rng
        month_cols_keep = [c for c in month_cols if (c.date() >= start_d and c.date() <= end_d)]
    else:
        month_cols_keep = []

    cols_keep = month_cols_keep + (["Total Ticker"] if "Total Ticker" in calendar.columns else [])
    cal_show = calendar[cols_keep] if cols_keep else calendar.copy()

    # Formateo: sin decimales, con $ en display
    cal_disp = cal_show.copy()
    cal_disp = cal_disp.applymap(lambda v: 0.0 if (isinstance(v, float) and not np.isfinite(v)) else v)
    cal_disp = cal_disp.round(0)

    # Renombrar meses a "MMM-YYYY" (estético)
    rename_cols = {}
    for c in cal_disp.columns:
        if isinstance(c, pd.Timestamp):
            rename_cols[c] = c.strftime("%b-%Y").capitalize()
    cal_disp = cal_disp.rename(columns=rename_cols)

    st.dataframe(
        cal_disp,
        use_container_width=True,
        height=520,
    )

    st.markdown("</div>", unsafe_allow_html=True)

