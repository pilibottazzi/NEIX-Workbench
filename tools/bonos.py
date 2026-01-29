# tools/bonos.py
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

# ✅ SIEMPRE activo (sin toggle)
TIR_MIN = -10.0
TIR_MAX = 15.0

# ✅ Los flujos están en USD → al buscar precios en IOL usamos Ticker + "D"
# (ej: GD30 -> GD30D)
PRICE_SUFFIX = "D"

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
    # normalizo a medianoche para evitar edge por hora
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


def tab_label(norm: str) -> str:
    if norm == "ARG":
        return "Ley local (ARG)"
    if norm == "NY":
        return "Ley NY (NY/NYC)"
    return f"Ley {norm}"


def law_cell_label(norm: str) -> str:
    if norm == "ARG":
        return "ARG (Ley local)"
    if norm == "NY":
        return "NY (NY/NYC)"
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
# Load cashflows (BONOS)
# =========================
def load_cashflows_bonos(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_completos.xlsx).")

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
# Precios BONOS (IOL)
# =========================
def fetch_iol_bonos_prices() -> pd.DataFrame:
    """
    Index: Ticker
    Columns: Precio, Volumen
    """
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    bonos = pd.read_html(url)[0]

    df = pd.DataFrame()
    df["Ticker"] = bonos["Símbolo"].astype(str).str.strip().str.upper()

    df["Precio"] = (
        bonos["Último Operado"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")

    try:
        df["Volumen"] = (
            bonos["Monto Operado"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["Volumen"] = pd.to_numeric(df["Volumen"], errors="coerce").fillna(0)
    except Exception:
        df["Volumen"] = 0

    df = df.dropna(subset=["Precio"]).set_index("Ticker")

    # si IOL trae duplicados, me quedo con el de mayor volumen
    df = df.sort_values("Volumen", ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    return df


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
# UI helpers
# =========================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .top-title{ font-size: 26px; font-weight: 850; letter-spacing: .04em; color:#111827; margin-bottom: 2px;}
  .top-sub{ color: rgba(17,24,39,.60); font-size: 13px; margin-top: 0px; }

  .section-title{ font-size: 18px; font-weight: 800; color:#111827; margin: 16px 0 8px; }

  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  label { margin-bottom: .18rem !important; }

  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 700; }
  .stSelectbox div[data-baseweb="select"]{ border-radius: 14px; }
  .stMultiSelect div[data-baseweb="select"]{ border-radius: 14px; }

  div[data-baseweb="tag"]{ border-radius: 999px !important; font-weight: 650; }

  div[data-testid="stExpander"] > details {
    border-radius: 16px;
    border: 1px solid rgba(17,24,39,.10);
    background: rgba(17,24,39,.015);
  }
  div[data-testid="stExpander"] summary { font-weight: 750; }

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
    """
    Multi-select con opción "Seleccionar todo" (sin emoji).
    Si está marcada, devuelve todas las opciones reales.
    """
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


def _pick_price(prices: pd.DataFrame, species: str) -> tuple[float, float, str]:
    """
    ✅ Busca el precio usando species + 'D' (USD).
    Devuelve (precio, volumen, ticker_usado).
    """
    sp = str(species).strip().upper()

    sym_usd = f"{sp}{PRICE_SUFFIX}"
    if sym_usd in prices.index:
        return float(prices.loc[sym_usd, "Precio"]), float(prices.loc[sym_usd, "Volumen"]), sym_usd

    # Fallback (por si algún ticker viene ya con D, o IOL no lo lista)
    if sp in prices.index:
        return float(prices.loc[sp, "Precio"]), float(prices.loc[sp, "Volumen"]), sp

    return np.nan, np.nan, ""


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        px, vol, px_ticker = _pick_price(prices, species)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        t = tir(cf, px, plazo_dias=plazo)

        rows.append(
            {
                "Ticker": species,               # ticker “base” (cashflow)
                "Ticker precio": px_ticker,      # ticker usado para precio (ej GD30D)
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
    out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

    # ✅ filtro interno SIEMPRE activo
    out = out[out["TIR (%)"].between(TIR_MIN, TIR_MAX, inclusive="both")].copy()
    return out


# =========================
# Render
# =========================
def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with c1:
        st.markdown('<div class="top-title">NEIX · Bonos</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="top-sub">Rendimientos, duration y filtros por Ley / Issuer / Descripción. Precios desde IOL (USD: Ticker + D).</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if back_to_home is not None:
            if st.button("← Volver", use_container_width=True):
                back_to_home()

    st.divider()

    # Load cashflows
    df_cf = load_cashflows_bonos(CASHFLOW_PATH)

    # Controls row
    top = st.columns([0.24, 0.22, 0.22, 0.32], vertical_alignment="bottom")
    with top[0]:
        plazo = st.selectbox(
            "Plazo de liquidación",
            [1, 0],
            index=0,  # ✅ T1 default
            format_func=lambda x: f"T{x}",
            key="bonos_plazo",
        )
    with top[1]:
        traer_precios = st.button("Actualizar precios", use_container_width=True, key="bonos_refresh")
    with top[2]:
        calcular = st.button("Calcular", type="primary", use_container_width=True, key="bonos_calc")
    with top[3]:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Prices cache
    if traer_precios or "bonos_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            st.session_state["bonos_iol_prices"] = fetch_iol_bonos_prices()

    prices = st.session_state.get("bonos_iol_prices")
    if prices is None or prices.empty:
        st.warning("No hay precios cargados.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.divider()

    # Tabs por ley
    tab_arg, tab_ny = st.tabs([tab_label("ARG"), tab_label("NY")])

    for tab, law_norm in [(tab_arg, "ARG"), (tab_ny, "NY")]:
        with tab:
            df_law = df_cf[df_cf["law_norm"] == law_norm].copy()
            if df_law.empty:
                st.info("No hay instrumentos para esta ley en el cashflow.")
                continue

            # -------------------------
            # Filtros (cerrados)
            # -------------------------
            st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)

            issuers = sorted(df_law["issuer_norm"].dropna().unique().tolist())
            descs = sorted(df_law["desc_norm"].dropna().unique().tolist())
            tickers_all = sorted(df_law["species"].dropna().unique().tolist())

            with st.expander("Issuer / Descripción (desplegable)", expanded=False):
                a, b = st.columns(2)
                with a:
                    issuer_sel = _multiselect_with_all("Issuer", issuers, key=f"bonos_issuer_{law_norm}", default_all=True)
                with b:
                    desc_sel = _multiselect_with_all("Descripción", descs, key=f"bonos_desc_{law_norm}", default_all=True)

            with st.expander("Ticker (desplegable)", expanded=False):
                ticker_sel = _multiselect_with_all("Ticker", tickers_all, key=f"bonos_ticker_{law_norm}", default_all=True)

            df_use = df_law.copy()
            if issuer_sel:
                df_use = df_use[df_use["issuer_norm"].isin(issuer_sel)]
            if desc_sel:
                df_use = df_use[df_use["desc_norm"].isin(desc_sel)]
            if ticker_sel:
                df_use = df_use[df_use["species"].isin(ticker_sel)]

            # -------------------------
            # Calcular + salida
            # -------------------------
            if not calcular:
                st.info("Elegí filtros (opcional) y tocá **Calcular**.")
                continue

            out = _compute_table(df_use, prices, plazo)
            if out.empty:
                st.warning("No se encontraron bonos para esta selección (con precio y dentro del rango interno de TIR).")
                continue

            # -------------------------
            # Columnas (desplegable)
            # -------------------------
            st.markdown(f"## NEIX · Bonos · {tab_label(law_norm)}")
            st.caption("Columnas configurables. (Filtro interno de TIR siempre activo.)")

            all_cols = ["Ticker", "Issuer", "Descripción", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]
            defaults = ["Ticker", "Issuer", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen", "Descripción"]

            with st.expander("Columnas (desplegable)", expanded=False):
                cols_pick = _multiselect_with_all(
                    "Columnas a mostrar",
                    options=all_cols,
                    key=f"bonos_cols_{law_norm}",
                    default_all=False,
                )
                if not cols_pick:
                    cols_pick = defaults.copy()
                if "Ticker" not in cols_pick:
                    cols_pick = ["Ticker"] + cols_pick
                if "Vencimiento" not in cols_pick:
                    cols_pick = cols_pick + ["Vencimiento"]

            show = out.copy()
            show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date

            base = 520
            row_h = 28
            max_h = 1050
            height_df = int(min(max_h, base + row_h * len(show)))

            st.dataframe(
                show[cols_pick],
                hide_index=True,
                use_container_width=True,
                height=height_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Issuer": st.column_config.TextColumn("Issuer"),
                    "Descripción": st.column_config.TextColumn("Descripción"),
                    "Precio": st.column_config.NumberColumn("Precio", format="%.2f"),
                    "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
                    "MD": st.column_config.NumberColumn("MD", format="%.2f"),
                    "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
                    "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
                    "Volumen": st.column_config.NumberColumn("Volumen", format="%.0f"),
                },
            )

    st.markdown("</div>", unsafe_allow_html=True)
