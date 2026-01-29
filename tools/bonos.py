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

# ✅ Filtro interno (NO se muestra en UI)
TIR_MIN = -10.0
TIR_MAX = 15.0


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
    return dt.datetime.today() + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])
    df = df[df["date"] > settlement].sort_values("date")
    return df


# =========================
# Normalizaciones (ley/issuer/desc)
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


def law_label(norm: str) -> str:
    if norm == "ARG":
        return "ARG (Ley Local)"
    if norm == "NY":
        return "NY (NY/NYC)"
    if norm == "NA":
        return "Sin Ley"
    return norm


def normalize_issuer(x: str) -> str:
    s = (x or "").strip().upper()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    # Si querés mapear variantes, agregalo acá
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

    # columnas esperadas (según tu excel)
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
    Precios de BONOS desde IOL (tabla completa).
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

    df = df.dropna(subset=["Precio"])
    df = df.set_index("Ticker")

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
# UI
# =========================
def _ui_css():
    st.markdown(
        """
<style>
  /* Layout general */
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .top-title{ font-size: 24px; font-weight: 850; letter-spacing: .04em; color:#111827; margin-bottom: 2px;}
  .top-sub{ color: rgba(17,24,39,.60); font-size: 13px; margin-top: 0px; }
  .section-title{ font-size: 18px; font-weight: 800; color:#111827; margin: 12px 0 6px; }

  /* Compactar paddings arriba */
  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  label { margin-bottom: .20rem !important; }
  .stCaption { margin-top: 0px !important; }

  /* Inputs / botones */
  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 700; }
  .stSelectbox div[data-baseweb="select"]{ border-radius: 14px; }
  div[data-baseweb="tag"]{ border-radius: 999px !important; }

  /* Expander prolijo */
  div[data-testid="stExpander"] > details { border-radius: 16px; }
  div[data-testid="stExpander"] summary { font-weight: 700; }

  /* Dataframe */
  div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        if species not in prices.index:
            continue

        px = float(prices.loc[species, "Precio"])
        vol = float(prices.loc[species, "Volumen"])

        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        t = tir(cf, px, plazo_dias=plazo)

        rows.append(
            {
                "Ticker": species,
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
    return out


def _multiselect_with_all(label: str, options: list[str], default_all: bool = True, key: str = "") -> list[str]:
    """
    Multiselect con pseudo-opción "Seleccionar todo" (como en tu captura).
    """
    if not options:
        return []

    options_sorted = sorted(set([str(x) for x in options if str(x).strip() != ""]))
    all_token = "✅ Seleccionar todo"
    opts = [all_token] + options_sorted

    default = [all_token] if default_all else []

    selected = st.multiselect(label, options=opts, default=default, key=key)

    if all_token in selected:
        # Si está "seleccionar todo", devolvemos todas menos el token
        return options_sorted

    return selected


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # ========= Header =========
    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with c1:
        st.markdown('<div class="top-title">NEIX · Bonos</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="top-sub">Rendimientos, duration y filtros por Ley / Issuer / Descripción. Precios desde IOL. </div>',
            unsafe_allow_html=True,
        )
    with c2:
        if back_to_home is not None:
            if st.button("← Volver", use_container_width=True):
                back_to_home()

    st.divider()

    # ========= Load cashflows =========
    try:
        df_cf = load_cashflows_bonos(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.info("El Excel debe tener: date, species, description, law, issuer y flujo_total.")
        return

    # ========= Top controls =========
    f1, f2, f3, f4 = st.columns([0.20, 0.22, 0.20, 0.38], vertical_alignment="bottom")

    with f1:
        plazo = st.selectbox("Plazo de liquidación", [1, 0], index=0, format_func=lambda x: f"T{x}")  # ✅ T1 default
    with f2:
        traer_precios = st.button("Actualizar PRECIOS", use_container_width=True)
    with f3:
        calcular = st.button("Calcular", type="primary", use_container_width=True)
    with f4:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # ========= Prices cache =========
    if traer_precios or "bonos_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            try:
                st.session_state["bonos_iol_prices"] = fetch_iol_bonos_prices()
            except Exception as e:
                st.error(f"No pude leer IOL: {e}")
                st.session_state["bonos_iol_prices"] = None

    prices = st.session_state.get("bonos_iol_prices")
    if prices is None:
        st.warning("No hay precios cargados.")
        return

    # ========= Filtros (cerrados) =========
    st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)

    # Opciones dependen del universo del cashflow (no del cálculo)
    laws = [law_label(x) for x in sorted(df_cf["law_norm"].dropna().unique().tolist())]
    # Guardamos mapping label->norm para filtrar
    law_map = {law_label(x): x for x in sorted(df_cf["law_norm"].dropna().unique().tolist())}

    issuers = sorted(df_cf["issuer_norm"].dropna().unique().tolist())
    descs = sorted(df_cf["desc_norm"].dropna().unique().tolist())
    tickers_all = sorted(df_cf["species"].dropna().unique().tolist())

    with st.expander("Ley / Issuer / Descripción (desplegable)", expanded=False):
        a, b, c = st.columns(3)
        with a:
            law_sel_labels = _multiselect_with_all(
                "Ley",
                options=laws,
                default_all=True,
                key="bonos_f_law",
            )
            law_sel = [law_map[x] for x in law_sel_labels] if law_sel_labels else []
        with b:
            issuer_sel = _multiselect_with_all(
                "Issuer",
                options=issuers,
                default_all=True,
                key="bonos_f_issuer",
            )
        with c:
            desc_sel = _multiselect_with_all(
                "Descripción",
                options=descs,
                default_all=True,
                key="bonos_f_desc",
            )

    with st.expander("Ticker (desplegable)", expanded=False):
        ticker_sel = _multiselect_with_all(
            "Ticker",
            options=tickers_all,
            default_all=True,
            key="bonos_f_ticker",
        )

    # ========= Aplicar filtros al cashflow (para acotar universo real) =========
    df_use = df_cf.copy()

    if law_sel:
        df_use = df_use[df_use["law_norm"].isin(law_sel)]
    if issuer_sel:
        df_use = df_use[df_use["issuer_norm"].isin(issuer_sel)]
    if desc_sel:
        df_use = df_use[df_use["desc_norm"].isin(desc_sel)]
    if ticker_sel:
        df_use = df_use[df_use["species"].isin(ticker_sel)]

    # ========= Calcular =========
    if not calcular:
        st.info("Elegí filtros (opcional) y tocá **Calcular**.")
        return

    out = _compute_table(df_use, prices, plazo)

    if out.empty:
        # diagnostico simple para que no te vuelvas loca
        inter = len(set(df_use["species"].unique()) & set(prices.index))
        st.warning("No se encontraron bonos con precio para la selección.")
        st.caption(f"Tickers en selección (cashflow): {df_use['species'].nunique()} · Tickers con precio en IOL: {inter}")
        return

    # ✅ filtro interno TIR
    out = out[out["TIR (%)"].between(TIR_MIN, TIR_MAX, inclusive="both")].copy()
    if out.empty:
        st.info("No quedaron bonos tras aplicar filtros internos.")
        return

    # ========= Tabla =========
    st.markdown("## NEIX · Bonos")
    st.caption("Columnas configurables. (Filtro interno de TIR activo, no visible.)")

    all_cols = ["Ticker", "Ley", "Issuer", "Descripción", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]
    defaults = ["Ticker", "Issuer", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen", "Descripción"]

    cols_pick = st.multiselect(
        "Columnas a mostrar",
        options=all_cols,
        default=defaults,
        key="bonos_cols",
    )

    if "Ticker" not in cols_pick:
        cols_pick = ["Ticker"] + cols_pick
    if "Vencimiento" not in cols_pick:
        cols_pick = cols_pick + ["Vencimiento"]

    show = out.copy()
    show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date
    show["Ley"] = show["Ley"].apply(law_label)

    # ✅ más alto (más filas visibles)
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
            "Ley": st.column_config.TextColumn("Ley"),
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
