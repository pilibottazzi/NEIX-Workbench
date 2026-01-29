# tools/bonos.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

CASHFLOW_PATH = os.path.join("data", "cashflows_completos.xlsx")

# (si querés, lo dejamos como ONs: filtro interno silencioso)
TIR_MIN = -10.0
TIR_MAX = 15.0


# =========================
# 1) XNPV / XIRR
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
# 2) Helpers cashflow
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
# 3) Ley: normalización
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
        return "Ley Local (ARG)"
    if norm == "NY":
        return "Ley NY (NY/NYC)"
    if norm == "NA":
        return "Sin Ley"
    return f"Ley {norm}"


# =========================
# 4) Load cashflows (Bonos)
# =========================
def load_cashflows_bonos(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_completos.xlsx)."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    req = {"date", "species", "law", "issuer", "flujo_total", "maturity"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    df["species"] = df["species"].astype(str).str.strip().str.upper()
    df["issuer"] = df["issuer"].astype(str).str.strip().str.upper()
    df["law"] = df["law"].astype(str).str.strip().str.upper()
    df["law_norm"] = df["law"].apply(normalize_law)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")

    df = df.dropna(subset=["species", "date", "flujo_total"]).sort_values(["species", "date"])
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("species", sort=False):
        out[str(k)] = g[["date", "flujo_total"]].copy().sort_values("date")
    return out


def build_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.groupby("species")
        .agg(
            law_norm=("law_norm", lambda s: s.value_counts().index[0]),
            issuer=("issuer", lambda s: s.value_counts().index[0]),
            maturity=("maturity", "max"),
            minPiece=("minPiece", lambda s: s.dropna().iloc[0] if "minPiece" in df.columns and s.dropna().shape[0] else np.nan),
            isin=("isin", lambda s: s.dropna().iloc[0] if "isin" in df.columns and s.dropna().shape[0] else ""),
            description=("description", lambda s: s.dropna().iloc[0] if "description" in df.columns and s.dropna().shape[0] else ""),
        )
        .reset_index()
    )
    return meta


# =========================
# 5) Precios IOL (Bonos)
# =========================
def to_float_iol(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return np.nan

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except Exception:
        return np.nan


def fetch_iol_bonos_prices() -> pd.DataFrame:
    """
    IOL suele tener una tabla de Bonos.
    Si por algún motivo cambia la url, dejé fallback.
    """
    urls = [
        "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos",
        "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos%20en%20dolares",
        "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos%20en%20pesos",
    ]

    last_err = None
    for url in urls:
        try:
            tb = pd.read_html(url)[0]
            df = pd.DataFrame(
                {
                    "Ticker": tb["Símbolo"].astype(str).str.strip().str.upper(),
                    "UltimoOperado": tb["Último Operado"].apply(to_float_iol),
                    "MontoOperado": tb.get("Monto Operado", pd.Series([0] * len(tb))).apply(to_float_iol).fillna(0),
                }
            ).dropna(subset=["UltimoOperado"])
            return df.set_index("Ticker")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"No pude leer tabla de Bonos desde IOL. Último error: {last_err}")


def pick_price_by_species(prices: pd.DataFrame, species: str) -> tuple[float, float]:
    sym = str(species).strip().upper()
    if sym in prices.index:
        px = float(prices.loc[sym, "UltimoOperado"])
        vol = float(prices.loc[sym, "MontoOperado"])
        if np.isfinite(px) and px > 1000:
            px = px / 100.0
        return px, vol
    return np.nan, np.nan


# =========================
# 6) Métricas
# =========================
def tir(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
    if not np.isfinite(precio) or precio <= 0:
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow.rename(columns={"date": "date", "flujo_total": "flujo_total"}), settlement)
    if cf.empty:
        return np.nan

    flujos = [(settlement, -float(precio))]
    for _, r in cf.iterrows():
        flujos.append((r["date"].to_pydatetime(), float(r["flujo_total"])))

    v = xirr(flujos, guess=0.10)
    return round(v, 2) if np.isfinite(v) else np.nan


def duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
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
        flujo = float(row["flujo_total"])
        pv = flujo / (1 + ytm / 100.0) ** tiempo
        denom.append(pv)
        numer.append(tiempo * pv)

    if np.sum(denom) == 0:
        return np.nan
    return round(float(np.sum(numer) / np.sum(denom)), 2)


def modified_duration(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
    dur = duration(cashflow, precio, plazo_dias=plazo_dias)
    ytm = tir(cashflow, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return round(dur / (1 + ytm / 100.0), 2)


# =========================
# 7) UI
# =========================
def _ui_css():
    st.markdown(
        """
    <style>
      .wrap{ max-width: 1400px; margin: 0 auto; }
      .title{ font-size:22px; font-weight:800; letter-spacing:.04em; color:#111827; }
      .sub{ color:rgba(17,24,39,.60); font-size:13px; margin-top:2px; }
      div[data-baseweb="tag"]{ border-radius:999px !important; }
      .block-container { padding-top: 1.2rem; }
      label { margin-bottom: 0.25rem !important; }

      .stButton > button { border-radius: 12px; padding: 0.60rem 1.0rem; }
      .stSelectbox div[data-baseweb="select"]{ border-radius: 12px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        px, vol = pick_price_by_species(prices, species)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        t = tir(cf.rename(columns={"date": "date", "flujo_total": "flujo_total"}), px, plazo_dias=plazo)

        rows.append(
            {
                "Ticker": species,
                "Ley": meta.loc[species, "law_norm"],
                "Issuer": meta.loc[species, "issuer"],
                "Precio": px,
                "TIR (%)": t,
                "MD": modified_duration(cf.rename(columns={"date": "date", "flujo_total": "flujo_total"}), px, plazo_dias=plazo),
                "Duration": duration(cf.rename(columns={"date": "date", "flujo_total": "flujo_total"}), px, plazo_dias=plazo),
                "Vencimiento": meta.loc[species, "maturity"],
                "Volumen": vol,
                "ISIN": meta.loc[species, "isin"],
                "Descripción": meta.loc[species, "description"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

    # filtro interno silencioso
    out = out[out["TIR (%)"].between(TIR_MIN, TIR_MAX, inclusive="both")].copy()
    return out


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    h1, h2 = st.columns([0.78, 0.22])
    with h1:
        st.markdown('<div class="title">NEIX · Bonos</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Tabs por ley (ARG / NY) + filtro por Issuer.</div>', unsafe_allow_html=True)

    st.divider()

    # Load cashflows
    try:
        df_cf = load_cashflows_bonos(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.info("El Excel debe tener al menos: date, species, law, issuer, flujo_total, maturity.")
        return

    # Header controls
    f1, f2, f3, f4 = st.columns([0.22, 0.18, 0.18, 0.42], vertical_alignment="bottom")

    with f1:
        plazo = st.selectbox("Plazo", [0, 1], index=0, format_func=lambda x: f"T{x}")
    with f2:
        traer_precios = st.button("Actualizar PRECIOS", use_container_width=True)
    with f3:
        calcular = st.button("Calcular", type="primary", use_container_width=True)
    with f4:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Cache precios
    if traer_precios or "bonos_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios de Bonos desde IOL..."):
            try:
                st.session_state["bonos_iol_prices"] = fetch_iol_bonos_prices()
            except Exception as e:
                st.error(f"No pude leer IOL Bonos: {e}")
                st.session_state["bonos_iol_prices"] = None

    prices = st.session_state.get("bonos_iol_prices")
    if prices is None:
        st.warning("No hay precios cargados.")
        return

    st.divider()

    tab_arg, tab_ny = st.tabs([law_label("ARG"), law_label("NY")])

    for tab, law_norm in [(tab_arg, "ARG"), (tab_ny, "NY")]:
        with tab:
            df_law = df_cf[df_cf["law_norm"] == law_norm].copy()
            if df_law.empty:
                st.info("No hay instrumentos para esta ley.")
                continue

            # Filtro issuer
            issuers = sorted(df_law["issuer"].dropna().astype(str).unique().tolist())
            issuer_sel = st.multiselect("Issuer", issuers, default=issuers, key=f"issuer_{law_norm}")

            df_law2 = df_law[df_law["issuer"].astype(str).isin(issuer_sel)].copy()
            if df_law2.empty:
                st.info("No hay bonos para esta selección de issuer.")
                continue

            tickers = sorted(df_law2["species"].unique().tolist())
            sel = st.multiselect("Ticker", tickers, default=tickers, key=f"tick_{law_norm}")

            if calcular:
                df_use = df_law2[df_law2["species"].isin(sel)].copy()
                out = _compute_table(df_use, prices, plazo)

                if out.empty:
                    st.info("No quedaron bonos con precio / filtros internos.")
                    continue

                st.markdown(f"### NEIX · Bonos · {law_label(law_norm)}")

                all_cols = ["Ticker", "Issuer", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen", "ISIN", "Descripción"]
                defaults = ["Ticker", "Issuer", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]

                cols_pick = st.multiselect(
                    "Columnas a mostrar",
                    options=all_cols,
                    default=defaults,
                    key=f"cols_{law_norm}",
                )

                if "Ticker" not in cols_pick:
                    cols_pick = ["Ticker"] + cols_pick
                if "Vencimiento" not in cols_pick:
                    cols_pick = cols_pick + ["Vencimiento"]

                show = out.copy()
                show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date

                # más alto (muchas filas)
                base = 420
                row_h = 28
                max_h = 950
                height_df = int(min(max_h, base + row_h * len(show)))

                st.dataframe(
                    show[cols_pick],
                    hide_index=True,
                    use_container_width=True,
                    height=height_df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Issuer": st.column_config.TextColumn("Issuer"),
                        "Precio": st.column_config.NumberColumn("Precio", format="%.2f"),
                        "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
                        "MD": st.column_config.NumberColumn("MD", format="%.2f"),
                        "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
                        "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
                        "Volumen": st.column_config.NumberColumn("Volumen", format="%.0f"),
                        "ISIN": st.column_config.TextColumn("ISIN"),
                        "Descripción": st.column_config.TextColumn("Descripción"),
                    },
                )
