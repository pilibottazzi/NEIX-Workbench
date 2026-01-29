# tools/ons.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

CASHFLOW_PATH = os.path.join("data", "cashflows_ON.xlsx")

# (opcional) filtro interno de TIR
DEFAULT_FILTRAR_TIR = False
TIR_MIN = -10.0
TIR_MAX = 15.0


# ======================================================
# 1) XNPV / XIRR
# ======================================================
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


# ======================================================
# 2) Cashflows
# ======================================================
def _settlement(plazo_dias: int) -> dt.datetime:
    # normalizo a medianoche para evitar efectos raros por horas
    base = pd.Timestamp.today().normalize().to_pydatetime()
    return base + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Cupon"])
    df = df[df["Fecha"] > settlement].sort_values("Fecha")
    return df


def load_cashflows_from_repo(path: str) -> pd.DataFrame:
    """
    Excel esperado:
      - species
      - root_key
      - Fecha
      - FlujoTotal (o Cupon)
      - law (opcional)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_ON.xlsx)."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    req_min = {"species", "root_key", "Fecha"}
    missing = req_min - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {path}: {sorted(missing)} (mínimas: {sorted(req_min)})"
        )

    if "FlujoTotal" in df.columns:
        flujo_col = "FlujoTotal"
    elif "Cupon" in df.columns:
        flujo_col = "Cupon"
    else:
        raise ValueError("Falta columna 'FlujoTotal' (recomendado) o 'Cupon' en el cashflow.")

    if "law" not in df.columns:
        df["law"] = "NA"

    df["species"] = df["species"].astype(str).str.strip().str.upper()
    df["root_key"] = df["root_key"].astype(str).str.strip().str.upper()
    df["law"] = df["law"].astype(str).str.strip().str.upper()

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df[flujo_col], errors="coerce")

    df = df.dropna(subset=["species", "root_key", "Fecha", "Cupon"]).sort_values(["species", "Fecha"])
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("species", sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


# ======================================================
# 3) Ley: normalización (ARG / NY)
# ======================================================
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


def build_species_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.groupby("species")
        .agg(
            root_key=("root_key", lambda s: s.value_counts().index[0]),
            law_norm=("law_norm", lambda s: s.value_counts().index[0]),
            Vencimiento=("Fecha", "max"),
        )
        .reset_index()
    )
    return meta


# ======================================================
# 4) Precios IOL
# ======================================================
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


def fetch_iol_on_prices() -> pd.DataFrame:
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"
    on = pd.read_html(url)[0]

    df = pd.DataFrame(
        {
            "Ticker": on["Símbolo"].astype(str).str.strip().str.upper(),
            "UltimoOperado": on["Último Operado"].apply(to_float_iol),
            "MontoOperado": on.get("Monto Operado", pd.Series([0] * len(on))).apply(to_float_iol).fillna(0),
        }
    ).dropna(subset=["UltimoOperado"])

    return df.set_index("Ticker")


def pick_usd_price_by_root(prices: pd.DataFrame, root_key: str) -> tuple[float, float, str]:
    rk = str(root_key).strip().upper()
    for sym, src in [(f"{rk}D", "D"), (f"{rk}C", "C")]:
        if sym in prices.index:
            px = float(prices.loc[sym, "UltimoOperado"])
            vol = float(prices.loc[sym, "MontoOperado"])
            # si viene en centavos (a veces) lo normalizamos
            if np.isfinite(px) and px > 1000:
                px = px / 100.0
            return px, vol, src
    return np.nan, np.nan, ""


# ======================================================
# 5) Métricas
# ======================================================
def tir(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    if not np.isfinite(precio) or precio <= 0:
        return np.nan

    settlement = _settlement(plazo_dias)
    cf = _future_cashflows(cashflow, settlement)
    if cf.empty:
        return np.nan

    flujos = [(settlement, -float(precio))]
    for _, r in cf.iterrows():
        flujos.append((r["Fecha"].to_pydatetime(), float(r["Cupon"])))

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
        t = row["Fecha"].to_pydatetime()
        tiempo = (t - settlement).days / 365.0
        cupon = float(row["Cupon"])
        pv = cupon / (1 + ytm / 100.0) ** tiempo
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


# ======================================================
# 6) UI helpers
# ======================================================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .top-title{ font-size: 24px; font-weight: 850; letter-spacing: .04em; color:#111827; margin-bottom: 2px;}
  .top-sub{ color: rgba(17,24,39,.60); font-size: 13px; margin-top: 0px; }

  .section-title{ font-size: 18px; font-weight: 800; color:#111827; margin: 12px 0 6px; }

  .block-container { padding-top: 1.10rem; padding-bottom: 2.2rem; }
  label { margin-bottom: .20rem !important; }

  .stButton > button { border-radius: 14px; padding: .68rem 1.0rem; font-weight: 700; }
  .stSelectbox div[data-baseweb="select"]{ border-radius: 14px; }
  div[data-baseweb="tag"]{ border-radius: 999px !important; }

  div[data-testid="stExpander"] > details { border-radius: 16px; }
  div[data-testid="stExpander"] summary { font-weight: 700; }

  div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def _multiselect_with_all(label: str, options: list[str], key: str, default_all: bool = True) -> list[str]:
    if not options:
        return []
    options_sorted = sorted(set([str(x) for x in options if str(x).strip() != ""]))
    all_token = "✅ Seleccionar todo"
    opts = [all_token] + options_sorted
    default = [all_token] if default_all else []
    sel = st.multiselect(label, options=opts, default=default, key=key)
    if all_token in sel:
        return options_sorted
    return sel


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> tuple[pd.DataFrame, dict]:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    sin_precio = []
    sin_cf = 0
    for species in meta.index:
        rk = meta.loc[species, "root_key"]
        law_norm = meta.loc[species, "law_norm"]
        venc = meta.loc[species, "Vencimiento"]

        px, vol, src = pick_usd_price_by_root(prices, rk)
        if not np.isfinite(px) or px <= 0:
            sin_precio.append((species, rk))
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            sin_cf += 1
            continue

        t = tir(cf, px, plazo_dias=plazo)

        rows.append(
            {
                "Ticker": species,
                "Ley": law_norm,
                "USD": src,
                "Precio USD": px,
                "TIR (%)": t,
                "MD": modified_duration(cf, px, plazo_dias=plazo),
                "Duration": duration(cf, px, plazo_dias=plazo),
                "Vencimiento": venc,
                "Volumen": vol,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
        out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)

    diag = {
        "tickers_seleccion": int(len(meta.index)),
        "con_precio": int(len(out)),
        "sin_precio": int(len(sin_precio)),
        "sin_cashflow": int(sin_cf),
        "ejemplos_sin_precio": sin_precio[:20],
        "tir_nan": int(out["TIR (%)"].isna().sum()) if not out.empty else 0,
    }
    return out, diag


# ======================================================
# 7) Render
# ======================================================
def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Header (mismo look que Bonos)
    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with c1:
        st.markdown('<div class="top-title">NEIX · ONs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="top-sub">Rendimientos y duration con filtros por Ley. Precios desde IOL.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if back_to_home is not None:
            if st.button("← Volver", use_container_width=True):
                back_to_home()

    st.divider()

    # Load cashflows
    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.info("El Excel debe tener: species, root_key, Fecha y FlujoTotal/Cupon (+ law opcional).")
        return

    df_cf = df_cf.copy()
    df_cf["law_norm"] = df_cf["law"].apply(normalize_law)

    # Controles superiores (✅ T1 default)
    f1, f2, f3, f4 = st.columns([0.20, 0.22, 0.20, 0.38], vertical_alignment="bottom")
    with f1:
        plazo = st.selectbox("Plazo de liquidación", [1, 0], index=0, format_func=lambda x: f"T{x}")
    with f2:
        traer_precios = st.button("Actualizar PRECIOS", use_container_width=True)
    with f3:
        calcular = st.button("Calcular", type="primary", use_container_width=True)
    with f4:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Cache precios
    if traer_precios or "ons_iol_prices" not in st.session_state:
        with st.spinner("Leyendo precios desde IOL..."):
            try:
                st.session_state["ons_iol_prices"] = fetch_iol_on_prices()
            except Exception as e:
                st.error(f"No pude leer IOL: {e}")
                st.session_state["ons_iol_prices"] = None

    prices = st.session_state.get("ons_iol_prices")
    if prices is None:
        st.warning("No hay precios cargados.")
        return

    st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)

    # Filtro por ley (sin tabs)
    law_opts = sorted(df_cf["law_norm"].dropna().unique().tolist())
    law_labels = [law_label(x) for x in law_opts]
    inv = {law_label(x): x for x in law_opts}

    with st.expander("Ley (desplegable)", expanded=False):
        ley_sel_labels = _multiselect_with_all("Ley", law_labels, key="ons_law", default_all=True)
        ley_sel = [inv[x] for x in ley_sel_labels] if ley_sel_labels else []

    with st.expander("Ticker (desplegable)", expanded=False):
        tickers_all = sorted(df_cf["species"].dropna().unique().tolist())
        ticker_sel = _multiselect_with_all("Ticker", tickers_all, key="ons_ticker", default_all=True)

    with st.expander("Ajustes avanzados (opcional)", expanded=False):
        filtrar_tir = st.checkbox(
            "Aplicar filtro interno de TIR (oculta instrumentos fuera de rango)",
            value=DEFAULT_FILTRAR_TIR,
        )
        if filtrar_tir:
            st.caption(f"Rango interno: {TIR_MIN} a {TIR_MAX}")

    # Aplicar filtros al DF
    df_use = df_cf.copy()
    if ley_sel:
        df_use = df_use[df_use["law_norm"].isin(ley_sel)]
    if ticker_sel:
        df_use = df_use[df_use["species"].isin(ticker_sel)]

    if not calcular:
        st.info("Elegí filtros (opcional) y tocá **Calcular**.")
        return

    out, diag = _compute_table(df_use, prices, plazo)

    # Diagnóstico (oculto)
    with st.expander("Diagnóstico (si falta info)", expanded=False):
        st.write(
            {
                "Tickers (selección)": diag["tickers_seleccion"],
                "Con precio (D/C)": diag["con_precio"],
                "Sin precio (no match root_keyD/root_keyC)": diag["sin_precio"],
                "Sin cashflow": diag["sin_cashflow"],
                "TIR NaN": diag["tir_nan"],
            }
        )
        if diag["ejemplos_sin_precio"]:
            st.write("Ejemplos sin precio (Ticker, root_key):")
            st.dataframe(pd.DataFrame(diag["ejemplos_sin_precio"], columns=["Ticker", "root_key"]), hide_index=True)

    if out.empty:
        st.warning("No se encontraron ONs con precio para la selección.")
        return

    # filtro interno TIR (opcional)
    if filtrar_tir:
        out = out[out["TIR (%)"].between(TIR_MIN, TIR_MAX, inclusive="both")].copy()
        if out.empty:
            st.info("No quedaron ONs tras aplicar filtros internos de TIR.")
            return

    # Tabla
    st.markdown("## NEIX · ONs")
    st.caption("Columnas configurables.")

    all_cols = ["Ticker", "Ley", "USD", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]
    defaults = ["Ticker", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]

    cols_pick = st.multiselect("Columnas a mostrar", options=all_cols, default=defaults, key="ons_cols")
    if "Ticker" not in cols_pick:
        cols_pick = ["Ticker"] + cols_pick
    if "Vencimiento" not in cols_pick:
        cols_pick = cols_pick + ["Vencimiento"]

    show = out.copy()
    show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date
    show["Ley"] = show["Ley"].apply(law_label)

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
            "USD": st.column_config.TextColumn("USD (D/C)"),
            "Precio USD": st.column_config.NumberColumn("Precio USD", format="%.2f"),
            "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
            "MD": st.column_config.NumberColumn("MD", format="%.2f"),
            "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
            "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
            "Volumen": st.column_config.NumberColumn("Volumen", format="%.0f"),
        },
    )

    st.markdown("</div>", unsafe_allow_html=True)
