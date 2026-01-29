# tools/ons.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

CASHFLOW_PATH = os.path.join("data", "cashflows_ON.xlsx")

# ✅ Filtro interno (SIEMPRE ACTIVO - NO se muestra en UI)
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
    return dt.datetime.today() + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df["Cupon"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Cupon"])
    df = df[df["Fecha"] > settlement].sort_values("Fecha")
    return df


def load_cashflows_from_repo(path: str) -> pd.DataFrame:
    """
    Excel requerido:
      - species
      - root_key
      - Fecha
      - FlujoTotal (o Cupon)
      - law (ARG / NYC / NY / etc.) opcional
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

    df = df.dropna(subset=["species", "root_key", "Fecha", "Cupon"]).sort_values(
        ["species", "Fecha"]
    )
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("species", sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


# ======================================================
# 3) Ley: normalización (NYC -> NY)
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
        return "Ley local (ARG)"
    if norm == "NY":
        return "Ley NY (NY/NYC)"
    if norm == "NA":
        return "Sin ley"
    return f"Ley {norm}"


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
            if np.isfinite(px) and px > 1000:
                px = px / 100.0
            return px, vol, src
    return np.nan, np.nan, ""


# ======================================================
# 5) Métricas
# ======================================================
def tir(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
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
        t = row["Fecha"].to_pydatetime()
        tiempo = (t - settlement).days / 365.0
        cupon = float(row["Cupon"])
        pv = cupon / (1 + ytm / 100.0) ** tiempo
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


# ======================================================
# 6) UI (layout prolijo + buscador + seleccionar todo)
# ======================================================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1250px; margin: 0 auto; }
  .title{ font-size:22px; font-weight:800; letter-spacing:.05em; color:#111827; }
  .sub{ color:rgba(17,24,39,.65); font-size:13px; margin-top:2px; }
  .muted{ color:rgba(17,24,39,.55); font-size:12px; }

  .block-container { padding-top: 1.2rem; }
  label { margin-bottom: 0.25rem !important; }

  /* Inputs */
  .stSelectbox div[data-baseweb="select"]{ border-radius: 12px; }
  .stMultiSelect div[data-baseweb="select"]{ border-radius: 12px; }

  /* Tags pills */
  div[data-baseweb="tag"]{ border-radius:999px !important; }

  /* Buttons */
  .stButton > button { border-radius: 14px; padding: 0.60rem 1.0rem; }

  /* Subtle separators */
  .soft-hr{ height:1px; background:rgba(17,24,39,.08); margin: 10px 0 14px; }
</style>
""",
        unsafe_allow_html=True,
    )


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        rk = meta.loc[species, "root_key"]
        law_norm = meta.loc[species, "law_norm"]
        venc = meta.loc[species, "Vencimiento"]

        px, vol, src = pick_usd_price_by_root(prices, rk)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        t = tir(cf, px, plazo_dias=plazo)

        rows.append(
            {
                "Ticker": species,
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
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)
    return out


def _select_all_multiselect(label: str, options: list[str], default: list[str], key_base: str) -> list[str]:
    """
    Multiselect con:
    - buscador (nativo del multiselect)
    - checkbox "Seleccionar todo" (sin emoji)
    """
    c1, c2 = st.columns([0.22, 0.78], vertical_alignment="bottom")
    with c1:
        sel_all = st.checkbox("Seleccionar todo", value=(len(default) == len(options)), key=f"{key_base}_all")
    with c2:
        if sel_all:
            picked = st.multiselect(label, options=options, default=options, key=f"{key_base}_ms")
        else:
            picked = st.multiselect(label, options=options, default=default, key=f"{key_base}_ms")
    return picked


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Header
    h1, h2 = st.columns([0.78, 0.22])
    with h1:
        st.markdown('<div class="title">NEIX · ONs</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Rendimientos y duration con precios desde IOL.</div>', unsafe_allow_html=True)
    with h2:
        if back_to_home is not None:
            st.button("← Volver", on_click=back_to_home, use_container_width=True)

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    # Load cashflows
    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.info("El Excel debe tener: species, root_key, Fecha y FlujoTotal/Cupon (+ law opcional).")
        return

    df_cf = df_cf.copy()
    df_cf["law_norm"] = df_cf["law"].apply(normalize_law)

    # ======================================================
    # Controles en la MISMA FILA: Ticker + Plazo + Columnas
    # (y debajo: botones actualizar precios / calcular)
    # ======================================================
    # Tabs ABAJO (como pediste): primero el panel de controles, luego tabs.
    st.markdown("### Controles")

    c_ticker, c_plazo, c_cols = st.columns([0.46, 0.18, 0.36], vertical_alignment="bottom")

    # Placeholder de selecciones por ley (se resuelve dentro de cada tab, pero mantenemos UI consistente)
    # Mostramos una ayuda arriba, y en cada tab se maneja el detalle.
    with c_plazo:
        plazo = st.selectbox(
            "Plazo de liquidación",
            [1, 0],
            index=0,  # default T1
            format_func=lambda x: f"T{x}",
            key="plazo_global",
        )

    with c_cols:
        all_cols = ["Ticker", "USD", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]
        defaults = ["Ticker", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento"]
        cols_pick = st.multiselect(
            "Columnas a mostrar",
            options=all_cols,
            default=defaults,
            key="cols_global",
        )
        if "Ticker" not in cols_pick:
            cols_pick = ["Ticker"] + cols_pick
        if "Vencimiento" not in cols_pick:
            cols_pick = cols_pick + ["Vencimiento"]

    # Botones abajo (alineados)
    b1, b2, b3 = st.columns([0.24, 0.24, 0.52], vertical_alignment="bottom")
    with b1:
        traer_precios = st.button("Actualizar precios", use_container_width=True)
    with b2:
        calcular = st.button("Calcular", type="primary", use_container_width=True)
    with b3:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Cache precios (global)
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

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    # ======================================================
    # TABS (ABAJO)
    # ======================================================
    tab_arg, tab_ny = st.tabs([law_label("ARG"), law_label("NY")])

    def _render_tab(law_norm: str):
        df_law = df_cf[df_cf["law_norm"] == law_norm].copy()
        if df_law.empty:
            st.info("No hay instrumentos para esta ley (revisá columna law en el cashflow).")
            return

        tickers = sorted(df_law["species"].unique().tolist())

        # ticker selector (con buscar + seleccionar todo)
        # lo ponemos dentro del tab pero visualmente “manda” el control de arriba
        # (no hay forma nativa de que el control viva fuera del tab y afecte dentro sin duplicar)
        sel = _select_all_multiselect(
            "Ticker (selección)",
            options=tickers,
            default=tickers,
            key_base=f"tick_{law_norm}",
        )

        if not calcular:
            st.info("Elegí tickers (podés buscar) y tocá **Calcular**.")
            return

        df_use = df_law[df_law["species"].isin(sel)].copy()
        out = _compute_table(df_use, prices, plazo)

        if out.empty:
            st.warning("No hay ONs con precio para esta selección.")
            return

        # ✅ filtro interno TIR (siempre)
        out = out[out["TIR (%)"].between(TIR_MIN, TIR_MAX, inclusive="both")].copy()
        if out.empty:
            st.warning("No quedaron ONs tras aplicar filtros internos.")
            return

        show = out.copy()
        show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date

        st.markdown(f"### NEIX · ONs · {law_label(law_norm)}")

        # altura dinámica
        base = 420
        row_h = 28
        max_h = 950
        height_df = int(min(max_h, base + row_h * len(show)))

        # Centrados: TODO menos Ticker y Vencimiento
        # (Streamlit no ofrece align directo por columna en dataframe;
        #  lo resolvemos vía column_config y CSS, pero el centrado real depende del renderer.
        #  Aun así, lo dejamos lo más cercano posible con widths consistentes.)
        st.dataframe(
            show[cols_pick],
            hide_index=True,
            use_container_width=True,
            height=height_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "USD": st.column_config.TextColumn("USD (D/C)", width="small"),
                "Precio USD": st.column_config.NumberColumn("Precio USD", format="%.2f", width="small"),
                "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f", width="small"),
                "MD": st.column_config.NumberColumn("MD", format="%.2f", width="small"),
                "Duration": st.column_config.NumberColumn("Duration", format="%.2f", width="small"),
                "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY", width="small"),
                "Volumen": st.column_config.NumberColumn("Volumen", format="%.0f", width="small"),
            },
        )

    with tab_arg:
        _render_tab("ARG")
    with tab_ny:
        _render_tab("NY")

    st.markdown("</div>", unsafe_allow_html=True)

