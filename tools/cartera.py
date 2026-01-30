from __future__ import annotations

import os
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

# =========================
# Config
# =========================
CASHFLOW_PATH = os.path.join("data", "cashflows_completos.xlsx")

# TIR fija (no UI)
TIR_MIN = -15.0
TIR_MAX = 20.0

# Precios USD MEP
PRICE_SUFFIX = "D"

# Excepciones PESOS -> USD (cuando no es solo + "D")
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
    # si algún día aparecen así en cashflows
    "BPA7": "BPA7D",
    "BPB7": "BPB7D",
    "BPC7": "BPC7D",
    "BPA8": "BPA8D",
    "BPB8": "BPB8D",
}

# =========================
# Utils parse num AR
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
    Fix para tickers USD terminados en D.
    Caso típico: viene "6097" y era 60.97 (VN100).
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


# =========================
# XNPV / XIRR
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
# Cashflows helpers
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
# Normalizaciones meta
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
# Precios de mercado (HTML table)
# =========================
def fetch_market_prices() -> pd.DataFrame:
    """
    Devuelve DataFrame indexado por Ticker con columnas:
      - Precio
      - Volumen

    Nota: NO mencionamos el proveedor en UI.
    """
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"

    try:
        tables = pd.read_html(url)  # intenta parser default (normalmente lxml)
    except ImportError as e:
        # Si falla por parsers, damos mensaje accionable
        raise ImportError(
            "Faltan dependencias para leer tablas HTML. "
            "En tu requirements.txt agregá: lxml y html5lib."
        ) from e
    except Exception as e:
        raise RuntimeError(f"No pude leer la tabla de precios. Error: {e}") from e

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


# =========================
# Ticker USD resolver
# =========================
def resolve_usd_ticker(species: str) -> str:
    sp = str(species).strip().upper()
    if sp.endswith("D"):
        return sp
    if sp in PESOS_TO_USD_OVERRIDES:
        return PESOS_TO_USD_OVERRIDES[sp]
    return f"{sp}{PRICE_SUFFIX}"


def pick_price_usd(prices: pd.DataFrame, species: str) -> tuple[float, float, str]:
    usd_ticker = resolve_usd_ticker(species)
    if usd_ticker in prices.index:
        px = float(prices.loc[usd_ticker, "Precio"])
        vol = float(prices.loc[usd_ticker, "Volumen"])
        return px, vol, usd_ticker
    return np.nan, np.nan, ""


# =========================
# Métricas por instrumento
# =========================
def calc_tir(cf: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    if not np.isfinite(precio) or precio <= 0:
        return np.nan

    settlement = _settlement(plazo_dias)
    fut = _future_cashflows(cf, settlement)
    if fut.empty:
        return np.nan

    flujos = [(settlement, -float(precio))]
    for _, r in fut.iterrows():
        flujos.append((r["date"].to_pydatetime(), float(r["flujo_total"])))

    v = xirr(flujos, guess=0.10)
    return v


def calc_duration(cf: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    ytm = calc_tir(cf, precio, plazo_dias=plazo_dias)
    if not np.isfinite(ytm):
        return np.nan

    settlement = _settlement(plazo_dias)
    fut = _future_cashflows(cf, settlement)
    if fut.empty:
        return np.nan

    denom = 0.0
    numer = 0.0
    for _, row in fut.iterrows():
        t = row["date"].to_pydatetime()
        tiempo = (t - settlement).days / 365.0
        monto = float(row["flujo_total"])
        pv = monto / (1 + ytm / 100.0) ** tiempo
        denom += pv
        numer += tiempo * pv

    if denom == 0:
        return np.nan
    return numer / denom


def calc_md(cf: pd.DataFrame, precio: float, plazo_dias: int = 1) -> float:
    dur = calc_duration(cf, precio, plazo_dias=plazo_dias)
    ytm = calc_tir(cf, precio, plazo_dias=plazo_dias)
    if not np.isfinite(dur) or not np.isfinite(ytm):
        return np.nan
    return dur / (1 + ytm / 100.0)


# =========================
# Cartera
# =========================
@dataclass
class AssetRow:
    ticker: str
    pct: float
    usd: float
    price: float
    vn: float
    tir: float
    md: float
    dur: float
    venc: dt.date | None
    ley: str
    issuer: str


def fmt_money_int(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"$ {x:,.0f}".replace(",", ".")


def fmt_num_2(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.2f}"


def fmt_pct_2(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.2f}%"


def build_eligible_universe(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int = 1) -> pd.DataFrame:
    """
    Universe elegible: tiene precio USD y TIR dentro del rango fijo.
    """
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for sp in meta.index:
        px, vol, px_ticker = pick_price_usd(prices, sp)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(sp)
        if cf is None or cf.empty:
            continue

        # tiene que tener flujos futuros
        settlement = _settlement(plazo)
        fut = _future_cashflows(cf, settlement)
        if fut.empty:
            continue

        y = calc_tir(cf, px, plazo_dias=plazo)
        if not np.isfinite(y):
            continue
        if not (TIR_MIN <= y <= TIR_MAX):
            continue

        rows.append(
            {
                "Ticker": sp,
                "Ley": meta.loc[sp, "law_norm"],
                "Issuer": meta.loc[sp, "issuer_norm"],
                "Descripción": meta.loc[sp, "desc_norm"],
                "Vencimiento": meta.loc[sp, "vencimiento"],
                "Precio (USD, VN100)": float(px),
                "Ticker precio": px_ticker,
                "Volumen": float(vol),
                "TIR (%)": float(y),
                "MD": float(calc_md(cf, px, plazo_dias=plazo)),
                "Duration": float(calc_duration(cf, px, plazo_dias=plazo)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)
    return out


def build_portfolio_table(
    df_cf: pd.DataFrame,
    prices: pd.DataFrame,
    selected: list[str],
    pct_map: dict[str, float],
    capital_usd: float,
    plazo: int = 1,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """
    Retorna:
      - tabla cartera (bonita)
      - resumen (tir/md/dur ponderadas)
      - tabla flujos (pivot con totales)
    """
    df_cf = df_cf.copy()
    df_cf["species"] = df_cf["species"].astype(str).str.upper().str.strip()
    selected = [str(x).upper().strip() for x in selected if str(x).strip()]

    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    # normalizar % (si no suma 100, igual escala por suma)
    pcts = np.array([max(0.0, float(pct_map.get(t, 0.0))) for t in selected], dtype=float)
    s = float(np.sum(pcts))
    if s <= 0:
        pcts = np.zeros_like(pcts)
    else:
        pcts = pcts / s * 100.0

    assets: list[AssetRow] = []
    for t, pct in zip(selected, pcts):
        if pct <= 0:
            continue

        px, _, _ = pick_price_usd(prices, t)
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(t)
        if cf is None or cf.empty:
            continue

        # VN estimada = USD / (Precio/100) asumiendo precio por VN100
        usd_amt = capital_usd * (pct / 100.0)
        vn = usd_amt / (px / 100.0) if px > 0 else np.nan

        y = calc_tir(cf, px, plazo_dias=plazo)
        md = calc_md(cf, px, plazo_dias=plazo)
        dur = calc_duration(cf, px, plazo_dias=plazo)

        venc = None
        if t in meta.index:
            vv = pd.to_datetime(meta.loc[t, "vencimiento"], errors="coerce")
            venc = vv.date() if pd.notna(vv) else None

        ley = meta.loc[t, "law_norm"] if t in meta.index else "NA"
        issuer = meta.loc[t, "issuer_norm"] if t in meta.index else "NA"

        assets.append(
            AssetRow(
                ticker=t,
                pct=float(pct),
                usd=float(usd_amt),
                price=float(px),
                vn=float(vn),
                tir=float(y) if np.isfinite(y) else np.nan,
                md=float(md) if np.isfinite(md) else np.nan,
                dur=float(dur) if np.isfinite(dur) else np.nan,
                venc=venc,
                ley=str(ley),
                issuer=str(issuer),
            )
        )

    if not assets:
        return pd.DataFrame(), {"tir": np.nan, "md": np.nan, "dur": np.nan}, pd.DataFrame()

    # Resumen ponderado por USD asignado
    w = np.array([a.usd for a in assets], dtype=float)
    wsum = float(np.sum(w)) if float(np.sum(w)) > 0 else 1.0

    tir_total = float(np.nansum([a.tir * a.usd for a in assets]) / wsum)
    md_total = float(np.nansum([a.md * a.usd for a in assets]) / wsum)
    dur_total = float(np.nansum([a.dur * a.usd for a in assets]) / wsum)

    resumen = {"tir": tir_total, "md": md_total, "dur": dur_total}

    # Tabla cartera
    df = pd.DataFrame(
        {
            "Ticker": [a.ticker for a in assets],
            "%": [a.pct for a in assets],
            "USD": [a.usd for a in assets],
            "Precio (USD, VN100)": [a.price for a in assets],
            "VN estimada": [a.vn for a in assets],
            "TIR (%)": [a.tir for a in assets],
            "MD": [a.md for a in assets],
            "Duration": [a.dur for a in assets],
            "Vencimiento": [a.venc for a in assets],
            "Ley": [law_cell_label(a.ley) for a in assets],
            "Issuer": [a.issuer for a in assets],
        }
    )

    # Flujos por mes (tabla calendario) + totales
    settlement = _settlement(plazo)
    flow_rows = []
    for a in assets:
        cf = cashflows.get(a.ticker)
        if cf is None or cf.empty:
            continue
        fut = _future_cashflows(cf, settlement)
        if fut.empty:
            continue

        # escalamos flujo_total (VN100 base) a VN estimada:
        # flujo_total está por VN100 -> para VN estimada:
        # factor = VN / 100
        factor = a.vn / 100.0 if np.isfinite(a.vn) else np.nan
        for _, r in fut.iterrows():
            flow_rows.append(
                {
                    "Ticker": a.ticker,
                    "Fecha": pd.to_datetime(r["date"]).date(),
                    "Monto": float(r["flujo_total"]) * float(factor),
                }
            )

    flows = pd.DataFrame(flow_rows)
    if flows.empty:
        flows_pivot = pd.DataFrame()
    else:
        flows["Mes"] = pd.to_datetime(flows["Fecha"]).dt.to_period("M").dt.to_timestamp()
        flows_pivot = (
            flows.pivot_table(index="Ticker", columns="Mes", values="Monto", aggfunc="sum", fill_value=0.0)
            .sort_index(axis=1)
        )

        # Totales fila/columna
        flows_pivot["Total Ticker"] = flows_pivot.sum(axis=1)
        totals_row = pd.DataFrame([flows_pivot.sum(axis=0)], index=["Totales"])
        flows_pivot = pd.concat([flows_pivot, totals_row], axis=0)

    return df, resumen, flows_pivot


# =========================
# UI
# =========================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1180px; margin: 0 auto; }
  .block-container { padding-top: 0.9rem; padding-bottom: 1.6rem; }

  /* Títulos: más empresa (más chicos y con aire) */
   .wrap h2 { font-size: 1.55rem !important; margin-bottom: .15rem !important; }
   .wrap h3 { font-size: 1.15rem !important; margin-top: 1.0rem !important; }


  /* Chips */
  div[data-baseweb="tag"]{
    background: rgba(17,24,39,.06) !important;
    color:#111827 !important;
    border: 1px solid rgba(17,24,39,.10) !important;
    border-radius: 999px !important;
    font-weight: 650 !important;
  }

  /* Dataframes */
  div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }

  /* KPIs: más minimal + tipografía más chica */
  .kpi{
    border: 1px solid rgba(17,24,39,.10);
    border-radius: 16px;
    padding: 12px 14px;
    background: white;
  }
  .kpi .lbl{ color: rgba(17,24,39,.60); font-size: 12px; margin-bottom: 6px; }
  .kpi .val{ font-size: 24px; font-weight: 800; color:#111827; letter-spacing: .01em; }
</style>
""",
        unsafe_allow_html=True,
    )



def _height_for_rows(n: int, base: int = 220, row_h: int = 28, max_h: int = 520) -> int:
    return int(min(max_h, base + row_h * max(1, n)))


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    # ===== Header: título + botón en la misma línea
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.markdown("## NEIX · Cartera Comercial")
        st.caption("Arma tu cartera con precios online.")
    with right:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_refresh")

    st.divider()

    # =====: cargar cashflows
    try:
        df_cf = load_cashflows(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # cache precios
    if refresh or "cartera_prices" not in st.session_state:
        with st.spinner("Actualizando precios..."):
            try:
                st.session_state["cartera_prices"] = fetch_market_prices()
            except Exception as e:
                st.error(str(e))
                st.markdown("</div>", unsafe_allow_html=True)
                return

    prices = st.session_state.get("cartera_prices")
    if prices is None or prices.empty:
        st.warning("No pude cargar precios de mercado (tabla vacía o cambió el formato).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Universe elegible (con TIR en rango fijo)
    universe = build_eligible_universe(df_cf, prices, plazo=1)
    if universe.empty:
        st.warning("No hay activos elegibles con TIR dentro del rango y precio disponible.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ===== Inputs: selección + capital + asignación (%)
    st.markdown("### Selección de activos")

    opts = universe["Ticker"].tolist()
    selected = st.multiselect(
        "Activos (bonos + ONs)",
        options=opts,
        default=opts[:6] if len(opts) >= 6 else opts,
        key="cartera_selected",
    )

    c1, c2 = st.columns([0.42, 0.58], vertical_alignment="bottom")
    with c1:
        capital = st.number_input(
            "Capital (USD)",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            format="%.0f",
            key="cartera_capital",
        )
    with c2:
        calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_calc")
    
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("### Asignación por activo")
    st.caption("Editá la columna %. Ideal: que sume 100% (si no, escala automáticamente).")

    if not selected:
        st.info("Seleccioná al menos un activo.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # tabla editable de %
    default_pct = round(100.0 / len(selected), 2) if selected else 0.0
    df_pct = pd.DataFrame({"Ticker": selected, "%": [default_pct] * len(selected)})

    edited = st.data_editor(
        df_pct,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "%": st.column_config.NumberColumn("%", min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
        },
        key="cartera_pct_editor",
    )

    pct_map = {r["Ticker"]: float(r["%"]) for _, r in edited.iterrows()}

    st.divider()

    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ===== Cálculo
    cartera_df, resumen, flows_pivot = build_portfolio_table(
        df_cf=df_cf,
        prices=prices,
        selected=selected,
        pct_map=pct_map,
        capital_usd=float(capital),
        plazo=1,  # ✅ siempre T1
    )

    if cartera_df.empty:
        st.warning("No pude armar cartera con la selección actual (faltan precios o flujos).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ===== Resumen (con 2 decimales en TIR/MD/Duration)
    st.markdown("## NEIX · Cartera Comercial")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
<div class="kpi">
  <div class="lbl">Capital</div>
  <div class="val">{fmt_money_int(float(capital))}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
<div class="kpi">
  <div class="lbl">TIR total (pond.)</div>
  <div class="val">{fmt_pct_2(float(resumen["tir"]))}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
<div class="kpi">
  <div class="lbl">MD total (pond.)</div>
  <div class="val">{fmt_num_2(float(resumen["md"]))}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
<div class="kpi">
  <div class="lbl">Duration total (pond.)</div>
  <div class="val">{fmt_num_2(float(resumen["dur"]))}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ===== Tabla Cartera (formateada)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    show = cartera_df.copy()

    # Formatos solicitados
    show["%"] = pd.to_numeric(show["%"], errors="coerce").round(2)
    show["USD"] = pd.to_numeric(show["USD"], errors="coerce").round(0)
    show["Precio (USD, VN100)"] = pd.to_numeric(show["Precio (USD, VN100)"], errors="coerce").round(2)
    show["VN estimada"] = pd.to_numeric(show["VN estimada"], errors="coerce").round(0)

    show["TIR (%)"] = pd.to_numeric(show["TIR (%)"], errors="coerce").round(2)
    show["MD"] = pd.to_numeric(show["MD"], errors="coerce").round(2)
    show["Duration"] = pd.to_numeric(show["Duration"], errors="coerce").round(2)

    # ✅ Vencimiento bien
    show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date

    # Dataframe height prolijo
    h = _height_for_rows(len(show), base=220, max_h=520)

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=h,
        column_config={
            "%": st.column_config.NumberColumn("%", format="%.2f"),
            "USD": st.column_config.NumberColumn("USD", format="$ %.0f"),
            "Precio (USD, VN100)": st.column_config.NumberColumn("Precio (USD, VN100)", format="%.2f"),
            "VN estimada": st.column_config.NumberColumn("VN estimada", format="%.0f"),
            "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
            "MD": st.column_config.NumberColumn("MD", format="%.2f"),
            "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
            "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
        },
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    
    # ===== Flujo de fondos (totales fila y columna)
    st.markdown("## Flujo de fondos")

    if flows_pivot is None or flows_pivot.empty:
        st.info("No hay flujos futuros para mostrar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    flows = flows_pivot.copy()

    # columnas de fecha lindas
    new_cols = []
    for c in flows.columns:
        if isinstance(c, (pd.Timestamp, dt.datetime)):
            new_cols.append(pd.to_datetime(c).strftime("%b-%Y").capitalize())
        else:
            new_cols.append(str(c))
    flows.columns = new_cols

    flows = flows.round(0)

    h2 = _height_for_rows(len(flows), base=240, max_h=560)

    st.dataframe(
        flows,
        use_container_width=True,
        height=h2,
        column_config={col: st.column_config.NumberColumn(col, format="$ %.0f") for col in flows.columns},
    )

    st.markdown("</div>", unsafe_allow_html=True)

