# tools/ons.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

CASHFLOW_PATH = os.path.join("data", "cashflows_ON.xlsx")


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
# 2) Cashflows (nuevo formato con LAW)
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
    Espera el Excel "nuevo" que generaste (como la captura):
    species | root_key | Fecha | FlujoTotal (o Cupon) | law (ej: ARG / NYC) | ...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_ON.xlsx)."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    # columnas mínimas
    if "species" not in df.columns:
        raise ValueError("Falta columna 'species' en el cashflow.")
    if "root_key" not in df.columns:
        raise ValueError("Falta columna 'root_key' en el cashflow.")
    if "Fecha" not in df.columns:
        raise ValueError("Falta columna 'Fecha' en el cashflow.")

    # flujo total: puede venir como FlujoTotal (tu export) o como Cupon (viejo)
    if "FlujoTotal" in df.columns:
        flujo_col = "FlujoTotal"
    elif "Cupon" in df.columns:
        flujo_col = "Cupon"
    else:
        raise ValueError("Falta columna 'FlujoTotal' o 'Cupon' en el cashflow.")

    if "law" not in df.columns:
        # si no está, igual funciona pero no habrá tabs por ley
        df["law"] = "NA"

    df["species"] = df["species"].astype(str).str.strip().str.upper()
    df["root_key"] = df["root_key"].astype(str).str.strip().str.upper()
    df["law"] = df["law"].astype(str).str.strip().str.upper()

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Cupon"] = pd.to_numeric(df[flujo_col], errors="coerce")

    df = df.dropna(subset=["species", "root_key", "Fecha", "Cupon"]).sort_values(["species", "Fecha"])
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Cashflows por especie (species) para calcular TIR/Duration.
    """
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("species", sort=False):
        out[str(k)] = g[["Fecha", "Cupon"]].copy().sort_values("Fecha")
    return out


def build_species_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Meta por species:
    - root_key más frecuente
    - law más frecuente
    - vencimiento (max Fecha)
    """
    meta = (
        df.groupby("species")
        .agg(
            root_key=("root_key", lambda s: s.value_counts().index[0]),
            law=("law", lambda s: s.value_counts().index[0]),
            Vencimiento=("Fecha", "max"),
        )
        .reset_index()
    )
    return meta


# ======================================================
# 3) Precios IOL (por root_keyD/root_keyC)
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
    return float(s)


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
    candidates = [(f"{rk}D", "D"), (f"{rk}C", "C")]

    for sym, src in candidates:
        if sym in prices.index:
            px = float(prices.loc[sym, "UltimoOperado"])
            vol = float(prices.loc[sym, "MontoOperado"])

            # normalización escala (si quedó x100)
            if np.isfinite(px) and px > 1000:
                px = px / 100.0

            return px, vol, src

    return np.nan, np.nan, ""


# ======================================================
# 4) Métricas
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
# 5) UI (tabs por ley + selector columnas)
# ======================================================
def _ui_css():
    st.markdown(
        """
    <style>
      .wrap{ max-width: 1200px; margin: 0 auto; }
      .head{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:10px; }
      .title{ font-size:22px; font-weight:800; letter-spacing:.04em; color:#111827; }
      .sub{ color:rgba(17,24,39,.60); font-size:13px; margin-top:2px; }
      .card{
        border:1px solid rgba(17,24,39,0.08);
        border-radius:14px;
        padding:14px 14px;
        background:#fff;
        box-shadow: 0 8px 26px rgba(17,24,39,0.05);
      }
      .muted{ color:rgba(17,24,39,.55); font-size:12px; }
      div[data-baseweb="tag"]{ border-radius:999px !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _law_label(law: str) -> str:
    law = (law or "").strip().upper()
    if law == "ARG":
        return "Ley Local (ARG)"
    if law in {"NYC", "NY", "NEW YORK"}:
        return "Ley NY (NYC)"
    if law == "NA":
        return "Sin Ley"
    return f"Ley {law}"


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    """
    df_cf: cashflows completo (con species/root_key/law/Fecha/Cupon)
    prices: tabla de IOL index=TICKER
    """
    cashflows = build_cashflow_dict(df_cf)
    meta = build_species_meta(df_cf).set_index("species")

    rows = []
    for species in meta.index:
        rk = meta.loc[species, "root_key"]
        law = meta.loc[species, "law"]
        venc = meta.loc[species, "Vencimiento"]

        px, vol, src = pick_usd_price_by_root(prices, rk)

        # solo con precio
        if not np.isfinite(px) or px <= 0:
            continue

        cf = cashflows.get(species)
        if cf is None or cf.empty:
            continue

        rows.append(
            {
                "Ticker": species,
                "Ley": law,
                "USD": src,
                "Precio USD": px,
                "TIR (%)": tir(cf, px, plazo_dias=plazo),
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


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    left, right = st.columns([0.75, 0.25])
    with left:
        st.markdown(
            """
            <div class="head">
              <div>
                <div class="title">ONs · Rendimientos</div>
                <div class="sub">Tabs por ley (ARG / NYC). Precios desde IOL por <b>root_key</b>: rootD (MEP) o rootC (Cable). Solo se muestran ONs con precio.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if back_to_home is not None:
            st.button("← Volver", on_click=back_to_home)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.info("Solución: el Excel debe tener columnas: species, root_key, Fecha y FlujoTotal/Cupon (+ law opcional).")
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    # filtros globales
    c1, c2, c3 = st.columns([0.22, 0.22, 0.56])
    with c1:
        plazo = st.selectbox("Plazo", [0, 1], index=0, format_func=lambda x: f"T{x}")
    with c2:
        traer_precios = st.button("Actualizar IOL")
    with c3:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # cache precios
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
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    # tabs por ley
    laws_present = sorted(df_cf["law"].dropna().astype(str).str.upper().unique().tolist())
    # orden preferido
    ordered = []
    for x in ["ARG", "NYC"]:
        if x in laws_present:
            ordered.append(x)
    for x in laws_present:
        if x not in ordered:
            ordered.append(x)

    if not ordered:
        ordered = ["NA"]

    tab_objs = st.tabs([_law_label(x) for x in ordered])

    for tab, law in zip(tab_objs, ordered):
        with tab:
            df_law = df_cf[df_cf["law"].astype(str).str.upper() == law].copy()

            # filtro ticker dentro de la ley (para no llenar de chips rojos)
            tickers = sorted(df_law["species"].unique().tolist())
            sel = st.multiselect("Ticker", tickers, default=tickers)

            # calcular tabla final
            if st.button("Calcular", type="primary", key=f"calc_{law}"):
                df_use = df_law[df_law["species"].isin(sel)].copy()
                out = _compute_table(df_use, prices, plazo)

                if out.empty:
                    st.info("No hay ONs con precio para esta ley / selección.")
                    continue

                # selector de columnas
                base_cols = ["Ticker", "USD", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen"]
                # siempre mostramos Ticker y Vencimiento por defecto
                defaults = ["Ticker", "Precio USD", "TIR (%)", "MD", "Duration", "Vencimiento"]

                cols_pick = st.multiselect(
                    "Columnas a mostrar",
                    options=base_cols,
                    default=defaults,
                    key=f"cols_{law}",
                )

                if "Ticker" not in cols_pick:
                    cols_pick = ["Ticker"] + cols_pick
                if "Vencimiento" not in cols_pick:
                    cols_pick = cols_pick + ["Vencimiento"]

                title = f"Tabla comercial · {_law_label(law)}"
                st.markdown(f"### {title}")

                show = out.copy()
                show["Vencimiento"] = pd.to_datetime(show["Vencimiento"], errors="coerce").dt.date

                st.dataframe(
                    show[cols_pick],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "USD": st.column_config.TextColumn("USD (D/C)"),
                        "Precio USD": st.column_config.NumberColumn("Precio USD", format="%.2f"),
                        "TIR (%)": st.column_config.NumberColumn("TIR (%)", format="%.2f"),
                        "MD": st.column_config.NumberColumn("MD", format="%.2f"),
                        "Duration": st.column_config.NumberColumn("Duration", format="%.2f"),
                        "Vencimiento": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
                        "Volumen": st.column_config.NumberColumn("Volumen", format="%.0f"),
                    },
                )

    st.markdown("</div></div>", unsafe_allow_html=True)
