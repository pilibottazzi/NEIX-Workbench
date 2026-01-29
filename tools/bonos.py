# tools/bonos.py
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize

CASHFLOW_PATH = os.path.join("data", "cashflows_completos.xlsx")

# ✅ Filtro interno (NO se muestra en UI)
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
# 2) Ley: normalización (ARG vs NY/NYC)
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
        return "Ley Local (ARG)"
    if norm == "NY":
        return "Ley NY (NY/NYC)"
    if norm == "NA":
        return "Sin Ley"
    return f"Ley {norm}"


# ======================================================
# 3) Load cashflows BONOS (excel completo)
# ======================================================
def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    return None


def load_cashflows_from_repo(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Subilo al repo (ej: data/cashflows_completos.xlsx)."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    # Columnas mínimas (permito aliases razonables)
    col_date = _pick_first_existing(df, ["date", "Fecha"])
    col_species = _pick_first_existing(df, ["species", "ticker", "Ticker"])
    col_desc = _pick_first_existing(df, ["description", "Descripcion", "Descripción"])
    col_law = _pick_first_existing(df, ["law", "Ley"])
    col_issuer = _pick_first_existing(df, ["issuer", "Emisor"])
    col_rent = _pick_first_existing(df, ["rent", "Renta"])
    col_amort = _pick_first_existing(df, ["amortization", "Amortizacion", "Amortización"])
    col_flujo = _pick_first_existing(df, ["flujo_total", "FlujoTotal", "Flujo Total", "flow_total"])

    required = {
        "date": col_date,
        "species": col_species,
        "description": col_desc,
        "law": col_law,
        "issuer": col_issuer,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Faltan columnas mínimas en {path}: {missing}. "
            f"Encontré: {list(df.columns)}"
        )

    # Normalizo nombres
    df = df.rename(
        columns={
            col_date: "date",
            col_species: "species",
            col_desc: "description",
            col_law: "law",
            col_issuer: "issuer",
            **({col_rent: "rent"} if col_rent else {}),
            **({col_amort: "amortization"} if col_amort else {}),
            **({col_flujo: "flujo_total"} if col_flujo else {}),
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["species"] = df["species"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype(str).str.strip()
    df["law"] = df["law"].astype(str).str.strip().str.upper()
    df["issuer"] = df["issuer"].astype(str).str.strip().str.upper()

    # Si no hay flujo_total, lo armo con rent + amortization cuando existan
    if "flujo_total" not in df.columns:
        df["flujo_total"] = np.nan

    if "rent" in df.columns:
        df["rent"] = pd.to_numeric(df["rent"], errors="coerce")
    else:
        df["rent"] = np.nan

    if "amortization" in df.columns:
        df["amortization"] = pd.to_numeric(df["amortization"], errors="coerce")
    else:
        df["amortization"] = np.nan

    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    # si flujo_total viene vacío en algunas filas, completo con rent+amort
    df["flujo_total"] = df["flujo_total"].fillna(df["rent"].fillna(0) + df["amortization"].fillna(0))

    # ley normalizada
    df["law_norm"] = df["law"].apply(normalize_law)

    # limpieza
    df = df.dropna(subset=["date", "species"]).sort_values(["species", "date"]).reset_index(drop=True)
    return df


def build_cashflow_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, g in df.groupby("species", sort=False):
        out[str(k)] = g[["date", "flujo_total"]].copy().sort_values("date")
    return out


# ======================================================
# 4) Métricas (con flujo_total)
# ======================================================
def _settlement(plazo_dias: int) -> dt.datetime:
    return dt.datetime.today() + dt.timedelta(days=int(plazo_dias))


def _future_cashflows(df: pd.DataFrame, settlement: dt.datetime) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flujo_total"] = pd.to_numeric(df["flujo_total"], errors="coerce")
    df = df.dropna(subset=["date", "flujo_total"])
    df = df[df["date"] > settlement].sort_values("date")
    return df


def tir(cashflow: pd.DataFrame, precio: float, plazo_dias: int = 0) -> float:
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


# ======================================================
# 5) Precios (placeholder simple)
#    Si ya tenés tu fetch de precios para bonos, reemplazá esta parte.
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


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_bonos_prices_iol() -> pd.DataFrame:
    """
    ⚠️ Esto es un placeholder para BONOS.
    Si ya tenés tu función, usá la tuya y devolvé un DF indexado por Ticker con columnas:
      - UltimoOperado
      - MontoOperado
    """
    urls = [
        "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos",
        "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos%20en%20dolares",
        "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos%20en%20pesos",
    ]
    last_err = None
    for url in urls:
        try:
            t = pd.read_html(url)[0]
            # Intento mapear columnas típicas
            sym_col = "Símbolo" if "Símbolo" in t.columns else None
            last_col = "Último Operado" if "Último Operado" in t.columns else None
            vol_col = "Monto Operado" if "Monto Operado" in t.columns else None
            if not sym_col or not last_col:
                continue

            df = pd.DataFrame(
                {
                    "Ticker": t[sym_col].astype(str).str.strip().str.upper(),
                    "UltimoOperado": t[last_col].apply(to_float_iol),
                    "MontoOperado": t[vol_col].apply(to_float_iol) if vol_col else 0.0,
                }
            ).dropna(subset=["UltimoOperado"])
            return df.set_index("Ticker")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"No pude leer precios de IOL para Bonos. Último error: {last_err}")


# ======================================================
# 6) UI
# ======================================================
def _ui_css():
    st.markdown(
        """
    <style>
      .wrap{ max-width: 1200px; margin: 0 auto; }
      .title{ font-size:22px; font-weight:800; letter-spacing:.04em; color:#111827; }
      .sub{ color:rgba(17,24,39,.60); font-size:13px; margin-top:2px; }
      div[data-baseweb="tag"]{ border-radius:999px !important; }

      /* Compacto */
      .block-container { padding-top: 1.2rem; }
      label { margin-bottom: 0.25rem !important; }

      /* Inputs */
      .stButton > button { border-radius: 12px; padding: 0.60rem 1.0rem; }
      .stSelectbox div[data-baseweb="select"]{ border-radius: 12px; }
      .stMultiSelect div[data-baseweb="select"]{ border-radius: 12px; }

      /* Que no quede “coso blanco” arriba: le saco aire extra */
      section[data-testid="stSidebar"]{ padding-top: 0.8rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _compute_table(df_cf: pd.DataFrame, prices: pd.DataFrame, plazo: int) -> pd.DataFrame:
    cashflows = build_cashflow_dict(df_cf)

    # meta por ticker: issuer/desc/vencimiento
    meta = (
        df_cf.groupby("species")
        .agg(
            issuer=("issuer", lambda s: s.value_counts().index[0] if len(s.dropna()) else ""),
            description=("description", lambda s: s.value_counts().index[0] if len(s.dropna()) else ""),
            law_norm=("law_norm", lambda s: s.value_counts().index[0] if len(s.dropna()) else "NA"),
            vencimiento=("date", "max"),
        )
        .reset_index()
    )

    rows = []
    for _, r in meta.iterrows():
        ticker = r["species"]
        if ticker not in prices.index:
            continue

        px = float(prices.loc[ticker, "UltimoOperado"])
        vol = float(prices.loc[ticker, "MontoOperado"]) if "MontoOperado" in prices.columns else 0.0

        cf = cashflows.get(ticker)
        if cf is None or cf.empty:
            continue

        t = tir(cf, px, plazo_dias=plazo)

        rows.append(
            {
                "Ticker": ticker,
                "Issuer": r["issuer"],
                "Descripción": r["description"],
                "Ley": r["law_norm"],
                "Precio": px,
                "TIR (%)": t,
                "MD": modified_duration(cf, px, plazo_dias=plazo),
                "Duration": duration(cf, px, plazo_dias=plazo),
                "Vencimiento": r["vencimiento"],
                "Volumen": vol,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Vencimiento"] = pd.to_datetime(out["Vencimiento"], errors="coerce")
    out = out.sort_values(["Vencimiento", "Ticker"], na_position="last").reset_index(drop=True)
    return out


def _clean_options(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return sorted(s.unique().tolist())


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    h1, h2 = st.columns([0.78, 0.22])
    with h1:
        st.markdown('<div class="title">NEIX · Bonos</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Filtros por Ley / Issuer / Descripción + Ticker. (Opciones encadenadas).</div>', unsafe_allow_html=True)

    st.divider()

    # Load cashflows
    try:
        df_cf = load_cashflows_from_repo(CASHFLOW_PATH)
    except Exception as e:
        st.error(str(e))
        st.info("El Excel debe tener (mínimo): date, species, description, law, issuer.")
        return

    # Top controls
    f1, f2, f3, f4 = st.columns([0.22, 0.18, 0.18, 0.42], vertical_alignment="bottom")
    with f1:
        plazo = st.selectbox("Plazo", [0, 1], index=0, format_func=lambda x: f"T{x}")
    with f2:
        traer_precios = st.button("Actualizar PRECIOS", use_container_width=True)
    with f3:
        calcular = st.button("Calcular", type="primary", use_container_width=True)
    with f4:
        st.caption(f"Cashflows: `{CASHFLOW_PATH}`")

    # Prices cache
    if traer_precios or "bonos_prices" not in st.session_state:
        with st.spinner("Leyendo precios..."):
            try:
                st.session_state["bonos_prices"] = fetch_bonos_prices_iol()
            except Exception as e:
                st.error(str(e))
                st.session_state["bonos_prices"] = None

    prices = st.session_state.get("bonos_prices")
    if prices is None:
        st.warning("No hay precios cargados todavía.")
        return

    st.divider()

    # Tabs por ley
    tab_arg, tab_ny = st.tabs([law_label("ARG"), law_label("NY")])

    for tab, law_norm in [(tab_arg, "ARG"), (tab_ny, "NY")]:
        with tab:
            # 1) filtro base por ley
            df_law = df_cf[df_cf["law_norm"] == law_norm].copy()
            if df_law.empty:
                st.info("No hay instrumentos para esta ley (revisá columna law).")
                continue

            # 2) Filtros prolijos (desplegables / encadenados)
            st.markdown("#### Filtros")

            # Armamos opciones siempre desde lo disponible *en este tab*
            all_issuers = _clean_options(df_law["issuer"])
            all_desc = _clean_options(df_law["description"])

            # defaults = todo
            if f"issuer_sel_{law_norm}" not in st.session_state:
                st.session_state[f"issuer_sel_{law_norm}"] = all_issuers
            if f"desc_sel_{law_norm}" not in st.session_state:
                st.session_state[f"desc_sel_{law_norm}"] = all_desc

            with st.expander("Issuer / Descripción (desplegable)", expanded=True):
                c1, c2 = st.columns([0.5, 0.5])
                with c1:
                    sel_issuers = st.multiselect(
                        "Issuer",
                        options=all_issuers,
                        default=st.session_state.get(f"issuer_sel_{law_norm}", all_issuers),
                        key=f"issuer_sel_{law_norm}",
                    )
                with c2:
                    # ✅ descripción se recalcula después de issuer (encadenado)
                    df_tmp = df_law[df_law["issuer"].isin(sel_issuers)] if sel_issuers else df_law.iloc[0:0]
                    desc_options = _clean_options(df_tmp["description"]) if not df_tmp.empty else []
                    # si lo guardado tiene cosas que ya no existen, lo limpiamos
                    prev = st.session_state.get(f"desc_sel_{law_norm}", desc_options)
                    prev = [x for x in prev if x in desc_options]
                    if not prev and desc_options:
                        prev = desc_options
                    sel_desc = st.multiselect(
                        "Descripción",
                        options=desc_options,
                        default=prev,
                        key=f"desc_sel_{law_norm}",
                    )

            # 3) Con issuer + desc, definimos universo real
            df_pool = df_law.copy()
            if sel_issuers:
                df_pool = df_pool[df_pool["issuer"].isin(sel_issuers)]
            else:
                df_pool = df_pool.iloc[0:0]

            if sel_desc:
                df_pool = df_pool[df_pool["description"].isin(sel_desc)]
            else:
                df_pool = df_pool.iloc[0:0]

            if df_pool.empty:
                st.warning("Con esos filtros no quedó ningún bono. Probá abrir issuer/descripcion.")
                continue

            # 4) Tickers (encadenado al pool)
            tickers = sorted(df_pool["species"].unique().tolist())

            with st.expander("Ticker (desplegable)", expanded=True):
                all_on = st.checkbox("Seleccionar todo", value=True, key=f"selall_{law_norm}")
                default_tickers = tickers if all_on else tickers[: min(10, len(tickers))]
                sel_tickers = st.multiselect(
                    "Ticker",
                    options=tickers,
                    default=default_tickers,
                    key=f"tick_{law_norm}",
                )

            if not sel_tickers:
                st.info("Elegí al menos 1 ticker.")
                continue

            if not calcular:
                st.caption("Ajustá filtros y tocá **Calcular**.")
                continue

            # 5) Computo y tabla
            df_use = df_pool[df_pool["species"].isin(sel_tickers)].copy()
            out = _compute_table(df_use, prices, plazo)

            if out.empty:
                st.info("No hay bonos con precio para esta selección (según fuente de precios).")
                continue

            # ✅ filtro interno de TIR (no se muestra)
            out = out[out["TIR (%)"].between(TIR_MIN, TIR_MAX, inclusive="both")].copy()
            if out.empty:
                st.info("No quedaron bonos tras aplicar filtros internos.")
                continue

            st.markdown(f"### NEIX · Bonos · {law_label(law_norm)}")

            all_cols = ["Ticker", "Issuer", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen", "Descripción"]
            defaults = ["Ticker", "Issuer", "Precio", "TIR (%)", "MD", "Duration", "Vencimiento", "Volumen", "Descripción"]

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

            # ✅ más alto (más filas visibles)
            base = 460
            row_h = 28
            max_h = 980
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
