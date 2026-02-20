# tools/mesa/cartera2.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
TICKER_LIST_XLSX = os.path.join("data", "Especies.xlsx")
TICKER_LIST_SHEET: str | int | None = None  # None => primera hoja

COL_PESOS = "Pesos"
COL_USD = "Usd"

# ✅ IOL URLs (con tu fix de Acciones)
IOL_SOURCES = [
    ("Acción", "https://iol.invertironline.com/mercado/cotizaciones"),
    ("CEDEAR", "https://iol.invertironline.com/mercado/cotizaciones/argentina/cedears/todos"),
    ("Bono",   "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"),
    ("ON",     "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"),
]


# =========================
# HELPERS
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


def base_ticker(symbol_raw: str) -> str:
    s = (symbol_raw or "").strip().upper()
    return s.split()[0] if s else ""


def display_label(symbol_raw: str) -> str:
    s = (symbol_raw or "").strip().upper()
    toks = s.split()
    if not toks:
        return ""
    if len(toks) >= 2 and toks[1] == "CEDEAR":
        return toks[0]
    if len(toks) >= 2:
        return f"{toks[0]} {toks[1]}"
    return toks[0]


def _clean_tickers(series: pd.Series) -> List[str]:
    out = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "-": np.nan, "NAN": np.nan, "NONE": np.nan})
        .dropna()
        .tolist()
    )
    # dedup preservando orden
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# =========================
# EXCEL: lista de tickers (Excel manda)
# =========================
def load_ticker_map_from_excel(
    path: str = TICKER_LIST_XLSX,
    sheet_name: str | int | None = TICKER_LIST_SHEET,
    prefer: str = "USD",  # si un ticker está en ambas columnas
) -> Tuple[Dict[str, str], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    xls = pd.ExcelFile(path)
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    if COL_PESOS not in df.columns or COL_USD not in df.columns:
        raise ValueError(
            f"El Excel debe tener columnas '{COL_PESOS}' y '{COL_USD}'. Encontradas: {list(df.columns)}"
        )

    pesos = _clean_tickers(df[COL_PESOS])
    usd = _clean_tickers(df[COL_USD])

    dup = sorted(list(set(pesos) & set(usd)))

    mapping: Dict[str, str] = {}
    for t in pesos:
        mapping[t] = "ARS"
    for t in usd:
        mapping[t] = "USD"

    pref = prefer.strip().upper()
    if pref not in {"USD", "ARS"}:
        raise ValueError("prefer debe ser 'USD' o 'ARS'")

    for t in dup:
        mapping[t] = pref

    return mapping, dup


# =========================
# IOL: fetch table estándar
# =========================
def _fetch_iol_table(url: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    t = tables[0].copy()
    t.columns = [str(c).strip() for c in t.columns]

    # estructura esperada en IOL
    if "Símbolo" not in t.columns or "Último Operado" not in t.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["SymbolRaw"] = t["Símbolo"].astype(str).str.strip().str.upper()
    out["Ticker"] = out["SymbolRaw"].apply(base_ticker)
    out["Label"] = out["SymbolRaw"].apply(display_label)
    out["Precio"] = t["Último Operado"].apply(parse_ar_number)

    if "Monto Operado" in t.columns:
        out["Volumen"] = t["Monto Operado"].apply(parse_ar_number).fillna(0.0)
    else:
        out["Volumen"] = 0.0

    out = out.dropna(subset=["Ticker", "Precio"])
    out = out[out["Ticker"].astype(str).str.len() > 0]

    out = out.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    out = out.drop_duplicates(subset=["Ticker"], keep="first")

    return out[["SymbolRaw", "Ticker", "Label", "Precio", "Volumen"]]


def fetch_universe_prices_iol() -> pd.DataFrame:
    frames = []
    for tipo, url in IOL_SOURCES:
        df = _fetch_iol_table(url)
        if df.empty:
            continue
        df["Tipo"] = tipo
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")
    allp = allp.set_index("Ticker")

    return allp[["Precio", "Volumen", "Tipo", "Label"]].sort_values("Volumen", ascending=False)


def fetch_iol_prices_filtered_by_excel(
    excel_path: str = TICKER_LIST_XLSX,
    sheet_name: str | int | None = TICKER_LIST_SHEET,
    prefer: str = "USD",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    ticker_to_ccy, dup = load_ticker_map_from_excel(excel_path, sheet_name, prefer=prefer)
    universe = fetch_universe_prices_iol()

    if universe.empty:
        return pd.DataFrame(), sorted(list(ticker_to_ccy.keys())), dup

    rows = []
    missing = []

    for tk, ccy in ticker_to_ccy.items():
        if tk in universe.index:
            r = universe.loc[tk]
            rows.append(
                {
                    "Ticker": tk,
                    "Precio": float(r["Precio"]) if pd.notna(r["Precio"]) else np.nan,
                    "Volumen": float(r["Volumen"]) if pd.notna(r["Volumen"]) else 0.0,
                    "Tipo": str(r["Tipo"]),
                    "Label": str(r["Label"]),
                    "Moneda": ccy,  # ✅ moneda definida por Excel
                }
            )
        else:
            missing.append(tk)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, missing, dup

    df = df.set_index("Ticker").sort_values("Volumen", ascending=False)
    return df, missing, dup


# =========================
# STREAMLIT UI
# =========================
def _ui_css() -> None:
    st.markdown(
        """
<style>
  .wrap{ max-width: 1180px; margin: 0 auto; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  .title{ font-size: 28px; font-weight: 850; letter-spacing: .02em; color:#111827; margin: 0; }
  .sub{ color: rgba(17,24,39,.62); font-size: 13px; margin-top: 6px; }
  .soft-hr{ height:1px; background:rgba(17,24,39,.10); margin: 14px 0 18px; }
  div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60 * 10, show_spinner=False)  # 10 min
def _cached_fetch(excel_path: str, sheet_name: str | int | None, prefer: str):
    return fetch_iol_prices_filtered_by_excel(excel_path, sheet_name, prefer)


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.markdown('<div class="title">Herramienta para armar carteras (ARG)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Precios IOL filtrados por tu Excel (Pesos / Usd)</div>', unsafe_allow_html=True)
    with right:
        refresh = st.button("Actualizar precios", use_container_width=True)

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    # Settings
    c1, c2 = st.columns([0.65, 0.35], vertical_alignment="bottom")
    with c1:
        excel_path = st.text_input("Excel de especies", value=TICKER_LIST_XLSX)
    with c2:
        prefer = st.selectbox("Si un ticker está en ambas columnas…", ["USD", "ARS"], index=0)

    # Sheet option simple (primera hoja o nombre)
    sheet = TICKER_LIST_SHEET
    sheet_hint = st.text_input("Hoja (vacío = primera hoja)", value="")
    if sheet_hint.strip():
        sheet = sheet_hint.strip()

    if refresh:
        _cached_fetch.clear()

    # Load
    try:
        with st.spinner("Cargando tickers del Excel y consultando IOL..."):
            df, missing, dup = _cached_fetch(excel_path, sheet, prefer)
    except Exception as e:
        st.error("Error cargando la herramienta.")
        st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Summary
    k1, k2, k3 = st.columns(3)
    k1.metric("Encontrados", f"{len(df):,}".replace(",", ".") if isinstance(df, pd.DataFrame) else "0")
    k2.metric("Faltantes", f"{len(missing):,}".replace(",", "."))
    k3.metric("Duplicados", f"{len(dup):,}".replace(",", "."))

    if dup:
        with st.expander("Ver duplicados (estaban en Pesos y en Usd)"):
            st.write(dup)

    if df is None or df.empty:
        st.warning("No se encontraron precios para los tickers del Excel (o IOL devolvió vacío).")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            height=760,
        )

    if missing:
        st.markdown("### Faltantes (no encontrados en IOL)")
        st.caption("Estos tickers están en tu Excel pero no aparecieron en el universo de IOL.")
        st.dataframe(pd.DataFrame({"Ticker": missing}), use_container_width=True, height=320)

    st.caption("Requiere: pip install lxml pandas numpy openpyxl")
    st.markdown("</div>", unsafe_allow_html=True)
