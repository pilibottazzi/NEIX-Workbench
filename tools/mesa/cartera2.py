# tools/cartera2.py
from __future__ import annotations

import io
import os
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
LOGO_PATH = os.path.join("data", "Neix_logo.png")

# Para carteras USD (simple)
DEFAULT_CAPITAL_USD = 100000.0

# =========================
# Utils num/format
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


def fmt_money_int(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return f"$ {v:,.0f}".replace(",", ".")


def fmt_num_ar(x: float, dec: int = 2) -> str:
    """
    Formato AR:
      - miles con '.'
      - decimales con ','
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    s = f"{v:,.{dec}f}"  # 12,345.67
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# =========================
# Fetch precios (IOL via read_html)
# =========================
def _fetch_iol_table(url: str) -> pd.DataFrame:
    """
    Devuelve un df estandarizado:
      cols: Ticker, Precio, Volumen
    Si no puede parsear, devuelve df vacío.
    """
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    t = tables[0].copy()
    t.columns = [str(c).strip() for c in t.columns]

    # IOL suele usar: Símbolo, Último Operado, Monto Operado
    if "Símbolo" not in t.columns or "Último Operado" not in t.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Ticker"] = t["Símbolo"].astype(str).str.strip().str.upper()
    out["RawPrecio"] = t["Último Operado"].astype(str).str.strip()
    out["Precio"] = t["Último Operado"].apply(parse_ar_number)

    if "Monto Operado" in t.columns:
        out["Volumen"] = t["Monto Operado"].apply(parse_ar_number).fillna(0)
    else:
        out["Volumen"] = 0.0

    out = out.dropna(subset=["Precio"])
    out = out[~out["Ticker"].duplicated(keep="first")]
    return out[["Ticker", "Precio", "Volumen"]]


def fetch_universe_prices() -> pd.DataFrame:
    """
    Une varias categorías para tener universo amplio:
    - Acciones
    - CEDEARs
    - Bonos
    - Obligaciones Negociables
    - FCIs
    - ETFs (si existe en IOL, si no queda vacío)

    Output:
      index = ticker
      cols  = Precio, Volumen, Tipo, Mercado
    """
    sources = [
        ("Acción", "Local", "https://iol.invertironline.com/mercado/cotizaciones/argentina/acciones/todas"),
        ("CEDEAR", "Local", "https://iol.invertironline.com/mercado/cotizaciones/argentina/cedears/todos"),
        ("Bono", "Local", "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"),
        ("ON", "Local", "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"),
        ("FCI", "Local", "https://iol.invertironline.com/mercado/cotizaciones/argentina/fondoscomunes/todos"),
        # Si IOL cambia esto o no existe, simplemente queda vacío:
        ("ETF", "Local", "https://iol.invertironline.com/mercado/cotizaciones/argentina/etf/todos"),
    ]

    frames = []
    for tipo, mercado, url in sources:
        df = _fetch_iol_table(url)
        if df.empty:
            continue
        df["Tipo"] = tipo
        df["Mercado"] = mercado
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)

    # Si un ticker aparece en más de una fuente, prioriza mayor volumen
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo", "Mercado"]].sort_values("Volumen", ascending=False)


# =========================
# Construcción cartera simple
# =========================
@dataclass
class SimpleAssetRow:
    ticker: str
    pct: float
    usd: float
    precio: float
    vn: float
    tipo: str
    mercado: str
    ticker_precio: str


def build_simple_portfolio(
    prices: pd.DataFrame,
    selected: list[str],
    pct_map: dict[str, float],
    capital_usd: float,
) -> pd.DataFrame:
    """
    Devuelve DF con:
      Ticker, %, USD, Precio, VN, Tipo, Mercado, Ticker precio
    """
    selected = [str(x).upper().strip() for x in selected if str(x).strip()]
    if not selected:
        return pd.DataFrame()

    # normalizar % (si no suma 100, escala)
    pcts = np.array([max(0.0, float(pct_map.get(t, 0.0))) for t in selected], dtype=float)
    s = float(np.sum(pcts))
    if s <= 0:
        pcts = np.zeros_like(pcts)
    else:
        pcts = pcts / s * 100.0

    rows: list[SimpleAssetRow] = []

    for t, pct in zip(selected, pcts):
        if pct <= 0:
            continue

        if t not in prices.index:
            # si no tiene precio, lo salteamos
            continue

        px = float(prices.loc[t, "Precio"])
        tipo = str(prices.loc[t, "Tipo"]) if "Tipo" in prices.columns else "NA"
        mercado = str(prices.loc[t, "Mercado"]) if "Mercado" in prices.columns else "NA"

        usd_amt = float(capital_usd) * (float(pct) / 100.0)

        # VN = cantidad estimada (unidades) = USD / Precio
        vn = (usd_amt / px) if px > 0 else np.nan

        rows.append(
            SimpleAssetRow(
                ticker=t,
                pct=float(pct),
                usd=float(usd_amt),
                precio=float(px),
                vn=float(vn) if np.isfinite(vn) else np.nan,
                tipo=tipo,
                mercado=mercado,
                ticker_precio=t,
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Ticker": [r.ticker for r in rows],
            "%": [r.pct for r in rows],
            "USD": [r.usd for r in rows],
            "Precio": [r.precio for r in rows],
            "VN": [r.vn for r in rows],
            "Tipo": [r.tipo for r in rows],
            "Mercado": [r.mercado for r in rows],
            "Ticker precio": [r.ticker_precio for r in rows],
        }
    )

    return df


# =========================
# UI
# =========================
def _ui_css():
    st.markdown(
        """
<style>
  .wrap{ max-width: 1180px; margin: 0 auto; }
  .block-container { padding-top: 1.1rem; padding-bottom: 1.8rem; }
  .title{ font-size: 28px; font-weight: 850; letter-spacing: .02em; color:#111827; margin: 0; }
  .sub{ color: rgba(17,24,39,.62); font-size: 13px; margin-top: 4px; }
  .soft-hr{ height:1px; background:rgba(17,24,39,.10); margin: 14px 0 18px; }

  .kpi{
    border: 1px solid rgba(17,24,39,.10);
    border-radius: 16px;
    padding: 12px 14px;
    background: white;
  }
  .kpi .lbl{ color: rgba(17,24,39,.60); font-size: 12px; margin-bottom: 6px; }
  .kpi .val{ font-size: 26px; font-weight: 850; color:#111827; letter-spacing: .01em; }
</style>
""",
        unsafe_allow_html=True,
    )


def _height_for_rows(n: int, row_h: int = 34, header: int = 42, pad: int = 18, max_h: int = 900) -> int:
    n = int(max(0, n))
    h = header + pad + row_h * max(1, n + 1)
    return int(min(max_h, h))


def _spacer(px: int = 14):
    st.markdown(f'<div style="height:{int(px)}px"></div>', unsafe_allow_html=True)


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.markdown('<div class="title">NEIX · Cartera Simple</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Sin TIR / MD. Solo asignación y cantidades estimadas.</div>', unsafe_allow_html=True)
    with right:
        refresh = st.button("Actualizar universo", use_container_width=True, key="cartera2_refresh")

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    # Cache precios
    if refresh or "cartera2_prices" not in st.session_state:
        with st.spinner("Actualizando universo de precios..."):
            st.session_state["cartera2_prices"] = fetch_universe_prices()

    prices = st.session_state.get("cartera2_prices")
    if prices is None or prices.empty:
        st.warning("No pude cargar el universo de precios (tabla vacía o cambió el formato).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Inputs
    c1, c2 = st.columns([0.42, 0.58], vertical_alignment="bottom")
    with c1:
        capital = st.number_input(
            "Capital (USD)",
            min_value=0.0,
            value=float(DEFAULT_CAPITAL_USD),
            step=1000.0,
            format="%.0f",
            key="cartera2_capital",
        )
    with c2:
        calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera2_calc")

    _spacer(8)

    # Selector de tickers (universo)
    st.markdown("### Selección de activos (universo completo)")
    opts = prices.index.tolist()

    selected = st.multiselect(
        "Tickers (Acciones / CEDEARs / Bonos / ONs / FCI / ETF)",
        options=opts,
        default=opts[:6] if len(opts) >= 6 else opts,
        key="cartera2_selected",
    )

    if not selected:
        st.info("Seleccioná al menos un ticker.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Asignación
    st.markdown("### Asignación por activo")
    st.caption("Editá la columna %. Ideal: que sume 100% (si no, escala automáticamente).")

    default_pct = round(100.0 / len(selected), 2)
    df_pct = pd.DataFrame({"Ticker": selected, "%": [default_pct] * len(selected)})

    edited = st.data_editor(
        df_pct,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "%": st.column_config.NumberColumn("%", min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
        },
        key="cartera2_pct_editor",
    )

    pct_map = {r["Ticker"]: float(r["%"]) for _, r in edited.iterrows()}

    _spacer(10)
    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)
    _spacer(6)

    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Construir cartera simple
    df = build_simple_portfolio(
        prices=prices,
        selected=selected,
        pct_map=pct_map,
        capital_usd=float(capital),
    )

    if df.empty:
        st.warning("No se pudo construir la cartera (probablemente faltan precios para los tickers elegidos).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # KPIs simples
    st.markdown("### Resumen")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(
            f"""
<div class="kpi">
  <div class="lbl">Capital (USD)</div>
  <div class="val">{fmt_money_int(float(capital))}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
<div class="kpi">
  <div class="lbl">Activos seleccionados</div>
  <div class="val">{len(selected)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    _spacer(12)

    # Tabla
    show = df.copy()
    show["%"] = pd.to_numeric(show["%"], errors="coerce").round(2)
    show["USD"] = pd.to_numeric(show["USD"], errors="coerce").round(0)
    show["Precio"] = pd.to_numeric(show["Precio"], errors="coerce").round(4)
    show["VN"] = pd.to_numeric(show["VN"], errors="coerce").round(4)

    h_tbl = _height_for_rows(len(show), row_h=34, header=42, pad=12, max_h=820)

    st.dataframe(
        show.drop(columns=["Ticker precio"], errors="ignore"),
        hide_index=True,
        use_container_width=True,
        height=h_tbl,
        column_config={
            "%": st.column_config.NumberColumn("%", format="%.2f"),
            "USD": st.column_config.NumberColumn("USD", format="$ %.0f"),
            "Precio": st.column_config.NumberColumn("Precio", format="%.4f"),
            "VN": st.column_config.NumberColumn("VN", format="%.4f"),
        },
    )

    _spacer(12)

    # Descargar Excel (simple)
    try:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            show.to_excel(writer, index=False, sheet_name="cartera_simple")
        out.seek(0)

        fname = f"NEIX_Cartera_Simple_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        st.download_button(
            "Descargar Excel",
            data=out.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="cartera2_xlsx",
        )
    except Exception as e:
        st.warning(f"No pude generar el Excel: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
