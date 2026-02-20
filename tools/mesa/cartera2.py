# tools/cartera_pesos.py
from __future__ import annotations

import io
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
DEFAULT_CAPITAL_ARS = 100_000_000.0  # ajustá a lo que usen

# =========================
# Parsing números AR (IOL)
# =========================
def parse_ar_number(x) -> float:
    """
    Convierte strings estilo AR:
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


# =========================
# Fetch IOL (PESOS)
# =========================
def _fetch_iol_table(url: str) -> pd.DataFrame:
    """
    Devuelve DF estandarizado:
      cols: Ticker, Precio, Volumen
    Si no puede parsear, df vacío.
    """
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    t = tables[0].copy()
    t.columns = [str(c).strip() for c in t.columns]

    # Formato típico IOL:
    # Símbolo | Último Operado | Monto Operado
    if "Símbolo" not in t.columns or "Último Operado" not in t.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Ticker"] = t["Símbolo"].astype(str).str.strip().str.upper()
    out["Precio"] = t["Último Operado"].apply(parse_ar_number)

    if "Monto Operado" in t.columns:
        out["Volumen"] = t["Monto Operado"].apply(parse_ar_number).fillna(0.0)
    else:
        out["Volumen"] = 0.0

    out = out.dropna(subset=["Precio"])
    out = out[~out["Ticker"].duplicated(keep="first")]
    return out[["Ticker", "Precio", "Volumen"]]


def fetch_universe_prices_pesos() -> pd.DataFrame:
    """
    Universo SOLO PESOS con coherencia por tipo:
    - Acciones
    - CEDEARs
    - Bonos (ARS)
    - ONs (ARS)

    Output:
      index = Ticker
      cols  = Precio, Volumen, Tipo
    """
    # URLs de IOL por categoría (PESOS)
    # (Si alguna cambia, esa categoría quedará vacía pero no rompe todo.)
    sources = [
        ("Acción",  "https://iol.invertironline.com/mercado/cotizaciones/argentina/acciones/todas"),
        ("CEDEAR",  "https://iol.invertironline.com/mercado/cotizaciones/argentina/cedears/todos"),
        ("Bono",    "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"),
        ("ON",      "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"),
    ]

    frames = []
    for tipo, url in sources:
        df = _fetch_iol_table(url)
        if df.empty:
            continue
        df["Tipo"] = tipo
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)

    # Si un ticker aparece repetido (raro, pero puede pasar),
    # priorizamos mayor volumen.
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo"]].sort_values("Volumen", ascending=False)


# =========================
# Cartera simple (ARS)
# =========================
@dataclass
class SimpleRowARS:
    ticker: str
    pct: float
    ars: float
    precio: float
    vn: float
    tipo: str


def build_simple_portfolio_ars(
    prices: pd.DataFrame,
    selected: list[str],
    pct_map: dict[str, float],
    capital_ars: float,
) -> pd.DataFrame:
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

    rows: list[SimpleRowARS] = []
    for t, pct in zip(selected, pcts):
        if pct <= 0:
            continue
        if t not in prices.index:
            continue

        px = float(prices.loc[t, "Precio"])
        tipo = str(prices.loc[t, "Tipo"]) if "Tipo" in prices.columns else "NA"

        ars_amt = float(capital_ars) * (float(pct) / 100.0)
        vn = (ars_amt / px) if px > 0 else np.nan

        rows.append(
            SimpleRowARS(
                ticker=t,
                pct=float(pct),
                ars=float(ars_amt),
                precio=float(px),
                vn=float(vn) if np.isfinite(vn) else np.nan,
                tipo=tipo,
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Ticker": [r.ticker for r in rows],
            "%": [r.pct for r in rows],
            "$": [r.ars for r in rows],
            "Precio": [r.precio for r in rows],
            "VN": [r.vn for r in rows],
            "Tipo": [r.tipo for r in rows],
        }
    )
    return df


# =========================
# UI (simple / prolija)
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
</style>
""",
        unsafe_allow_html=True,
    )


def _height_for_rows(n: int, row_h: int = 34, header: int = 42, pad: int = 18, max_h: int = 900) -> int:
    n = int(max(0, n))
    h = header + pad + row_h * max(1, n + 1)
    return int(min(max_h, h))


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.markdown('<div class="title">Cartera (Pesos)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Acciones / CEDEARs / Bonos / ONs</div>', unsafe_allow_html=True)
    with right:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_pesos_refresh")

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    # Cache universo
    if refresh or "cartera_pesos_prices" not in st.session_state:
        with st.spinner("Actualizando precios (Pesos)..."):
            st.session_state["cartera_pesos_prices"] = fetch_universe_prices_pesos()

    prices = st.session_state.get("cartera_pesos_prices")
    if prices is None or prices.empty:
        st.warning("No pude cargar el universo de precios (Pesos). Puede haber cambiado el formato de IOL.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Capital + botón
    c1, c2 = st.columns([0.42, 0.58], vertical_alignment="bottom")
    with c1:
        capital = st.number_input(
            "Capital (ARS)",
            min_value=0.0,
            value=float(DEFAULT_CAPITAL_ARS),
            step=1_000_000.0,
            format="%.0f",
            key="cartera_pesos_capital",
        )
    with c2:
        calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_pesos_calc")

    opts = prices.index.tolist()

    selected = st.multiselect(
        "Tickers (Acciones / CEDEARs / Bonos / ONs)",
        options=opts,
        default=opts[:6] if len(opts) >= 6 else opts,
        key="cartera_pesos_selected",
    )

    if not selected:
        st.info("Seleccioná al menos un ticker.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

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
        key="cartera_pesos_pct_editor",
    )

    pct_map = {r["Ticker"]: float(r["%"]) for _, r in edited.iterrows()}

    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = build_simple_portfolio_ars(
        prices=prices,
        selected=selected,
        pct_map=pct_map,
        capital_ars=float(capital),
    )

    if df.empty:
        st.warning("No se pudo construir la cartera (faltan precios para los tickers elegidos).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Formato display (evita confusión miles/decimales)
    show = df.copy()
    show["%"] = pd.to_numeric(show["%"], errors="coerce").round(2)
    show["$"] = pd.to_numeric(show["$"], errors="coerce").round(0)
    show["Precio"] = pd.to_numeric(show["Precio"], errors="coerce")
    show["VN"] = pd.to_numeric(show["VN"], errors="coerce")

    h_tbl = _height_for_rows(len(show), max_h=820)

    st.markdown("### Detalle de cartera")

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=h_tbl,
        column_config={
            "%": st.column_config.NumberColumn("%", format="%.2f"),
            "$": st.column_config.NumberColumn("$", format="$ %.0f"),
            # Precio/VN con decimales “visibles” sin mezclar miles:
            "Precio": st.column_config.NumberColumn("Precio", format="%.4f"),
            "VN": st.column_config.NumberColumn("VN", format="%.6f"),
        },
    )

    # Descargar Excel
    try:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            show.to_excel(writer, index=False, sheet_name="cartera_pesos")
        out.seek(0)

        fname = f"NEIX_Cartera_Pesos_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        st.download_button(
            "Descargar Excel",
            data=out.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="cartera_pesos_xlsx",
        )
    except Exception as e:
        st.warning(f"No pude generar el Excel: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
