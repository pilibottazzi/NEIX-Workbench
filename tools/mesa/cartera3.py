# tools/cartera_dolares_mep.py
from __future__ import annotations

import io
import os
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# PDF (ReportLab)
# =========================
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# Config
# =========================
DEFAULT_CAPITAL_USD = 100_000.0
LOGO_PATH = os.path.join("data", "Neix_logo.png")

MONEY_COL_NAME = "US$ (MEP)"  # ✅ header pedido para dólares MEP

# Excel mapping Pesos -> USD
ESPECIES_XLSX_PATH = os.path.join("data", "Especies.xlsx")
ESPECIES_SHEET = 0  # o "Hoja1"

# Tipos que cotizan "por 100 V/N"
PRICE_PER_100_VN_TYPES = {"BONO", "ON"}


# =========================
# Helpers
# =========================
def parse_ar_number(x) -> float:
    """
    Convierte strings estilo AR a float:
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


def base_ticker(symbol_raw: str) -> str:
    """Primer token antes del primer espacio (ej: 'AL30D CI' -> 'AL30D')."""
    s = (symbol_raw or "").strip().upper()
    if not s:
        return ""
    return s.split()[0]


def short_label(symbol_raw: str) -> str:
    """Display corto: 2 tokens como máximo."""
    s = (symbol_raw or "").strip().upper()
    toks = s.split()
    if len(toks) >= 2:
        return f"{toks[0]} {toks[1]}"
    return toks[0] if toks else ""


def display_label(symbol_raw: str) -> str:
    """
    Display final:
      - Si el 2do token es "CEDEAR", lo sacamos: "VIST CEDEAR ..." -> "VIST"
      - Si no, short_label: "AL30D CI" -> "AL30D CI"
    """
    s = (symbol_raw or "").strip().upper()
    toks = s.split()
    if not toks:
        return ""
    if len(toks) >= 2 and toks[1] == "CEDEAR":
        return toks[0]
    return short_label(symbol_raw)


def is_mep_ticker(ticker: str) -> bool:
    """MEP = especie D (termina en D). Excluye CCL (C)."""
    t = (ticker or "").strip().upper()
    return bool(t) and t.endswith("D")


def fmt_ar_int(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = int(round(float(x)))
    except Exception:
        return ""
    return f"{v:,}".replace(",", ".")


def fmt_usd_money(x: float) -> str:
    """US$ con miles punto (sin decimales)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return "US$ " + f"{v:,.0f}".replace(",", ".")


def fmt_ar_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return f"{v:.2f}".replace(".", ",") + "%"


def fmt_ar_2dec(x: float) -> str:
    """2 decimales, miles punto y coma decimal."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    s = f"{v:,.2f}"            # 12,345.67
    s = s.replace(",", "X")    # 12X345.67
    s = s.replace(".", ",")    # 12X345,67
    s = s.replace("X", ".")    # 12.345,67
    return s


def unit_price_for_vn(*, tipo: str, precio_cotizado: float) -> float:
    """
    Precio unitario para VN:
      - Bonos/ON: cotizan cada 100 V/N => unit = precio/100
      - Acciones/CEDEAR: unit = precio
    """
    t = (tipo or "").strip().upper()
    px = float(precio_cotizado) if np.isfinite(precio_cotizado) else np.nan
    if not np.isfinite(px) or px <= 0:
        return np.nan
    if t in PRICE_PER_100_VN_TYPES:
        return px / 100.0
    return px


# =========================
# Pesos -> USD mapping (Especies.xlsx)
# =========================
def load_especies_map(path: str = ESPECIES_XLSX_PATH) -> dict[str, str]:
    """
    Lee data/Especies.xlsx con columnas: Pesos | Usd
    Devuelve dict: { 'AL30': 'AL30D', ... }
    """
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_excel(path, sheet_name=ESPECIES_SHEET, dtype=str)
    except Exception:
        return {}

    cols = {str(c).strip().lower(): c for c in df.columns}
    col_pesos = cols.get("pesos")
    col_usd = cols.get("usd")

    if not col_pesos or not col_usd:
        return {}

    out: dict[str, str] = {}
    for _, r in df.iterrows():
        p = (r.get(col_pesos) or "").strip().upper()
        u = (r.get(col_usd) or "").strip().upper()
        if p and u:
            out[p] = u
    return out


def resolve_usd_ticker_strict(ticker_input: str, especies_map: dict[str, str], prices: pd.DataFrame) -> str:
    """
    ✅ REGLA PEDIDA (estricta):
    1) SIEMPRE intentar pasar por Especies.xlsx (Pesos -> USD).
    2) Si no está en el Excel: tratarlo como si fuera PESOS y buscar su versión USD:
         - base = ticker sin 'D' si la trae
         - usd = base + 'D'
       (si existe en prices, mejor; si no, igual devolvemos ese para que quede como faltante)
    """
    t = (ticker_input or "").strip().upper()
    if not t:
        return ""

    # 1) Siempre prioriza el Excel (match exacto)
    if t in especies_map:
        return especies_map[t]

    # 1b) Si viene con D, igual se interpreta como "posible input en pesos mal tipeado"
    #     => probamos el Excel con el base sin D
    if t.endswith("D"):
        base = t[:-1]
        if base in especies_map:
            return especies_map[base]
    else:
        base = t

    # 2) Si no está en Excel => "como si estuviera en pesos": base + D
    usd = base + "D"

    # Si el universo lo trae, perfecto (igual devolvemos el mismo string)
    if prices is not None and not prices.empty and usd in prices.index:
        return usd

    return usd


# =========================
# Fetch IOL (DÓLAR MEP = especie D)
# =========================
def _fetch_iol_table(url: str) -> pd.DataFrame:
    """
    Devuelve DF estandarizado:
      cols: SymbolRaw, Ticker, Label, Precio, Volumen
    """
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    t = tables[0].copy()
    t.columns = [str(c).strip() for c in t.columns]

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

    # si se repite ticker dentro de la misma tabla, dejamos el mayor volumen
    out = out.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    out = out.drop_duplicates(subset=["Ticker"], keep="first")

    return out[["SymbolRaw", "Ticker", "Label", "Precio", "Volumen"]]


def fetch_universe_prices_mep() -> pd.DataFrame:
    """
    Universo DÓLAR MEP (solo especie D) con tipos:
      - Acción (tickers ...D)
      - CEDEAR (si existiera ...D)
      - Bono (AL30D, GD30D, etc.)
      - ON (si existiera ...D)

    Output index = Ticker
      cols = Precio, Volumen, Tipo, Label
    """
    sources = [
        ("Acción", "https://iol.invertironline.com/mercado/cotizaciones"),
        ("CEDEAR", "https://iol.invertironline.com/mercado/cotizaciones/argentina/cedears/todos"),
        ("Bono", "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos"),
        ("ON", "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones-negociables/todos"),
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

    # ✅ SOLO MEP: especie D
    allp = allp[allp["Ticker"].astype(str).apply(is_mep_ticker)].copy()
    if allp.empty:
        return pd.DataFrame()

    # Si un ticker aparece en más de una categoría, priorizamos mayor volumen.
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo", "Label"]].sort_values("Volumen", ascending=False)


# =========================
# Cartera simple (USD MEP)
# =========================
@dataclass
class SimpleRowUSD:
    ticker: str
    label: str
    tipo: str
    pct: float
    usd: float
    precio_cotizado: float
    precio_unitario: float
    vn: float


def build_simple_portfolio_usd(
    prices: pd.DataFrame,
    selected_pesos: list[str],
    pct_map: dict[str, float],
    capital_usd: float,
    especies_map: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    selected_pesos: tickers elegidos (en formato pesos, según el Excel)
    retorna: (df_cartera, missing_usd_tickers)
    """
    selected_raw = [str(x).upper().strip() for x in selected_pesos if str(x).strip()]
    if not selected_raw:
        return pd.DataFrame(), []

    # ✅ resolver SIEMPRE via Excel; si no existe, base + D
    selected_usd = [resolve_usd_ticker_strict(t, especies_map, prices) for t in selected_raw]

    # Pcts se cargan por lo que el usuario editó (los tickers “pesos” de la UI)
    pcts = np.array([max(0.0, float(pct_map.get(t, 0.0))) for t in selected_raw], dtype=float)
    s = float(np.sum(pcts))
    pcts = (pcts / s * 100.0) if s > 0 else np.zeros_like(pcts)

    missing = [t_usd for t_usd in selected_usd if t_usd and t_usd not in prices.index]

    rows: list[SimpleRowUSD] = []
    for t_pesos, t_usd, pct in zip(selected_raw, selected_usd, pcts):
        if pct <= 0:
            continue
        if t_usd not in prices.index:
            continue

        px = float(prices.loc[t_usd, "Precio"])
        tipo = str(prices.loc[t_usd, "Tipo"]) if "Tipo" in prices.columns else "NA"
        label = str(prices.loc[t_usd, "Label"]) if "Label" in prices.columns else t_usd

        usd_amt = float(capital_usd) * (float(pct) / 100.0)

        px_unit = unit_price_for_vn(tipo=tipo, precio_cotizado=px)
        vn = (usd_amt / px_unit) if (np.isfinite(px_unit) and px_unit > 0) else np.nan

        rows.append(
            SimpleRowUSD(
                ticker=t_usd,  # USD real
                label=label,
                tipo=tipo,
                pct=float(pct),
                usd=float(usd_amt),
                precio_cotizado=float(px),
                precio_unitario=float(px_unit) if np.isfinite(px_unit) else np.nan,
                vn=float(vn) if np.isfinite(vn) else np.nan,
            )
        )

    if not rows:
        return pd.DataFrame(), missing

    df = pd.DataFrame(
        {
            "Ticker": [r.label for r in rows],
            "Tipo": [r.tipo for r in rows],
            "%": [r.pct for r in rows],
            MONEY_COL_NAME: [r.usd for r in rows],
            "Precio": [r.precio_cotizado for r in rows],
            "VN": [r.vn for r in rows],
            "__ticker_usd": [r.ticker for r in rows],  # para debug interno si necesitás
        }
    )
    return df, missing


# =========================
# PDF
# =========================
def build_cartera_mep_pdf_bytes(*, capital_usd: float, table_df: pd.DataFrame, logo_path: str | None = None) -> bytes:
    buff = io.BytesIO()

    left = right = 1.3 * cm
    top = bottom = 1.2 * cm
    page_w, _ = A4
    usable_w = page_w - left - right

    doc = SimpleDocTemplate(
        buff,
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
    )
    styles = getSampleStyleSheet()
    story = []

    if logo_path and os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=6.2 * cm, height=1.6 * cm)
            tlogo = Table([[logo]], colWidths=[usable_w])
            tlogo.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.append(tlogo)
        except Exception:
            pass

    story.append(Paragraph("Cartera recomendada (Dólar MEP)", styles["Heading2"]))
    story.append(Paragraph(f"Capital: {fmt_usd_money(capital_usd)}", styles["Normal"]))
    story.append(Spacer(1, 10))

    df = table_df.copy()

    cols = ["Ticker", "Tipo", "%", MONEY_COL_NAME, "Precio", "VN"]
    df = df[cols].copy()

    df["%"] = pd.to_numeric(df["%"], errors="coerce").apply(fmt_ar_pct)
    df[MONEY_COL_NAME] = pd.to_numeric(df[MONEY_COL_NAME], errors="coerce").apply(fmt_usd_money)
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").apply(fmt_ar_2dec)
    df["VN"] = pd.to_numeric(df["VN"], errors="coerce").apply(lambda v: fmt_ar_int(v))

    data = [cols] + df.fillna("").astype(str).values.tolist()

    col_widths = [
        usable_w * 0.22,  # Ticker
        usable_w * 0.14,  # Tipo
        usable_w * 0.10,  # %
        usable_w * 0.20,  # US$ (MEP)
        usable_w * 0.18,  # Precio
        usable_w * 0.16,  # VN
    ]

    t = Table(data, repeatRows=1, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                ("ALIGN", (4, 1), (4, -1), "RIGHT"),
                ("ALIGN", (5, 1), (5, -1), "RIGHT"),
            ]
        )
    )

    story.append(t)
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Nota: Bonos y ON cotizan por cada 100 de V/N (para VN se usa Precio/100). "
            "Acciones y CEDEARs cotizan por unidad (no se ajusta el precio).",
            styles["Normal"],
        )
    )

    doc.build(story)
    pdf = buff.getvalue()
    buff.close()
    return pdf


# =========================
# Excel export
# =========================
def build_excel_bytes(table_df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    df = table_df.copy()

    # limpieza de columna interna
    df = df.drop(columns=["__ticker_usd"], errors="ignore")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cartera_mep")
        ws = writer.sheets["cartera_mep"]

        headers = {cell.value: idx + 1 for idx, cell in enumerate(ws[1])}  # 1-based
        col_pct = headers.get("%")
        col_money = headers.get(MONEY_COL_NAME)
        col_price = headers.get("Precio")
        col_vn = headers.get("VN")

        for r in range(2, ws.max_row + 1):
            if col_pct:
                ws.cell(r, col_pct).number_format = "0.00"
            if col_money:
                ws.cell(r, col_money).number_format = "#,##0"
            if col_price:
                ws.cell(r, col_price).number_format = "#,##0.00"
            if col_vn:
                ws.cell(r, col_vn).number_format = "#,##0"

        for name, w in [("Ticker", 18), ("Tipo", 12), ("%", 8), (MONEY_COL_NAME, 16), ("Precio", 14), ("VN", 12)]:
            c = headers.get(name)
            if c:
                ws.column_dimensions[chr(64 + c)].width = w

    out.seek(0)
    return out.getvalue()


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
  div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }
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
        st.markdown('<div class="title">Herramienta para armar carteras (Dólar MEP)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">SOLO USD MEP. Siempre convierte vía Especies.xlsx (Pesos → USD). Si no existe, usa ticker + D.</div>', unsafe_allow_html=True)
    with right:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_mep_refresh")

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    if refresh or "cartera_mep_prices" not in st.session_state:
        with st.spinner("Actualizando precios (MEP / especie D)..."):
            st.session_state["cartera_mep_prices"] = fetch_universe_prices_mep()

    prices = st.session_state.get("cartera_mep_prices")
    if prices is None or prices.empty:
        st.warning("No pude cargar el universo MEP (especie D). Puede haber cambiado el formato de IOL.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ✅ mapa pesos -> usd (desde Excel)
    especies_map = load_especies_map()
    if not especies_map:
        st.warning("No pude leer data/Especies.xlsx (necesito columnas: Pesos | Usd).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    c1, c2 = st.columns([0.42, 0.58], vertical_alignment="bottom")
    with c1:
        capital = st.number_input(
            "Capital (USD MEP)",
            min_value=0.0,
            value=float(DEFAULT_CAPITAL_USD),
            step=1_000.0,
            format="%.0f",
            key="cartera_mep_capital",
        )
    with c2:
        calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_mep_calc")

    # ✅ UI: solo tickers en PESOS (los del Excel). Internamente calcula en USD MEP.
    opts_pesos = sorted(especies_map.keys())
    label_map = {p: f"{p} → {especies_map.get(p, p + 'D')}" for p in opts_pesos}

    selected = st.multiselect(
        "Activos (selección en PESOS; cálculo SIEMPRE en USD MEP)",
        options=opts_pesos,
        default=opts_pesos[:6] if len(opts_pesos) >= 6 else opts_pesos,
        format_func=lambda tk: label_map.get(tk, tk),
        key="cartera_mep_selected",
    )

    if not selected:
        st.info("Seleccioná al menos un activo.")
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
        key="cartera_mep_pct_editor",
    )

    pct_map = {r["Ticker"]: float(r["%"]) for _, r in edited.iterrows()}

    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df, missing = build_simple_portfolio_usd(
        prices=prices,
        selected_pesos=selected,
        pct_map=pct_map,
        capital_usd=float(capital),
        especies_map=especies_map,
    )

    if missing:
        st.warning("No encontré precio USD (MEP) para: " + ", ".join(sorted(set(missing))))

    if df.empty:
        st.warning("No se pudo construir la cartera (ningún ticker resolvió a un precio USD MEP disponible).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### Detalle de cartera (USD MEP)")
    st.caption(
        "Regla de VN: Bonos/ON cotizan por cada 100 de V/N ⇒ para calcular VN se usa Precio/100. "
        "Acciones/CEDEARs cotizan por unidad ⇒ no se ajusta el precio."
    )

    show = df.copy().drop(columns=["__ticker_usd"], errors="ignore")

    show["%"] = pd.to_numeric(show["%"], errors="coerce").apply(fmt_ar_pct)
    show[MONEY_COL_NAME] = pd.to_numeric(show[MONEY_COL_NAME], errors="coerce").apply(fmt_usd_money)
    show["Precio"] = pd.to_numeric(show["Precio"], errors="coerce").apply(fmt_ar_2dec)
    show["VN"] = pd.to_numeric(show["VN"], errors="coerce").apply(lambda v: fmt_ar_int(v))

    h_tbl = _height_for_rows(len(show), max_h=820)

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=h_tbl,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Tipo": st.column_config.TextColumn("Tipo"),
            "%": st.column_config.TextColumn("%"),
            MONEY_COL_NAME: st.column_config.TextColumn(MONEY_COL_NAME),
            "Precio": st.column_config.TextColumn("Precio"),
            "VN": st.column_config.TextColumn("VN"),
        },
    )

    cxl, cpdf = st.columns(2)
    with cxl:
        try:
            xlsx = build_excel_bytes(df)
            fname = f"NEIX_Cartera_MEP_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            st.download_button(
                "Descargar Excel",
                data=xlsx,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="cartera_mep_xlsx",
            )
        except Exception as e:
            st.warning(f"No pude generar el Excel: {e}")

    with cpdf:
        try:
            pdf = build_cartera_mep_pdf_bytes(
                capital_usd=float(capital),
                table_df=df.drop(columns=["__ticker_usd"], errors="ignore"),
                logo_path=LOGO_PATH,
            )
            fname = f"NEIX_Cartera_MEP_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                "Descargar PDF",
                data=pdf,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                key="cartera_mep_pdf",
            )
        except Exception as e:
            st.warning(f"No pude generar el PDF: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
