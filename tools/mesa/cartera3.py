# tools/cartera_dolares.py
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# Config
# =========================
DEFAULT_CAPITAL_USD = 100_000.0
LOGO_PATH = os.path.join("data", "Neix_logo.png")  # tu logo en /data

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
    """Ticker 'real' = primer token antes del primer espacio (ej: 'VIST CEDEAR ...' -> 'VIST')."""
    s = (symbol_raw or "").strip().upper()
    if not s:
        return ""
    return s.split()[0]


def short_label(symbol_raw: str) -> str:
    """
    Display corto:
      - si tiene >= 2 tokens => "TOKEN1 TOKEN2"
      - si no => "TOKEN1"
    """
    s = (symbol_raw or "").strip().upper()
    toks = s.split()
    if len(toks) >= 2:
        return f"{toks[0]} {toks[1]}"
    return toks[0] if toks else ""


def fmt_ar_int(x: float) -> str:
    """Miles con punto, sin decimales."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = int(round(float(x)))
    except Exception:
        return ""
    return f"{v:,}".replace(",", ".")


def fmt_usd_money(x: float) -> str:
    """US$ con miles punto."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return "US$ " + f"{v:,.0f}".replace(",", ".")


def fmt_ar_pct(x: float) -> str:
    """Porcentaje con coma decimal."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return f"{v:.2f}".replace(".", ",") + "%"


def fmt_ar_2dec(x: float) -> str:
    """Número con 2 decimales, miles punto y coma decimal."""
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


# =========================
# Fetch IOL (USD via sufijos C/D)
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
    out["Label"] = out["SymbolRaw"].apply(short_label)

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


def fetch_universe_prices_usd(*, mode: str = "MEP") -> pd.DataFrame:
    """
    Universo USD tomando especies con sufijo:
      - mode="MEP" => tickers terminan en D
      - mode="CCL" => tickers terminan en C
      - mode="AMBOS" => C o D

    Output index = Ticker
      cols = Precio, Volumen, Tipo, Label
    """
    mode = (mode or "MEP").upper().strip()
    if mode not in {"MEP", "CCL", "AMBOS"}:
        mode = "MEP"

    if mode == "MEP":
        allowed_suffix = ("D",)
    elif mode == "CCL":
        allowed_suffix = ("C",)
    else:
        allowed_suffix = ("C", "D")

    # OJO: CEDEAR generalmente no tiene "MELID/MELIC"; por eso en USD suele quedar más chico el universo.
    sources = [
        ("Acción", "https://iol.invertironline.com/mercado/cotizaciones/argentina/acciones/todas"),
        ("CEDEAR", "https://iol.invertironline.com/mercado/cotizaciones/argentina/cedears/todos"),
        ("Bono",   "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos"),
        ("ON",     "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones-negociables/todos"),
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

    # Filtrar por sufijo USD
    allp = allp[allp["Ticker"].astype(str).str.upper().str.endswith(allowed_suffix)]

    if allp.empty:
        return pd.DataFrame()

    # Si un ticker aparece en más de una categoría, priorizamos mayor volumen.
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo", "Label"]].sort_values("Volumen", ascending=False)


# =========================
# Cartera simple (USD)
# =========================
@dataclass
class SimpleRowUSD:
    ticker: str
    label: str
    tipo: str
    pct: float
    usd: float
    precio: float
    vn: float


def build_simple_portfolio_usd(
    prices: pd.DataFrame,
    selected: list[str],
    pct_map: dict[str, float],
    capital_usd: float,
) -> pd.DataFrame:
    selected = [str(x).upper().strip() for x in selected if str(x).strip()]
    if not selected:
        return pd.DataFrame()

    pcts = np.array([max(0.0, float(pct_map.get(t, 0.0))) for t in selected], dtype=float)
    s = float(np.sum(pcts))
    pcts = (pcts / s * 100.0) if s > 0 else np.zeros_like(pcts)

    rows: list[SimpleRowUSD] = []
    for t, pct in zip(selected, pcts):
        if pct <= 0:
            continue
        if t not in prices.index:
            continue

        px = float(prices.loc[t, "Precio"])
        tipo = str(prices.loc[t, "Tipo"]) if "Tipo" in prices.columns else "NA"
        label = str(prices.loc[t, "Label"]) if "Label" in prices.columns else t

        usd_amt = float(capital_usd) * (float(pct) / 100.0)
        vn = (usd_amt / px) if px > 0 else np.nan

        rows.append(
            SimpleRowUSD(
                ticker=t,
                label=label,
                tipo=tipo,
                pct=float(pct),
                usd=float(usd_amt),
                precio=float(px),
                vn=float(vn) if np.isfinite(vn) else np.nan,
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "Ticker": [r.label for r in rows],  # display corto
            "Tipo": [r.tipo for r in rows],
            "%": [r.pct for r in rows],
            "US$": [r.usd for r in rows],
            "Precio": [r.precio for r in rows],  # precio en “USD del panel” (C/D)
            "VN": [r.vn for r in rows],
            "__ticker_base": [r.ticker for r in rows],
        }
    )


# =========================
# PDF (simple / prolijo)
# =========================
def build_cartera_usd_pdf_bytes(*, capital_usd: float, table_df: pd.DataFrame, logo_path: str | None = None) -> bytes:
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

    story.append(Paragraph("Cartera recomendada (USD)", styles["Heading2"]))
    story.append(Paragraph(f"Capital: {fmt_usd_money(capital_usd)}", styles["Normal"]))
    story.append(Spacer(1, 10))

    df = table_df.copy()
    cols = ["Ticker", "Tipo", "%", "US$", "Precio", "VN"]
    df = df[cols].copy()

    df["%"] = pd.to_numeric(df["%"], errors="coerce").apply(fmt_ar_pct)
    df["US$"] = pd.to_numeric(df["US$"], errors="coerce").apply(fmt_usd_money)
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").apply(fmt_ar_2dec)
    df["VN"] = pd.to_numeric(df["VN"], errors="coerce").apply(lambda v: fmt_ar_int(v))

    data = [cols] + df.fillna("").astype(str).values.tolist()

    col_widths = [
        usable_w * 0.20,  # Ticker
        usable_w * 0.16,  # Tipo
        usable_w * 0.10,  # %
        usable_w * 0.20,  # US$
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
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),  # %
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),  # US$
                ("ALIGN", (4, 1), (4, -1), "RIGHT"),  # Precio
                ("ALIGN", (5, 1), (5, -1), "RIGHT"),  # VN
            ]
        )
    )

    story.append(t)
    doc.build(story)
    pdf = buff.getvalue()
    buff.close()
    return pdf


# =========================
# Excel export (con formatos)
# =========================
def build_excel_bytes(table_df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    df = table_df.copy()

    if "__ticker_base" in df.columns:
        df = df.drop(columns=["__ticker_base"], errors="ignore")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cartera_usd")
        ws = writer.sheets["cartera_usd"]

        headers = {cell.value: idx + 1 for idx, cell in enumerate(ws[1])}
        col_pct = headers.get("%")
        col_money = headers.get("US$")
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

        for name, w in [("Ticker", 18), ("Tipo", 12), ("%", 8), ("US$", 16), ("Precio", 14), ("VN", 12)]:
            c = headers.get(name)
            if c:
                ws.column_dimensions[chr(64 + c)].width = w

    out.seek(0)
    return out.getvalue()


# =========================
# UI (minimal / ordenada)
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
        st.markdown('<div class="title">Herramienta para armar carteras (USD)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Panel dólar vía sufijo: MEP (D) / CCL (C)</div>', unsafe_allow_html=True)
    with right:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_usd_refresh")

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns([0.26, 0.37, 0.37], vertical_alignment="bottom")
    with m1:
        mode = st.selectbox(
            "Tipo de dólar",
            options=["MEP", "CCL", "AMBOS"],
            index=0,
            key="cartera_usd_mode",
        )
    with m2:
        capital = st.number_input(
            "Capital (USD)",
            min_value=0.0,
            value=float(DEFAULT_CAPITAL_USD),
            step=1_000.0,
            format="%.0f",
            key="cartera_usd_capital",
        )
    with m3:
        calc = st.button("Calcular cartera", type="primary", use_container_width=True, key="cartera_usd_calc")

    # Cache universo
    cache_key = f"cartera_usd_prices_{mode}"
    if refresh or cache_key not in st.session_state:
        with st.spinner(f"Actualizando precios (USD - {mode})..."):
            st.session_state[cache_key] = fetch_universe_prices_usd(mode=mode)

    prices = st.session_state.get(cache_key)
    if prices is None or prices.empty:
        st.warning("No pude cargar el universo USD (C/D). Puede haber cambiado el formato de IOL o no haber volumen.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    opts = prices.index.tolist()
    label_map = {tk: str(prices.loc[tk, "Label"]) for tk in opts}

    selected = st.multiselect(
        "Tickers (USD)",
        options=opts,
        default=opts[:6] if len(opts) >= 6 else opts,
        format_func=lambda tk: label_map.get(tk, tk),
        key="cartera_usd_selected",
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
        key="cartera_usd_pct_editor",
    )

    pct_map = {r["Ticker"]: float(r["%"]) for _, r in edited.iterrows()}

    if not calc:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = build_simple_portfolio_usd(
        prices=prices,
        selected=selected,
        pct_map=pct_map,
        capital_usd=float(capital),
    )

    if df.empty:
        st.warning("No se pudo construir la cartera (faltan precios para los tickers elegidos).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### Detalle de cartera")

    show = df.copy().drop(columns=["__ticker_base"], errors="ignore")
    show["%"] = pd.to_numeric(show["%"], errors="coerce").apply(fmt_ar_pct)
    show["US$"] = pd.to_numeric(show["US$"], errors="coerce").apply(fmt_usd_money)
    show["Precio"] = pd.to_numeric(show["Precio"], errors="coerce").apply(fmt_ar_2dec)
    show["VN"] = pd.to_numeric(show["VN"], errors="coerce").apply(fmt_ar_int)

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
            "US$": st.column_config.TextColumn("US$"),
            "Precio": st.column_config.TextColumn("Precio"),
            "VN": st.column_config.TextColumn("VN"),
        },
    )

    cxl, cpdf = st.columns(2)
    with cxl:
        try:
            xlsx = build_excel_bytes(df)
            fname = f"NEIX_Cartera_USD_{mode}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            st.download_button(
                "Descargar Excel",
                data=xlsx,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="cartera_usd_xlsx",
            )
        except Exception as e:
            st.warning(f"No pude generar el Excel: {e}")

    with cpdf:
        try:
            pdf = build_cartera_usd_pdf_bytes(
                capital_usd=float(capital),
                table_df=df.drop(columns=["__ticker_base"], errors="ignore"),
                logo_path=LOGO_PATH,
            )
            fname = f"NEIX_Cartera_USD_{mode}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                "Descargar PDF",
                data=pdf,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                key="cartera_usd_pdf",
            )
        except Exception as e:
            st.warning(f"No pude generar el PDF: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
