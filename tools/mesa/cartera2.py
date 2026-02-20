# tools/cartera_pesos.py
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
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

# =========================
# Config
# =========================
DEFAULT_CAPITAL_ARS = 100_000_000.0
LOGO_PATH = os.path.join("data", "Neix_logo.png")  # si está en /data

# Prioridad para resolver duplicados entre categorías
# (si un ticker aparece como Acción y CEDEAR, se queda Acción)
TYPE_PRIORITY = {
    "Acción": 1,
    "CEDEAR": 2,
    "ON": 3,
    "Bono": 4,
}

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
# Formateo AR (para evitar confusión . / ,)
# =========================
def _fmt_ar_int(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = int(round(float(x)))
    except Exception:
        return ""
    # miles con punto
    return f"{v:,}".replace(",", ".")


def _fmt_ar_money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = int(round(float(x)))
    except Exception:
        return ""
    return "$ " + f"{v:,}".replace(",", ".")


def _fmt_ar_num(x, dec: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    # formato AR: miles '.' y decimales ','
    s = f"{v:,.{dec}f}"              # 12,345.6789 (US)
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # 12.345,6789
    return s


def _fmt_pct(x, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return _fmt_ar_num(v, dec) + "%"


# =========================
# Fetch IOL (PESOS)
# =========================
def _fetch_iol_table(url: str) -> pd.DataFrame:
    """
    Devuelve DF estandarizado:
      cols: Ticker, Precio, Volumen, Nombre(opc)
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
    # Símbolo | Último Operado | Monto Operado | (a veces Nombre/Descripción)
    if "Símbolo" not in t.columns or "Último Operado" not in t.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Ticker"] = t["Símbolo"].astype(str).str.strip().str.upper()
    out["Precio"] = t["Último Operado"].apply(parse_ar_number)

    if "Monto Operado" in t.columns:
        out["Volumen"] = t["Monto Operado"].apply(parse_ar_number).fillna(0.0)
    else:
        out["Volumen"] = 0.0

    # Nombre opcional (si aparece)
    name_col = None
    for c in ["Nombre", "Especie", "Descripción", "Descripcion", "Instrumento"]:
        if c in t.columns:
            name_col = c
            break
    if name_col:
        out["Nombre"] = t[name_col].astype(str).str.strip()
    else:
        out["Nombre"] = ""

    out = out.dropna(subset=["Precio"])
    out = out[~out["Ticker"].duplicated(keep="first")]
    return out[["Ticker", "Precio", "Volumen", "Nombre"]]


def fetch_universe_prices_pesos() -> pd.DataFrame:
    """
    Universo SOLO PESOS con coherencia por tipo:
    - Acciones
    - CEDEARs
    - Bonos (ARS)
    - ONs (ARS)

    Output:
      index = Ticker
      cols  = Precio, Volumen, Tipo, Nombre
    """
    sources = [
        ("Acción", "https://iol.invertironline.com/mercado/cotizaciones/argentina/acciones/todas"),
        ("CEDEAR", "https://iol.invertironline.com/mercado/cotizaciones/argentina/cedears/todos"),
        ("Bono",   "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"),
        ("ON",     "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"),
    ]

    frames = []
    for tipo, url in sources:
        df = _fetch_iol_table(url)
        if df.empty:
            continue
        df["Tipo"] = tipo
        df["TipoRank"] = TYPE_PRIORITY.get(tipo, 99)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)

    # Dedup por ticker: primero tipo (prioridad), luego volumen
    allp = allp.sort_values(["Ticker", "TipoRank", "Volumen"], ascending=[True, True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo", "Nombre"]].sort_values("Volumen", ascending=False)


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

    return pd.DataFrame(
        {
            "Ticker": [r.ticker for r in rows],
            "Tipo": [r.tipo for r in rows],
            "%": [r.pct for r in rows],
            "$": [r.ars for r in rows],
            "Precio": [r.precio for r in rows],
            "VN": [r.vn for r in rows],
        }
    )


# =========================
# PDF helpers
# =========================
def _df_to_table_data(df: pd.DataFrame, max_rows: int = 200) -> list[list[str]]:
    if df is None or df.empty:
        return [["(sin datos)"]]
    d = df.copy().head(max_rows).fillna("")
    cols = list(d.columns)
    data = [cols]
    for _, r in d.iterrows():
        data.append([str(r[c]) for c in cols])
    return data


def _pie_by_tipo(df_display: pd.DataFrame, usable_w: float) -> Drawing | None:
    """
    Pie por Tipo ponderado por $ asignado.
    df_display debe traer columnas: Tipo, $ (formato string "$ 1.234.567")
    """
    if df_display is None or df_display.empty:
        return None
    if "Tipo" not in df_display.columns or "$" not in df_display.columns:
        return None

    def _to_float_ars(s: str) -> float:
        ss = str(s).replace("$", "").strip()
        ss = ss.replace(".", "").replace(",", ".")
        try:
            return float(ss)
        except Exception:
            return 0.0

    tmp = df_display.copy()
    tmp["_ars"] = tmp["$"].apply(_to_float_ars)
    g = tmp.groupby("Tipo")["_ars"].sum().sort_values(ascending=False)

    if g.sum() <= 0:
        return None

    labels = [f"{k} ({(v / g.sum() * 100):.0f}%)" for k, v in g.items()]
    values = [float(v) for v in g.values]

    d = Drawing(usable_w, 2.8 * cm)
    pie = Pie()
    pie.x = 0
    pie.y = 0
    pie.width = 7.0 * cm
    pie.height = 2.8 * cm
    pie.data = values
    pie.labels = labels
    pie.sideLabels = 1
    pie.simpleLabels = 0
    pie.slices.strokeWidth = 0.25
    d.add(pie)
    return d


def build_cartera_pesos_pdf_bytes(
    *,
    capital_ars: float,
    df_display: pd.DataFrame,
    logo_path: str | None = None,
    include_pie: bool = True,
) -> bytes:
    """
    PDF minimal:
    - Logo
    - "Cartera (Pesos)" + Capital
    - Un solo cuadro (tabla) ajustado al ancho
    - VN entero (sin decimales)
    - Sin columna Nombre
    - Pie opcional por Tipo
    """
    buff = io.BytesIO()

    left = right = 1.35 * cm
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

    # Logo
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

    story.append(Paragraph("Cartera (Pesos)", styles["Heading2"]))
    story.append(Paragraph(f"Capital: {_fmt_ar_money(capital_ars)}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Pie chart por Tipo (opcional)
    if include_pie:
        pie = _pie_by_tipo(df_display, usable_w)
        if pie:
            story.append(pie)
            story.append(Spacer(1, 8))

    # Tabla (un solo cuadro) — sin Nombre
    cols_order = ["Ticker", "Tipo", "%", "$", "Precio", "VN"]
    dft = df_display.copy()
    dft = dft[[c for c in cols_order if c in dft.columns]]

    data = _df_to_table_data(dft, max_rows=200)

    # colWidths exactos (suman usable_w)
    col_widths = [
        usable_w * 0.14,  # Ticker
        usable_w * 0.16,  # Tipo
        usable_w * 0.10,  # %
        usable_w * 0.20,  # $
        usable_w * 0.20,  # Precio
        usable_w * 0.20,  # VN
    ]

    t = Table(data, repeatRows=1, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8.8),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                # Alinear números a derecha
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (1, -1), "LEFT"),
            ]
        )
    )

    story.append(t)
    doc.build(story)

    pdf = buff.getvalue()
    buff.close()
    return pdf


# =========================
# Excel export helper
# =========================
def build_excel_bytes(df: pd.DataFrame, sheet_name: str = "cartera_pesos") -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
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

  /* Dataframes */
  div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }

  /* Buttons */
  .stButton > button { border-radius: 14px; padding: 0.60rem 1.0rem; }

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
        st.markdown('<div class="sub">Acciones / CEDEARs / Bonos / ONs — sin cálculos, solo asignación</div>', unsafe_allow_html=True)
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

    # Selector
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

    # ===== Display formateado (AR) =====
    show = df.copy()
    show["%"] = pd.to_numeric(show["%"], errors="coerce").round(2)
    show["$"] = pd.to_numeric(show["$"], errors="coerce").round(0)
    show["Precio"] = pd.to_numeric(show["Precio"], errors="coerce")
    show["VN"] = pd.to_numeric(show["VN"], errors="coerce").round(0)

    # Version "bonita" para PDF/tabla (strings con AR)
    show_fmt = show.copy()
    show_fmt["%"] = show_fmt["%"].apply(lambda v: _fmt_ar_num(v, 2))
    show_fmt["$"] = show_fmt["$"].apply(_fmt_ar_money)
    show_fmt["Precio"] = show_fmt["Precio"].apply(lambda v: _fmt_ar_num(v, 4))
    show_fmt["VN"] = show_fmt["VN"].apply(_fmt_ar_int)

    # Tabla (en pantalla) — mantenemos números como números para filtros,
    # pero mostramos formato AR con st.dataframe config
    st.markdown("### Detalle de cartera")

    h_tbl = _height_for_rows(len(show), max_h=820)

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=h_tbl,
        column_config={
            "%": st.column_config.NumberColumn("%", format="%.2f"),
            "$": st.column_config.NumberColumn("$", format="$ %.0f"),
            "Precio": st.column_config.NumberColumn("Precio", format="%.4f"),
            "VN": st.column_config.NumberColumn("VN", format="%.0f"),
        },
    )

    # ===== Descargas: Excel y PDF =====
    cxl, cpdf = st.columns(2, vertical_alignment="center")

    with cxl:
        try:
            xlsx_bytes = build_excel_bytes(show)
            fname = f"NEIX_Cartera_Pesos_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            st.download_button(
                "Descargar Excel",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="cartera_pesos_xlsx",
            )
        except Exception as e:
            st.warning(f"No pude generar el Excel: {e}")

    with cpdf:
        try:
            pdf_bytes = build_cartera_pesos_pdf_bytes(
                capital_ars=float(capital),
                df_display=show_fmt.drop(columns=["Nombre"], errors="ignore"),  # por si existiera
                logo_path=LOGO_PATH,
                include_pie=True,  # pie por Tipo ponderado por $
            )
            fname = f"NEIX_Cartera_Pesos_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                "Descargar PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                key="cartera_pesos_pdf",
            )
        except Exception as e:
            st.warning(f"No pude generar el PDF: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
