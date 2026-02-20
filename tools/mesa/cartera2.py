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

# =========================
# Config
# =========================
DEFAULT_CAPITAL_ARS = 100_000_000.0
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
      - si tiene >= 2 tokens => "TOKEN1 TOKEN2" (ej: "VIST CEDEAR ... " -> "VIST CEDEAR")
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


def fmt_ar_money(x: float) -> str:
    """$ con miles punto."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    return "$ " + f"{v:,.0f}".replace(",", ".")


def fmt_ar_pct(x: float) -> str:
    """Porcentaje con coma decimal."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    # 2 decimales, reemplazar punto por coma
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
# Fetch IOL (PESOS)
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


def fetch_universe_prices_pesos() -> pd.DataFrame:
    """
    Universo PESOS con tipos:
      - Acción
      - CEDEAR
      - Bono
      - ON

    Output index = Ticker
      cols = Precio, Volumen, Tipo, Label
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
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)

    # Si un ticker aparece en más de una categoría, priorizamos mayor volumen.
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo", "Label"]].sort_values("Volumen", ascending=False)


# =========================
# Cartera simple (ARS)
# =========================
@dataclass
class SimpleRowARS:
    ticker: str
    label: str
    tipo: str
    pct: float
    ars: float
    precio: float
    vn: float


def build_simple_portfolio_ars(
    prices: pd.DataFrame,
    selected: list[str],
    pct_map: dict[str, float],
    capital_ars: float,
) -> pd.DataFrame:
    selected = [str(x).upper().strip() for x in selected if str(x).strip()]
    if not selected:
        return pd.DataFrame()

    pcts = np.array([max(0.0, float(pct_map.get(t, 0.0))) for t in selected], dtype=float)
    s = float(np.sum(pcts))
    pcts = (pcts / s * 100.0) if s > 0 else np.zeros_like(pcts)

    rows: list[SimpleRowARS] = []
    for t, pct in zip(selected, pcts):
        if pct <= 0:
            continue
        if t not in prices.index:
            continue

        px = float(prices.loc[t, "Precio"])
        tipo = str(prices.loc[t, "Tipo"]) if "Tipo" in prices.columns else "NA"
        label = str(prices.loc[t, "Label"]) if "Label" in prices.columns else t

        ars_amt = float(capital_ars) * (float(pct) / 100.0)
        vn = (ars_amt / px) if px > 0 else np.nan

        rows.append(
            SimpleRowARS(
                ticker=t,
                label=label,
                tipo=tipo,
                pct=float(pct),
                ars=float(ars_amt),
                precio=float(px),
                vn=float(vn) if np.isfinite(vn) else np.nan,
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "Ticker": [r.label for r in rows],  # ✅ display corto (ej: "VIST CEDEAR")
            "Tipo": [r.tipo for r in rows],
            "%": [r.pct for r in rows],
            "$": [r.ars for r in rows],
            "Precio": [r.precio for r in rows],
            "VN": [r.vn for r in rows],
            "__ticker_base": [r.ticker for r in rows],  # interno (para exports si querés)
        }
    )


# =========================
# PDF (simple / prolijo)
# =========================
def build_cartera_pesos_pdf_bytes(*, capital_ars: float, table_df: pd.DataFrame, logo_path: str | None = None) -> bytes:
    """
    PDF:
    - Logo
    - Título Cartera (Pesos) + Capital
    - UNA sola tabla ajustada al ancho útil
    - Sin gráficos
    """
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

    story.append(Paragraph("Cartera recomendada", styles["Heading2"]))
    story.append(Paragraph(f"Capital: {fmt_ar_money(capital_ars)}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Preparar tabla (strings AR)
    df = table_df.copy()

    # columnas finales (sin internos)
    cols = ["Ticker", "Tipo", "%", "$", "Precio", "VN"]
    df = df[cols].copy()

    # VN entero, Precio 2 dec, % con coma, $ miles con punto
    df["%"] = pd.to_numeric(df["%"], errors="coerce").apply(fmt_ar_pct)
    df["$"] = pd.to_numeric(df["$"], errors="coerce").apply(fmt_ar_money)
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").apply(fmt_ar_2dec)
    df["VN"] = pd.to_numeric(df["VN"], errors="coerce").apply(lambda v: fmt_ar_int(v))

    data = [cols] + df.fillna("").astype(str).values.tolist()

    # Anchos por proporción (evita que se pisen)
    col_widths = [
        usable_w * 0.20,  # Ticker
        usable_w * 0.16,  # Tipo
        usable_w * 0.10,  # %
        usable_w * 0.20,  # $
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
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),  # $
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

    # export sin columna interna
    if "__ticker_base" in df.columns:
        df = df.drop(columns=["__ticker_base"], errors="ignore")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cartera_pesos")

        # Formatos (Excel los verá según locale del usuario)
        ws = writer.sheets["cartera_pesos"]

        # Buscar columnas por header
        headers = {cell.value: idx + 1 for idx, cell in enumerate(ws[1])}  # 1-based
        col_pct = headers.get("%")
        col_money = headers.get("$")
        col_price = headers.get("Precio")
        col_vn = headers.get("VN")

        # aplicar formatos desde fila 2
        for r in range(2, ws.max_row + 1):
            if col_pct:
                ws.cell(r, col_pct).number_format = "0.00"
            if col_money:
                ws.cell(r, col_money).number_format = "#,##0"
            if col_price:
                ws.cell(r, col_price).number_format = "#,##0.00"
            if col_vn:
                ws.cell(r, col_vn).number_format = "#,##0"

        # ancho columnas (prolijo)
        for name, w in [("Ticker", 18), ("Tipo", 12), ("%", 8), ("$", 16), ("Precio", 14), ("VN", 12)]:
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

  /* Cards suaves */
  .card{
    border: 1px solid rgba(17,24,39,.10);
    border-radius: 16px;
    padding: 12px 14px;
    background: #fff;
  }
  .kpi_lbl{ color: rgba(17,24,39,.60); font-size: 12px; margin-bottom: 6px; }
  .kpi_val{ font-size: 26px; font-weight: 850; color:#111827; letter-spacing: .01em; }

  /* Dataframes */
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
        st.markdown('<div class="title">Herramienta para armar carteras</div>', unsafe_allow_html=True)
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

    # Multiselect: options = tickers base, label = "VIST CEDEAR" (hasta 2do espacio)
    opts = prices.index.tolist()
    label_map = {tk: str(prices.loc[tk, "Label"]) for tk in opts}

    selected = st.multiselect(
        "Tickers",
        options=opts,
        default=opts[:6] if len(opts) >= 6 else opts,
        format_func=lambda tk: label_map.get(tk, tk),
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


    st.markdown("### Detalle de cartera")

    # Formato UI como strings AR para evitar confusión . y ,
    show = df.copy()

    # eliminar columna interna para UI
    show = show.drop(columns=["__ticker_base"], errors="ignore")

    show["%"] = pd.to_numeric(show["%"], errors="coerce").apply(lambda v: fmt_ar_pct(v))
    show["$"] = pd.to_numeric(show["$"], errors="coerce").apply(lambda v: fmt_ar_money(v))
    show["Precio"] = pd.to_numeric(show["Precio"], errors="coerce").apply(lambda v: fmt_ar_2dec(v))
    show["VN"] = pd.to_numeric(show["VN"], errors="coerce").apply(lambda v: fmt_ar_int(v))  # ✅ entero

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
            "$": st.column_config.TextColumn("$"),
            "Precio": st.column_config.TextColumn("Precio"),
            "VN": st.column_config.TextColumn("VN"),
        },
    )

    # =========================
    # Descargas: Excel + PDF
    # =========================
    cxl, cpdf = st.columns(2)
    with cxl:
        try:
            xlsx = build_excel_bytes(df)
            fname = f"NEIX_Cartera_Pesos_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            st.download_button(
                "Descargar Excel",
                data=xlsx,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="cartera_pesos_xlsx",
            )
        except Exception as e:
            st.warning(f"No pude generar el Excel: {e}")

    with cpdf:
        try:
            # Para PDF usamos df original (num) para formatear adentro
            pdf = build_cartera_pesos_pdf_bytes(
                capital_ars=float(capital),
                table_df=df.drop(columns=["__ticker_base"], errors="ignore"),
                logo_path=LOGO_PATH,
            )
            fname = f"NEIX_Cartera_Pesos_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                "Descargar PDF",
                data=pdf,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                key="cartera_pesos_pdf",
            )
        except Exception as e:
            st.warning(f"No pude generar el PDF: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
