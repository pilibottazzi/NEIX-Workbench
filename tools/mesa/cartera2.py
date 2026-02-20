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
LOGO_PATH = os.path.join("data", "Neix_logo.png")  # opcional

# IOL: nombres de columnas que pueden contener descripción
NAME_COL_CANDIDATES = [
    "Descripción",
    "Denominación",
    "Especie",
    "Instrumento",
    "Nombre",
    "Ticker Descripción",
]


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
# Formateo AR (pantalla/PDF)
# =========================
def _fmt_ar_int(n: float) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return ""
    try:
        n = float(n)
    except Exception:
        return ""
    s = f"{n:,.0f}"          # 1,234,567
    s = s.replace(",", "X")  # 1X234X567
    s = s.replace(".", ",")  # (por si hubiera)
    s = s.replace("X", ".")  # 1.234.567
    return s


def _fmt_ar_num(n: float, dec: int = 2) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return ""
    try:
        n = float(n)
    except Exception:
        return ""
    s = f"{n:,.{dec}f}"      # 1,234,567.89
    s = s.replace(",", "X")  # 1X234X567.89
    s = s.replace(".", ",")  # 1X234X567,89
    s = s.replace("X", ".")  # 1.234.567,89
    return s


def fmt_ars(n: float) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return ""
    return f"$ {_fmt_ar_int(n)}"


def fmt_pct(n: float, dec: int = 2) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return ""
    return f"{_fmt_ar_num(n, dec)}%"


# =========================
# Fetch IOL (PESOS)
# =========================
def _pick_name_column(t: pd.DataFrame) -> str | None:
    cols = [str(c).strip() for c in t.columns]
    for cand in NAME_COL_CANDIDATES:
        if cand in cols:
            return cand
    return None


def _fetch_iol_table(url: str) -> pd.DataFrame:
    """
    Devuelve DF estandarizado:
      cols: Ticker, Nombre, Precio, Volumen
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
    # Símbolo | Último Operado | Monto Operado | (a veces Descripción/Denominación)
    if "Símbolo" not in t.columns or "Último Operado" not in t.columns:
        return pd.DataFrame()

    name_col = _pick_name_column(t)

    out = pd.DataFrame()
    out["Ticker"] = t["Símbolo"].astype(str).str.strip().str.upper()

    if name_col:
        out["Nombre"] = (
            t[name_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        out["Nombre"] = ""

    out["Precio"] = t["Último Operado"].apply(parse_ar_number)

    if "Monto Operado" in t.columns:
        out["Volumen"] = t["Monto Operado"].apply(parse_ar_number).fillna(0.0)
    else:
        out["Volumen"] = 0.0

    out = out.dropna(subset=["Precio"])
    out = out[~out["Ticker"].duplicated(keep="first")]
    return out[["Ticker", "Nombre", "Precio", "Volumen"]]


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
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)

    # Si un ticker aparece repetido, priorizamos el de mayor volumen
    allp = allp.sort_values(["Ticker", "Volumen"], ascending=[True, False])
    allp = allp.drop_duplicates(subset=["Ticker"], keep="first")

    # Nombre fallback: si viene vacío, dejamos ""
    allp["Nombre"] = allp["Nombre"].fillna("").astype(str)

    allp = allp.set_index("Ticker")
    return allp[["Precio", "Volumen", "Tipo", "Nombre"]].sort_values("Volumen", ascending=False)


# =========================
# Cartera simple (ARS)
# =========================
@dataclass
class SimpleRowARS:
    ticker: str
    nombre: str
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
    if s > 0:
        pcts = pcts / s * 100.0
    else:
        pcts = np.zeros_like(pcts)

    rows: list[SimpleRowARS] = []
    for t, pct in zip(selected, pcts):
        if pct <= 0:
            continue
        if t not in prices.index:
            continue

        px = float(prices.loc[t, "Precio"])
        tipo = str(prices.loc[t, "Tipo"]) if "Tipo" in prices.columns else "NA"
        nombre = str(prices.loc[t, "Nombre"]) if "Nombre" in prices.columns else ""

        ars_amt = float(capital_ars) * (float(pct) / 100.0)
        vn = (ars_amt / px) if px > 0 else np.nan

        rows.append(
            SimpleRowARS(
                ticker=t,
                nombre=nombre,
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
            "Ticker": [r.ticker for r in rows],
            "Nombre": [r.nombre for r in rows],
            "Tipo":   [r.tipo for r in rows],
            "%":      [r.pct for r in rows],
            "$":      [r.ars for r in rows],
            "Precio": [r.precio for r in rows],
            "VN":     [r.vn for r in rows],
        }
    )


# =========================
# Excel export (con formatos)
# =========================
def build_excel_bytes(df_numeric: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_numeric.to_excel(writer, index=False, sheet_name="cartera_pesos")

        ws = writer.sheets["cartera_pesos"]

        # Ajustes de ancho (simple)
        widths = {
            "A": 10,  # Ticker
            "B": 42,  # Nombre
            "C": 10,  # Tipo
            "D": 10,  # %
            "E": 16,  # $
            "F": 14,  # Precio
            "G": 16,  # VN
        }
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

        # Formatos numéricos
        # Nota: Excel usa separadores según locale del usuario.
        for r in range(2, 2 + len(df_numeric)):
            ws[f"D{r}"].number_format = "0.00"      # %
            ws[f"E{r}"].number_format = '#,##0'     # $
            ws[f"F{r}"].number_format = "0.0000"    # Precio
            ws[f"G{r}"].number_format = "0.000000"  # VN

        # Header bold + freeze
        ws.freeze_panes = "A2"

    out.seek(0)
    return out.getvalue()


# =========================
# PDF export (minimal)
# =========================
def _df_to_table_data(df: pd.DataFrame) -> list[list[str]]:
    cols = list(df.columns)
    data = [cols]
    for _, r in df.iterrows():
        data.append([str(r[c]) for c in cols])
    return data


def build_pdf_bytes(
    *,
    df_display: pd.DataFrame,
    capital_ars: float,
    logo_path: str | None = None,
) -> bytes:
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

    # Logo centrado (sin título)
    if logo_path and os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=6.2 * cm, height=1.6 * cm)
            tlogo = Table([[logo]], colWidths=[usable_w])
            tlogo.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(tlogo)
            story.append(Spacer(1, 10))
        except Exception:
            pass

    story.append(Paragraph("Cartera (Pesos)", styles["Heading2"]))
    story.append(Paragraph(f"Capital: {fmt_ars(capital_ars)}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Tabla
    # Orden de columnas para PDF (más compacto y claro)
    cols_order = ["Ticker", "Nombre", "Tipo", "%", "$", "Precio", "VN"]
    pdf_df = df_display.copy()
    pdf_df = pdf_df[[c for c in cols_order if c in pdf_df.columns]]

    data = _df_to_table_data(pdf_df)

    # Anchos: dejar Nombre más ancho
    # sum = usable_w
    col_widths = [
        usable_w * 0.10,  # Ticker
        usable_w * 0.40,  # Nombre
        usable_w * 0.10,  # Tipo
        usable_w * 0.08,  # %
        usable_w * 0.12,  # $
        usable_w * 0.10,  # Precio
        usable_w * 0.10,  # VN
    ]

    t = Table(data, repeatRows=1, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8.6),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(t)

    doc.build(story)
    pdf = buff.getvalue()
    buff.close()
    return pdf


# =========================
# UI (minimal / NEIX-ish)
# =========================
NEIX_RED = "#ff3b30"

def _ui_css():
    st.markdown(
        f"""
<style>
  .wrap{{ max-width: 1180px; margin: 0 auto; }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 1.8rem; }}

  .title-row{{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; }}
  .title{{ font-size: 28px; font-weight: 850; letter-spacing: .01em; color:#111827; margin:0; }}
  .sub{{ color: rgba(17,24,39,.62); font-size: 13px; margin-top: 4px; }}
  .soft-hr{{ height:1px; background:rgba(17,24,39,.10); margin: 14px 0 18px; }}

  /* Botones */
  .stButton > button {{
    border-radius: 14px;
    padding: .62rem 1rem;
  }}
  .stButton > button[kind="primary"] {{
    background: {NEIX_RED};
    border: 1px solid {NEIX_RED};
  }}

  /* Dataframe border */
  div[data-testid="stDataFrame"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(17,24,39,.10);
  }}

  /* Pills/chips (multiselect) */
  div[data-baseweb="tag"]{{
    background: rgba(255,59,48,.10) !important;
    color:#111827 !important;
    border: 1px solid rgba(255,59,48,.25) !important;
    border-radius: 999px !important;
    font-weight: 650 !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


def _height_for_rows(n: int, row_h: int = 34, header: int = 42, pad: int = 18, max_h: int = 900) -> int:
    n = int(max(0, n))
    h = header + pad + row_h * max(1, n + 1)
    return int(min(max_h, h))


def _short_name(name: str, max_len: int = 42) -> str:
    s = (name or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def render(back_to_home=None):
    _ui_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Header
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.markdown('<div class="title">Cartera (Pesos)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Acciones / CEDEARs / Bonos / ONs — sin TIR/MD (solo asignación)</div>', unsafe_allow_html=True)
    with right:
        refresh = st.button("Actualizar precios", use_container_width=True, key="cartera_pesos_refresh")

    st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

    # Cache universo
    if refresh or "cartera_pesos_prices" not in st.session_state:
        with st.spinner("Actualizando universo (Pesos)..."):
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

    # Selector con nombre/tipo (mejora CEDEAR)
    opts = prices.index.tolist()
    label_map = {}
    for tk in opts:
        tipo = str(prices.loc[tk, "Tipo"])
        nm = str(prices.loc[tk, "Nombre"])
        nm = _short_name(nm, 44)
        if nm:
            label_map[tk] = f"{tk} · {nm} ({tipo})"
        else:
            label_map[tk] = f"{tk} ({tipo})"

    selected = st.multiselect(
        "Tickers",
        options=opts,
        default=opts[:6] if len(opts) >= 6 else opts,
        format_func=lambda x: label_map.get(x, x),
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

    df_num = build_simple_portfolio_ars(
        prices=prices,
        selected=selected,
        pct_map=pct_map,
        capital_ars=float(capital),
    )

    if df_num.empty:
        st.warning("No se pudo construir la cartera (faltan precios para los tickers elegidos).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---- DISPLAY (strings AR) ----
    show = df_num.copy()
    show["Nombre"] = show["Nombre"].apply(lambda s: _short_name(str(s), 46))
    show["%"] = show["%"].apply(lambda v: fmt_pct(v, 2))
    show["$"] = show["$"].apply(fmt_ars)
    show["Precio"] = show["Precio"].apply(lambda v: _fmt_ar_num(v, 4))
    show["VN"] = show["VN"].apply(lambda v: _fmt_ar_num(v, 6))

    # Reordenar columnas (más lindo)
    show = show[["Ticker", "Nombre", "Tipo", "%", "$", "Precio", "VN"]]

    st.markdown("### Detalle de cartera")
    h_tbl = _height_for_rows(len(show), max_h=820)

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=h_tbl,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Nombre": st.column_config.TextColumn("Nombre"),
            "Tipo": st.column_config.TextColumn("Tipo"),
            "%": st.column_config.TextColumn("%"),
            "$": st.column_config.TextColumn("$"),
            "Precio": st.column_config.TextColumn("Precio"),
            "VN": st.column_config.TextColumn("VN"),
        },
    )

    # =========================
    # Descargas (Excel + PDF)
    # =========================
    cxl, cpdf = st.columns([0.5, 0.5], vertical_alignment="center")

    with cxl:
        try:
            xlsx_bytes = build_excel_bytes(df_num)
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
            pdf_bytes = build_pdf_bytes(
                df_display=show,
                capital_ars=float(capital),
                logo_path=LOGO_PATH,
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
