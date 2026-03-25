from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# BRAND
# ─────────────────────────────────────────────────────────────────────────────
NEIX_RED    = "#E8192C"
NEIX_DARK   = "#0A0E1A"
NEIX_CARD   = "#111827"
NEIX_MUTED  = "#6B7280"
NEIX_BORDER = "rgba(255,255,255,0.07)"
NEIX_BG     = "#0D1117"
GREEN       = "#10B981"
BLUE        = "#3B82F6"
AMBER       = "#F59E0B"

NEIX_LOGO_SVG = """
<svg viewBox="0 0 120 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:26px;display:block;">
  <rect x="0" y="0" width="120" height="36" rx="6" fill="#E8192C"/>
  <text x="10" y="26" font-family="Arial Black, Arial" font-weight="900" font-size="22"
        letter-spacing="-1" fill="white">NEIX</text>
</svg>
"""

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=DM+Mono:wght@400;500&display=swap');

  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: {NEIX_BG} !important;
    font-family: 'DM Sans', sans-serif;
    color: #E5E7EB;
  }}
  [data-testid="stHeader"] {{ background: transparent !important; }}
  .block-container {{ max-width: 1480px; padding-top: 1.4rem; padding-bottom: 3rem; }}
  section[data-testid="stSidebar"] > div {{ background: {NEIX_DARK} !important; }}

  /* ─── Hero ─── */
  .hero {{
    background: linear-gradient(135deg, #0D1117 0%, #141B2D 55%, #1c0d12 100%);
    border: 1px solid {NEIX_BORDER};
    border-radius: 20px;
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.4rem;
    box-shadow: 0 0 80px rgba(232,25,44,0.07), 0 24px 60px rgba(0,0,0,0.45);
  }}
  .hero h1 {{
    font-size: 1.85rem; font-weight: 800; color: #F9FAFB;
    letter-spacing: -0.035em; margin: 0.6rem 0 0; line-height: 1.1;
  }}
  .hero p {{ color: {NEIX_MUTED}; font-size: 0.86rem; margin: 0.35rem 0 0; }}
  .hero-pills {{ display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.7rem; }}
  .hero-pill {{
    border-radius: 999px; font-size: 0.72rem; font-weight: 700;
    padding: 0.28rem 0.65rem; letter-spacing: 0.07em; text-transform: uppercase;
  }}
  .pill-red {{ background:rgba(232,25,44,0.12); border:1px solid rgba(232,25,44,0.25); color:{NEIX_RED}; }}
  .pill-green {{ background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); color:{GREEN}; }}
  .pill-blue {{ background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); color:{BLUE}; }}
  .pill-amber {{ background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.25); color:{AMBER}; }}

  /* ─── Upload zone ─── */
  .upload-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
  .upload-label {{
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:{NEIX_MUTED}; margin-bottom:0.4rem;
  }}
  div[data-testid="stFileUploader"] {{
    background: {NEIX_CARD};
    border: 1px dashed rgba(232,25,44,0.3) !important;
    border-radius: 14px; padding: 0.2rem;
  }}

  /* ─── KPI strip ─── */
  .kpi-strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:0.8rem; margin-bottom:1.2rem; }}
  .kpi {{
    background: {NEIX_CARD}; border: 1px solid {NEIX_BORDER};
    border-radius: 16px; padding: 1rem 1.1rem; position:relative; overflow:hidden;
  }}
  .kpi::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,transparent,{NEIX_RED},transparent);
  }}
  .kpi.green::before {{ background: linear-gradient(90deg,transparent,{GREEN},transparent); }}
  .kpi.blue::before  {{ background: linear-gradient(90deg,transparent,{BLUE},transparent); }}
  .kpi.amber::before {{ background: linear-gradient(90deg,transparent,{AMBER},transparent); }}
  .kpi-label {{ color:{NEIX_MUTED}; font-size:0.69rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.45rem; }}
  .kpi-val {{ color:#F9FAFB; font-size:1.45rem; font-weight:800; letter-spacing:-0.04em; font-family:'DM Mono',monospace; line-height:1.1; }}
  .kpi-sub {{ font-size:0.77rem; font-weight:600; margin-top:0.3rem; }}
  .pos {{ color:{GREEN}; }} .neg {{ color:{NEIX_RED}; }} .neu {{ color:{NEIX_MUTED}; }}

  /* ─── Section title ─── */
  .stitle {{
    font-size:0.72rem; font-weight:700; color:{NEIX_MUTED};
    text-transform:uppercase; letter-spacing:0.11em; margin:0.4rem 0 0.75rem;
  }}

  /* ─── Cards ─── */
  .card {{
    background:{NEIX_CARD}; border:1px solid {NEIX_BORDER};
    border-radius:16px; padding:1.2rem 1.4rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28);
  }}

  /* ─── Divider ─── */
  .divider {{ border:none; border-top:1px solid {NEIX_BORDER}; margin:0.8rem 0; }}

  /* ─── P&L bridge row ─── */
  .bridge {{
    display:flex; gap:0; align-items:stretch; border-radius:14px; overflow:hidden;
    border:1px solid {NEIX_BORDER}; margin-bottom:1.2rem;
  }}
  .bridge-cell {{
    flex:1; padding:1rem 1.2rem; text-align:center; border-right:1px solid {NEIX_BORDER};
  }}
  .bridge-cell:last-child {{ border-right:none; }}
  .bridge-cell .bc-label {{ color:{NEIX_MUTED}; font-size:0.69rem; text-transform:uppercase; letter-spacing:0.09em; font-weight:700; margin-bottom:0.4rem; }}
  .bridge-cell .bc-val {{ font-size:1.2rem; font-weight:800; font-family:'DM Mono',monospace; letter-spacing:-0.03em; }}
  .bridge-cell .bc-sub {{ font-size:0.73rem; color:{NEIX_MUTED}; margin-top:0.25rem; }}
  .bc-green {{ color:{GREEN}; }}
  .bc-red   {{ color:{NEIX_RED}; }}
  .bc-amber {{ color:{AMBER}; }}
  .bc-blue  {{ color:{BLUE}; }}

  /* ─── Tables ─── */
  .ct {{ width:100%; border-collapse:collapse; font-size:0.86rem; }}
  .ct th {{
    color:{NEIX_MUTED}; font-size:0.69rem; text-transform:uppercase;
    letter-spacing:0.09em; padding:0.5rem 0.7rem; text-align:left;
    border-bottom:1px solid {NEIX_BORDER};
  }}
  .ct th.r {{ text-align:right; }}
  .ct td {{ padding:0.5rem 0.7rem; border-bottom:1px solid rgba(255,255,255,0.025); color:#E5E7EB; }}
  .ct td.r {{ text-align:right; font-family:'DM Mono',monospace; font-size:0.83rem; }}
  .ct tr:last-child td {{ border-bottom:none; }}
  .ct tr.total td {{ font-weight:700; color:#F9FAFB; border-top:1px solid {NEIX_BORDER}; background:rgba(255,255,255,0.02); }}
  .badge {{
    display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
    font-size:0.7rem; font-weight:700; letter-spacing:0.04em;
  }}
  .b-pos {{ background:rgba(16,185,129,0.12); color:{GREEN}; }}
  .b-neg {{ background:rgba(232,25,44,0.12);  color:{NEIX_RED}; }}
  .b-neu {{ background:rgba(107,114,128,0.12); color:{NEIX_MUTED}; }}
  .b-amber {{ background:rgba(245,158,11,0.12); color:{AMBER}; }}

  /* ─── Info chips ─── */
  .chips {{ display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.9rem; }}
  .chip {{ background:rgba(255,255,255,0.04); border:1px solid {NEIX_BORDER}; color:#9CA3AF; font-size:0.76rem; padding:0.25rem 0.6rem; border-radius:8px; }}

  /* ─── Tabs ─── */
  div[data-testid="stTabs"] button {{
    background:transparent !important; border:1px solid {NEIX_BORDER} !important;
    border-radius:999px !important; color:{NEIX_MUTED} !important;
    font-weight:600 !important; font-size:0.82rem !important;
    padding:0.36rem 1rem !important; transition:all 0.18s !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    background:{NEIX_RED} !important; border-color:{NEIX_RED} !important; color:white !important;
  }}

  /* ─── Dataframe ─── */
  .stDataFrame {{ border-radius:14px; overflow:hidden; }}

  /* ─── Download btn ─── */
  div[data-testid="stDownloadButton"] > button {{
    width:100% !important; border-radius:12px !important;
    background:{NEIX_RED} !important; color:white !important;
    border:none !important; font-weight:700 !important; font-family:'DM Sans',sans-serif !important;
  }}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _to_float(s) -> float:
    if s is None: return 0.0
    s = str(s).strip()
    if not s or s in {"-","—"}: return 0.0
    s = re.sub(r"\.(?=\d{3})", "", s).replace(",", ".")
    try: return float(s)
    except ValueError: return 0.0

def _ars(v: float, dec=0) -> str:
    fmt = f"{{:,.{dec}f}}"
    return "$ " + fmt.format(v).replace(",","X").replace(".",",").replace("X",".")

def _usd(v: float) -> str: return f"U$S {v:,.2f}"
def _pct(v: float) -> str: return f"{v:+.1f}%"
def _pct2(v: float) -> str: return f"{v:+.2f}%"

def _sub(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")

def _badge(v: float, fmt_fn=_ars) -> str:
    cls = "b-pos" if v > 0 else ("b-neg" if v < 0 else "b-neu")
    return f'<span class="badge {cls}">{fmt_fn(v)}</span>'

CATEGORIA = {
    "acciones": {"EDN","GGAL","YPFD","SUPV"},
    "cedears":  {"AXP","GOOGL","MELI","META","NVDA","ADBE","TSLA","GLOB",
                 "BABA","VIST","UNH","SPY","COIN","LAC","NU","SPCE","MSFT"},
    "bonos":    {"AL30","AL35","GD35"},
    "fondos":   {"CAUCION","MEGA PES A","FIMA PRE A"},
    "lecaps":   {"S29G5","S16A5","S12S5","S30S5","S15G5","S31O5"},
}

def _cat(ticker: str) -> str:
    t = ticker.upper().strip()
    for cat, tickers in CATEGORIA.items():
        if t in tickers: return cat
    return "otros"


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_historico(path: str) -> Tuple[pd.DataFrame, Dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    soup = BeautifulSoup(content, "html.parser")

    meta: Dict = {"usuario":"","comitente":"","fecha_desde":"","fecha_hasta":""}
    enc = soup.find("div", id="encabezadoExcel")
    if enc:
        rows = enc.find_all("tr")
        if len(rows) >= 2:
            ths = [th.get_text(strip=True) for th in rows[0].find_all("th")]
            tds = [td.get_text(strip=True) for td in rows[1].find_all("td")]
            m = dict(zip(ths, tds))
            meta.update({
                "usuario":     m.get("Usuario","").strip(),
                "comitente":   m.get("Comitente","").strip(),
                "fecha_desde": m.get("Fecha Desde","").strip(),
                "fecha_hasta": m.get("Fecha Hasta","").strip(),
            })

    records: List[Dict] = []
    for box in soup.find_all("div", class_="box box-default"):
        h3 = box.find("h3")
        full_name = h3.get_text(strip=True) if h3 else "?"
        ticker = full_name.split()[0].upper()
        nombre = " ".join(full_name.split()[1:])

        table = box.find("table")
        if not table: continue

        total_ars = total_usd = 0.0
        movimientos = []
        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["th","td"])]
            if not cells: continue
            cpbt = cells[0].upper()
            if cpbt == "TOTAL":
                total_ars = _to_float(cells[6]) if len(cells)>6 else 0.0
                total_usd = _to_float(cells[8]) if len(cells)>8 else 0.0
            elif len(cells) >= 9:
                movimientos.append({
                    "Ticker": ticker, "Nombre": nombre, "Cpbt": cpbt,
                    "Numero": cells[1], "Fecha_Concertacion": cells[2],
                    "Fecha_Liquidacion": cells[3],
                    "Cantidad": _to_float(cells[4]), "Precio": _to_float(cells[5]),
                    "Neto_ARS": _to_float(cells[6]), "Moneda": cells[7],
                    "Neto_USD": _to_float(cells[8]),
                })

        compras_ars  = sum(abs(m["Neto_ARS"]) for m in movimientos if m["Cpbt"]=="COMPRA" and m["Neto_ARS"]<0)
        ventas_ars   = sum(m["Neto_ARS"] for m in movimientos if m["Cpbt"]=="VENTA" and m["Neto_ARS"]>0)
        divs_ars     = sum(m["Neto_ARS"] for m in movimientos if m["Cpbt"] in {"DIVIDENDOS","RTA/AMORT","CREDITO RTA"} and m["Neto_ARS"]>0)
        qty_comprada = sum(m["Cantidad"] for m in movimientos if m["Cpbt"]=="COMPRA" and m["Cantidad"]>0)
        qty_vendida  = sum(abs(m["Cantidad"]) for m in movimientos if m["Cpbt"]=="VENTA" and m["Cantidad"]<0)

        records.append({
            "Ticker": ticker, "Nombre": nombre, "Categoria": _cat(ticker),
            "Qty_Comprada": qty_comprada, "Qty_Vendida": qty_vendida,
            "Saldo_Hist": qty_comprada - qty_vendida,
            "Compras_ARS": compras_ars, "Ventas_ARS": ventas_ars,
            "Dividendos_ARS": divs_ars,
            "Total_ARS": total_ars, "Total_USD": total_usd,
            "N_Movimientos": len(movimientos),
            "_movs": movimientos,
        })

    return pd.DataFrame(records), meta


def parse_resultados(path: str) -> Tuple[pd.DataFrame, Dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    soup = BeautifulSoup(content, "html.parser")

    meta: Dict = {"usuario":"","comitente":"","fecha_desde":"","fecha_hasta":""}
    enc = soup.find("div", id="encabezadoExcel")
    if enc:
        rows = enc.find_all("tr")
        if len(rows) >= 2:
            ths = [th.get_text(strip=True) for th in rows[0].find_all("th")]
            tds = [td.get_text(strip=True) for td in rows[1].find_all("td")]
            m = dict(zip(ths, tds))
            meta.update({
                "usuario":     m.get("Usuario","").strip(),
                "comitente":   m.get("Comitente","").strip(),
                "fecha_desde": m.get("Fecha Desde","").strip(),
                "fecha_hasta": m.get("Fecha Hasta","").strip(),
            })

    table = soup.find("table", class_="table-consultas")
    if table is None:
        return pd.DataFrame(), meta

    records = []
    for row in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
        if not cells or len(cells) < 7: continue
        especie_raw = cells[0].strip()
        if especie_raw.lower() in {"perdida","ganancia","total",""}: continue
        if not cells[1].strip(): continue  # skip summary rows

        parts = especie_raw.split()
        ticker = parts[0].upper()
        nombre = " ".join(parts[1:])

        records.append({
            "Ticker":    ticker,
            "Nombre":    nombre,
            "Categoria": _cat(ticker),
            "Cantidad":  _to_float(cells[1]),
            "PPP":       _to_float(cells[2]),
            "Inversion": _to_float(cells[3]),   # costo actual de la posición abierta
            "Precio_Actual": _to_float(cells[4]),
            "Valuacion": _to_float(cells[5]),   # valor a mercado hoy
            "Diferencia": _to_float(cells[6]),  # ganancia/pérdida no realizada
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["Rend_Pct"] = df.apply(
            lambda r: (r["Diferencia"] / r["Inversion"] * 100) if r["Inversion"] else 0.0, axis=1
        )
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9CA3AF", family="DM Sans"),
    margin=dict(l=8, r=8, t=44, b=8),
    title_font=dict(size=13, color="#E5E7EB"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

def _hbar(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, height=None):
    colors = [NEIX_RED if v < 0 else GREEN for v in df_plot[x_col]]
    fig = go.Figure(go.Bar(
        x=df_plot[x_col], y=df_plot[y_col], orientation="h",
        marker_color=colors,
        text=df_plot[x_col].apply(lambda v: _ars(v)),
        textposition="outside", textfont=dict(size=9, color="#9CA3AF"),
    ))
    fig.update_layout(
        **BASE_LAYOUT, title=title,
        height=height or max(320, 38*len(df_plot)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=True,
                   zerolinecolor=NEIX_RED, zerolinewidth=1),
        yaxis=dict(gridcolor="rgba(255,255,255,0.02)"),
    )
    return fig

def chart_bridge(df_res: pd.DataFrame):
    """Waterfall: inversión → valuación → diferencia por ticker"""
    df = df_res.sort_values("Diferencia")
    fig = go.Figure(go.Waterfall(
        name="P&L abierto", orientation="v",
        x=df["Ticker"].tolist(),
        y=df["Diferencia"].tolist(),
        connector=dict(line=dict(color="rgba(255,255,255,0.08)")),
        increasing=dict(marker_color=GREEN),
        decreasing=dict(marker_color=NEIX_RED),
        totals=dict(marker_color=AMBER),
        text=[_ars(v) for v in df["Diferencia"]],
        textposition="outside",
    ))
    fig.update_layout(
        **BASE_LAYOUT, title="P&L no realizado por especie",
        height=340,
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False,
    )
    return fig

def chart_inv_vs_val(df_res: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Inversión (costo)", x=df_res["Ticker"], y=df_res["Inversion"],
                         marker_color="#374151", text=df_res["Inversion"].apply(_ars),
                         textposition="inside", textfont=dict(size=9)))
    fig.add_trace(go.Bar(name="Valuación actual", x=df_res["Ticker"], y=df_res["Valuacion"],
                         marker_color=BLUE, text=df_res["Valuacion"].apply(_ars),
                         textposition="inside", textfont=dict(size=9)))
    fig.update_layout(
        **BASE_LAYOUT, title="Inversión vs valuación actual",
        height=340, barmode="group",
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    return fig

def chart_donut(labels, values, title):
    palette = [NEIX_RED, AMBER, BLUE, "#8B5CF6", GREEN, "#EC4899", "#06B6D4"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker_colors=palette[:len(labels)],
        textinfo="label+percent", textfont=dict(size=11, color="#E5E7EB"),
    ))
    fig.update_layout(**BASE_LAYOUT, title=title, height=310, showlegend=False)
    return fig

def chart_treemap_open(df_res: pd.DataFrame):
    df = df_res[df_res["Valuacion"] > 0].copy()
    if df.empty: return None
    fig = px.treemap(
        df, path=["Categoria","Ticker"], values="Valuacion",
        color="Diferencia",
        color_continuous_scale=[NEIX_RED, "#374151", GREEN],
        color_continuous_midpoint=0,
        title="Portafolio abierto — tamaño = valuación, color = P&L",
    )
    fig.update_layout(**BASE_LAYOUT, height=420)
    return fig

def chart_scatter(df_res: pd.DataFrame):
    fig = px.scatter(
        df_res, x="Inversion", y="Diferencia", size="Valuacion",
        color="Rend_Pct", color_continuous_scale=[NEIX_RED,"#374151",GREEN],
        hover_name="Ticker", text="Ticker",
        title="Riesgo / retorno posiciones abiertas",
        labels={"Inversion":"Capital invertido ($)","Diferencia":"P&L no realizado ($)"},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=10, color="#E5E7EB"))
    fig.update_layout(
        **BASE_LAYOUT, height=380,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=True,
                   zerolinecolor=NEIX_RED, zerolinewidth=1),
    )
    return fig

def chart_hist_vs_open(df_hist_agg: pd.DataFrame, df_res_agg: pd.DataFrame):
    """Combined bar: realizado vs no-realizado"""
    all_tickers = sorted(set(df_hist_agg["Ticker"]) | set(df_res_agg["Ticker"]))
    h_map = dict(zip(df_hist_agg["Ticker"], df_hist_agg["Total_ARS"]))
    r_map = dict(zip(df_res_agg["Ticker"], df_res_agg["Diferencia"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Realizado (histórico)", x=all_tickers,
        y=[h_map.get(t,0) for t in all_tickers], marker_color=GREEN,
    ))
    fig.add_trace(go.Bar(
        name="No realizado (posición abierta)", x=all_tickers,
        y=[r_map.get(t,0) for t in all_tickers], marker_color=AMBER,
    ))
    fig.update_layout(
        **BASE_LAYOUT, title="P&L realizado vs no realizado por especie",
        height=380, barmode="relative",
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=True,
                   zerolinecolor=NEIX_RED, zerolinewidth=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def build_excel(df_hist: pd.DataFrame, df_res: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        cols_h = ["Ticker","Nombre","Categoria","Qty_Comprada","Qty_Vendida","Saldo_Hist",
                  "Compras_ARS","Ventas_ARS","Dividendos_ARS","Total_ARS","Total_USD"]
        df_hist[[c for c in cols_h if c in df_hist.columns]].to_excel(
            writer, sheet_name="Historico", index=False)
        if not df_res.empty:
            df_res[["Ticker","Nombre","Categoria","Cantidad","PPP","Inversion",
                    "Precio_Actual","Valuacion","Diferencia","Rend_Pct"]].to_excel(
                writer, sheet_name="Posicion Actual", index=False)
        # combined
        if not df_res.empty:
            merged = df_hist[["Ticker","Total_ARS"]].merge(
                df_res[["Ticker","Diferencia","Valuacion"]], on="Ticker", how="outer"
            ).fillna(0)
            merged["Total_Combinado"] = merged["Total_ARS"] + merged["Diferencia"]
            merged.to_excel(writer, sheet_name="Combinado", index=False)
    out.seek(0)
    return out.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def render():
    st.set_page_config(page_title="NEIX · Portfolio", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero">
      <div>
        {NEIX_LOGO_SVG}
        <h1>Dashboard Integral de Portafolio</h1>
        <p>P&L realizado · Posición abierta valorizada · Vista consolidada</p>
        <div class="hero-pills">
          <span class="hero-pill pill-red">Realizado</span>
          <span class="hero-pill pill-amber">No realizado</span>
          <span class="hero-pill pill-blue">Valuación actual</span>
          <span class="hero-pill pill-green">Multi-activo</span>
        </div>
      </div>
      <div style="text-align:right;color:{NEIX_MUTED};font-size:0.8rem;line-height:1.8;">
        <div style="color:#E5E7EB;font-size:0.95rem;font-weight:700;">Acciones · CEDEARs · Bonos</div>
        LECAPs · Fondos · Cauciones<br/>
        ARS &amp; USD
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploaders ───────────────────────────────────────────────────────
    col_u1, col_u2 = st.columns(2, gap="medium")
    with col_u1:
        st.markdown('<div class="upload-label">📂 Histórico por especie (operaciones)</div>', unsafe_allow_html=True)
        up_hist = st.file_uploader("hist", type=["xls","xlsx"], label_visibility="collapsed", key="hist")
    with col_u2:
        st.markdown('<div class="upload-label">📈 Resultados por especie (posición abierta)</div>', unsafe_allow_html=True)
        up_res  = st.file_uploader("res",  type=["xls","xlsx"], label_visibility="collapsed", key="res")

    # paths
    default_hist = "/mnt/user-data/uploads/Historico_por_Especie.xls"
    default_res  = "/mnt/user-data/uploads/Resultados_por_Especie.xls"

    if up_hist:
        p = Path("/tmp/hist.xls"); p.write_bytes(up_hist.read()); hist_path = str(p)
    else:
        hist_path = default_hist

    if up_res:
        p = Path("/tmp/res.xls"); p.write_bytes(up_res.read()); res_path = str(p)
    else:
        res_path = default_res

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        df_hist, meta = parse_historico(hist_path)
    except Exception as e:
        st.error(f"Error leyendo histórico: {e}"); return

    df_res = pd.DataFrame()
    try:
        df_res, _ = parse_resultados(res_path)
    except Exception as e:
        st.warning(f"No se pudo leer Resultados por Especie: {e}")

    if df_hist.empty:
        st.warning("Sin datos en el histórico."); return

    # ── Meta strip ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="chips">
      <span class="chip">👤 {meta.get('usuario','—')}</span>
      <span class="chip">🏦 Comitente {meta.get('comitente','—')}</span>
      <span class="chip">📅 Período: {meta.get('fecha_desde','—')} → {meta.get('fecha_hasta','?')}</span>
      <span class="chip">📊 {len(df_hist)} especies operadas</span>
      {"<span class='chip'>🟢 " + str(len(df_res)) + " posiciones abiertas</span>" if not df_res.empty else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Consolidated numbers ─────────────────────────────────────────────────
    # Realizado (histórico)
    pnl_realizado     = df_hist["Total_ARS"].sum()
    capital_invertido = df_hist["Compras_ARS"].sum()
    dividendos        = df_hist["Dividendos_ARS"].sum()
    rend_pct_real     = (pnl_realizado / capital_invertido * 100) if capital_invertido else 0.0

    # No realizado (posiciones abiertas)
    pnl_no_real   = df_res["Diferencia"].sum() if not df_res.empty else 0.0
    valuacion_hoy = df_res["Valuacion"].sum()   if not df_res.empty else 0.0
    inversion_abi = df_res["Inversion"].sum()   if not df_res.empty else 0.0
    rend_pct_abie = (pnl_no_real / inversion_abi * 100) if inversion_abi else 0.0

    # Combined
    pnl_total      = pnl_realizado + pnl_no_real
    rend_pct_total = (pnl_total / capital_invertido * 100) if capital_invertido else 0.0

    # ── P&L Bridge ───────────────────────────────────────────────────────────
    def _bc(label, val, cls, sub=""):
        return f"""
        <div class="bridge-cell">
          <div class="bc-label">{label}</div>
          <div class="bc-val {cls}">{_ars(val)}</div>
          {"<div class='bc-sub'>" + sub + "</div>" if sub else ""}
        </div>"""

    st.markdown(f"""
    <div class="bridge">
      {_bc("Capital total invertido", capital_invertido, "bc-blue", f"{len(df_hist)} especies operadas")}
      {_bc("P&L realizado", pnl_realizado, "bc-green" if pnl_realizado>=0 else "bc-red", _pct(rend_pct_real)+" s/capital")}
      {_bc("P&L no realizado", pnl_no_real, "bc-green" if pnl_no_real>=0 else "bc-red", _pct(rend_pct_abie)+" s/abierto")}
      {_bc("Dividendos / carry", dividendos, "bc-green" if dividendos>=0 else "bc-red", "cobrado")}
      {_bc("Valuación posición abierta", valuacion_hoy, "bc-amber", f"{len(df_res)} posiciones")}
      {_bc("P&L TOTAL combinado", pnl_total, "bc-green" if pnl_total>=0 else "bc-red", _pct(rend_pct_total)+" s/capital total")}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    n_pos_hist = int((df_hist["Total_ARS"] > 0).sum())
    n_neg_hist = int((df_hist["Total_ARS"] < 0).sum())
    mejor_hist = df_hist.loc[df_hist["Total_ARS"].idxmax(), "Ticker"]
    peor_hist  = df_hist.loc[df_hist["Total_ARS"].idxmin(), "Ticker"]

    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi">
        <div class="kpi-label">P&amp;L realizado</div>
        <div class="kpi-val">{_ars(pnl_realizado)}</div>
        <div class="kpi-sub {_sub(pnl_realizado)}">{_pct(rend_pct_real)} s/capital</div>
      </div>
      <div class="kpi {'green' if pnl_no_real>=0 else ''}">
        <div class="kpi-label">P&amp;L no realizado</div>
        <div class="kpi-val">{_ars(pnl_no_real)}</div>
        <div class="kpi-sub {_sub(pnl_no_real)}">{_pct(rend_pct_abie)} s/posición abierta</div>
      </div>
      <div class="kpi blue">
        <div class="kpi-label">Valuación hoy</div>
        <div class="kpi-val">{_ars(valuacion_hoy)}</div>
        <div class="kpi-sub neu">{len(df_res)} posiciones abiertas</div>
      </div>
      <div class="kpi amber">
        <div class="kpi-label">P&amp;L total</div>
        <div class="kpi-val">{_ars(pnl_total)}</div>
        <div class="kpi-sub {_sub(pnl_total)}">{_pct(rend_pct_total)} combinado</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Dividendos / carry</div>
        <div class="kpi-val">{_ars(dividendos)}</div>
        <div class="kpi-sub pos">cobrado en el período</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Win / Loss rate</div>
        <div class="kpi-val">{n_pos_hist}/{len(df_hist)}</div>
        <div class="kpi-sub neg">{n_neg_hist} en negativo</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠  Vista General",
        "📋  Histórico completo",
        "🟢  Posición abierta",
        "🔀  Realizado vs Papel",
        "📈  Gráficos",
        "📁  Movimientos",
    ])

    # ── TAB 1: General view ───────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([1.05, 0.95], gap="large")
        with c1:
            st.markdown('<div class="stitle">Lectura ejecutiva del portafolio</div>', unsafe_allow_html=True)
            mejor_v = df_hist.loc[df_hist["Total_ARS"].idxmax()]
            peor_v  = df_hist.loc[df_hist["Total_ARS"].idxmin()]
            pct_win = n_pos_hist / len(df_hist) * 100

            res_texto = ""
            if not df_res.empty:
                mejor_ab = df_res.loc[df_res["Diferencia"].idxmax()]
                peor_ab  = df_res.loc[df_res["Diferencia"].idxmin()]
                res_texto = f"""
                <br/>Las posiciones abiertas registran un P&L no realizado de
                <strong style="color:{'#10B981' if pnl_no_real>=0 else NEIX_RED}">{_ars(pnl_no_real)}</strong>
                ({_pct(rend_pct_abie)}) sobre una inversión abierta de <strong>{_ars(inversion_abi)}</strong>,
                con valuación actual de <strong style="color:{BLUE}">{_ars(valuacion_hoy)}</strong>.<br/>
                Mejor posición abierta: <span style="color:{GREEN}">{mejor_ab['Ticker']}</span>
                ({_ars(mejor_ab['Diferencia'])}) &nbsp;|&nbsp;
                Peor: <span style="color:{NEIX_RED}">{peor_ab['Ticker']}</span>
                ({_ars(peor_ab['Diferencia'])})
                """

            st.markdown(f"""
            <div class="card">
              <p style="color:#E5E7EB;line-height:1.9;margin:0">
                El portafolio acumula un <strong>P&L realizado de
                <span style="color:{'#10B981' if pnl_realizado>=0 else NEIX_RED}">
                {_ars(pnl_realizado)}</span></strong>
                sobre {len(df_hist)} especies operadas entre
                <strong>{meta.get('fecha_desde','?')}</strong> y
                <strong>{meta.get('fecha_hasta','hoy')}</strong>,
                equivalente a <strong style="color:{'#10B981' if rend_pct_real>=0 else NEIX_RED}">
                {_pct(rend_pct_real)}</strong> sobre el capital total invertido
                de <strong>{_ars(capital_invertido)}</strong>.<br/><br/>
                <strong>{n_pos_hist} de {len(df_hist)}</strong> especies
                ({pct_win:.0f}%) cerraron operaciones en positivo.
                Mejor contribución: <span style="color:{GREEN}">{mejor_v['Ticker']}</span>
                ({_ars(mejor_v['Total_ARS'])}) /
                Peor: <span style="color:{NEIX_RED}">{peor_v['Ticker']}</span>
                ({_ars(peor_v['Total_ARS'])}).
                {res_texto}
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Resumen table
            st.markdown('<div class="stitle" style="margin-top:1.1rem">Cuadro de resultados consolidado</div>', unsafe_allow_html=True)
            rows_data = [
                ("Capital total desplegado (ARS)", _ars(capital_invertido), ""),
                ("Total ventas ARS", _ars(df_hist["Ventas_ARS"].sum()), ""),
                ("Dividendos / rentas cobradas", _ars(dividendos), "pos"),
                ("─── P&L realizado ───", _ars(pnl_realizado), "pos" if pnl_realizado>=0 else "neg"),
                ("Inversión posiciones abiertas", _ars(inversion_abi), ""),
                ("Valuación a mercado hoy", _ars(valuacion_hoy), "bc-blue"),
                ("─── P&L no realizado ───", _ars(pnl_no_real), "pos" if pnl_no_real>=0 else "neg"),
                ("═══ P&L TOTAL COMBINADO ═══", _ars(pnl_total), "pos" if pnl_total>=0 else "neg"),
                ("Rendimiento combinado s/ capital", _pct(rend_pct_total), "pos" if rend_pct_total>=0 else "neg"),
            ]
            def _color(c):
                if c == "pos": return "#10B981"
                if c == "neg": return NEIX_RED
                return "#E5E7EB"
            rows_html = "".join(
                f"<tr><td style='color:{NEIX_MUTED}'>{k}</td>"
                f"<td class='r' style='color:{_color(c)}'>{v}</td></tr>"
                for k,v,c in rows_data
            )
            st.markdown(f'<div class="card" style="margin-top:0.4rem"><table class="ct">{rows_html}</table></div>', unsafe_allow_html=True)

        with c2:
            if not df_res.empty:
                st.plotly_chart(chart_bridge(df_res), use_container_width=True)
            st.plotly_chart(_hbar(df_hist.sort_values("Total_ARS"), "Total_ARS", "Ticker",
                                  "P&L realizado por especie"), use_container_width=True)

    # ── TAB 2: Histórico ──────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="stitle">Histórico completo · todas las especies operadas</div>', unsafe_allow_html=True)
        f1, f2 = st.columns([1.5, 1], gap="medium")
        with f1:
            cats = sorted(df_hist["Categoria"].unique())
            sel  = st.multiselect("Categoría", cats, default=cats, key="hcat")
        with f2:
            solo_neg = st.toggle("Solo perdedoras 🔴", key="hneg")

        dh = df_hist[df_hist["Categoria"].isin(sel)]
        if solo_neg: dh = dh[dh["Total_ARS"] < 0]

        show = dh[["Ticker","Nombre","Categoria","Qty_Comprada","Qty_Vendida","Saldo_Hist",
                    "Compras_ARS","Ventas_ARS","Dividendos_ARS","Total_ARS","Total_USD"]].copy()
        st.dataframe(
            show.style.format({
                "Qty_Comprada":"{:,.0f}", "Qty_Vendida":"{:,.0f}", "Saldo_Hist":"{:,.0f}",
                "Compras_ARS":"$ {:,.0f}", "Ventas_ARS":"$ {:,.0f}",
                "Dividendos_ARS":"$ {:,.0f}", "Total_ARS":"$ {:,.0f}", "Total_USD":"U$S {:,.2f}",
            }).applymap(
                lambda v: f"color:{GREEN};font-weight:700" if isinstance(v,(int,float)) and v>0
                else (f"color:{NEIX_RED};font-weight:700" if isinstance(v,(int,float)) and v<0 else ""),
                subset=["Total_ARS","Total_USD"]
            ),
            use_container_width=True, hide_index=True, height=min(580, 40+38*len(show))
        )

    # ── TAB 3: Posición abierta ───────────────────────────────────────────────
    with tab3:
        if df_res.empty:
            st.info("No se cargó el archivo de Resultados por Especie.")
        else:
            st.markdown('<div class="stitle">Posiciones abiertas — valuadas a precio actual</div>', unsafe_allow_html=True)
            c1, c2 = st.columns([1.1, 0.9], gap="large")
            with c1:
                show_r = df_res[["Ticker","Nombre","Categoria","Cantidad","PPP",
                                  "Inversion","Precio_Actual","Valuacion","Diferencia","Rend_Pct"]].copy()
                st.dataframe(
                    show_r.style.format({
                        "Cantidad":"{:,.0f}", "PPP":"$ {:,.3f}",
                        "Inversion":"$ {:,.0f}", "Precio_Actual":"$ {:,.3f}",
                        "Valuacion":"$ {:,.0f}", "Diferencia":"$ {:,.0f}",
                        "Rend_Pct":"{:+.2f}%",
                    }).applymap(
                        lambda v: f"color:{GREEN};font-weight:700" if isinstance(v,(int,float)) and v>0
                        else (f"color:{NEIX_RED};font-weight:700" if isinstance(v,(int,float)) and v<0 else ""),
                        subset=["Diferencia","Rend_Pct"]
                    ),
                    use_container_width=True, hide_index=True
                )

                # category summary for open positions
                cat_r = df_res.groupby("Categoria").agg(
                    N=("Ticker","count"),
                    Inversion=("Inversion","sum"),
                    Valuacion=("Valuacion","sum"),
                    Diferencia=("Diferencia","sum"),
                ).reset_index()
                cat_r["Rend_Pct"] = cat_r.apply(
                    lambda r: r["Diferencia"]/r["Inversion"]*100 if r["Inversion"] else 0, axis=1)

                hdr = "<tr>" + "".join(f"<th {'class=r' if i>1 else ''}>{h}</th>" for i,h in
                      enumerate(["Categoría","N","Inversión","Valuación","Diferencia","Rend %"])) + "</tr>"
                rws = ""
                for _,r in cat_r.iterrows():
                    rws += f"""<tr>
                      <td><strong>{r['Categoria'].upper()}</strong></td>
                      <td class=r>{int(r['N'])}</td>
                      <td class=r>{_ars(r['Inversion'])}</td>
                      <td class=r style="color:{BLUE}">{_ars(r['Valuacion'])}</td>
                      <td class=r>{_badge(r['Diferencia'])}</td>
                      <td class=r style="color:{'#10B981' if r['Rend_Pct']>=0 else NEIX_RED}">{_pct2(r['Rend_Pct'])}</td>
                    </tr>"""
                rws += f"""<tr class=total>
                  <td>TOTAL</td><td class=r>{len(df_res)}</td>
                  <td class=r>{_ars(inversion_abi)}</td>
                  <td class=r style="color:{BLUE}">{_ars(valuacion_hoy)}</td>
                  <td class=r>{_badge(pnl_no_real)}</td>
                  <td class=r style="color:{'#10B981' if rend_pct_abie>=0 else NEIX_RED}">{_pct2(rend_pct_abie)}</td>
                </tr>"""
                st.markdown(f'<div class="card" style="margin-top:1rem"><table class="ct"><thead>{hdr}</thead><tbody>{rws}</tbody></table></div>',
                            unsafe_allow_html=True)

            with c2:
                st.plotly_chart(chart_inv_vs_val(df_res), use_container_width=True)
                st.plotly_chart(chart_donut(df_res["Ticker"].tolist(), df_res["Valuacion"].tolist(),
                                            "Composición de la posición abierta"), use_container_width=True)

    # ── TAB 4: Realizado vs Papel ─────────────────────────────────────────────
    with tab4:
        if df_res.empty:
            st.info("Cargá el archivo de Resultados para ver esta vista combinada.")
        else:
            st.markdown('<div class="stitle">P&L realizado + no realizado · vista combinada</div>', unsafe_allow_html=True)

            # Merge: tickers in both
            merged = df_hist[["Ticker","Nombre","Categoria","Total_ARS","Total_USD"]].merge(
                df_res[["Ticker","Diferencia","Valuacion","Rend_Pct","Inversion"]],
                on="Ticker", how="outer"
            ).fillna(0)
            merged["PnL_Total"] = merged["Total_ARS"] + merged["Diferencia"]
            merged["En_Historico"] = merged["Total_ARS"] != 0
            merged["En_Abierto"]   = merged["Diferencia"] != 0

            st.dataframe(
                merged[["Ticker","Nombre","Total_ARS","Diferencia","PnL_Total","Valuacion","En_Historico","En_Abierto"]]
                .rename(columns={"Total_ARS":"P&L Realizado","Diferencia":"P&L Papel",
                                  "PnL_Total":"P&L Total","Valuacion":"Valuación hoy"})
                .style.format({
                    "P&L Realizado":"$ {:,.0f}", "P&L Papel":"$ {:,.0f}",
                    "P&L Total":"$ {:,.0f}", "Valuación hoy":"$ {:,.0f}",
                }).applymap(
                    lambda v: f"color:{GREEN};font-weight:700" if isinstance(v,(int,float)) and v>0
                    else (f"color:{NEIX_RED};font-weight:700" if isinstance(v,(int,float)) and v<0 else ""),
                    subset=["P&L Realizado","P&L Papel","P&L Total"]
                ),
                use_container_width=True, hide_index=True
            )

            st.plotly_chart(
                chart_hist_vs_open(
                    df_hist[["Ticker","Total_ARS"]],
                    df_res[["Ticker","Diferencia"]]
                ),
                use_container_width=True
            )

    # ── TAB 5: Charts ────────────────────────────────────────────────────────
    with tab5:
        if not df_res.empty:
            r1c1, r1c2 = st.columns(2, gap="large")
            with r1c1:
                st.plotly_chart(chart_bridge(df_res), use_container_width=True)
            with r1c2:
                st.plotly_chart(chart_inv_vs_val(df_res), use_container_width=True)
            st.plotly_chart(chart_scatter(df_res), use_container_width=True)
            tree = chart_treemap_open(df_res)
            if tree: st.plotly_chart(tree, use_container_width=True)

        st.plotly_chart(
            _hbar(df_hist.sort_values("Total_ARS"), "Total_ARS", "Ticker",
                  "P&L realizado completo · todas las especies"),
            use_container_width=True
        )
        cat_hist = df_hist.groupby("Categoria")["Total_ARS"].sum().reset_index().sort_values("Total_ARS")
        st.plotly_chart(
            _hbar(cat_hist, "Total_ARS", "Categoria", "P&L realizado por categoría", height=280),
            use_container_width=True
        )

    # ── TAB 6: Movimientos ───────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="stitle">Movimientos crudos del histórico</div>', unsafe_allow_html=True)
        all_movs = []
        for _, row in df_hist.iterrows():
            for m in row.get("_movs", []):
                all_movs.append(m)
        if all_movs:
            mov_df = pd.DataFrame(all_movs)
            tick_f = st.selectbox("Filtrar por ticker", ["Todos"] + sorted(df_hist["Ticker"].unique()), key="mvtick")
            if tick_f != "Todos": mov_df = mov_df[mov_df["Ticker"] == tick_f]
            st.dataframe(
                mov_df[["Ticker","Nombre","Cpbt","Numero","Fecha_Concertacion",
                         "Fecha_Liquidacion","Cantidad","Precio","Neto_ARS","Moneda","Neto_USD"]],
                use_container_width=True, hide_index=True, height=500
            )

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    dl_col, note_col = st.columns([0.22, 0.78], gap="large")
    with dl_col:
        st.download_button(
            "⬇  Exportar todo a Excel",
            data=build_excel(df_hist, df_res),
            file_name="neix_portafolio_completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with note_col:
        st.caption(
            "Dashboard consolidado: P&L realizado (Histórico por Especie) + P&L no realizado (Resultados por Especie). "
            "El rendimiento total combina operaciones cerradas + valuación actual de posiciones abiertas."
        )


if __name__ == "__main__":
    render()
