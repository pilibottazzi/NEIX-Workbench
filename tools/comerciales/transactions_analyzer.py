from __future__ import annotations
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup

R   = "#C8102E"
INK = "#0F172A"
SUB = "#64748B"
BDR = "#E2E8F0"
BG  = "#F8FAFC"
CRD = "#FFFFFF"
GRN = "#059669"
AMB = "#D97706"
BLU = "#2563EB"

LOGO = '<svg viewBox="0 0 72 22" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:20px;display:inline-block;vertical-align:middle;"><rect width="72" height="22" rx="4" fill="#C8102E"/><text x="6" y="16" font-family="Arial Black,Arial" font-weight="900" font-size="14" letter-spacing="-0.5" fill="white">NEIX</text></svg>'

def _write_config():
    d = Path.home() / ".streamlit"
    d.mkdir(exist_ok=True)
    (d / "config.toml").write_text(
        '[theme]\nbase="light"\nbackgroundColor="#F8FAFC"\n'
        'secondaryBackgroundColor="#FFFFFF"\ntextColor="#0F172A"\n'
        'primaryColor="#C8102E"\n[server]\nheadless=true\n'
        '[browser]\ngatherUsageStats=false\n'
    )

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#F8FAFC!important;color:#0F172A!important;font-family:'Inter',sans-serif!important;}
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"]>div,[data-testid="stVerticalBlock"]{background:transparent!important;color:#0F172A!important;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span,[data-testid="stMarkdownContainer"] div{color:inherit!important;}
[data-testid="stHeader"]{background:transparent!important;}
.block-container{max-width:1440px!important;padding-top:2rem!important;padding-bottom:3rem!important;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 0 1.4rem;border-bottom:1px solid #E2E8F0;margin-bottom:1.8rem;}
.topbar-left{display:flex;align-items:center;gap:1rem;}
.topbar-title{font-size:1rem;font-weight:700;color:#0F172A;letter-spacing:-0.01em;}
.topbar-sub{font-size:0.78rem;color:#64748B;margin-top:0.1rem;}
.topbar-right{font-size:0.76rem;color:#64748B;text-align:right;line-height:1.7;}
div[data-testid="stFileUploader"]{background:transparent!important;border:1px dashed #E2E8F0!important;border-radius:8px!important;}
.upload-label{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:0.5rem;}
.upload-hint{font-size:0.74rem;color:#64748B;margin-top:0.35rem;}
.empty-state{text-align:center;padding:4rem 2rem;}
.empty-icon{font-size:2.2rem;margin-bottom:0.8rem;opacity:0.35;}
.empty-title{font-size:1rem;font-weight:700;color:#0F172A;margin-bottom:0.35rem;}
.empty-sub{font-size:0.82rem;color:#64748B;}
.meta-row{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.6rem;align-items:center;}
.meta-chip{font-size:0.71rem;color:#64748B;font-weight:500;padding:0.2rem 0.6rem;background:#FFFFFF;border:1px solid #E2E8F0;border-radius:5px;}
.bridge{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid #E2E8F0;border-radius:10px;overflow:hidden;background:#FFFFFF;margin-bottom:1.6rem;}
.bc{padding:1rem;border-right:1px solid #E2E8F0;position:relative;}
.bc:last-child{border-right:none;}
.bc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.bc.red::before{background:#C8102E;} .bc.green::before{background:#059669;}
.bc.blue::before{background:#2563EB;} .bc.ink::before{background:#0F172A;}
.bc-label{font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#64748B;margin-bottom:0.45rem;}
.bc-val{font-size:1.05rem;font-weight:800;letter-spacing:-0.03em;font-family:'JetBrains Mono',monospace;line-height:1.1;}
.bc-sub{font-size:0.68rem;color:#64748B;margin-top:0.25rem;font-weight:500;}
.c-ink{color:#0F172A;} .c-green{color:#059669;} .c-red{color:#C8102E;} .c-blue{color:#2563EB;}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:0.85rem;margin-bottom:1.6rem;}
.kpi{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:9px;padding:0.9rem 1rem;}
.kpi-label{font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#64748B;margin-bottom:0.4rem;}
.kpi-val{font-size:1.3rem;font-weight:800;letter-spacing:-0.04em;font-family:'JetBrains Mono',monospace;color:#0F172A;line-height:1.1;}
.kpi-sub{font-size:0.73rem;font-weight:600;margin-top:0.22rem;}
.slabel{font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin:0 0 0.7rem;}
.exec-card{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:9px;padding:1.3rem 1.5rem;}
.exec-card p{color:#0F172A;line-height:1.85;margin:0;font-size:0.88rem;}
.st{width:100%;border-collapse:collapse;font-size:0.84rem;}
.st th{font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#64748B;padding:0.42rem 0.6rem;text-align:left;border-bottom:1px solid #E2E8F0;}
.st th.r{text-align:right;}
.st td{padding:0.45rem 0.6rem;border-bottom:1px solid #F1F5F9;color:#0F172A;}
.st td.r{text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.78rem;}
.st tr:last-child td{border-bottom:none;}
.st tr.tot td{font-weight:700;border-top:1px solid #E2E8F0;background:#F8FAFC;}
.badge{display:inline-block;padding:0.13rem 0.48rem;border-radius:4px;font-size:0.7rem;font-weight:700;}
.bg-pos{background:#ECFDF5;color:#059669;} .bg-neg{background:#FEF2F2;color:#C8102E;} .bg-neu{background:#F1F5F9;color:#64748B;}
div[data-testid="stTabs"] button{background:transparent!important;border:1px solid #E2E8F0!important;border-radius:6px!important;color:#64748B!important;font-weight:600!important;font-size:0.79rem!important;padding:0.3rem 0.85rem!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{background:#0F172A!important;border-color:#0F172A!important;color:white!important;}
div[data-testid="stDownloadButton"]>button{width:100%!important;border-radius:7px!important;background:#0F172A!important;color:white!important;border:none!important;font-weight:700!important;font-family:'Inter',sans-serif!important;font-size:0.8rem!important;}
.stDataFrame{border-radius:9px;overflow:hidden;border:1px solid #E2E8F0!important;}
</style>
"""

def _n(s) -> float:
    if s is None: return 0.0
    s = re.sub(r"\.(?=\d{3})", "", str(s).strip()).replace(",", ".")
    try: return float(s)
    except: return 0.0

def _ars(v: float) -> str:
    s = "{:,.0f}".format(abs(v)).replace(",", "X").replace(".", ",").replace("X", ".")
    return "-$ " + s if v < 0 else "$ " + s

def _pct(v: float) -> str: return "{:+.1f}%".format(v)
def _cls(v: float) -> str: return "c-green" if v > 0 else ("c-red" if v < 0 else "c-ink")

def _badge(v: float) -> str:
    cls = "bg-pos" if v > 0 else ("bg-neg" if v < 0 else "bg-neu")
    return '<span class="badge ' + cls + '">' + _ars(v) + '</span>'

CATS = {
    "acciones": {"EDN","GGAL","YPFD","SUPV"},
    "cedears":  {"AXP","GOOGL","MELI","META","NVDA","ADBE","TSLA","GLOB",
                 "BABA","VIST","UNH","SPY","COIN","LAC","NU","SPCE","MSFT"},
    "bonos":    {"AL30","AL35","GD35"},
    "fondos":   {"CAUCION","MEGA PES A","FIMA PRE A"},
    "lecaps":   {"S29G5","S16A5","S12S5","S30S5","S15G5","S31O5"},
}

def _cat(t: str) -> str:
    for c, ts in CATS.items():
        if t.upper() in ts: return c
    return "otros"

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_historico(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict]:
    soup = BeautifulSoup(file_bytes.decode("utf-8-sig"), "html.parser")
    meta = {"usuario": "", "comitente": "", "fecha_desde": "", "fecha_hasta": ""}
    enc = soup.find("div", id="encabezadoExcel")
    if enc:
        rows = enc.find_all("tr")
        if len(rows) >= 2:
            ths = [t.get_text(strip=True) for t in rows[0].find_all("th")]
            tds = [t.get_text(strip=True) for t in rows[1].find_all("td")]
            m = dict(zip(ths, tds))
            meta = {
                "usuario":     m.get("Usuario", "").strip(),
                "comitente":   m.get("Comitente", "").strip(),
                "fecha_desde": m.get("Fecha Desde", "").strip(),
                "fecha_hasta": m.get("Fecha Hasta", "").strip(),
            }
    records = []
    for box in soup.find_all("div", class_="box box-default"):
        h3 = box.find("h3")
        full = h3.get_text(strip=True) if h3 else "?"
        ticker = full.split()[0].upper()
        nombre = " ".join(full.split()[1:])
        table = box.find("table")
        if not table: continue
        total_ars = total_usd = 0.0
        movs: List[Dict] = []
        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if not cells: continue
            cpbt = cells[0].upper()
            if cpbt == "TOTAL":
                total_ars = _n(cells[6]) if len(cells) > 6 else 0.0
                total_usd = _n(cells[8]) if len(cells) > 8 else 0.0
            elif len(cells) >= 9:
                movs.append({
                    "Ticker": ticker, "Nombre": nombre, "Cpbt": cpbt,
                    "Numero": cells[1], "Fecha_Concertacion": cells[2],
                    "Fecha_Liquidacion": cells[3],
                    "Cantidad": _n(cells[4]), "Precio": _n(cells[5]),
                    "Neto_ARS": _n(cells[6]), "Moneda": cells[7], "Neto_USD": _n(cells[8]),
                })
        compras = sum(abs(m["Neto_ARS"]) for m in movs if m["Cpbt"] == "COMPRA" and m["Neto_ARS"] < 0)
        ventas  = sum(m["Neto_ARS"] for m in movs if m["Cpbt"] == "VENTA" and m["Neto_ARS"] > 0)
        divs    = sum(m["Neto_ARS"] for m in movs if m["Cpbt"] in {"DIVIDENDOS","RTA/AMORT","CREDITO RTA"} and m["Neto_ARS"] > 0)
        qc      = sum(m["Cantidad"] for m in movs if m["Cpbt"] == "COMPRA" and m["Cantidad"] > 0)
        qv      = sum(abs(m["Cantidad"]) for m in movs if m["Cpbt"] == "VENTA" and m["Cantidad"] < 0)
        records.append({
            "Ticker": ticker, "Nombre": nombre, "Categoria": _cat(ticker),
            "Qty_Comprada": qc, "Qty_Vendida": qv, "Saldo": qc - qv,
            "Compras_ARS": compras, "Ventas_ARS": ventas, "Dividendos_ARS": divs,
            "Total_ARS": total_ars, "Total_USD": total_usd, "N_Movs": len(movs),
            "_movs": movs,
        })
    return pd.DataFrame(records), meta


def parse_resultados(file_bytes: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(file_bytes.decode("utf-8-sig"), "html.parser")
    table = soup.find("table", class_="table-consultas")
    if table is None: return pd.DataFrame()
    records = []
    for row in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if not cells or len(cells) < 7: continue
        raw = cells[0].strip()
        if not cells[1].strip() or raw.lower() in {"perdida", "ganancia", "total", ""}: continue
        parts  = raw.split()
        ticker = parts[0].upper()
        nombre = " ".join(parts[1:])
        inv = _n(cells[3]); val = _n(cells[5]); dif = _n(cells[6])
        records.append({
            "Ticker": ticker, "Nombre": nombre, "Categoria": _cat(ticker),
            "Cantidad": _n(cells[1]), "PPP": _n(cells[2]),
            "Inversion": inv, "Precio_Actual": _n(cells[4]),
            "Valuacion": val, "Diferencia": dif,
            "Rend_Pct": (dif / inv * 100 if inv else 0.0),
        })
    return pd.DataFrame(records)


# ── Charts ────────────────────────────────────────────────────────────────────
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=SUB, family="Inter"),
    margin=dict(l=0, r=0, t=36, b=0),
    title_font=dict(size=12, color=INK, family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

def _hbar(df, x, y, title, h=None):
    colors = [R if v < 0 else GRN for v in df[x]]
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], orientation="h", marker_color=colors,
        text=df[x].apply(_ars), textposition="outside",
        textfont=dict(size=9, color=SUB),
    ))
    fig.update_layout(**BASE, title=title,
                      height=h or max(280, 34 * len(df)),
                      xaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor=BDR, zerolinewidth=1),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig

def _waterfall(df_res):
    d = df_res.sort_values("Diferencia")
    fig = go.Figure(go.Waterfall(
        orientation="v", x=d["Ticker"].tolist(), y=d["Diferencia"].tolist(),
        connector=dict(line=dict(color=BDR, width=1)),
        increasing=dict(marker_color=GRN), decreasing=dict(marker_color=R),
        text=[_ars(v) for v in d["Diferencia"]], textposition="outside",
    ))
    fig.update_layout(**BASE, title="P&L no realizado por especie", height=300,
                      xaxis=dict(gridcolor="#F1F5F9"),
                      yaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor=BDR),
                      showlegend=False)
    return fig

def _inv_vs_val(df_res):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Inversión", x=df_res["Ticker"], y=df_res["Inversion"],
                         marker_color="#CBD5E1", text=df_res["Inversion"].apply(_ars),
                         textposition="inside", textfont=dict(size=8)))
    fig.add_trace(go.Bar(name="Valuación actual", x=df_res["Ticker"], y=df_res["Valuacion"],
                         marker_color=BLU, text=df_res["Valuacion"].apply(_ars),
                         textposition="inside", textfont=dict(size=8)))
    fig.update_layout(**BASE, title="Inversión vs valuación actual",
                      height=300, barmode="group",
                      xaxis=dict(gridcolor="#F1F5F9"),
                      yaxis=dict(gridcolor="#F1F5F9"))
    return fig

def _combined_bar(df_h, df_r):
    all_t = sorted(set(df_h["Ticker"]) | set(df_r["Ticker"]))
    hm = dict(zip(df_h["Ticker"], df_h["Total_ARS"]))
    rm = dict(zip(df_r["Ticker"], df_r["Diferencia"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Realizado", x=all_t,
                         y=[hm.get(t, 0) for t in all_t], marker_color=GRN))
    fig.add_trace(go.Bar(name="No realizado (papel)", x=all_t,
                         y=[rm.get(t, 0) for t in all_t], marker_color=AMB))
    fig.update_layout(**BASE, title="P&L realizado vs papel por especie",
                      height=320, barmode="relative",
                      xaxis=dict(gridcolor="#F1F5F9"),
                      yaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor=BDR, zerolinewidth=1))
    return fig

def _treemap(df_res):
    d = df_res[df_res["Valuacion"] > 0].copy()
    if d.empty: return None
    fig = px.treemap(d, path=["Categoria", "Ticker"], values="Valuacion",
                     color="Diferencia",
                     color_continuous_scale=[R, "#CBD5E1", GRN],
                     color_continuous_midpoint=0,
                     title="Posición abierta — tamaño=valuación, color=P&L")
    fig.update_layout(**BASE, height=340)
    return fig


# ── Export ────────────────────────────────────────────────────────────────────
def _export(df_h: pd.DataFrame, df_r: pd.DataFrame) -> bytes:
    out = BytesIO()
    cols_h = ["Ticker","Nombre","Categoria","Qty_Comprada","Qty_Vendida","Saldo",
              "Compras_ARS","Ventas_ARS","Dividendos_ARS","Total_ARS","Total_USD"]
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_h[[c for c in cols_h if c in df_h.columns]].to_excel(w, sheet_name="Historico", index=False)
        if not df_r.empty:
            df_r[["Ticker","Nombre","Categoria","Cantidad","PPP","Inversion",
                  "Precio_Actual","Valuacion","Diferencia","Rend_Pct"]].to_excel(
                w, sheet_name="Posicion Actual", index=False)
            mg = df_h[["Ticker","Total_ARS"]].merge(
                df_r[["Ticker","Diferencia","Valuacion"]], on="Ticker", how="outer").fillna(0)
            mg["PnL_Total"] = mg["Total_ARS"] + mg["Diferencia"]
            mg.to_excel(w, sheet_name="Combinado", index=False)
    out.seek(0)
    return out.getvalue()


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    _write_config()
    st.set_page_config(
        page_title="NEIX · Portfolio", page_icon="📊",
        layout="wide", initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Topbar
    st.markdown(
        '<div class="topbar">'
        '<div class="topbar-left">' + LOGO +
        '<div><div class="topbar-title">Dashboard de Portafolio</div>'
        '<div class="topbar-sub">P&amp;L realizado &middot; posición abierta &middot; vista consolidada</div>'
        '</div></div>'
        '<div class="topbar-right">Acciones &middot; CEDEARs &middot; Bonos &middot; LECAPs &middot; Fondos<br/>ARS &amp; USD</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Uploaders
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="upload-label">Histórico por especie</div>', unsafe_allow_html=True)
        up_hist = st.file_uploader("Histórico", type=["xls","xlsx"],
                                   label_visibility="collapsed", key="u_hist")
        st.markdown('<div class="upload-hint">Operaciones realizadas &middot; P&amp;L realizado</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="upload-label">Resultados por especie <span style="font-weight:400;text-transform:none;letter-spacing:0">(opcional)</span></div>',
                    unsafe_allow_html=True)
        up_res = st.file_uploader("Resultados", type=["xls","xlsx"],
                                  label_visibility="collapsed", key="u_res")
        st.markdown('<div class="upload-hint">Posición abierta &middot; valuación actual &middot; P&amp;L no realizado</div>',
                    unsafe_allow_html=True)

    if not up_hist:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">📂</div>'
            '<div class="empty-title">Subí el Histórico por Especie para comenzar</div>'
            '<div class="empty-sub">El archivo de Resultados es opcional — agrega la valuación de posiciones abiertas</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Parse
    try:
        df_h, meta = parse_historico(up_hist.getvalue())
    except Exception as e:
        st.error("No se pudo leer el Histórico: " + str(e)); return

    df_r = pd.DataFrame()
    if up_res:
        try:
            df_r = parse_resultados(up_res.getvalue())
        except Exception as e:
            st.warning("No se pudo leer Resultados: " + str(e))

    if df_h.empty:
        st.warning("El archivo no contiene datos reconocibles."); return

    # Meta strip
    open_chip = ('<span class="meta-chip">&#128994; ' + str(len(df_r)) + ' posiciones abiertas</span>') if not df_r.empty else ""
    st.markdown(
        '<div class="meta-row">'
        '<span class="meta-chip">&#128100; ' + meta.get("usuario","—") + '</span>'
        '<span class="meta-chip">Comitente ' + meta.get("comitente","—") + '</span>'
        '<span class="meta-chip">' + meta.get("fecha_desde","—") + ' &#8594; ' + meta.get("fecha_hasta","hoy") + '</span>'
        '<span class="meta-chip">' + str(len(df_h)) + ' especies operadas</span>'
        + open_chip + '</div>',
        unsafe_allow_html=True,
    )

    # Numbers
    pnl_r   = df_h["Total_ARS"].sum()
    cap     = df_h["Compras_ARS"].sum()
    divs    = df_h["Dividendos_ARS"].sum()
    rp      = pnl_r / cap * 100 if cap else 0.0
    pnl_nr  = df_r["Diferencia"].sum()  if not df_r.empty else 0.0
    val_hoy = df_r["Valuacion"].sum()   if not df_r.empty else 0.0
    inv_ab  = df_r["Inversion"].sum()   if not df_r.empty else 0.0
    rp_ab   = pnl_nr / inv_ab * 100     if inv_ab else 0.0
    pnl_tot = pnl_r + pnl_nr
    rp_tot  = pnl_tot / cap * 100       if cap else 0.0
    n_pos   = int((df_h["Total_ARS"] > 0).sum())
    n_neg   = int((df_h["Total_ARS"] < 0).sum())
    mejor   = df_h.loc[df_h["Total_ARS"].idxmax()]
    peor    = df_h.loc[df_h["Total_ARS"].idxmin()]

    # Bridge
    def _bc(border_cls, val_cls, label, val_str, sub=""):
        return (
            '<div class="bc ' + border_cls + '">'
            '<div class="bc-label">' + label + '</div>'
            '<div class="bc-val ' + val_cls + '">' + val_str + '</div>'
            + ('<div class="bc-sub">' + sub + '</div>' if sub else '') +
            '</div>'
        )

    no_r_bc = _bc("red" if pnl_nr < 0 else "green", _cls(pnl_nr), "P&amp;L no realizado",
                  _ars(pnl_nr), _pct(rp_ab) + " s/abierto") if not df_r.empty else _bc("ink","c-ink","P&amp;L no realizado","—","sin datos")
    val_bc  = _bc("blue","c-blue","Valuación hoy", _ars(val_hoy), str(len(df_r)) + " posiciones") if not df_r.empty else _bc("ink","c-ink","Valuación hoy","—","sin datos")

    st.markdown(
        '<div class="bridge">'
        + _bc("ink","c-ink","Capital invertido", _ars(cap), str(len(df_h)) + " especies")
        + _bc("red" if pnl_r < 0 else "green", _cls(pnl_r), "P&amp;L realizado", _ars(pnl_r), _pct(rp) + " s/capital")
        + _bc("green" if divs >= 0 else "red", _cls(divs), "Dividendos / carry", _ars(divs), "cobrado")
        + no_r_bc + val_bc
        + _bc("red" if pnl_tot < 0 else "green", _cls(pnl_tot), "P&amp;L TOTAL", _ars(pnl_tot), _pct(rp_tot) + " combinado")
        + '</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    nr_val = _ars(pnl_nr) if not df_r.empty else "—"
    nr_sub = _pct(rp_ab) if not df_r.empty else "sin datos"
    nr_col = (R if pnl_nr < 0 else (GRN if pnl_nr > 0 else SUB)) if not df_r.empty else SUB
    v_val  = _ars(val_hoy) if not df_r.empty else "—"

    st.markdown(
        '<div class="kpi-row">'
        '<div class="kpi"><div class="kpi-label">P&amp;L realizado</div>'
        '<div class="kpi-val" style="color:' + (R if pnl_r < 0 else GRN) + '">' + _ars(pnl_r) + '</div>'
        '<div class="kpi-sub" style="color:' + (R if rp < 0 else GRN) + '">' + _pct(rp) + ' s/capital</div></div>'

        '<div class="kpi"><div class="kpi-label">P&amp;L no realizado</div>'
        '<div class="kpi-val" style="color:' + nr_col + '">' + nr_val + '</div>'
        '<div class="kpi-sub" style="color:' + SUB + '">' + nr_sub + '</div></div>'

        '<div class="kpi"><div class="kpi-label">Valuación actual</div>'
        '<div class="kpi-val" style="color:' + BLU + '">' + v_val + '</div>'
        '<div class="kpi-sub" style="color:' + SUB + '">' + str(len(df_r)) + ' posiciones abiertas</div></div>'

        '<div class="kpi"><div class="kpi-label">P&amp;L total combinado</div>'
        '<div class="kpi-val" style="color:' + (R if pnl_tot < 0 else GRN) + '">' + _ars(pnl_tot) + '</div>'
        '<div class="kpi-sub" style="color:' + (R if rp_tot < 0 else GRN) + '">' + _pct(rp_tot) + '</div></div>'

        '<div class="kpi"><div class="kpi-label">Win rate</div>'
        '<div class="kpi-val">' + str(n_pos) + '/' + str(len(df_h)) + '</div>'
        '<div class="kpi-sub" style="color:' + R + '">' + str(n_neg) + ' en negativo</div></div>'

        '<div class="kpi"><div class="kpi-label">Top / Bottom</div>'
        '<div class="kpi-val" style="font-size:1rem">' + str(mejor["Ticker"]) + '</div>'
        '<div class="kpi-sub" style="color:' + R + '">Peor: ' + str(peor["Ticker"]) + '</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tabs = st.tabs(["Resumen", "Histórico", "Posición abierta", "Real. vs Papel", "Gráficos", "Movimientos"])

    # TAB 0 Resumen
    with tabs[0]:
        c1, c2 = st.columns([1.05, 0.95], gap="large")
        with c1:
            st.markdown('<div class="slabel">Lectura ejecutiva</div>', unsafe_allow_html=True)
            pct_win = n_pos / len(df_h) * 100
            res_txt = ""
            if not df_r.empty:
                mb = df_r.loc[df_r["Diferencia"].idxmax()]
                pb = df_r.loc[df_r["Diferencia"].idxmin()]
                res_txt = (
                    " Las posiciones abiertas registran un P&amp;L no realizado de "
                    "<strong style='color:" + (R if pnl_nr < 0 else GRN) + "'>" + _ars(pnl_nr) + "</strong>"
                    " (" + _pct(rp_ab) + ") sobre " + _ars(inv_ab) + " invertidos,"
                    " con valuación actual de <strong style='color:" + BLU + "'>" + _ars(val_hoy) + "</strong>."
                    " Mejor posición abierta: <strong>" + str(mb["Ticker"]) + "</strong> (" + _ars(float(mb["Diferencia"])) + "),"
                    " peor: <strong>" + str(pb["Ticker"]) + "</strong> (" + _ars(float(pb["Diferencia"])) + ")."
                )
            narrative = (
                "El portafolio acumula un P&amp;L realizado de "
                "<strong style='color:" + (R if pnl_r < 0 else GRN) + "'>" + _ars(pnl_r) + "</strong>"
                " sobre " + str(len(df_h)) + " especies operadas"
                " (" + meta.get("fecha_desde","?") + " &#8594; " + meta.get("fecha_hasta","hoy") + "),"
                " equivalente a <strong style='color:" + (R if rp < 0 else GRN) + "'>" + _pct(rp) + "</strong>"
                " sobre el capital desplegado de <strong>" + _ars(cap) + "</strong>."
                "<br/><br/>"
                + str(n_pos) + " de " + str(len(df_h)) + " especies (" + "{:.0f}".format(pct_win) + "%) cerraron en positivo."
                " Mejor contribución: <strong style='color:" + GRN + "'>" + str(mejor["Ticker"]) + "</strong>"
                " (" + _ars(float(mejor["Total_ARS"])) + ")."
                " Peor: <strong style='color:" + R + "'>" + str(peor["Ticker"]) + "</strong>"
                " (" + _ars(float(peor["Total_ARS"])) + ")."
                " El carry cobrado sumó <strong>" + _ars(divs) + "</strong>."
                + res_txt
            )
            st.markdown('<div class="exec-card"><p>' + narrative + '</p></div>', unsafe_allow_html=True)

            # Consolidated table
            st.markdown('<div class="slabel" style="margin-top:1.4rem">Cuadro consolidado</div>', unsafe_allow_html=True)
            rows_data = [
                ("Capital invertido (ARS)", _ars(cap), INK),
                ("Total ventas ARS", _ars(df_h["Ventas_ARS"].sum()), INK),
                ("Dividendos / rentas cobrados", _ars(divs), GRN),
                ("P&amp;L realizado", _ars(pnl_r), GRN if pnl_r >= 0 else R),
            ]
            if not df_r.empty:
                rows_data += [
                    ("Inversión posiciones abiertas", _ars(inv_ab), INK),
                    ("Valuación a mercado hoy", _ars(val_hoy), BLU),
                    ("P&amp;L no realizado", _ars(pnl_nr), GRN if pnl_nr >= 0 else R),
                ]
            rows_data += [
                ("P&amp;L TOTAL COMBINADO", _ars(pnl_tot), GRN if pnl_tot >= 0 else R),
                ("Rendimiento s/capital", _pct(rp_tot), GRN if rp_tot >= 0 else R),
            ]
            trs = "".join(
                '<tr><td style="color:' + SUB + '">' + k + '</td>'
                '<td class="r" style="color:' + c + '">' + v + '</td></tr>'
                for k, v, c in rows_data
            )
            st.markdown('<table class="st">' + trs + '</table>', unsafe_allow_html=True)

        with c2:
            if not df_r.empty:
                st.plotly_chart(_waterfall(df_r), use_container_width=True)
            st.plotly_chart(_hbar(df_h.sort_values("Total_ARS"), "Total_ARS", "Ticker",
                                  "P&L realizado por especie"), use_container_width=True)

    # TAB 1 Historico
    with tabs[1]:
        fa, fb = st.columns([2, 1], gap="medium")
        with fa:
            cats = sorted(df_h["Categoria"].unique())
            sel = st.multiselect("Categoría", cats, default=cats, key="hc")
        with fb:
            solo_neg = st.toggle("Solo perdedoras", key="hn")
        dh = df_h[df_h["Categoria"].isin(sel)]
        if solo_neg: dh = dh[dh["Total_ARS"] < 0]
        show = dh[["Ticker","Nombre","Categoria","Qty_Comprada","Qty_Vendida","Saldo",
                   "Compras_ARS","Ventas_ARS","Dividendos_ARS","Total_ARS","Total_USD"]].copy()
        st.dataframe(
            show.style.format({
                "Qty_Comprada":"{:,.0f}","Qty_Vendida":"{:,.0f}","Saldo":"{:,.0f}",
                "Compras_ARS":"$ {:,.0f}","Ventas_ARS":"$ {:,.0f}",
                "Dividendos_ARS":"$ {:,.0f}","Total_ARS":"$ {:,.0f}","Total_USD":"U$S {:,.2f}",
            }).map(
                lambda v: "color:" + GRN + ";font-weight:700" if isinstance(v, (int, float)) and v > 0
                else ("color:" + R + ";font-weight:700" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["Total_ARS","Total_USD"]
            ),
            use_container_width=True, hide_index=True, height=min(560, 40+38*len(show))
        )
        # Category summary
        st.markdown('<div class="slabel" style="margin-top:1.2rem">Por categoría</div>', unsafe_allow_html=True)
        cat_h = df_h.groupby("Categoria").agg(
            N=("Ticker","count"), Capital=("Compras_ARS","sum"),
            Ventas=("Ventas_ARS","sum"), Dividendos=("Dividendos_ARS","sum"),
            Total=("Total_ARS","sum"),
        ).reset_index().sort_values("Total", ascending=False)
        hdr = '<tr><th>Categoría</th><th class="r">N</th><th class="r">Capital</th><th class="r">Ventas</th><th class="r">Divid.</th><th class="r">Total ARS</th></tr>'
        trs = ""
        for _, row in cat_h.iterrows():
            trs += ('<tr><td><strong>' + str(row["Categoria"]).upper() + '</strong></td>'
                    '<td class="r">' + str(int(row["N"])) + '</td>'
                    '<td class="r">' + _ars(row["Capital"]) + '</td>'
                    '<td class="r">' + _ars(row["Ventas"]) + '</td>'
                    '<td class="r" style="color:' + GRN + '">' + _ars(row["Dividendos"]) + '</td>'
                    '<td class="r">' + _badge(row["Total"]) + '</td></tr>')
        trs += ('<tr class="tot"><td>TOTAL</td><td class="r">' + str(len(df_h)) + '</td>'
                '<td class="r">' + _ars(cat_h["Capital"].sum()) + '</td>'
                '<td class="r">' + _ars(cat_h["Ventas"].sum()) + '</td>'
                '<td class="r" style="color:' + GRN + '">' + _ars(cat_h["Dividendos"].sum()) + '</td>'
                '<td class="r">' + _badge(pnl_r) + '</td></tr>')
        st.markdown('<table class="st">' + hdr + trs + '</table>', unsafe_allow_html=True)

    # TAB 2 Posicion abierta
    with tabs[2]:
        if df_r.empty:
            st.markdown('<div class="empty-state"><div class="empty-icon">📈</div><div class="empty-title">No cargaste el archivo de Resultados</div><div class="empty-sub">Subí el archivo para ver la valuación actual de tus posiciones abiertas</div></div>', unsafe_allow_html=True)
        else:
            c1, c2 = st.columns([1.1, 0.9], gap="large")
            with c1:
                st.dataframe(
                    df_r[["Ticker","Nombre","Categoria","Cantidad","PPP",
                           "Inversion","Precio_Actual","Valuacion","Diferencia","Rend_Pct"]]
                    .style.format({
                        "Cantidad":"{:,.0f}","PPP":"$ {:,.3f}","Inversion":"$ {:,.0f}",
                        "Precio_Actual":"$ {:,.3f}","Valuacion":"$ {:,.0f}",
                        "Diferencia":"$ {:,.0f}","Rend_Pct":"{:+.2f}%",
                    }).map(
                        lambda v: "color:" + GRN + ";font-weight:700" if isinstance(v, (int, float)) and v > 0
                        else ("color:" + R + ";font-weight:700" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=["Diferencia","Rend_Pct"]
                    ),
                    use_container_width=True, hide_index=True
                )
                st.markdown(
                    '<div style="margin-top:1rem"><table class="st">'
                    '<tr><td style="color:' + SUB + '">Total invertido (posiciones abiertas)</td><td class="r">' + _ars(inv_ab) + '</td></tr>'
                    '<tr><td style="color:' + SUB + '">Valuación total a mercado</td><td class="r" style="color:' + BLU + '">' + _ars(val_hoy) + '</td></tr>'
                    '<tr class="tot"><td>P&amp;L no realizado</td><td class="r" style="color:' + (R if pnl_nr < 0 else GRN) + '">' + _ars(pnl_nr) + ' (' + _pct(rp_ab) + ')</td></tr>'
                    '</table></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.plotly_chart(_inv_vs_val(df_r), use_container_width=True)
                t = _treemap(df_r)
                if t: st.plotly_chart(t, use_container_width=True)

    # TAB 3 Real vs Papel
    with tabs[3]:
        if df_r.empty:
            st.info("Cargá el archivo de Resultados para ver la vista combinada.")
        else:
            mg = df_h[["Ticker","Nombre","Total_ARS"]].merge(
                df_r[["Ticker","Diferencia","Valuacion"]], on="Ticker", how="outer"
            ).fillna(0)
            mg["PnL_Total"] = mg["Total_ARS"] + mg["Diferencia"]
            st.dataframe(
                mg.rename(columns={"Total_ARS":"P&L Realizado","Diferencia":"P&L Papel",
                                   "PnL_Total":"P&L Total","Valuacion":"Valuación"})
                .style.format({
                    "P&L Realizado":"$ {:,.0f}","P&L Papel":"$ {:,.0f}",
                    "P&L Total":"$ {:,.0f}","Valuación":"$ {:,.0f}",
                }).map(
                    lambda v: "color:" + GRN + ";font-weight:700" if isinstance(v, (int, float)) and v > 0
                    else ("color:" + R + ";font-weight:700" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["P&L Realizado","P&L Papel","P&L Total"]
                ),
                use_container_width=True, hide_index=True
            )
            st.plotly_chart(_combined_bar(df_h[["Ticker","Total_ARS"]], df_r[["Ticker","Diferencia"]]),
                            use_container_width=True)

    # TAB 4 Graficos
    with tabs[4]:
        if not df_r.empty:
            g1, g2 = st.columns(2, gap="large")
            with g1: st.plotly_chart(_waterfall(df_r), use_container_width=True)
            with g2: st.plotly_chart(_inv_vs_val(df_r), use_container_width=True)
            st.plotly_chart(_combined_bar(df_h[["Ticker","Total_ARS"]], df_r[["Ticker","Diferencia"]]),
                            use_container_width=True)
            t = _treemap(df_r)
            if t: st.plotly_chart(t, use_container_width=True)
        st.plotly_chart(_hbar(df_h.sort_values("Total_ARS"), "Total_ARS", "Ticker",
                              "P&L realizado — todas las especies"), use_container_width=True)
        cat_agg = df_h.groupby("Categoria")["Total_ARS"].sum().reset_index().sort_values("Total_ARS")
        st.plotly_chart(_hbar(cat_agg, "Total_ARS", "Categoria", "P&L por categoría", h=260),
                        use_container_width=True)

    # TAB 5 Movimientos
    with tabs[5]:
        all_movs = [m for _, row in df_h.iterrows() for m in row.get("_movs", [])]
        if all_movs:
            mv = pd.DataFrame(all_movs)
            tf = st.selectbox("Ticker", ["Todos"] + sorted(df_h["Ticker"].unique()), key="mvt")
            if tf != "Todos": mv = mv[mv["Ticker"] == tf]
            st.dataframe(
                mv[["Ticker","Nombre","Cpbt","Numero","Fecha_Concertacion",
                    "Fecha_Liquidacion","Cantidad","Precio","Neto_ARS","Moneda","Neto_USD"]],
                use_container_width=True, hide_index=True, height=480
            )

    # Download
    st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:1.5rem 0 1rem'/>", unsafe_allow_html=True)
    dc, nc = st.columns([0.2, 0.8], gap="large")
    with dc:
        st.download_button(
            "↓  Exportar Excel",
            data=_export(df_h, df_r),
            file_name="neix_portafolio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with nc:
        st.caption("Exporta Histórico · Posición Actual · Combinado en un único archivo Excel.")


if __name__ == "__main__":
    main()
