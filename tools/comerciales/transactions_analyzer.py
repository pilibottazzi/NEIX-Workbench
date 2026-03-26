from __future__ import annotations
import re, io
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup

# ─── reportlab ───────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ─── Palette ─────────────────────────────────────────────────────────────────
R   = "#C8102E"
INK = "#0F172A"
SUB = "#64748B"
BDR = "#E2E8F0"
BG  = "#F8FAFC"
GRN = "#059669"
BLU = "#2563EB"
AMB = "#D97706"

RL_RED   = colors.HexColor("#C8102E")
RL_INK   = colors.HexColor("#0F172A")
RL_SUB   = colors.HexColor("#64748B")
RL_BDR   = colors.HexColor("#E2E8F0")
RL_BG    = colors.HexColor("#F8FAFC")
RL_GRN   = colors.HexColor("#059669")
RL_GRN_L = colors.HexColor("#ECFDF5")
RL_RED_L = colors.HexColor("#FEF2F2")
RL_BLU   = colors.HexColor("#2563EB")
RL_AMB   = colors.HexColor("#D97706")
RL_WHITE = colors.white

LOGO_HTML = '<svg viewBox="0 0 72 22" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:20px;display:inline-block;vertical-align:middle;"><rect width="72" height="22" rx="4" fill="#C8102E"/><text x="6" y="16" font-family="Arial Black,Arial" font-weight="900" font-size="14" letter-spacing="-0.5" fill="white">NEIX</text></svg>'

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
.block-container{max-width:1400px!important;padding-top:2rem!important;padding-bottom:3rem!important;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding-bottom:1.4rem;border-bottom:2px solid #0F172A;margin-bottom:2rem;}
.topbar-left{display:flex;align-items:center;gap:1.2rem;}
.topbar-name{font-size:0.95rem;font-weight:700;color:#0F172A;letter-spacing:-0.01em;}
.topbar-sub{font-size:0.75rem;color:#64748B;margin-top:0.08rem;}
.topbar-right{font-size:0.73rem;color:#64748B;text-align:right;line-height:1.8;}
.upload-label{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#64748B;margin-bottom:0.5rem;}
.upload-hint{font-size:0.72rem;color:#94A3B8;margin-top:0.3rem;}
div[data-testid="stFileUploader"]{background:#FFFFFF!important;border:1px solid #E2E8F0!important;border-radius:8px!important;}
.empty-wrap{text-align:center;padding:5rem 2rem;}
.empty-title{font-size:0.95rem;font-weight:700;color:#0F172A;margin-bottom:0.4rem;}
.empty-sub{font-size:0.82rem;color:#64748B;}
.meta-row{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.8rem;}
.meta-chip{font-size:0.7rem;color:#64748B;font-weight:500;padding:0.2rem 0.65rem;background:#FFFFFF;border:1px solid #E2E8F0;border-radius:5px;}
.bridge{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid #E2E8F0;border-radius:10px;overflow:hidden;background:#FFFFFF;margin-bottom:1.8rem;box-shadow:0 1px 3px rgba(0,0,0,0.04);}
.bc{padding:1.1rem 1rem;border-right:1px solid #E2E8F0;position:relative;}
.bc:last-child{border-right:none;}
.bc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:#E2E8F0;}
.bc.red::before{background:#C8102E;} .bc.grn::before{background:#059669;} .bc.blu::before{background:#2563EB;} .bc.ink::before{background:#0F172A;}
.bc-lbl{font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:0.5rem;}
.bc-val{font-size:1rem;font-weight:800;letter-spacing:-0.03em;font-family:'JetBrains Mono',monospace;line-height:1.1;}
.bc-sub{font-size:0.65rem;color:#64748B;margin-top:0.25rem;}
.c-ink{color:#0F172A;} .c-grn{color:#059669;} .c-red{color:#C8102E;} .c-blu{color:#2563EB;}
.krow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.9rem;margin-bottom:1.8rem;}
.kpi{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:0.9rem 1rem;box-shadow:0 1px 2px rgba(0,0,0,0.03);}
.kpi-lbl{font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:0.4rem;}
.kpi-val{font-size:1.25rem;font-weight:800;letter-spacing:-0.04em;font-family:'JetBrains Mono',monospace;color:#0F172A;line-height:1.1;}
.kpi-sub{font-size:0.71rem;font-weight:600;margin-top:0.2rem;}
.slbl{font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.11em;color:#64748B;margin:0 0 0.7rem;}
.section-card{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:1.3rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.04);margin-bottom:1rem;}
.narrative{color:#0F172A;line-height:1.9;margin:0;font-size:0.88rem;}
.tbl{width:100%;border-collapse:collapse;font-size:0.83rem;}
.tbl th{font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#64748B;padding:0.4rem 0.65rem;text-align:left;border-bottom:1px solid #E2E8F0;white-space:nowrap;}
.tbl th.r{text-align:right;}
.tbl td{padding:0.44rem 0.65rem;border-bottom:1px solid #F1F5F9;color:#0F172A;}
.tbl td.r{text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.78rem;white-space:nowrap;}
.tbl tr:last-child td{border-bottom:none;}
.tbl tr.tot td{font-weight:700;border-top:1px solid #E2E8F0;background:#F8FAFC;}
.badge{display:inline-block;padding:0.12rem 0.48rem;border-radius:4px;font-size:0.69rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.b-pos{background:#ECFDF5;color:#059669;} .b-neg{background:#FEF2F2;color:#C8102E;} .b-neu{background:#F1F5F9;color:#64748B;}
div[data-testid="stTabs"] button{background:#FFFFFF!important;border:1px solid #E2E8F0!important;border-radius:6px!important;color:#64748B!important;font-weight:600!important;font-size:0.78rem!important;padding:0.28rem 0.85rem!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{background:#0F172A!important;border-color:#0F172A!important;color:white!important;}
div[data-testid="stDownloadButton"]>button{width:100%!important;border-radius:7px!important;background:#0F172A!important;color:white!important;border:none!important;font-weight:700!important;font-family:'Inter',sans-serif!important;font-size:0.79rem!important;letter-spacing:0.01em!important;}
.stDataFrame{border-radius:8px;overflow:hidden;border:1px solid #E2E8F0!important;}
hr.divider{border:none;border-top:1px solid #E2E8F0;margin:1.5rem 0 1rem;}
</style>
"""

# ─── Number helpers ───────────────────────────────────────────────────────────
def _n(s) -> float:
    if s is None: return 0.0
    s = re.sub(r"\.(?=\d{3})", "", str(s).strip()).replace(",", ".")
    try: return float(s)
    except: return 0.0

def _ars(v: float, dec=0) -> str:
    fmt = "{:,."+str(dec)+"f}"
    s = fmt.format(abs(v)).replace(",","X").replace(".",",").replace("X",".")
    return ("-$ " if v < 0 else "$ ") + s

def _pct(v: float) -> str: return "{:+.1f}%".format(v)
def _cls(v: float) -> str: return "c-grn" if v > 0 else ("c-red" if v < 0 else "c-ink")
def _col(v: float) -> str: return GRN if v > 0 else (R if v < 0 else INK)

def _badge(v: float, fn=None) -> str:
    fn = fn or _ars
    cls = "b-pos" if v > 0 else ("b-neg" if v < 0 else "b-neu")
    return '<span class="badge ' + cls + '">' + fn(v) + '</span>'

CATS = {
    "Acciones":{"EDN","GGAL","YPFD","SUPV"},
    "CEDEARs": {"AXP","GOOGL","MELI","META","NVDA","ADBE","TSLA","GLOB",
                "BABA","VIST","UNH","SPY","COIN","LAC","NU","SPCE","MSFT"},
    "Bonos":   {"AL30","AL35","GD35"},
    "Fondos":  {"CAUCION","MEGA PES A","FIMA PRE A"},
    "LECAPs":  {"S29G5","S16A5","S12S5","S30S5","S15G5","S31O5"},
}
def _cat(t: str) -> str:
    for c, ts in CATS.items():
        if t.upper() in ts: return c
    return "Otros"

# ─── Parsers ──────────────────────────────────────────────────────────────────
def parse_historico(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    soup = BeautifulSoup(file_bytes.decode("utf-8-sig"), "html.parser")
    meta = {"usuario":"","comitente":"","fecha_desde":"","fecha_hasta":""}
    enc = soup.find("div", id="encabezadoExcel")
    if enc:
        rows = enc.find_all("tr")
        if len(rows) >= 2:
            ths = [t.get_text(strip=True) for t in rows[0].find_all("th")]
            tds = [t.get_text(strip=True) for t in rows[1].find_all("td")]
            m = dict(zip(ths, tds))
            meta = {"usuario": m.get("Usuario","").strip(),
                    "comitente": m.get("Comitente","").strip(),
                    "fecha_desde": m.get("Fecha Desde","").strip(),
                    "fecha_hasta": m.get("Fecha Hasta","").strip()}

    summary_rows, all_movs = [], []
    for box in soup.find_all("div", class_="box box-default"):
        h3 = box.find("h3")
        full = h3.get_text(strip=True) if h3 else "?"
        ticker = full.split()[0].upper()
        nombre = " ".join(full.split()[1:])
        table = box.find("table")
        if not table: continue
        total_ars = total_usd = 0.0
        movs = []
        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["th","td"])]
            if not cells: continue
            cpbt = cells[0].upper()
            if cpbt == "TOTAL":
                total_ars = _n(cells[6]) if len(cells) > 6 else 0.0
                total_usd = _n(cells[8]) if len(cells) > 8 else 0.0
            elif len(cells) >= 9:
                mov = {"Ticker":ticker,"Especie":nombre,"Comprobante":cpbt,
                       "Numero":cells[1],"Fecha Concertacion":cells[2],
                       "Fecha Liquidacion":cells[3],
                       "Cantidad":_n(cells[4]),"Precio":_n(cells[5]),
                       "Neto ARS":_n(cells[6]),"Moneda":cells[7],"Neto USD":_n(cells[8])}
                movs.append(mov)
                all_movs.append(mov)

        compras = sum(abs(m["Neto ARS"]) for m in movs if m["Comprobante"]=="COMPRA" and m["Neto ARS"]<0)
        ventas  = sum(m["Neto ARS"] for m in movs if m["Comprobante"]=="VENTA" and m["Neto ARS"]>0)
        divs    = sum(m["Neto ARS"] for m in movs if m["Comprobante"] in {"DIVIDENDOS","RTA/AMORT","CREDITO RTA"} and m["Neto ARS"]>0)
        qc = sum(m["Cantidad"] for m in movs if m["Comprobante"]=="COMPRA" and m["Cantidad"]>0)
        qv = sum(abs(m["Cantidad"]) for m in movs if m["Comprobante"]=="VENTA" and m["Cantidad"]<0)
        summary_rows.append({
            "Ticker":ticker,"Especie":nombre,"Categoria":_cat(ticker),
            "Qty Comprada":qc,"Qty Vendida":qv,"Saldo":qc-qv,
            "Compras ARS":compras,"Ventas ARS":ventas,"Dividendos ARS":divs,
            "PnL ARS":total_ars,"PnL USD":total_usd,"N Movimientos":len(movs),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(all_movs), meta


def parse_resultados(file_bytes: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(file_bytes.decode("utf-8-sig"), "html.parser")
    table = soup.find("table", class_="table-consultas")
    if table is None: return pd.DataFrame()
    records = []
    for row in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
        if not cells or len(cells) < 7: continue
        raw = cells[0].strip()
        if not cells[1].strip() or raw.lower() in {"perdida","ganancia","total",""}: continue
        parts = raw.split(); ticker = parts[0].upper(); nombre = " ".join(parts[1:])
        inv = _n(cells[3]); val = _n(cells[5]); dif = _n(cells[6])
        records.append({
            "Ticker":ticker,"Especie":nombre,"Categoria":_cat(ticker),
            "Cantidad":_n(cells[1]),"PPP":_n(cells[2]),
            "Inversion":inv,"Precio Actual":_n(cells[4]),
            "Valuacion":val,"Diferencia":dif,
            "Rend %":(dif/inv*100 if inv else 0.0),
        })
    return pd.DataFrame(records)


# ─── Charts ───────────────────────────────────────────────────────────────────
BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=SUB, family="Inter"),
            margin=dict(l=0, r=10, t=36, b=0),
            title_font=dict(size=12, color=INK, family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))

def _hbar(df, x, y, title, h=None):
    colors_list = [R if v < 0 else GRN for v in df[x]]
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], orientation="h", marker_color=colors_list,
        text=df[x].apply(_ars), textposition="outside", textfont=dict(size=9, color=SUB),
    ))
    fig.update_layout(**BASE, title=title, height=h or max(280, 34*len(df)),
                      xaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#E2E8F0"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig

def _waterfall(df_r):
    d = df_r.sort_values("Diferencia")
    fig = go.Figure(go.Waterfall(
        orientation="v", x=d["Ticker"].tolist(), y=d["Diferencia"].tolist(),
        connector=dict(line=dict(color="#E2E8F0", width=1)),
        increasing=dict(marker_color=GRN), decreasing=dict(marker_color=R),
        text=[_ars(v) for v in d["Diferencia"]], textposition="outside",
    ))
    fig.update_layout(**BASE, title="P&L no realizado por especie (posición abierta)",
                      height=300, xaxis=dict(gridcolor="#F1F5F9"),
                      yaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#E2E8F0"),
                      showlegend=False)
    return fig

def _combined(df_h, df_r):
    all_t = sorted(set(df_h["Ticker"]) | set(df_r["Ticker"]))
    hm = dict(zip(df_h["Ticker"], df_h["PnL ARS"]))
    rm = dict(zip(df_r["Ticker"], df_r["Diferencia"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Realizado", x=all_t, y=[hm.get(t,0) for t in all_t], marker_color=GRN))
    fig.add_trace(go.Bar(name="No realizado", x=all_t, y=[rm.get(t,0) for t in all_t], marker_color=AMB))
    fig.update_layout(**BASE, title="P&L realizado vs no realizado", height=320, barmode="relative",
                      xaxis=dict(gridcolor="#F1F5F9"),
                      yaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#E2E8F0"))
    return fig


# ─── PDF Builder ──────────────────────────────────────────────────────────────
def build_pdf(df_h: pd.DataFrame, df_movs: pd.DataFrame, df_r: pd.DataFrame, meta: Dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.2*cm, bottomMargin=2*cm,
                            title="NEIX — Informe de Portafolio")

    W = A4[0] - 4*cm  # usable width

    # ── Styles ──
    def sty(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=9, textColor=RL_INK, leading=14)
        defaults.update(kw)
        base = ParagraphStyle(name, **defaults)
        return base

    S_TITLE  = sty("title",  fontName="Helvetica-Bold", fontSize=18, textColor=RL_INK, leading=22, spaceAfter=4)
    S_H1     = sty("h1",     fontName="Helvetica-Bold", fontSize=11, textColor=RL_INK, leading=14, spaceBefore=14, spaceAfter=6)
    S_H2     = sty("h2",     fontName="Helvetica-Bold", fontSize=9,  textColor=RL_SUB, leading=12, spaceBefore=8, spaceAfter=4)
    S_BODY   = sty("body",   fontSize=8.5, textColor=RL_SUB, leading=13, spaceAfter=4)
    S_SMALL  = sty("small",  fontSize=7.5, textColor=RL_SUB, leading=11)
    S_NUM    = sty("num",    fontName="Courier", fontSize=8.5, textColor=RL_INK, alignment=TA_RIGHT)
    S_NUM_G  = sty("num_g",  fontName="Courier-Bold", fontSize=8.5, textColor=RL_GRN, alignment=TA_RIGHT)
    S_NUM_R  = sty("num_r",  fontName="Courier-Bold", fontSize=8.5, textColor=RL_RED, alignment=TA_RIGHT)
    S_TH     = sty("th",     fontName="Helvetica-Bold", fontSize=7, textColor=RL_SUB, alignment=TA_LEFT)
    S_THR    = sty("thr",    fontName="Helvetica-Bold", fontSize=7, textColor=RL_SUB, alignment=TA_RIGHT)

    def p(text, style=None): return Paragraph(text, style or S_BODY)
    def sp(n=6): return Spacer(1, n)
    def hr(): return HRFlowable(width="100%", thickness=0.5, color=RL_BDR, spaceAfter=8, spaceBefore=8)

    def num_para(v, bold=False):
        s = _ars(v)
        if v > 0: return Paragraph(s, S_NUM_G if bold else sty("ng", fontName="Courier", fontSize=8.5, textColor=RL_GRN, alignment=TA_RIGHT))
        if v < 0: return Paragraph(s, S_NUM_R if bold else sty("nr", fontName="Courier", fontSize=8.5, textColor=RL_RED, alignment=TA_RIGHT))
        return Paragraph(s, S_NUM)

    def _tbl_style(has_total=True):
        cmds = [
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 7),
            ("TEXTCOLOR", (0,0), (-1,0), RL_SUB),
            ("ALIGN", (0,0), (-1,0), "LEFT"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [RL_WHITE, RL_BG]),
            ("LINEBELOW", (0,0), (-1,0), 0.5, RL_BDR),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("FONTSIZE", (0,1), (-1,-1), 8),
            ("TEXTCOLOR", (0,1), (-1,-1), RL_INK),
            ("GRID", (0,0), (-1,-1), 0.25, RL_BDR),
        ]
        if has_total:
            cmds += [
                ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
                ("LINEABOVE", (0,-1), (-1,-1), 0.75, RL_INK),
                ("BACKGROUND", (0,-1), (-1,-1), RL_BG),
            ]
        return TableStyle(cmds)

    story = []

    # ── COVER / HEADER ──────────────────────────────────────────────────────
    # Red accent bar
    logo_tbl = Table(
        [[Paragraph('<font name="Helvetica-Bold" size="16" color="#C8102E">NEIX</font>', sty("logo", alignment=TA_LEFT)),
          Paragraph("Informe de Portafolio", sty("ir", fontSize=8, textColor=RL_SUB, alignment=TA_RIGHT))]],
        colWidths=[W*0.5, W*0.5]
    )
    logo_tbl.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBELOW",(0,0),(-1,0),2,RL_INK),
        ("BOTTOMPADDING",(0,0),(-1,0),8),
    ]))
    story.append(logo_tbl)
    story.append(sp(16))

    story.append(p("INFORME DE RENDIMIENTO DE PORTAFOLIO",
                   sty("cover_title", fontName="Helvetica-Bold", fontSize=20,
                       textColor=RL_INK, leading=24, spaceAfter=8)))

    info_rows = [
        ["Comitente", meta.get("comitente","—"),  "Período desde", meta.get("fecha_desde","—")],
        ["Usuario",   meta.get("usuario","—"),    "Período hasta",  meta.get("fecha_hasta","—")],
        ["Fecha reporte", datetime.now().strftime("%d/%m/%Y"), "Especies operadas", str(len(df_h))],
    ]
    info_tbl = Table(info_rows, colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"), ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8), ("TEXTCOLOR",(0,0),(0,-1),RL_SUB),
        ("TEXTCOLOR",(2,0),(2,-1),RL_SUB), ("TEXTCOLOR",(1,0),(-1,-1),RL_INK),
        ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LINEBELOW",(0,-1),(-1,-1),0.5,RL_BDR),
    ]))
    story.append(info_tbl)
    story.append(sp(20))

    # ── SECTION 1: P&L CONSOLIDADO ──────────────────────────────────────────
    story.append(p("1. RESULTADOS CONSOLIDADOS", S_H1))
    story.append(hr())

    pnl_r   = df_h["PnL ARS"].sum()
    cap     = df_h["Compras ARS"].sum()
    ventas  = df_h["Ventas ARS"].sum()
    divs    = df_h["Dividendos ARS"].sum()
    rp      = pnl_r/cap*100 if cap else 0.0
    pnl_nr  = df_r["Diferencia"].sum() if not df_r.empty else 0.0
    val_hoy = df_r["Valuacion"].sum()  if not df_r.empty else 0.0
    inv_ab  = df_r["Inversion"].sum()  if not df_r.empty else 0.0
    rp_ab   = pnl_nr/inv_ab*100 if inv_ab else 0.0
    pnl_tot = pnl_r + pnl_nr
    rp_tot  = pnl_tot/cap*100 if cap else 0.0
    n_pos   = int((df_h["PnL ARS"]>0).sum())
    n_neg   = int((df_h["PnL ARS"]<0).sum())

    kpi_data = [
        ["CONCEPTO", "VALOR", "", "CONCEPTO", "VALOR"],
        ["Capital total invertido", _ars(cap), "", "Total ventas realizadas", _ars(ventas)],
        ["P&L realizado (ganancias/pérdidas)", _ars(pnl_r), "", "Dividendos / rentas cobrados", _ars(divs)],
        ["Rendimiento s/ capital invertido", _pct(rp), "", "Especies ganadoras / perdedoras", "{}/{}".format(n_pos,n_neg)],
    ]
    if not df_r.empty:
        kpi_data += [
            ["Inversión posiciones abiertas", _ars(inv_ab), "", "Valuación a mercado (hoy)", _ars(val_hoy)],
            ["P&L no realizado (papel)", _ars(pnl_nr), "", "Rendimiento s/ abierto", _pct(rp_ab)],
        ]
    kpi_data.append(["P&L TOTAL COMBINADO", _ars(pnl_tot), "", "Rendimiento total s/ capital", _pct(rp_tot)])

    def _kpi_color(v_str):
        if v_str.startswith("-") and "$" in v_str: return RL_RED
        if "$" in v_str and not v_str.startswith("-"): return RL_GRN
        if "+" in v_str: return RL_GRN
        if v_str.startswith("-"): return RL_RED
        return RL_INK

    kpi_tbl_data = []
    for i, row in enumerate(kpi_data):
        if i == 0:
            kpi_tbl_data.append([
                Paragraph(row[0], sty("kh", fontName="Helvetica-Bold", fontSize=7, textColor=RL_SUB)),
                Paragraph(row[1], sty("kh", fontName="Helvetica-Bold", fontSize=7, textColor=RL_SUB, alignment=TA_RIGHT)),
                Paragraph("", S_SMALL),
                Paragraph(row[3], sty("kh", fontName="Helvetica-Bold", fontSize=7, textColor=RL_SUB)),
                Paragraph(row[4], sty("kh", fontName="Helvetica-Bold", fontSize=7, textColor=RL_SUB, alignment=TA_RIGHT)),
            ])
        else:
            c2 = _kpi_color(row[1]); c4 = _kpi_color(row[4])
            bold2 = "Helvetica-Bold" if i == len(kpi_data)-1 else "Courier"
            kpi_tbl_data.append([
                Paragraph(row[0], sty("kl", fontSize=8, textColor=RL_INK, fontName="Helvetica-Bold" if i==len(kpi_data)-1 else "Helvetica")),
                Paragraph(row[1], sty("kv", fontName=bold2, fontSize=8.5, textColor=c2, alignment=TA_RIGHT)),
                Paragraph("", S_SMALL),
                Paragraph(row[3], sty("kl2", fontSize=8, textColor=RL_INK)),
                Paragraph(row[4], sty("kv2", fontName="Courier", fontSize=8.5, textColor=c4, alignment=TA_RIGHT)),
            ])

    kpi_tbl = Table(kpi_tbl_data, colWidths=[W*0.38, W*0.16, W*0.02, W*0.30, W*0.14])
    kpi_tbl.setStyle(TableStyle([
        ("LINEBELOW",(0,0),(-1,0),0.5,RL_BDR),
        ("LINEABOVE",(0,-1),(-1,-1),0.75,RL_INK),
        ("BACKGROUND",(0,-1),(-1,-1),RL_BG),
        ("ROWBACKGROUNDS",(0,1),(-1,-2),[RL_WHITE, RL_BG]),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("GRID",(0,0),(-1,-1),0.25,RL_BDR),
        ("LINEAFTER",(1,0),(1,-1),0.25,RL_BDR),("LINEAFTER",(2,0),(2,-1),0,RL_WHITE),
    ]))
    story.append(kpi_tbl)
    story.append(sp(20))

    # ── SECTION 2: G/P REALIZADAS POR ESPECIE ────────────────────────────────
    story.append(p("2. GANANCIAS Y PÉRDIDAS REALIZADAS POR ESPECIE", S_H1))
    story.append(hr())
    story.append(p("Resultado neto de las operaciones cerradas en el período. Incluye todas las compras, ventas, dividendos y rentas.", S_BODY))
    story.append(sp(8))

    # by category
    for cat in sorted(df_h["Categoria"].unique()):
        dfc = df_h[df_h["Categoria"]==cat].sort_values("PnL ARS")
        story.append(p(cat.upper(), sty("cat_h", fontName="Helvetica-Bold", fontSize=8, textColor=RL_SUB,
                                        spaceBefore=8, spaceAfter=4)))
        rows_gp = [[
            Paragraph("Ticker", S_TH), Paragraph("Especie", S_TH),
            Paragraph("Qty Comprada", S_THR), Paragraph("Qty Vendida", S_THR),
            Paragraph("Compras ARS", S_THR), Paragraph("Ventas ARS", S_THR),
            Paragraph("Dividendos", S_THR), Paragraph("P&L Realizado", S_THR),
        ]]
        for _, r in dfc.iterrows():
            vc = RL_GRN if r["PnL ARS"]>0 else RL_RED
            rows_gp.append([
                Paragraph(str(r["Ticker"]), sty("t",fontName="Helvetica-Bold",fontSize=8,textColor=RL_INK)),
                Paragraph(str(r["Especie"])[:35], sty("e",fontSize=7.5,textColor=RL_SUB)),
                Paragraph("{:,.0f}".format(r["Qty Comprada"]), S_NUM),
                Paragraph("{:,.0f}".format(r["Qty Vendida"]), S_NUM),
                Paragraph(_ars(r["Compras ARS"]), S_NUM),
                Paragraph(_ars(r["Ventas ARS"]), S_NUM),
                Paragraph(_ars(r["Dividendos ARS"]), S_NUM),
                Paragraph(_ars(r["PnL ARS"]), sty("pnl",fontName="Courier-Bold",fontSize=8.5,textColor=vc,alignment=TA_RIGHT)),
            ])
        # subtotal
        rows_gp.append([
            Paragraph("SUBTOTAL", sty("sub",fontName="Helvetica-Bold",fontSize=7.5,textColor=RL_INK)), Paragraph(" ", S_SMALL),
            Paragraph(" ", S_SMALL), Paragraph(" ", S_SMALL),
            Paragraph(_ars(dfc["Compras ARS"].sum()), sty("sn",fontName="Courier-Bold",fontSize=8,textColor=RL_INK,alignment=TA_RIGHT)),
            Paragraph(_ars(dfc["Ventas ARS"].sum()), sty("sn2",fontName="Courier-Bold",fontSize=8,textColor=RL_INK,alignment=TA_RIGHT)),
            Paragraph(_ars(dfc["Dividendos ARS"].sum()), sty("sn3",fontName="Courier-Bold",fontSize=8,textColor=RL_INK,alignment=TA_RIGHT)),
            Paragraph(_ars(dfc["PnL ARS"].sum()), sty("spnl",fontName="Courier-Bold",fontSize=8.5,
                textColor=RL_GRN if dfc["PnL ARS"].sum()>=0 else RL_RED, alignment=TA_RIGHT)),
        ])
        cw = [W*0.07, W*0.24, W*0.09, W*0.09, W*0.14, W*0.14, W*0.11, W*0.12]
        t = Table(rows_gp, colWidths=cw, repeatRows=1)
        t.setStyle(_tbl_style(True))
        story.append(KeepTogether(t))
        story.append(sp(6))

    # Grand total
    story.append(sp(4))
    gt = Table([
        [Paragraph("TOTAL G/P REALIZADAS", sty("gt",fontName="Helvetica-Bold",fontSize=9,textColor=RL_INK)),
         Paragraph(_ars(pnl_r), sty("gv",fontName="Courier-Bold",fontSize=11,
             textColor=RL_GRN if pnl_r>=0 else RL_RED, alignment=TA_RIGHT))],
    ], colWidths=[W*0.6, W*0.4])
    gt.setStyle(TableStyle([
        ("LINEABOVE",(0,0),(-1,0),1.5,RL_INK), ("LINEBELOW",(0,0),(-1,0),0.5,RL_BDR),
        ("LEFTPADDING",(0,0),(-1,-1),4), ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("BACKGROUND",(0,0),(-1,-1),RL_BG),
    ]))
    story.append(gt)
    story.append(PageBreak())

    # ── SECTION 3: TENENCIA ACTUAL ────────────────────────────────────────────
    if not df_r.empty:
        story.append(p("3. TENENCIA ACTUAL — POSICIÓN ABIERTA", S_H1))
        story.append(hr())
        story.append(p("Posiciones vigentes valoradas a precio de mercado. Incluye diferencia respecto al precio promedio ponderado de compra.", S_BODY))
        story.append(sp(8))

        rows_t = [[
            Paragraph("Ticker", S_TH), Paragraph("Especie", S_TH),
            Paragraph("Categoría", S_TH), Paragraph("Cantidad", S_THR),
            Paragraph("PPP", S_THR), Paragraph("Inversión", S_THR),
            Paragraph("Precio Actual", S_THR), Paragraph("Valuación", S_THR),
            Paragraph("Diferencia", S_THR), Paragraph("Rend %", S_THR),
        ]]
        for _, r in df_r.sort_values("Diferencia").iterrows():
            vc = RL_GRN if r["Diferencia"]>=0 else RL_RED
            rows_t.append([
                Paragraph(str(r["Ticker"]), sty("tt",fontName="Helvetica-Bold",fontSize=8,textColor=RL_INK)),
                Paragraph(str(r["Especie"])[:30], sty("te",fontSize=7.5,textColor=RL_SUB)),
                Paragraph(str(r["Categoria"]), sty("tc",fontSize=7.5,textColor=RL_SUB)),
                Paragraph("{:,.0f}".format(r["Cantidad"]), S_NUM),
                Paragraph(_ars(r["PPP"],2), S_NUM),
                Paragraph(_ars(r["Inversion"]), S_NUM),
                Paragraph(_ars(r["Precio Actual"],2), S_NUM),
                Paragraph(_ars(r["Valuacion"]), sty("tv",fontName="Courier-Bold",fontSize=8.5,textColor=RL_BLU,alignment=TA_RIGHT)),
                Paragraph(_ars(r["Diferencia"]), sty("td",fontName="Courier-Bold",fontSize=8.5,textColor=vc,alignment=TA_RIGHT)),
                Paragraph(_pct(r["Rend %"]), sty("tr",fontName="Courier-Bold",fontSize=8,textColor=vc,alignment=TA_RIGHT)),
            ])
        rows_t.append([
            Paragraph("TOTAL", sty("tot",fontName="Helvetica-Bold",fontSize=8,textColor=RL_INK)),
            Paragraph(" ", S_SMALL), Paragraph(" ", S_SMALL), Paragraph(" ", S_SMALL), Paragraph(" ", S_SMALL),
            Paragraph(_ars(inv_ab), sty("tn",fontName="Courier-Bold",fontSize=8.5,textColor=RL_INK,alignment=TA_RIGHT)),
            Paragraph(" ", S_SMALL),
            Paragraph(_ars(val_hoy), sty("tv2",fontName="Courier-Bold",fontSize=8.5,textColor=RL_BLU,alignment=TA_RIGHT)),
            Paragraph(_ars(pnl_nr), sty("td2",fontName="Courier-Bold",fontSize=8.5,
                textColor=RL_GRN if pnl_nr>=0 else RL_RED, alignment=TA_RIGHT)),
            Paragraph(_pct(rp_ab), sty("tr2",fontName="Courier-Bold",fontSize=8,
                textColor=RL_GRN if rp_ab>=0 else RL_RED, alignment=TA_RIGHT)),
        ])
        cw = [W*0.07, W*0.20, W*0.08, W*0.07, W*0.10, W*0.12, W*0.10, W*0.11, W*0.10, W*0.05]
        t = Table(rows_t, colWidths=cw, repeatRows=1)
        t.setStyle(_tbl_style(True))
        story.append(t)
        story.append(PageBreak())

    # ── SECTION 4: MOVIMIENTOS ─────────────────────────────────────────────
    story.append(p("4. DETALLE DE MOVIMIENTOS", S_H1))
    story.append(hr())
    story.append(p("Registro completo de todas las operaciones realizadas en el período, ordenadas por especie.", S_BODY))
    story.append(sp(8))

    # Group by ticker
    for ticker in sorted(df_movs["Ticker"].unique()):
        dfm = df_movs[df_movs["Ticker"]==ticker]
        story.append(p(ticker + " — " + str(dfm.iloc[0]["Especie"])[:50],
                       sty("mh",fontName="Helvetica-Bold",fontSize=8,textColor=RL_INK,spaceBefore=10,spaceAfter=4)))
        rows_m = [[
            Paragraph("Comprobante", S_TH), Paragraph("Número", S_TH),
            Paragraph("F. Concertación", S_TH), Paragraph("F. Liquidación", S_TH),
            Paragraph("Cantidad", S_THR), Paragraph("Precio", S_THR),
            Paragraph("Neto ARS", S_THR), Paragraph("Moneda", S_TH), Paragraph("Neto USD", S_THR),
        ]]
        for _, r in dfm.iterrows():
            na = r["Neto ARS"]; nu = r["Neto USD"]
            rows_m.append([
                Paragraph(str(r["Comprobante"]), sty("mc",fontSize=7.5,textColor=RL_INK,fontName="Helvetica-Bold")),
                Paragraph(str(r["Numero"]), sty("mn",fontSize=7.5,textColor=RL_SUB)),
                Paragraph(str(r["Fecha Concertacion"]), sty("mf",fontSize=7.5,textColor=RL_SUB)),
                Paragraph(str(r["Fecha Liquidacion"]), sty("ml",fontSize=7.5,textColor=RL_SUB)),
                Paragraph("{:,.2f}".format(r["Cantidad"]) if r["Cantidad"]!=0 else "—", S_NUM),
                Paragraph("{:,.3f}".format(r["Precio"]) if r["Precio"]!=0 else "—", S_NUM),
                Paragraph(_ars(na), sty("mna",fontName="Courier",fontSize=7.5,
                    textColor=RL_GRN if na>0 else (RL_RED if na<0 else RL_SUB), alignment=TA_RIGHT)),
                Paragraph(str(r["Moneda"]), sty("mm",fontSize=7.5,textColor=RL_SUB)),
                Paragraph(_ars(nu,2) if nu!=0 else "—", sty("mnu",fontName="Courier",fontSize=7.5,
                    textColor=RL_GRN if nu>0 else (RL_RED if nu<0 else RL_SUB), alignment=TA_RIGHT)),
            ])
        cw = [W*0.11, W*0.09, W*0.11, W*0.11, W*0.09, W*0.10, W*0.16, W*0.10, W*0.13]
        t = Table(rows_m, colWidths=cw, repeatRows=1)
        t.setStyle(_tbl_style(False))
        story.append(KeepTogether(t))
        story.append(sp(4))

    # Footer note
    story.append(sp(16))
    story.append(hr())
    story.append(p("Este informe es generado automáticamente a partir de los archivos exportados de la plataforma de trading. "
                   "Los valores históricos corresponden a operaciones realizadas. La valuación de posiciones abiertas es indicativa "
                   "y está sujeta a variaciones de mercado. NEIX — " + datetime.now().strftime("%d/%m/%Y"), S_SMALL))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(RL_SUB)
        canvas.drawString(2*cm, 1.2*cm, "NEIX — Informe de Portafolio — Comitente {}".format(meta.get("comitente","—")))
        canvas.drawRightString(A4[0]-2*cm, 1.2*cm, "Página {} ".format(doc.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.getvalue()


# ─── Excel Builder ────────────────────────────────────────────────────────────
def build_excel(df_h: pd.DataFrame, df_movs: pd.DataFrame, df_r: pd.DataFrame, meta: Dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                                  numbers as xl_numbers)
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    # colors
    CL_INK  = "0F172A"; CL_RED  = "C8102E"; CL_GRN  = "059669"
    CL_SUB  = "64748B"; CL_BG   = "F8FAFC"; CL_BDR  = "E2E8F0"
    CL_BLU  = "2563EB"; CL_WHT  = "FFFFFF"
    CL_GNLG = "ECFDF5"; CL_RDLG = "FEF2F2"

    thin  = Side(style="thin", color=CL_BDR)
    med   = Side(style="medium", color=CL_INK)
    bdr   = Border(left=thin, right=thin, top=thin, bottom=thin)
    bdr_b = Border(left=thin, right=thin, top=med, bottom=med)

    def _hdr(ws, row, cols, bg=CL_INK, fg=CL_WHT, bold=True):
        for c, (val, w) in enumerate(cols, 1):
            cell = ws.cell(row=row, column=c, value=val)
            cell.font = Font(name="Calibri", bold=bold, color=fg, size=9)
            cell.fill = PatternFill("solid", fgColor=bg)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = bdr
            ws.column_dimensions[get_column_letter(c)].width = w

    def _val_fmt(cell, v, fmt="$#,##0", bold=False):
        cell.value = v
        cell.number_format = fmt
        cell.font = Font(name="Calibri", size=9, bold=bold,
                         color=CL_GRN if v > 0 else (CL_RED if v < 0 else CL_INK))
        cell.alignment = Alignment(horizontal="right")
        cell.border = bdr

    def _txt(ws, r, c, val, bold=False, color=CL_INK, align="left"):
        cell = ws.cell(row=r, column=c, value=val)
        cell.font = Font(name="Calibri", size=9, bold=bold, color=color)
        cell.alignment = Alignment(horizontal=align, vertical="center")
        cell.border = bdr
        return cell

    # ── Sheet 1: Resumen consolidado ──────────────────────────────────────────
    ws1 = wb.active; ws1.title = "Resumen"
    ws1.row_dimensions[1].height = 40
    ws1.merge_cells("A1:F1")
    c = ws1["A1"]; c.value = "NEIX — INFORME DE PORTAFOLIO"
    c.font = Font(name="Calibri", bold=True, size=16, color=CL_WHT)
    c.fill = PatternFill("solid", fgColor=CL_INK)
    c.alignment = Alignment(horizontal="left", vertical="center")

    # Meta info
    ws1.row_dimensions[2].height = 18
    for col, (k, v) in enumerate([("Comitente", meta.get("comitente","—")),
                                   ("Usuario",   meta.get("usuario","—")),
                                   ("Desde",     meta.get("fecha_desde","—")),
                                   ("Hasta",     meta.get("fecha_hasta","—")),
                                   ("Generado",  datetime.now().strftime("%d/%m/%Y %H:%M")),
                                   ("Especies",  str(len(df_h)))], 1):
        ws1.cell(row=2, column=col).value = k + ": " + str(v)
        ws1.cell(row=2, column=col).font = Font(name="Calibri", size=8, color=CL_SUB, italic=True)
        ws1.column_dimensions[get_column_letter(col)].width = 22

    pnl_r  = df_h["PnL ARS"].sum(); cap = df_h["Compras ARS"].sum()
    divs   = df_h["Dividendos ARS"].sum(); ventas = df_h["Ventas ARS"].sum()
    rp     = pnl_r/cap*100 if cap else 0.0
    pnl_nr = df_r["Diferencia"].sum() if not df_r.empty else 0.0
    val_hoy= df_r["Valuacion"].sum()  if not df_r.empty else 0.0
    inv_ab = df_r["Inversion"].sum()  if not df_r.empty else 0.0
    rp_ab  = pnl_nr/inv_ab*100 if inv_ab else 0.0
    pnl_tot= pnl_r+pnl_nr; rp_tot = pnl_tot/cap*100 if cap else 0.0

    row = 4
    ws1.row_dimensions[row].height = 16
    for c2, h in enumerate(["Concepto","Valor ARS","","Concepto","Valor"],1):
        cell = ws1.cell(row=row, column=c2, value=h)
        cell.font = Font(name="Calibri", bold=True, size=8, color=CL_WHT)
        cell.fill = PatternFill("solid", fgColor=CL_INK)
        cell.border = bdr; cell.alignment = Alignment(horizontal="center", vertical="center")

    kpis = [
        ("Capital total invertido", cap, "Total ventas realizadas", ventas),
        ("P&L realizado", pnl_r, "Dividendos / rentas cobrados", divs),
        ("Rendimiento s/ capital", rp/100, "Especies ganadoras / perdedoras",
         "{}/{}".format(int((df_h["PnL ARS"]>0).sum()), int((df_h["PnL ARS"]<0).sum()))),
    ]
    if not df_r.empty:
        kpis += [
            ("Inversión posiciones abiertas", inv_ab, "Valuación a mercado (hoy)", val_hoy),
            ("P&L no realizado (papel)", pnl_nr, "Rendimiento s/ abierto", rp_ab/100),
        ]
    kpis.append(("P&L TOTAL COMBINADO", pnl_tot, "Rendimiento total s/ capital", rp_tot/100))

    for i, (k1,v1,k2,v2) in enumerate(kpis):
        r = row+1+i
        ws1.row_dimensions[r].height = 15
        is_last = (i == len(kpis)-1)
        _txt(ws1,r,1,k1,bold=is_last)
        cell = ws1.cell(row=r, column=2)
        if isinstance(v1, float):
            fmt = "0.00%" if abs(v1)<1 and "endimiento" in k1 else '#,##0'
            _val_fmt(cell, v1, fmt, bold=is_last)
        else:
            _txt(ws1,r,2,v1)
        ws1.cell(row=r,column=3).border = bdr
        _txt(ws1,r,4,k2,bold=is_last)
        cell = ws1.cell(row=r, column=5)
        if isinstance(v2, float):
            fmt = "0.00%" if abs(v2)<1 and "endimiento" in k2 else '#,##0'
            _val_fmt(cell, v2, fmt, bold=is_last)
        else:
            _txt(ws1,r,5,str(v2))

    # ── Sheet 2: G/P Realizadas ───────────────────────────────────────────────
    ws2 = wb.create_sheet("GP Realizadas")
    ws2.row_dimensions[1].height = 22
    ws2.merge_cells("A1:I1")
    c = ws2["A1"]; c.value = "GANANCIAS Y PÉRDIDAS REALIZADAS POR ESPECIE"
    c.font = Font(name="Calibri", bold=True, size=12, color=CL_WHT)
    c.fill = PatternFill("solid", fgColor=CL_INK)
    c.alignment = Alignment(horizontal="left", vertical="center")

    cols_gp = [("Ticker",10),("Especie",32),("Categoría",12),("Qty Comprada",12),
               ("Qty Vendida",12),("Saldo",10),("Compras ARS",16),
               ("Ventas ARS",16),("Dividendos ARS",14),("P&L ARS",16),("P&L USD",12)]
    _hdr(ws2, 2, cols_gp)
    for i, (_, r) in enumerate(df_h.sort_values(["Categoria","PnL ARS"]).iterrows()):
        row = 3 + i
        bg = CL_WHT if i%2==0 else CL_BG
        fill = PatternFill("solid", fgColor=bg)
        for c2, (val, fmt) in enumerate([
            (r["Ticker"],None),(r["Especie"],None),(r["Categoria"],None),
            (r["Qty Comprada"],"#,##0"),(r["Qty Vendida"],"#,##0"),(r["Saldo"],"#,##0"),
            (r["Compras ARS"],"#,##0"),(r["Ventas ARS"],"#,##0"),
            (r["Dividendos ARS"],"#,##0"),(r["PnL ARS"],"#,##0"),(r["PnL USD"],"#,##0.00"),
        ], 1):
            cell = ws2.cell(row=row, column=c2, value=val)
            cell.font = Font(name="Calibri", size=9,
                             color=CL_GRN if isinstance(val,float) and val>0 and c2>=7
                             else (CL_RED if isinstance(val,float) and val<0 and c2>=7 else CL_INK),
                             bold=(c2==1))
            cell.fill = fill
            cell.border = bdr
            if fmt: cell.number_format = fmt
            cell.alignment = Alignment(horizontal="right" if c2>=4 else "left")
    # Totals row
    tr = 3 + len(df_h)
    ws2.row_dimensions[tr].height = 16
    for c2 in range(1,12):
        cell = ws2.cell(row=tr, column=c2)
        cell.fill = PatternFill("solid", fgColor=CL_INK)
        cell.font = Font(name="Calibri", bold=True, size=9, color=CL_WHT)
        cell.border = bdr_b
        cell.alignment = Alignment(horizontal="right")
    ws2.cell(row=tr,column=1).value = "TOTAL"
    ws2.cell(row=tr,column=1).alignment = Alignment(horizontal="left")
    for c2,col in enumerate(["Qty Comprada","Qty Vendida","Saldo","Compras ARS","Ventas ARS","Dividendos ARS","PnL ARS","PnL USD"],4):
        ws2.cell(row=tr, column=c2).value = df_h[col].sum()
        ws2.cell(row=tr, column=c2).number_format = "#,##0" if "USD" not in col else "#,##0.00"

    # ── Sheet 3: Tenencia actual ──────────────────────────────────────────────
    if not df_r.empty:
        ws3 = wb.create_sheet("Tenencia Actual")
        ws3.row_dimensions[1].height = 22
        ws3.merge_cells("A1:J1")
        c = ws3["A1"]; c.value = "TENENCIA ACTUAL — POSICIÓN ABIERTA"
        c.font = Font(name="Calibri", bold=True, size=12, color=CL_WHT)
        c.fill = PatternFill("solid", fgColor=CL_INK)
        c.alignment = Alignment(horizontal="left", vertical="center")
        cols_t = [("Ticker",10),("Especie",32),("Categoría",12),("Cantidad",12),
                  ("PPP",14),("Inversión",16),("Precio Actual",14),
                  ("Valuación",16),("Diferencia",14),("Rend %",10)]
        _hdr(ws3, 2, cols_t)
        for i, (_, r) in enumerate(df_r.sort_values("Diferencia").iterrows()):
            row = 3+i
            bg = CL_WHT if i%2==0 else CL_BG
            fill = PatternFill("solid", fgColor=bg)
            dif = r["Diferencia"]
            for c2, (val, fmt) in enumerate([
                (r["Ticker"],None),(r["Especie"],None),(r["Categoria"],None),
                (r["Cantidad"],"#,##0"),(r["PPP"],"#,##0.000"),
                (r["Inversion"],"#,##0"),(r["Precio Actual"],"#,##0.000"),
                (r["Valuacion"],"#,##0"),(dif,"#,##0"),(r["Rend %"],"0.00%"),
            ], 1):
                cell = ws3.cell(row=row, column=c2, value=val)
                is_num_col = c2 >= 4
                is_pnl = c2 in {8,9,10}
                cell.font = Font(name="Calibri", size=9, bold=(c2==1),
                                 color=(CL_GRN if isinstance(val,(int,float)) and val>0 and is_pnl
                                        else CL_RED if isinstance(val,(int,float)) and val<0 and is_pnl
                                        else CL_BLU if c2==8 else CL_INK))
                cell.fill = fill; cell.border = bdr
                if fmt:
                    actual_fmt = fmt
                    if c2==10: actual_fmt="0.00%"
                    cell.number_format = actual_fmt
                    if c2==10 and isinstance(val,float): cell.value = val/100
                cell.alignment = Alignment(horizontal="right" if is_num_col else "left")
        # Total
        tr = 3+len(df_r)
        for c2 in range(1,11):
            cell = ws3.cell(row=tr, column=c2)
            cell.fill = PatternFill("solid", fgColor=CL_INK)
            cell.font = Font(name="Calibri", bold=True, size=9, color=CL_WHT)
            cell.border = bdr_b; cell.alignment = Alignment(horizontal="right")
        ws3.cell(row=tr,column=1).value="TOTAL"; ws3.cell(row=tr,column=1).alignment=Alignment(horizontal="left")
        ws3.cell(row=tr,column=6).value=inv_ab; ws3.cell(row=tr,column=6).number_format="#,##0"
        ws3.cell(row=tr,column=8).value=val_hoy; ws3.cell(row=tr,column=8).number_format="#,##0"
        ws3.cell(row=tr,column=9).value=pnl_nr; ws3.cell(row=tr,column=9).number_format="#,##0"
        ws3.cell(row=tr,column=10).value=rp_ab/100; ws3.cell(row=tr,column=10).number_format="0.00%"

    # ── Sheet 4: Movimientos ──────────────────────────────────────────────────
    ws4 = wb.create_sheet("Movimientos")
    ws4.row_dimensions[1].height = 22
    ws4.merge_cells("A1:I1")
    c = ws4["A1"]; c.value = "DETALLE DE MOVIMIENTOS"
    c.font = Font(name="Calibri", bold=True, size=12, color=CL_WHT)
    c.fill = PatternFill("solid", fgColor=CL_INK)
    c.alignment = Alignment(horizontal="left", vertical="center")
    cols_mv = [("Ticker",10),("Especie",30),("Comprobante",14),("Número",10),
               ("F. Concertación",14),("F. Liquidación",14),
               ("Cantidad",12),("Precio",14),("Neto ARS",16),("Moneda",12),("Neto USD",14)]
    _hdr(ws4, 2, cols_mv)
    for i, (_, r) in enumerate(df_movs.iterrows()):
        row = 3+i; bg = CL_WHT if i%2==0 else CL_BG
        fill = PatternFill("solid", fgColor=bg)
        for c2, (val, fmt) in enumerate([
            (r["Ticker"],None),(r["Especie"],None),(r["Comprobante"],None),(r["Numero"],None),
            (r["Fecha Concertacion"],None),(r["Fecha Liquidacion"],None),
            (r["Cantidad"],"#,##0.##"),(r["Precio"],"#,##0.000"),
            (r["Neto ARS"],"#,##0"),(r["Moneda"],None),(r["Neto USD"],"#,##0.00"),
        ], 1):
            cell = ws4.cell(row=row, column=c2, value=val)
            is_pnl = c2 in {9,11}
            cell.font = Font(name="Calibri", size=8.5, bold=(c2==1),
                             color=(CL_GRN if isinstance(val,(int,float)) and val>0 and is_pnl
                                    else CL_RED if isinstance(val,(int,float)) and val<0 and is_pnl
                                    else CL_INK))
            cell.fill = fill; cell.border = bdr
            if fmt: cell.number_format = fmt
            cell.alignment = Alignment(horizontal="right" if c2>=7 else "left")

    # Freeze panes and filters
    for ws in [ws2, ws4]:
        ws.freeze_panes = "A3"
        ws.auto_filter.ref = ws.dimensions
    if not df_r.empty:
        ws3.freeze_panes = "A3"
        ws3.auto_filter.ref = ws3.dimensions

    out = BytesIO(); wb.save(out); out.seek(0)
    return out.getvalue()


# ─── Streamlit App ────────────────────────────────────────────────────────────
def main():
    _write_config()
    st.set_page_config(page_title="NEIX · Portfolio", page_icon="📊",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    # Topbar
    st.markdown(
        '<div class="topbar">'
        '<div class="topbar-left">' + LOGO_HTML +
        '<div><div class="topbar-name">Rendimiento de Portafolio</div>'
        '<div class="topbar-sub">P&amp;L realizado &middot; tenencia actual &middot; movimientos &middot; exportación PDF &amp; Excel</div>'
        '</div></div>'
        '<div class="topbar-right">Acciones &middot; CEDEARs &middot; Bonos<br/>LECAPs &middot; Fondos &middot; Cauciones</div>'
        '</div>', unsafe_allow_html=True)

    # Uploaders
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="upload-label">Histórico por Especie</div>', unsafe_allow_html=True)
        up_h = st.file_uploader("h", type=["xls","xlsx"], label_visibility="collapsed", key="uh")
        st.markdown('<div class="upload-hint">Ganancias y pérdidas realizadas &middot; movimientos históricos</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="upload-label">Resultados / Tenencia Actual <span style="font-weight:400;text-transform:none;letter-spacing:0">(opcional)</span></div>', unsafe_allow_html=True)
        up_r = st.file_uploader("r", type=["xls","xlsx"], label_visibility="collapsed", key="ur")
        st.markdown('<div class="upload-hint">Posición abierta valuada a precio actual</div>', unsafe_allow_html=True)

    if not up_h:
        st.markdown(
            '<div class="empty-wrap"><div class="empty-title">Subí el Histórico por Especie para comenzar</div>'
            '<div class="empty-sub">El archivo de Resultados es opcional y agrega la valuación de posiciones abiertas al informe</div></div>',
            unsafe_allow_html=True)
        return

    # Parse
    try:
        df_h, df_movs, meta = parse_historico(up_h.getvalue())
    except Exception as e:
        st.error("No se pudo leer el Histórico: " + str(e)); return

    df_r = pd.DataFrame()
    if up_r:
        try: df_r = parse_resultados(up_r.getvalue())
        except Exception as e: st.warning("No se pudo leer Resultados: " + str(e))

    if df_h.empty:
        st.warning("El archivo no contiene datos reconocibles."); return

    # Meta
    open_chip = ('<span class="meta-chip">&#128994; ' + str(len(df_r)) + ' posiciones abiertas</span>') if not df_r.empty else ""
    st.markdown(
        '<div class="meta-row">'
        '<span class="meta-chip">&#128100; ' + meta.get("usuario","—") + '</span>'
        '<span class="meta-chip">Comitente ' + meta.get("comitente","—") + '</span>'
        '<span class="meta-chip">' + meta.get("fecha_desde","—") + ' &#8594; ' + meta.get("fecha_hasta","hoy") + '</span>'
        '<span class="meta-chip">' + str(len(df_h)) + ' especies operadas</span>'
        '<span class="meta-chip">' + str(len(df_movs)) + ' movimientos</span>'
        + open_chip + '</div>', unsafe_allow_html=True)

    # Numbers
    pnl_r   = df_h["PnL ARS"].sum(); cap = df_h["Compras ARS"].sum()
    divs    = df_h["Dividendos ARS"].sum(); rp = pnl_r/cap*100 if cap else 0.0
    pnl_nr  = df_r["Diferencia"].sum()  if not df_r.empty else 0.0
    val_hoy = df_r["Valuacion"].sum()   if not df_r.empty else 0.0
    inv_ab  = df_r["Inversion"].sum()   if not df_r.empty else 0.0
    rp_ab   = pnl_nr/inv_ab*100         if inv_ab else 0.0
    pnl_tot = pnl_r + pnl_nr
    rp_tot  = pnl_tot/cap*100           if cap else 0.0
    n_pos   = int((df_h["PnL ARS"]>0).sum()); n_neg = int((df_h["PnL ARS"]<0).sum())

    # Bridge
    def _bc(border_cls, val_cls, label, val_str, sub=""):
        return ('<div class="bc ' + border_cls + '">'
                '<div class="bc-lbl">' + label + '</div>'
                '<div class="bc-val ' + val_cls + '">' + val_str + '</div>'
                + ('<div class="bc-sub">' + sub + '</div>' if sub else '') + '</div>')

    no_r = _bc("red" if pnl_nr<0 else "grn", _cls(pnl_nr), "P&amp;L no realizado",
               _ars(pnl_nr), _pct(rp_ab)+" s/abierto") if not df_r.empty else _bc("ink","c-ink","P&amp;L no realizado","—","sin datos")
    val_b= _bc("blu","c-blu","Valuación hoy", _ars(val_hoy), str(len(df_r))+" pos.") if not df_r.empty else _bc("ink","c-ink","Valuación hoy","—","sin datos")

    st.markdown(
        '<div class="bridge">'
        + _bc("ink","c-ink","Capital invertido",_ars(cap),str(len(df_h))+" especies")
        + _bc("red" if pnl_r<0 else "grn", _cls(pnl_r),"P&amp;L realizado",_ars(pnl_r),_pct(rp)+" s/capital")
        + _bc("grn" if divs>=0 else "red", _cls(divs),"Dividendos / carry",_ars(divs),"cobrado")
        + no_r + val_b
        + _bc("red" if pnl_tot<0 else "grn", _cls(pnl_tot),"P&amp;L total",_ars(pnl_tot),_pct(rp_tot)+" combinado")
        + '</div>', unsafe_allow_html=True)

    # KPIs
    nr_v = _ars(pnl_nr) if not df_r.empty else "—"
    nr_c = _col(pnl_nr) if not df_r.empty else SUB
    st.markdown(
        '<div class="krow">'
        '<div class="kpi"><div class="kpi-lbl">P&amp;L realizado</div>'
        '<div class="kpi-val" style="color:'+_col(pnl_r)+'">'+_ars(pnl_r)+'</div>'
        '<div class="kpi-sub" style="color:'+_col(rp)+'">'+_pct(rp)+' s/capital</div></div>'

        '<div class="kpi"><div class="kpi-lbl">P&amp;L no realizado</div>'
        '<div class="kpi-val" style="color:'+nr_c+'">'+nr_v+'</div>'
        '<div class="kpi-sub" style="color:'+SUB+'">'+(_pct(rp_ab) if not df_r.empty else "sin datos")+'</div></div>'

        '<div class="kpi"><div class="kpi-lbl">Valuación actual</div>'
        '<div class="kpi-val" style="color:'+BLU+'">'+('' if df_r.empty else _ars(val_hoy))+'</div>'
        '<div class="kpi-sub" style="color:'+SUB+'">'+str(len(df_r))+' posiciones</div></div>'

        '<div class="kpi"><div class="kpi-lbl">P&amp;L total</div>'
        '<div class="kpi-val" style="color:'+_col(pnl_tot)+'">'+_ars(pnl_tot)+'</div>'
        '<div class="kpi-sub" style="color:'+_col(rp_tot)+'">'+_pct(rp_tot)+'</div></div>'

        '<div class="kpi"><div class="kpi-lbl">Dividendos / carry</div>'
        '<div class="kpi-val" style="color:'+_col(divs)+'">'+_ars(divs)+'</div>'
        '<div class="kpi-sub" style="color:'+SUB+'">cobrado en el período</div></div>'

        '<div class="kpi"><div class="kpi-lbl">Win rate</div>'
        '<div class="kpi-val">'+str(n_pos)+'/'+str(len(df_h))+'</div>'
        '<div class="kpi-sub" style="color:'+R+'">'+str(n_neg)+' en negativo</div></div>'
        '</div>', unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["Resumen", "G/P Realizadas", "Tenencia actual", "Movimientos", "Gráficos"])

    # ── TAB 0: Resumen ──
    with tabs[0]:
        col_l, col_r = st.columns([1, 1], gap="large")
        with col_l:
            st.markdown('<div class="slbl">Cuadro de resultados</div>', unsafe_allow_html=True)
            rows = [
                ("Capital total invertido (ARS)", _ars(cap), INK),
                ("Total ventas realizadas (ARS)", _ars(df_h["Ventas ARS"].sum()), INK),
                ("Dividendos / rentas cobradas", _ars(divs), GRN),
                ("P&amp;L realizado", _ars(pnl_r), _col(pnl_r)),
            ]
            if not df_r.empty:
                rows += [
                    ("Inversión posiciones abiertas", _ars(inv_ab), INK),
                    ("Valuación a mercado hoy", _ars(val_hoy), BLU),
                    ("P&amp;L no realizado (papel)", _ars(pnl_nr), _col(pnl_nr)),
                ]
            rows += [
                ("P&amp;L TOTAL COMBINADO", _ars(pnl_tot), _col(pnl_tot)),
                ("Rendimiento total s/ capital", _pct(rp_tot), _col(rp_tot)),
            ]
            trs = "".join('<tr><td style="color:'+SUB+'">'+k+'</td><td class="r" style="color:'+c+'">'+v+'</td></tr>'
                          for k,v,c in rows)
            st.markdown('<div class="section-card"><table class="tbl">'+trs+'</table></div>', unsafe_allow_html=True)

            # Category summary
            st.markdown('<div class="slbl" style="margin-top:1.2rem">Por categoría</div>', unsafe_allow_html=True)
            cat = df_h.groupby("Categoria").agg(N=("Ticker","count"), Capital=("Compras ARS","sum"),
                Ventas=("Ventas ARS","sum"), Divs=("Dividendos ARS","sum"), PnL=("PnL ARS","sum")).reset_index()
            hdr = '<tr><th>Categoría</th><th class="r">N</th><th class="r">Capital</th><th class="r">Ventas</th><th class="r">Dividendos</th><th class="r">P&amp;L ARS</th></tr>'
            trs = "".join(
                '<tr><td><strong>'+str(r["Categoria"])+'</strong></td>'
                '<td class="r">'+str(int(r["N"]))+'</td>'
                '<td class="r">'+_ars(r["Capital"])+'</td>'
                '<td class="r">'+_ars(r["Ventas"])+'</td>'
                '<td class="r" style="color:'+GRN+'">'+_ars(r["Divs"])+'</td>'
                '<td class="r">'+_badge(r["PnL"])+'</td></tr>'
                for _,r in cat.sort_values("PnL").iterrows()
            )
            trs += '<tr class="tot"><td>TOTAL</td><td class="r">'+str(len(df_h))+'</td>'
            trs += '<td class="r">'+_ars(cap)+'</td><td class="r">'+_ars(df_h["Ventas ARS"].sum())+'</td>'
            trs += '<td class="r" style="color:'+GRN+'">'+_ars(divs)+'</td>'
            trs += '<td class="r">'+_badge(pnl_r)+'</td></tr>'
            st.markdown('<div class="section-card"><table class="tbl">'+hdr+trs+'</table></div>', unsafe_allow_html=True)

        with col_r:
            if not df_r.empty:
                st.plotly_chart(_waterfall(df_r), use_container_width=True)
            st.plotly_chart(_hbar(df_h.sort_values("PnL ARS"), "PnL ARS","Ticker",
                                  "P&L realizado por especie"), use_container_width=True)

    # ── TAB 1: G/P Realizadas ──
    with tabs[1]:
        fa, fb, fc = st.columns([2,1,1], gap="medium")
        with fa:
            cats = sorted(df_h["Categoria"].unique())
            sel = st.multiselect("Categoría", cats, default=cats, key="gpc")
        with fb: solo_neg = st.toggle("Solo perdedoras", key="gpn")
        with fc: solo_pos = st.toggle("Solo ganadoras", key="gpp")
        dh = df_h[df_h["Categoria"].isin(sel)]
        if solo_neg: dh = dh[dh["PnL ARS"]<0]
        if solo_pos: dh = dh[dh["PnL ARS"]>0]

        hdr = '<tr><th>Ticker</th><th>Especie</th><th>Cat.</th><th class="r">Compradas</th><th class="r">Vendidas</th><th class="r">Saldo</th><th class="r">Compras ARS</th><th class="r">Ventas ARS</th><th class="r">Dividendos</th><th class="r">P&amp;L ARS</th><th class="r">P&amp;L USD</th></tr>'
        trs = "".join(
            '<tr>'
            '<td><strong>'+str(r["Ticker"])+'</strong></td>'
            '<td style="font-size:0.8rem;color:'+SUB+'">'+str(r["Especie"])[:35]+'</td>'
            '<td style="font-size:0.75rem;color:'+SUB+'">'+str(r["Categoria"])+'</td>'
            '<td class="r">'+"{:,.0f}".format(r["Qty Comprada"])+'</td>'
            '<td class="r">'+"{:,.0f}".format(r["Qty Vendida"])+'</td>'
            '<td class="r">'+"{:,.0f}".format(r["Saldo"])+'</td>'
            '<td class="r">'+_ars(r["Compras ARS"])+'</td>'
            '<td class="r">'+_ars(r["Ventas ARS"])+'</td>'
            '<td class="r" style="color:'+GRN+'">'+_ars(r["Dividendos ARS"])+'</td>'
            '<td class="r">'+_badge(r["PnL ARS"])+'</td>'
            '<td class="r" style="font-family:monospace;font-size:0.78rem;color:'+_col(r["PnL USD"])+'">'+_ars(r["PnL USD"],2)+'</td>'
            '</tr>'
            for _,r in dh.sort_values("PnL ARS").iterrows()
        )
        trs += ('<tr class="tot"><td colspan="6">TOTAL</td>'
                '<td class="r">'+_ars(dh["Compras ARS"].sum())+'</td>'
                '<td class="r">'+_ars(dh["Ventas ARS"].sum())+'</td>'
                '<td class="r" style="color:'+GRN+'">'+_ars(dh["Dividendos ARS"].sum())+'</td>'
                '<td class="r">'+_badge(dh["PnL ARS"].sum())+'</td>'
                '<td class="r">'+_ars(dh["PnL USD"].sum(),2)+'</td></tr>')
        st.markdown('<div class="section-card" style="overflow-x:auto"><table class="tbl">'+hdr+trs+'</table></div>', unsafe_allow_html=True)

    # ── TAB 2: Tenencia actual ──
    with tabs[2]:
        if df_r.empty:
            st.markdown('<div class="empty-wrap"><div class="empty-title">Subí el archivo de Resultados</div><div class="empty-sub">Para ver la posición abierta valorizada a precio de mercado</div></div>', unsafe_allow_html=True)
        else:
            hdr = '<tr><th>Ticker</th><th>Especie</th><th>Cat.</th><th class="r">Cantidad</th><th class="r">PPP</th><th class="r">Inversión</th><th class="r">Precio Actual</th><th class="r">Valuación</th><th class="r">Diferencia</th><th class="r">Rend %</th></tr>'
            trs = "".join(
                '<tr>'
                '<td><strong>'+str(r["Ticker"])+'</strong></td>'
                '<td style="font-size:0.8rem;color:'+SUB+'">'+str(r["Especie"])[:30]+'</td>'
                '<td style="font-size:0.75rem;color:'+SUB+'">'+str(r["Categoria"])+'</td>'
                '<td class="r">'+"{:,.0f}".format(r["Cantidad"])+'</td>'
                '<td class="r" style="font-size:0.78rem;font-family:monospace">'+_ars(r["PPP"],2)+'</td>'
                '<td class="r">'+_ars(r["Inversion"])+'</td>'
                '<td class="r" style="font-size:0.78rem;font-family:monospace">'+_ars(r["Precio Actual"],2)+'</td>'
                '<td class="r" style="color:'+BLU+'">'+_ars(r["Valuacion"])+'</td>'
                '<td class="r">'+_badge(r["Diferencia"])+'</td>'
                '<td class="r" style="color:'+_col(r["Rend %"])+'">'+_pct(r["Rend %"])+'</td>'
                '</tr>'
                for _,r in df_r.sort_values("Diferencia").iterrows()
            )
            trs += ('<tr class="tot"><td colspan="5">TOTAL</td>'
                    '<td class="r">'+_ars(inv_ab)+'</td><td></td>'
                    '<td class="r" style="color:'+BLU+'">'+_ars(val_hoy)+'</td>'
                    '<td class="r">'+_badge(pnl_nr)+'</td>'
                    '<td class="r" style="color:'+_col(rp_ab)+'">'+_pct(rp_ab)+'</td></tr>')
            st.markdown('<div class="section-card" style="overflow-x:auto"><table class="tbl">'+hdr+trs+'</table></div>', unsafe_allow_html=True)

    # ── TAB 3: Movimientos ──
    with tabs[3]:
        fa, fb = st.columns([2,1], gap="medium")
        with fa: tf = st.selectbox("Especie", ["Todas"] + sorted(df_h["Ticker"].unique()), key="mvt")
        with fb: cpbt_f = st.selectbox("Comprobante", ["Todos"] + sorted(df_movs["Comprobante"].unique()), key="mvc")
        mv = df_movs.copy()
        if tf != "Todas": mv = mv[mv["Ticker"]==tf]
        if cpbt_f != "Todos": mv = mv[mv["Comprobante"]==cpbt_f]

        hdr = '<tr><th>Ticker</th><th>Comprobante</th><th>Número</th><th>F. Concertación</th><th>F. Liquidación</th><th class="r">Cantidad</th><th class="r">Precio</th><th class="r">Neto ARS</th><th>Moneda</th><th class="r">Neto USD</th></tr>'
        trs = "".join(
            '<tr>'
            '<td><strong>'+str(r["Ticker"])+'</strong></td>'
            '<td style="font-size:0.78rem">'+str(r["Comprobante"])+'</td>'
            '<td style="font-size:0.78rem;color:'+SUB+'">'+str(r["Numero"])+'</td>'
            '<td style="font-size:0.78rem;color:'+SUB+'">'+str(r["Fecha Concertacion"])+'</td>'
            '<td style="font-size:0.78rem;color:'+SUB+'">'+str(r["Fecha Liquidacion"])+'</td>'
            '<td class="r">'+("{:,.2f}".format(r["Cantidad"]) if r["Cantidad"]!=0 else "—")+'</td>'
            '<td class="r" style="font-family:monospace;font-size:0.78rem">'+("{:,.3f}".format(r["Precio"]) if r["Precio"]!=0 else "—")+'</td>'
            '<td class="r" style="color:'+_col(r["Neto ARS"])+'">'+_ars(r["Neto ARS"])+'</td>'
            '<td style="font-size:0.75rem;color:'+SUB+'">'+str(r["Moneda"])+'</td>'
            '<td class="r" style="color:'+_col(r["Neto USD"])+'">'+(_ars(r["Neto USD"],2) if r["Neto USD"]!=0 else "—")+'</td>'
            '</tr>'
            for _,r in mv.iterrows()
        )
        st.markdown('<div class="section-card" style="overflow-x:auto"><table class="tbl">'+hdr+trs+'</table></div>', unsafe_allow_html=True)
        st.caption(str(len(mv)) + " movimientos")

    # ── TAB 4: Gráficos ──
    with tabs[4]:
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.plotly_chart(_hbar(df_h.sort_values("PnL ARS"), "PnL ARS","Ticker","P&L realizado por especie"), use_container_width=True)
        with g2:
            cat_agg = df_h.groupby("Categoria")["PnL ARS"].sum().reset_index().sort_values("PnL ARS")
            st.plotly_chart(_hbar(cat_agg,"PnL ARS","Categoria","P&L por categoría",h=260), use_container_width=True)
        if not df_r.empty:
            g3, g4 = st.columns(2, gap="large")
            with g3: st.plotly_chart(_waterfall(df_r), use_container_width=True)
            with g4: st.plotly_chart(_combined(df_h, df_r), use_container_width=True)

    # Exports
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="slbl">Exportar informe</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns([0.2, 0.2, 0.6], gap="medium")

    with e1:
        try:
            pdf_bytes = build_pdf(df_h, df_movs, df_r, meta)
            st.download_button("↓  Descargar PDF", data=pdf_bytes,
                               file_name="neix_portafolio_{}.pdf".format(meta.get("comitente","").replace(" ","_")),
                               mime="application/pdf")
        except Exception as ex:
            st.error("Error generando PDF: " + str(ex))
    with e2:
        try:
            xl_bytes = build_excel(df_h, df_movs, df_r, meta)
            st.download_button("↓  Descargar Excel", data=xl_bytes,
                               file_name="neix_portafolio_{}.xlsx".format(meta.get("comitente","").replace(" ","_")),
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as ex:
            st.error("Error generando Excel: " + str(ex))
    with e3:
        st.caption("PDF: portada + P&L consolidado + G/P por especie + tenencia actual + movimientos detallados\n"
                   "Excel: 4 hojas — Resumen · G/P Realizadas · Tenencia Actual · Movimientos")


if __name__ == "__main__":
    main()
