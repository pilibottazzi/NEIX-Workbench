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
html,body,stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],main{background:#F8FAFC!important;color:#0F172A!important;font-family:'Inter',sans-serif!important;}
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
.upload-label{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:0.35rem;}
.kpi-card{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:1rem 1rem 0.9rem 1rem;}
.kpi-label{font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;}
.kpi-value{font-size:1.55rem;font-weight:800;color:#0F172A;letter-spacing:-0.03em;margin-top:0.3rem;}
.kpi-sub{font-size:0.78rem;color:#64748B;margin-top:0.2rem;}
.badge{display:inline-block;padding:0.22rem 0.55rem;border-radius:999px;font-size:0.72rem;font-weight:700;}
.bg-pos{background:#ECFDF5;color:#059669;}
.bg-neg{background:#FEF2F2;color:#C8102E;}
.bg-neu{background:#F1F5F9;color:#64748B;}
.c-green{color:#059669;font-weight:700;}
.c-red{color:#C8102E;font-weight:700;}
.c-ink{color:#0F172A;font-weight:700;}
div[data-testid="stTabs"] button{background:transparent!important;border:1px solid #E2E8F0!important;border-radius:6px!important;color:#64748B!important;font-weight:600!important;font-size:0.79rem!important;padding:0.3rem 0.85rem!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{background:#0F172A!important;border-color:#0F172A!important;color:white!important;}
div[data-testid="stDownloadButton"]>button{width:100%!important;border-radius:7px!important;background:#0F172A!important;color:white!important;border:none!important;font-weight:700!important;font-family:'Inter',sans-serif!important;font-size:0.8rem!important;}
.stDataFrame{border-radius:9px;overflow:hidden;border:1px solid #E2E8F0!important;}
</style>
"""


def _n(s) -> float:
    if s is None:
        return 0.0
    s = re.sub(r"\.(?=\d{3})", "", str(s).strip()).replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def _ars(v: float) -> str:
    s = "{:,.0f}".format(abs(v)).replace(",", "X").replace(".", ",").replace("X", ".")
    return "-$ " + s if v < 0 else "$ " + s


def _pct(v: float) -> str:
    return "{:+.1f}%".format(v)


def _cls(v: float) -> str:
    return "c-green" if v > 0 else ("c-red" if v < 0 else "c-ink")


def _badge(v: float) -> str:
    cls = "bg-pos" if v > 0 else ("bg-neg" if v < 0 else "bg-neu")
    return '<span class="badge ' + cls + '">' + _ars(v) + '</span>'


CATS = {
    "acciones": {"EDN", "GGAL", "YPFD", "SUPV"},
    "cedears": {"AXP", "GOOGL", "MELI", "META", "NVDA", "ADBE", "TSLA", "GLOB",
                "BABA", "VIST", "UNH", "SPY", "COIN", "LAC", "NU", "SPCE", "MSFT"},
    "bonos": {"AL30", "AL35", "GD35"},
    "fondos": {"CAUCION", "MEGA PES A", "FIMA PRE A"},
    "lecaps": {"S29G5", "S16A5", "S12S5", "S30S5", "S15G5", "S31O5"},
}


def _cat(t: str) -> str:
    for c, ts in CATS.items():
        if t.upper() in ts:
            return c
    return "otros"


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
                "usuario": m.get("Usuario", "").strip(),
                "comitente": m.get("Comitente", "").strip(),
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
        if not table:
            continue

        total_ars = total_usd = 0.0
        movs: List[Dict] = []

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if not cells:
                continue

            cpbt = cells[0].upper()
            if cpbt == "TOTAL":
                total_ars = _n(cells[6]) if len(cells) > 6 else 0.0
                total_usd = _n(cells[8]) if len(cells) > 8 else 0.0
            elif len(cells) >= 9:
                movs.append({
                    "Ticker": ticker,
                    "Nombre": nombre,
                    "Cpbt": cpbt,
                    "Numero": cells[1],
                    "Fecha_Concertacion": cells[2],
                    "Fecha_Liquidacion": cells[3],
                    "Cantidad": _n(cells[4]),
                    "Precio": _n(cells[5]),
                    "Monto_ARS": _n(cells[6]),
                    "Monto_USD": _n(cells[8]),
                })

        compras = sum(m["Monto_ARS"] for m in movs if m["Cpbt"] == "C")
        ventas = sum(m["Monto_ARS"] for m in movs if m["Cpbt"] == "V")
        divs = sum(m["Monto_ARS"] for m in movs if "DIV" in m["Cpbt"])

        records.append({
            "Ticker": ticker,
            "Nombre": nombre,
            "Categoria": _cat(ticker),
            "Compras_ARS": compras,
            "Ventas_ARS": ventas,
            "Dividendos_ARS": divs,
            "Total_ARS": total_ars,
            "Total_USD": total_usd,
            "N_Movs": len(movs),
            "_movs": movs,
        })

    return pd.DataFrame(records), meta


def parse_resultados(file_bytes: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(file_bytes.decode("utf-8-sig"), "html.parser")
    table = soup.find("table", class_="table-consultas")
    if table is None:
        return pd.DataFrame()

    records = []
    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame()

    for row in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if not cells or len(cells) < 7:
            continue

        raw = cells[0].strip()
        if not cells[1].strip() or raw.lower() in {"perdida", "ganancia", "total", ""}:
            continue

        parts = raw.split()
        ticker = parts[0].upper()
        nombre = " ".join(parts[1:])
        inv = _n(cells[3])
        val = _n(cells[5])
        dif = _n(cells[6])

        records.append({
            "Ticker": ticker,
            "Nombre": nombre,
            "Categoria": _cat(ticker),
            "Cantidad": _n(cells[1]),
            "PPP": _n(cells[2]),
            "Inversion": inv,
            "Precio_Actual": _n(cells[4]),
            "Valuacion": val,
            "Diferencia": dif,
            "Rend_Pct": (dif / inv * 100 if inv else 0.0),
        })

    return pd.DataFrame(records)


BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=SUB, family="Inter"),
    margin=dict(l=0, r=0, t=36, b=0),
    title_font=dict(size=12, color=INK, family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


def _hbar(df, x, y, title, h=None):
    colors = [R if v < 0 else GRN for v in df[x]]
    fig = go.Figure(go.Bar(
        x=df[x],
        y=df[y],
        orientation="h",
        marker_color=colors,
        text=df[x].apply(_ars),
        textposition="outside",
        textfont=dict(size=9, color=SUB),
    ))
    fig.update_layout(
        **BASE,
        title=title,
        height=h or max(280, 34 * len(df)),
        xaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor=BDR, zerolinewidth=1),
        yaxis=dict(gridcolor="rgba(0,0,0,0)")
    )
    return fig


def _waterfall(df_res):
    d = df_res.sort_values("Diferencia")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=d["Ticker"].tolist(),
        y=d["Diferencia"].tolist(),
        connector=dict(line=dict(color=BDR, width=1)),
        increasing=dict(marker_color=GRN),
        decreasing=dict(marker_color=R),
        text=[_ars(v) for v in d["Diferencia"]],
        textposition="outside",
    ))
    fig.update_layout(
        **BASE,
        title="P&L no realizado por especie",
        height=300,
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", zeroline=True, zerolinecolor=BDR),
        showlegend=False
    )
    return fig


def _inv_vs_val(df_res):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Inversión",
        x=df_res["Ticker"],
        y=df_res["Inversion"],
        marker_color="#CBD5E1",
        text=df_res["Inversion"].apply(_ars),
        textposition="inside",
        textfont=dict(size=8),
    ))
    fig.add_trace(go.Bar(
        name="Valuación actual",
        x=df_res["Ticker"],
        y=df_res["Valuacion"],
        marker_color=BLU,
        text=df_res["Valuacion"].apply(_ars),
        textposition="inside",
        textfont=dict(size=8),
    ))
    fig.update_layout(
        **BASE,
        title="Inversión vs valuación actual",
        height=300,
        barmode="group",
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9"),
    )
    return fig


def render():
    _write_config()
    st.set_page_config(
        page_title="NEIX Workbench | Rendimiento de cartera",
        page_icon="📈",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <div>{LOGO}</div>
                <div>
                    <div class="topbar-title">Rendimiento de cartera</div>
                    <div class="topbar-sub">Herramienta integrada a NEIX Workbench</div>
                </div>
            </div>
            <div class="topbar-right">
                Carga de histórico por especie + resultados actuales<br>
                Análisis ejecutivo por comitente
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="upload-label">Histórico por especie</div>', unsafe_allow_html=True)
        hist_file = st.file_uploader(
            "Subí el histórico por especie",
            type=["html", "xls", "xlsx"],
            key="hist_file",
            label_visibility="collapsed",
        )

    with c2:
        st.markdown('<div class="upload-label">Resultados / tenencia actual</div>', unsafe_allow_html=True)
        res_file = st.file_uploader(
            "Subí el archivo de resultados",
            type=["html", "xls", "xlsx"],
            key="res_file",
            label_visibility="collapsed",
        )

    if not hist_file or not res_file:
        st.info("Subí ambos archivos para procesar el análisis.")
        return

    try:
        df_hist, meta = parse_historico(hist_file.getvalue())
        df_res = parse_resultados(res_file.getvalue())
    except Exception as e:
        st.error(f"Error procesando archivos: {e}")
        return

    if df_hist.empty or df_res.empty:
        st.warning("No se pudo obtener información suficiente desde los archivos cargados.")
        return

    total_inv = df_res["Inversion"].sum()
    total_val = df_res["Valuacion"].sum()
    total_dif = df_res["Diferencia"].sum()
    total_pct = (total_dif / total_inv * 100) if total_inv else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Comitente</div><div class="kpi-value">{meta.get("comitente","-")}</div><div class="kpi-sub">{meta.get("usuario","")}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Inversión</div><div class="kpi-value">{_ars(total_inv)}</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Valuación actual</div><div class="kpi-value">{_ars(total_val)}</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Resultado no realizado</div><div class="kpi-value">{_ars(total_dif)}</div><div class="kpi-sub"><span class="{_cls(total_pct)}">{_pct(total_pct)}</span></div></div>',
            unsafe_allow_html=True,
        )

    t1, t2, t3 = st.tabs(["Resumen", "Gráficos", "Detalle"])

    with t1:
        st.dataframe(
            df_res[["Ticker", "Nombre", "Categoria", "Cantidad", "PPP", "Precio_Actual", "Inversion", "Valuacion", "Diferencia", "Rend_Pct"]],
            use_container_width=True,
            hide_index=True,
        )

    with t2:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(_waterfall(df_res), use_container_width=True)
        with g2:
            st.plotly_chart(_inv_vs_val(df_res), use_container_width=True)

        top = df_res.sort_values("Diferencia").copy()
        st.plotly_chart(_hbar(top, "Diferencia", "Ticker", "Resultado por especie"), use_container_width=True)

    with t3:
        st.dataframe(df_hist.drop(columns=["_movs"], errors="ignore"), use_container_width=True, hide_index=True)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name="resultado_actual", index=False)
            df_hist.drop(columns=["_movs"], errors="ignore").to_excel(writer, sheet_name="historico", index=False)

        st.download_button(
            "Descargar Excel consolidado",
            data=output.getvalue(),
            file_name="rendimiento_cartera.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    render()
