from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CONFIG / ESTILO
# =========================================================
NEIX_RED = "#ff3b30"
NEIX_DARK = "#111827"
NEIX_MUTED = "#6b7280"
NEIX_BG = "#f6f7fb"
CARD_BG = "#ffffff"
BORDER = "rgba(17,24,39,0.08)"


# =========================================================
# UI / CSS
# =========================================================
def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1380px;
            padding-top: 1.15rem;
            padding-bottom: 2rem;
          }}

          html, body, [data-testid="stAppViewContainer"] {{
            background: {NEIX_BG};
          }}

          [data-testid="stHeader"] {{
            background: transparent;
          }}

          .neix-hero {{
            background: linear-gradient(135deg, #121926 0%, #1b2435 100%);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 24px;
            padding: 1.4rem 1.5rem 1.15rem 1.5rem;
            color: white;
            box-shadow: 0 16px 40px rgba(17,24,39,0.10);
            margin-bottom: 1rem;
          }}

          .neix-kicker {{
            color: rgba(255,255,255,0.72);
            font-size: 0.80rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.35rem;
          }}

          .neix-title {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.05;
          }}

          .neix-subtitle {{
            margin-top: 0.45rem;
            color: rgba(255,255,255,0.82);
            font-size: 0.96rem;
          }}

          .neix-pill-row {{
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
          }}

          .neix-pill {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.92);
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
          }}

          .neix-section-title {{
            font-size: 1.03rem;
            font-weight: 800;
            color: {NEIX_DARK};
            margin: 0.4rem 0 0.7rem 0;
            letter-spacing: -0.01em;
          }}

          .neix-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 30px rgba(17,24,39,0.04);
          }}

          .neix-kpi {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 28px rgba(17,24,39,0.04);
            min-height: 124px;
          }}

          .neix-kpi-label {{
            color: {NEIX_MUTED};
            font-size: 0.80rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.55rem;
          }}

          .neix-kpi-value {{
            color: {NEIX_DARK};
            font-size: 1.7rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.05;
          }}

          .neix-kpi-delta-pos {{
            color: #059669;
            font-size: 0.86rem;
            font-weight: 700;
            margin-top: 0.35rem;
          }}

          .neix-kpi-delta-neg {{
            color: {NEIX_RED};
            font-size: 0.86rem;
            font-weight: 700;
            margin-top: 0.35rem;
          }}

          .neix-kpi-delta-neutral {{
            color: {NEIX_MUTED};
            font-size: 0.86rem;
            font-weight: 700;
            margin-top: 0.35rem;
          }}

          div[data-testid="stFileUploader"] {{
            background: rgba(255,255,255,0.72);
            border: 1px dashed rgba(17,24,39,0.16);
            border-radius: 18px;
            padding: 0.35rem 0.5rem 0.1rem 0.5rem;
          }}

          div[data-testid="stTabs"] button {{
            border-radius: 999px !important;
            padding: 0.45rem 0.9rem !important;
            border: 1px solid rgba(17,24,39,0.08) !important;
            background: white !important;
            color: {NEIX_DARK} !important;
            font-weight: 700 !important;
          }}

          div[data-testid="stTabs"] button[aria-selected="true"] {{
            background: {NEIX_RED} !important;
            color: white !important;
            border-color: {NEIX_RED} !important;
          }}

          div[data-testid="stDownloadButton"] > button {{
            width: 100% !important;
            border-radius: 12px !important;
            background: {NEIX_RED} !important;
            color: white !important;
            border: none !important;
            font-weight: 700 !important;
          }}

          .stDataFrame {{
            border-radius: 18px;
            overflow: hidden;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS
# =========================================================
def _to_float(value) -> float:
    if pd.isna(value):
        return 0.0

    s = str(value).strip()
    if s == "":
        return 0.0

    s = s.replace("U$S", "").replace("USD", "").replace("ARS", "").strip()
    s = s.replace(".", "").replace(",", ".")
    s = s.replace("%", "")

    try:
        return float(s)
    except Exception:
        return 0.0


def _fmt_usd(value: float) -> str:
    return f"USD {value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:,.2f}%"


def _delta_class(value: float) -> str:
    if value > 0:
        return "neix-kpi-delta-pos"
    if value < 0:
        return "neix-kpi-delta-neg"
    return "neix-kpi-delta-neutral"


def _find_row_index(raw: pd.DataFrame, target: str) -> Optional[int]:
    target_up = str(target).strip().upper()

    for i in range(len(raw)):
        row_values = [str(x).strip().upper() for x in raw.iloc[i].tolist()]
        if target_up in row_values:
            return i
    return None


def _extract_table_block(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Busca el bloque que empieza en la fila donde está 'Ticker'
    y termina antes de 'TOTAL'.
    """
    header_idx = _find_row_index(raw, "Ticker")
    if header_idx is None:
        raise ValueError("No se encontró la fila de encabezados del bloque de posiciones.")

    headers = [str(x).strip() for x in raw.iloc[header_idx].tolist()]

    end_idx = None
    for i in range(header_idx + 1, len(raw)):
        first_col = str(raw.iloc[i, 0]).strip().upper()
        if first_col == "TOTAL":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("No se encontró la fila TOTAL del bloque de posiciones.")

    table = raw.iloc[header_idx + 1 : end_idx + 1].copy()
    table.columns = headers
    table = table.loc[:, ~table.columns.isna()]
    table.columns = [str(c).strip() for c in table.columns]

    # limpiar columnas vacías
    table = table[[c for c in table.columns if c != "" and c.lower() != "nan"]].copy()

    # renombre seguro
    rename_map = {
        "Cant. Comprada": "Cant_Comprada",
        "Cant. Vendida": "Cant_Vendida",
        "Saldo Tenencia": "Saldo_Tenencia",
        "Costo Total (USD)": "Costo_Total_USD",
        "Cobrado RTA/DIV (USD)": "Cobrado_RTA_DIV_USD",
        "Ventas (USD)": "Ventas_USD",
        "Rendimiento Neto (USD)": "Rendimiento_Neto_USD",
    }
    table = table.rename(columns=rename_map)

    expected = [
        "Ticker",
        "Cant_Comprada",
        "Cant_Vendida",
        "Saldo_Tenencia",
        "Costo_Total_USD",
        "Cobrado_RTA_DIV_USD",
        "Ventas_USD",
        "Rendimiento_Neto_USD",
    ]
    missing = [c for c in expected if c not in table.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el bloque principal: {missing}")

    for col in expected[1:]:
        table[col] = table[col].apply(_to_float)

    table["Ticker"] = table["Ticker"].astype(str).str.strip()
    return table


def _extract_financial_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Busca bloque 'RESUMEN FINANCIERO'.
    Toma concepto en col A/B y valor a la derecha.
    """
    start_idx = _find_row_index(raw, "RESUMEN FINANCIERO")
    if start_idx is None:
        return pd.DataFrame(columns=["Concepto", "Valor"])

    items: List[Dict[str, object]] = []

    for i in range(start_idx + 1, min(start_idx + 25, len(raw))):
        row = raw.iloc[i].tolist()
        non_null = [x for x in row if pd.notna(x) and str(x).strip() != ""]
        if len(non_null) == 0:
            continue

        concept = None
        value = None

        # primera celda textual útil
        for x in row[:4]:
            if pd.notna(x) and str(x).strip() != "":
                concept = str(x).strip()
                break

        # último valor útil de la fila
        for x in reversed(row):
            if pd.notna(x) and str(x).strip() != "":
                value = x
                break

        if concept is None:
            continue

        if str(concept).strip().upper() == "RESUMEN FINANCIERO":
            continue

        items.append({"Concepto": concept, "Valor": _to_float(value)})

    out = pd.DataFrame(items)
    if not out.empty:
        out["Concepto"] = out["Concepto"].astype(str).str.strip()
    return out


def _extract_period_text(raw: pd.DataFrame) -> Optional[str]:
    for i in range(min(8, len(raw))):
        row_values = [str(x).strip() for x in raw.iloc[i].tolist() if pd.notna(x) and str(x).strip() != ""]
        row_text = " | ".join(row_values)
        if "Período:" in row_text or "Periodo:" in row_text:
            return row_text
    return None


def _extract_comitente_from_title(raw: pd.DataFrame, fallback_name: str) -> str:
    for i in range(min(5, len(raw))):
        row_values = [str(x).strip() for x in raw.iloc[i].tolist() if pd.notna(x) and str(x).strip() != ""]
        row_text = " ".join(row_values)
        if "COMITENTE" in row_text.upper():
            return row_text.replace("— DETALLE DE POSICIONES Y RENDIMIENTO", "").strip()
    return Path(fallback_name).stem


def _parse_uploaded_report(file) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    raw = pd.read_excel(file, header=None)

    detail = _extract_table_block(raw)
    summary = _extract_financial_summary(raw)
    comitente = _extract_comitente_from_title(raw, file.name)
    period_text = _extract_period_text(raw)

    # quitar total del detalle
    detail = detail[detail["Ticker"].str.upper() != "TOTAL"].copy()
    detail["Comitente"] = comitente

    meta = {
        "comitente": comitente,
        "period_text": period_text or "",
        "filename": file.name,
    }

    if not summary.empty:
        summary["Comitente"] = comitente

    return detail, summary, meta


def _styled_detail(df: pd.DataFrame):
    show_cols = [
        "Comitente",
        "Ticker",
        "Cant_Comprada",
        "Cant_Vendida",
        "Saldo_Tenencia",
        "Costo_Total_USD",
        "Cobrado_RTA_DIV_USD",
        "Ventas_USD",
        "Rendimiento_Neto_USD",
    ]
    tmp = df[show_cols].copy()

    return (
        tmp.style.format(
            {
                "Cant_Comprada": "{:,.0f}",
                "Cant_Vendida": "{:,.0f}",
                "Saldo_Tenencia": "{:,.0f}",
                "Costo_Total_USD": "USD {:,.2f}",
                "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                "Ventas_USD": "USD {:,.2f}",
                "Rendimiento_Neto_USD": "USD {:,.2f}",
            }
        )
        .applymap(
            lambda v: "color: #ff3b30; font-weight: 700;" if isinstance(v, (int, float)) and v < 0 else "",
            subset=["Costo_Total_USD", "Rendimiento_Neto_USD"],
        )
        .applymap(
            lambda v: "color: #059669; font-weight: 700;" if isinstance(v, (int, float)) and v > 0 else "",
            subset=["Cobrado_RTA_DIV_USD", "Ventas_USD"],
        )
    )


def _build_export_excel(
    detail_all: pd.DataFrame,
    summary_all: pd.DataFrame,
    kpis: Dict[str, float],
    by_asset: pd.DataFrame,
    by_comitente: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail_all.to_excel(writer, sheet_name="Detalle Consolidado", index=False)
        summary_all.to_excel(writer, sheet_name="Resumen Financiero", index=False)
        by_asset.to_excel(writer, sheet_name="Por Activo", index=False)
        by_comitente.to_excel(writer, sheet_name="Por Comitente", index=False)

        pd.DataFrame(
            {
                "KPI": list(kpis.keys()),
                "Valor": list(kpis.values()),
            }
        ).to_excel(writer, sheet_name="KPIs", index=False)

    output.seek(0)
    return output.getvalue()


# =========================================================
# RENDER
# =========================================================
def render() -> None:
    _inject_css()

    st.markdown(
        """
        <div class="neix-hero">
            <div class="neix-kicker">NEIX Workbench · Mesa / Clientes</div>
            <h1 class="neix-title">Dashboard Ejecutivo — Rendimiento de Cartera</h1>
            <div class="neix-subtitle">
                Consolidado de comitentes en USD, con foco en posición actual, rendimiento neto,
                carry cobrado y lectura ejecutiva por activo y por cuenta.
            </div>
            <div class="neix-pill-row">
                <div class="neix-pill">Minimalista</div>
                <div class="neix-pill">Ejecutivo</div>
                <div class="neix-pill">Multi-comitente</div>
                <div class="neix-pill">WorkBench Ready</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.2, 0.8], gap="large")

    with top_left:
        uploaded_files = st.file_uploader(
            "Subí los 3 archivos de comitentes",
            type=["xls", "xlsx"],
            accept_multiple_files=True,
            help="Idealmente, subí los excels ya armados con el formato del reporte: Detalle de Posiciones y Rendimiento.",
        )

    with top_right:
        st.markdown('<div class="neix-card">', unsafe_allow_html=True)
        st.markdown('<div class="neix-section-title">Criterio del dashboard</div>', unsafe_allow_html=True)
        st.caption(
            "• Todo en USD\n"
            "\n• Consolidación por activo y por comitente"
            "\n\n• Posición actual = saldo de tenencia"
            "\n\n• Resultado = rendimiento neto reportado"
            "\n\n• Pensado para comité, mesa y seguimiento ejecutivo"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_files:
        st.info("Subí los archivos para visualizar el dashboard.")
        return

    detail_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    metas: List[Dict[str, str]] = []
    parse_errors: List[str] = []

    for file in uploaded_files:
        try:
            detail_df, summary_df, meta = _parse_uploaded_report(file)
            detail_frames.append(detail_df)
            if not summary_df.empty:
                summary_frames.append(summary_df)
            metas.append(meta)
        except Exception as e:
            parse_errors.append(f"{file.name}: {e}")

    if parse_errors:
        st.error("No pude leer correctamente algunos archivos:")
        for err in parse_errors:
            st.write(f"• {err}")
        if not detail_frames:
            return

    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    if detail_all.empty:
        st.warning("No se pudo construir el consolidado.")
        return

    # filtros
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1], gap="medium")
    with filter_col1:
        comitentes = sorted(detail_all["Comitente"].dropna().unique().tolist())
        selected_comitentes = st.multiselect(
            "Comitentes",
            options=comitentes,
            default=comitentes,
        )

    with filter_col2:
        only_open_positions = st.toggle("Solo posiciones abiertas", value=False)

    with filter_col3:
        sort_metric = st.selectbox(
            "Orden principal",
            options=[
                "Rendimiento_Neto_USD",
                "Saldo_Tenencia",
                "Costo_Total_USD",
                "Cobrado_RTA_DIV_USD",
                "Ventas_USD",
            ],
            index=0,
        )

    df = detail_all[detail_all["Comitente"].isin(selected_comitentes)].copy()

    if only_open_positions:
        df = df[df["Saldo_Tenencia"] > 0].copy()

    if df.empty:
        st.warning("Con los filtros aplicados no hay registros para mostrar.")
        return

    # KPIs
    capital_invertido = df["Costo_Total_USD"].sum()
    cobrado = df["Cobrado_RTA_DIV_USD"].sum()
    ventas = df["Ventas_USD"].sum()
    rendimiento = df["Rendimiento_Neto_USD"].sum()
    posiciones_abiertas = int((df["Saldo_Tenencia"] > 0).sum())
    capital_base = abs(capital_invertido) if capital_invertido != 0 else 0.0
    rendimiento_pct = (rendimiento / capital_base * 100) if capital_base else 0.0

    kpis = {
        "Capital invertido USD": capital_invertido,
        "Cobrado USD": cobrado,
        "Ventas USD": ventas,
        "Rendimiento neto USD": rendimiento,
        "% sobre capital USD": rendimiento_pct,
    }

    st.markdown("")

    k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

    k1.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Capital invertido</div>
            <div class="neix-kpi-value">{_fmt_usd(capital_invertido)}</div>
            <div class="{_delta_class(capital_invertido)}">Base consolidada</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k2.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Cobrado RTA / DIV</div>
            <div class="neix-kpi-value">{_fmt_usd(cobrado)}</div>
            <div class="{_delta_class(cobrado)}">Carry cobrado</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k3.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Ventas</div>
            <div class="neix-kpi-value">{_fmt_usd(ventas)}</div>
            <div class="{_delta_class(ventas)}">Realizado por ventas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k4.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Rendimiento neto</div>
            <div class="neix-kpi-value">{_fmt_usd(rendimiento)}</div>
            <div class="{_delta_class(rendimiento)}">Resultado consolidado</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k5.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">% sobre capital</div>
            <div class="neix-kpi-value">{_fmt_pct(rendimiento_pct)}</div>
            <div class="{_delta_class(rendimiento_pct)}">{posiciones_abiertas} posiciones abiertas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # tablas resumen
    by_asset = (
        df.groupby("Ticker", as_index=False)
        .agg(
            Cant_Comprada=("Cant_Comprada", "sum"),
            Cant_Vendida=("Cant_Vendida", "sum"),
            Saldo_Tenencia=("Saldo_Tenencia", "sum"),
            Costo_Total_USD=("Costo_Total_USD", "sum"),
            Cobrado_RTA_DIV_USD=("Cobrado_RTA_DIV_USD", "sum"),
            Ventas_USD=("Ventas_USD", "sum"),
            Rendimiento_Neto_USD=("Rendimiento_Neto_USD", "sum"),
        )
        .sort_values(sort_metric, ascending=True if sort_metric in ["Rendimiento_Neto_USD", "Costo_Total_USD"] else False)
        .reset_index(drop=True)
    )

    by_comitente = (
        df.groupby("Comitente", as_index=False)
        .agg(
            Posiciones_Abiertas=("Saldo_Tenencia", lambda s: int((s > 0).sum())),
            Cantidad_Activos=("Ticker", "nunique"),
            Costo_Total_USD=("Costo_Total_USD", "sum"),
            Cobrado_RTA_DIV_USD=("Cobrado_RTA_DIV_USD", "sum"),
            Ventas_USD=("Ventas_USD", "sum"),
            Rendimiento_Neto_USD=("Rendimiento_Neto_USD", "sum"),
        )
        .sort_values("Rendimiento_Neto_USD")
        .reset_index(drop=True)
    )

    current_positions = df[df["Saldo_Tenencia"] > 0].copy().sort_values("Rendimiento_Neto_USD")

    # charts
    chart_by_asset = px.bar(
        by_asset.sort_values("Rendimiento_Neto_USD"),
        x="Rendimiento_Neto_USD",
        y="Ticker",
        orientation="h",
        title="Rendimiento neto por activo",
        text_auto=".2s",
    )
    chart_by_asset.update_layout(
        height=max(380, 40 * len(by_asset)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(color=NEIX_DARK),
        title_font=dict(size=18),
        xaxis_title=None,
        yaxis_title=None,
    )
    chart_by_asset.update_traces(marker_color=NEIX_RED)

    chart_by_comitente = px.bar(
        by_comitente.sort_values("Rendimiento_Neto_USD"),
        x="Rendimiento_Neto_USD",
        y="Comitente",
        orientation="h",
        title="Rendimiento neto por comitente",
        text_auto=".2s",
    )
    chart_by_comitente.update_layout(
        height=max(320, 75 * len(by_comitente)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(color=NEIX_DARK),
        title_font=dict(size=18),
        xaxis_title=None,
        yaxis_title=None,
    )
    chart_by_comitente.update_traces(marker_color=NEIX_RED)

    position_mix = current_positions.copy()
    if position_mix.empty:
        position_mix = df.nlargest(min(10, len(df)), "Saldo_Tenencia").copy()

    chart_positions = px.bar(
        position_mix.sort_values("Saldo_Tenencia"),
        x="Saldo_Tenencia",
        y="Ticker",
        color="Comitente",
        orientation="h",
        title="Posición actual por activo",
        text_auto=".2s",
    )
    chart_positions.update_layout(
        height=max(380, 40 * len(position_mix)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(color=NEIX_DARK),
        title_font=dict(size=18),
        xaxis_title=None,
        yaxis_title=None,
    )

    # meta visual
    period_texts = [m["period_text"] for m in metas if m.get("period_text")]
    if period_texts:
        st.caption(" | ".join(period_texts[:3]))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Resumen Ejecutivo", "Posición Actual", "Rendimiento", "Por Comitente", "Detalle"]
    )

    with tab1:
        top_l, top_r = st.columns([1.1, 0.9], gap="large")

        with top_l:
            st.markdown('<div class="neix-section-title">Lectura ejecutiva</div>', unsafe_allow_html=True)
            st.markdown('<div class="neix-card">', unsafe_allow_html=True)

            worst_asset = by_asset.nsmallest(1, "Rendimiento_Neto_USD")
            best_asset = by_asset.nlargest(1, "Rendimiento_Neto_USD")
            worst_comitente = by_comitente.nsmallest(1, "Rendimiento_Neto_USD")
            best_comitente = by_comitente.nlargest(1, "Rendimiento_Neto_USD")

            st.markdown(
                f"""
                **Consolidado actual**
                
                El resultado neto consolidado asciende a **{_fmt_usd(rendimiento)}**, sobre una base de capital
                invertido de **{_fmt_usd(capital_invertido)}**, lo que implica un rendimiento de
                **{_fmt_pct(rendimiento_pct)}** sobre capital.

                **Activo con peor contribución:** {worst_asset.iloc[0]['Ticker']} ({_fmt_usd(worst_asset.iloc[0]['Rendimiento_Neto_USD'])})  
                **Activo con mejor contribución:** {best_asset.iloc[0]['Ticker']} ({_fmt_usd(best_asset.iloc[0]['Rendimiento_Neto_USD'])})

                **Comitente con peor resultado:** {worst_comitente.iloc[0]['Comitente']} ({_fmt_usd(worst_comitente.iloc[0]['Rendimiento_Neto_USD'])})  
                **Comitente con mejor resultado:** {best_comitente.iloc[0]['Comitente']} ({_fmt_usd(best_comitente.iloc[0]['Rendimiento_Neto_USD'])})
                """
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="neix-section-title">Resumen financiero consolidado</div>', unsafe_allow_html=True)
            summary_resume = pd.DataFrame(
                {
                    "Concepto": [
                        "Capital invertido",
                        "Cobrado RTA / DIV",
                        "Ventas",
                        "Rendimiento neto",
                        "% sobre capital",
                    ],
                    "Valor": [
                        capital_invertido,
                        cobrado,
                        ventas,
                        rendimiento,
                        rendimiento_pct,
                    ],
                }
            )

            st.dataframe(
                summary_resume.style.format(
                    {
                        "Valor": lambda x: _fmt_pct(x) if abs(x) < 500 and summary_resume.loc[summary_resume["Valor"] == x, "Concepto"].iloc[0] == "% sobre capital" else f"{x:,.2f}"
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with top_r:
            st.plotly_chart(chart_by_asset, use_container_width=True)

    with tab2:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown('<div class="neix-section-title">Posiciones abiertas</div>', unsafe_allow_html=True)
            if current_positions.empty:
                st.info("No hay posiciones abiertas con los filtros seleccionados.")
            else:
                pos_show = current_positions[
                    [
                        "Comitente",
                        "Ticker",
                        "Saldo_Tenencia",
                        "Costo_Total_USD",
                        "Cobrado_RTA_DIV_USD",
                        "Rendimiento_Neto_USD",
                    ]
                ].copy()

                st.dataframe(
                    pos_show.style.format(
                        {
                            "Saldo_Tenencia": "{:,.0f}",
                            "Costo_Total_USD": "USD {:,.2f}",
                            "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                            "Rendimiento_Neto_USD": "USD {:,.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        with right:
            st.plotly_chart(chart_positions, use_container_width=True)

    with tab3:
        left, right = st.columns([1.0, 1.0], gap="large")

        with left:
            st.markdown('<div class="neix-section-title">Ranking por activo</div>', unsafe_allow_html=True)
            st.dataframe(
                by_asset.style.format(
                    {
                        "Cant_Comprada": "{:,.0f}",
                        "Cant_Vendida": "{:,.0f}",
                        "Saldo_Tenencia": "{:,.0f}",
                        "Costo_Total_USD": "USD {:,.2f}",
                        "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                        "Ventas_USD": "USD {:,.2f}",
                        "Rendimiento_Neto_USD": "USD {:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.plotly_chart(chart_by_asset, use_container_width=True)

    with tab4:
        left, right = st.columns([0.95, 1.05], gap="large")

        with left:
            st.markdown('<div class="neix-section-title">Resumen por comitente</div>', unsafe_allow_html=True)
            st.dataframe(
                by_comitente.style.format(
                    {
                        "Costo_Total_USD": "USD {:,.2f}",
                        "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                        "Ventas_USD": "USD {:,.2f}",
                        "Rendimiento_Neto_USD": "USD {:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.plotly_chart(chart_by_comitente, use_container_width=True)

    with tab5:
        st.markdown('<div class="neix-section-title">Detalle consolidado</div>', unsafe_allow_html=True)
        st.dataframe(_styled_detail(df), use_container_width=True, hide_index=True)

        if not summary_all.empty:
            st.markdown('<div class="neix-section-title">Resumen financiero por comitente</div>', unsafe_allow_html=True)
            sum_filtered = summary_all[summary_all["Comitente"].isin(selected_comitentes)].copy()
            st.dataframe(
                sum_filtered.style.format({"Valor": "USD {:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

    # export
    export_bytes = _build_export_excel(
        detail_all=df,
        summary_all=summary_all[summary_all["Comitente"].isin(selected_comitentes)].copy() if not summary_all.empty else pd.DataFrame(),
        kpis=kpis,
        by_asset=by_asset,
        by_comitente=by_comitente,
    )

    st.markdown("")
    d1, d2 = st.columns([0.72, 0.28], gap="large")
    with d1:
        st.caption(
            "El dashboard consolida los reportes cargados y permite lectura ejecutiva por cuenta y por activo. "
            "La lógica asume que cada archivo conserva la estructura del reporte mostrado en tu ejemplo."
        )
    with d2:
        st.download_button(
            "Descargar consolidado Excel",
            data=export_bytes,
            file_name="consolidado_rendimiento_cartera.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# =========================================================
# MODO STANDALONE
# =========================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="NEIX · Rendimiento de Cartera",
        page_icon="📊",
        layout="wide",
    )
    render()
