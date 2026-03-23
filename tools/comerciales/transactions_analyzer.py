
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
            max-width: 1420px;
            padding-top: 1.1rem;
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
            border-radius: 26px;
            padding: 1.45rem 1.55rem 1.2rem 1.55rem;
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
            margin: 0.35rem 0 0.75rem 0;
            letter-spacing: -0.01em;
          }}

          .neix-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
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
            font-size: 0.79rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.55rem;
          }}

          .neix-kpi-value {{
            color: {NEIX_DARK};
            font-size: 1.65rem;
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
            background: rgba(255,255,255,0.74);
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
def _read_excel_robust(file) -> pd.DataFrame:
    file_name = getattr(file, "name", "").lower()

    try:
        if file_name.endswith(".xls"):
            file.seek(0)
            return pd.read_excel(file, header=None, engine="xlrd")
        if file_name.endswith(".xlsx"):
            file.seek(0)
            return pd.read_excel(file, header=None, engine="openpyxl")

        file.seek(0)
        return pd.read_excel(file, header=None)
    except Exception as first_error:
        try:
            file.seek(0)
            return pd.read_excel(file, header=None, engine="openpyxl")
        except Exception:
            try:
                file.seek(0)
                return pd.read_excel(file, header=None, engine="xlrd")
            except Exception:
                raise ValueError(
                    f"No se pudo leer '{getattr(file, 'name', 'archivo')}'. "
                    f"Error original: {first_error}"
                )


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_float(value) -> float:
    if pd.isna(value):
        return 0.0

    s = str(value).strip()
    if s == "":
        return 0.0

    s = s.replace("U$S", "").replace("USD", "").replace("ARS", "").replace("$", "").strip()
    s = s.replace(".", "").replace(",", ".")
    s = s.replace("%", "")

    try:
        return float(s)
    except Exception:
        return 0.0


def _fmt_usd(value: float) -> str:
    return f"USD {value:,.2f}"


def _fmt_num(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_pct(value: float) -> str:
    return f"{value:,.2f}%"


def _delta_class(value: float) -> str:
    if value > 0:
        return "neix-kpi-delta-pos"
    if value < 0:
        return "neix-kpi-delta-neg"
    return "neix-kpi-delta-neutral"


def _find_row_index_contains(raw: pd.DataFrame, target: str) -> Optional[int]:
    target_up = target.strip().upper()
    for i in range(len(raw)):
        vals = [_norm_text(x).upper() for x in raw.iloc[i].tolist()]
        if target_up in vals:
            return i
    return None


def _find_row_index_startswith(raw: pd.DataFrame, target: str) -> Optional[int]:
    target_up = target.strip().upper()
    for i in range(len(raw)):
        vals = [_norm_text(x).upper() for x in raw.iloc[i].tolist()]
        joined = " | ".join([v for v in vals if v])
        if joined.startswith(target_up):
            return i
    return None


def _extract_metadata_from_activity(raw: pd.DataFrame, fallback_name: str) -> Dict[str, str]:
    meta = {
        "usuario": "",
        "comitente": Path(fallback_name).stem,
        "fecha_desde": "",
        "fecha_hasta": "",
        "filename": fallback_name,
    }

    # en tu formato:
    # fila 3 encabezados: Usuario | Comitente | Fecha Desde | Fecha Hasta ...
    # fila 4 valores
    try:
        headers_row = raw.iloc[2].tolist()
        values_row = raw.iloc[3].tolist()

        headers = [_norm_text(x).upper() for x in headers_row]
        values = [_norm_text(x) for x in values_row]

        mapping = dict(zip(headers, values))

        if "USUARIO" in mapping:
            meta["usuario"] = mapping["USUARIO"]
        if "COMITENTE" in mapping:
            meta["comitente"] = mapping["COMITENTE"]
        if "FECHA DESDE" in mapping:
            meta["fecha_desde"] = mapping["FECHA DESDE"]
        if "FECHA HASTA" in mapping:
            meta["fecha_hasta"] = mapping["FECHA HASTA"]
    except Exception:
        pass

    return meta


def _parse_activity_detail(raw: pd.DataFrame) -> pd.DataFrame:
    header_idx = _find_row_index_contains(raw, "Fecha de Emisión")
    if header_idx is None:
        raise ValueError("No encontré la fila de encabezados del bloque Detalle.")

    headers = [_norm_text(x) for x in raw.iloc[header_idx].tolist()]
    detail = raw.iloc[header_idx + 1 :].copy()
    detail.columns = headers

    detail = detail.loc[:, [c for c in detail.columns if c != ""]].copy()

    expected_cols = [
        "Fecha de Emisión",
        "Fecha Liquidación",
        "Comprobante",
        "Nro. de Comprobante",
        "Ticker",
        "Cantidad",
        "Precio",
        "Moneda",
        "Importe Pesos",
        "Importe En Moneda",
    ]
    missing = [c for c in expected_cols if c not in detail.columns]
    if missing:
        raise ValueError(f"Faltan columnas en Activity: {missing}")

    # limpiar filas vacías reales
    detail = detail[
        detail[expected_cols]
        .astype(str)
        .apply(lambda row: any(str(x).strip() not in {"", "nan", "None"} for x in row), axis=1)
    ].copy()

    # normalizar
    detail["Comprobante"] = detail["Comprobante"].astype(str).str.strip().str.upper()
    detail["Ticker"] = detail["Ticker"].fillna("").astype(str).str.strip().str.upper()
    detail["Moneda"] = detail["Moneda"].fillna("").astype(str).str.strip()

    detail["Cantidad_num"] = detail["Cantidad"].apply(_to_float)
    detail["Precio_num"] = detail["Precio"].apply(_to_float)
    detail["Importe_Pesos_num"] = detail["Importe Pesos"].apply(_to_float)
    detail["Importe_Moneda_num"] = detail["Importe En Moneda"].apply(_to_float)

    detail["Fecha_Emision_dt"] = pd.to_datetime(detail["Fecha de Emisión"], dayfirst=True, errors="coerce")
    detail["Fecha_Liquidacion_dt"] = pd.to_datetime(detail["Fecha Liquidación"], dayfirst=True, errors="coerce")

    return detail


def _is_usd_row(moneda: str) -> bool:
    m = _norm_text(moneda).upper()
    return m in {"DOLAR LOCAL", "U$S EXTERIOR", "USD EXTERIOR", "USD", "US$"}


def _is_pesos_row(moneda: str) -> bool:
    return _norm_text(moneda).upper() == "PESOS"


def _classify_flow_columns(detail: pd.DataFrame) -> pd.DataFrame:
    df = detail.copy()

    df["USD_Flow"] = df.apply(
        lambda r: r["Importe_Moneda_num"] if _is_usd_row(r["Moneda"]) else 0.0,
        axis=1,
    )

    df["ARS_Flow"] = df.apply(
        lambda r: r["Importe_Pesos_num"] if _is_pesos_row(r["Moneda"]) else 0.0,
        axis=1,
    )

    return df


def _build_summary_from_activity(detail: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye tabla por ticker similar al reporte que querés:
    - Cant. Comprada
    - Cant. Vendida
    - Saldo Tenencia
    - Costo Total (USD)
    - Cobrado RTA/DIV (USD)
    - Ventas (USD)
    - Rendimiento Neto (USD)

    Supuestos prácticos:
    - Compras: COMPRA / LICITACION con flujo USD negativo
    - Ventas: VENTA con flujo USD positivo
    - Cobrado RTA/DIV: RTA/AMORT / CREDITO RTA con flujo USD positivo
    - Gastos USD: PAU$ / DEBITO RTA / similares => afectan rendimiento
    - Saldo Tenencia: compras - ventas, usando cantidad
    """
    df = _classify_flow_columns(detail)

    trade_tickers = df["Ticker"].replace("", pd.NA).dropna().unique().tolist()
    if not trade_tickers:
        raise ValueError("No encontré tickers operables dentro del Activity.")

    rows = []

    buy_types = {"COMPRA", "LICITACION"}
    sell_types = {"VENTA"}
    income_types = {"RTA/AMORT", "CREDITO RTA", "DIVIDENDO", "CREDITO DIV", "RTA", "AMORT"}
    expense_types = {"PAU$", "DEBITO RTA"}

    for ticker in sorted(trade_tickers):
        dft = df[df["Ticker"] == ticker].copy()

        qty_bought = dft.loc[
            (dft["Comprobante"].isin(buy_types)) & (dft["Cantidad_num"] > 0),
            "Cantidad_num",
        ].sum()

        qty_sold = abs(
            dft.loc[
                (dft["Comprobante"].isin(sell_types)) & (dft["Cantidad_num"] < 0),
                "Cantidad_num",
            ].sum()
        )

        # si alguna venta viene positiva, también la tomo
        qty_sold += dft.loc[
            (dft["Comprobante"].isin(sell_types)) & (dft["Cantidad_num"] > 0),
            "Cantidad_num",
        ].sum()

        cost_total_usd = abs(
            dft.loc[
                (dft["Comprobante"].isin(buy_types)) & (dft["USD_Flow"] < 0),
                "USD_Flow",
            ].sum()
        )

        ventas_usd = dft.loc[
            (dft["Comprobante"].isin(sell_types)) & (dft["USD_Flow"] > 0),
            "USD_Flow",
        ].sum()

        cobrado_rta_div_usd = dft.loc[
            (dft["Comprobante"].isin(income_types)) & (dft["USD_Flow"] > 0),
            "USD_Flow",
        ].sum()

        gastos_usd = abs(
            dft.loc[
                (dft["Comprobante"].isin(expense_types)) & (dft["USD_Flow"] < 0),
                "USD_Flow",
            ].sum()
        )

        # rend. neto "cash" por ticker
        rendimiento_neto_usd = ventas_usd + cobrado_rta_div_usd - cost_total_usd - gastos_usd

        saldo_tenencia = qty_bought - qty_sold

        rows.append(
            {
                "Ticker": ticker,
                "Cant_Comprada": qty_bought,
                "Cant_Vendida": qty_sold,
                "Saldo_Tenencia": saldo_tenencia,
                "Costo_Total_USD": cost_total_usd,
                "Cobrado_RTA_DIV_USD": cobrado_rta_div_usd,
                "Ventas_USD": ventas_usd,
                "Gastos_USD": gastos_usd,
                "Rendimiento_Neto_USD": rendimiento_neto_usd,
            }
        )

    by_ticker = pd.DataFrame(rows)

    # resumen financiero consolidado
    capital_ingresado_usd = df.loc[
        (df["Comprobante"] == "RECIBO") & (df["USD_Flow"] > 0),
        "USD_Flow",
    ].sum()

    capital_ingresado_ars = df.loc[
        (df["Comprobante"] == "RECIBO") & (df["ARS_Flow"] > 0),
        "ARS_Flow",
    ].sum()

    total_compras_usd = by_ticker["Costo_Total_USD"].sum()
    total_ventas_usd = by_ticker["Ventas_USD"].sum()
    total_cobrado_usd = by_ticker["Cobrado_RTA_DIV_USD"].sum()
    total_gastos_usd = by_ticker["Gastos_USD"].sum()

    gastos_custodia_ars = abs(
        df.loc[
            (df["Comprobante"] == "GS CUSTODIA") & (df["ARS_Flow"] < 0),
            "ARS_Flow",
        ].sum()
    )

    # cash flow puro en USD, sin ARS
    rendimiento_neto_cash_usd = capital_ingresado_usd - total_compras_usd + total_ventas_usd + total_cobrado_usd - total_gastos_usd
    pct_sobre_capital_usd = (rendimiento_neto_cash_usd / capital_ingresado_usd * 100) if capital_ingresado_usd else 0.0

    resumen_financiero = pd.DataFrame(
        [
            {"Concepto": "Capital ingresado (USD)", "Valor": capital_ingresado_usd},
            {"Concepto": "Capital ingresado (ARS)", "Valor": capital_ingresado_ars},
            {"Concepto": "Total compras (USD)", "Valor": total_compras_usd},
            {"Concepto": "Total ventas (USD)", "Valor": total_ventas_usd},
            {"Concepto": "Cobrado en RTA/Dividendos (USD)", "Valor": total_cobrado_usd},
            {"Concepto": "Gastos custodia (ARS)", "Valor": gastos_custodia_ars},
            {"Concepto": "Gastos PAU$ / Débitos (USD)", "Valor": total_gastos_usd},
            {"Concepto": "Rendimiento neto USD (cash flow)", "Valor": rendimiento_neto_cash_usd},
            {"Concepto": "% sobre capital USD", "Valor": pct_sobre_capital_usd},
        ]
    )

    return by_ticker, resumen_financiero


def _parse_uploaded_activity(file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    raw = _read_excel_robust(file)
    meta = _extract_metadata_from_activity(raw, file.name)
    detail = _parse_activity_detail(raw)
    summary_ticker, summary_fin = _build_summary_from_activity(detail)

    comitente = meta["comitente"]

    detail["Comitente"] = comitente
    summary_ticker["Comitente"] = comitente
    summary_fin["Comitente"] = comitente

    return detail, summary_ticker, summary_fin, meta


def _styled_asset_table(df: pd.DataFrame):
    cols = [
        "Comitente",
        "Ticker",
        "Cant_Comprada",
        "Cant_Vendida",
        "Saldo_Tenencia",
        "Costo_Total_USD",
        "Cobrado_RTA_DIV_USD",
        "Ventas_USD",
        "Gastos_USD",
        "Rendimiento_Neto_USD",
    ]
    tmp = df[cols].copy()

    return (
        tmp.style.format(
            {
                "Cant_Comprada": "{:,.0f}",
                "Cant_Vendida": "{:,.0f}",
                "Saldo_Tenencia": "{:,.0f}",
                "Costo_Total_USD": "USD {:,.2f}",
                "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                "Ventas_USD": "USD {:,.2f}",
                "Gastos_USD": "USD {:,.2f}",
                "Rendimiento_Neto_USD": "USD {:,.2f}",
            }
        )
    )


def _build_export_excel(
    movimientos: pd.DataFrame,
    resumen_tickers: pd.DataFrame,
    resumen_financiero: pd.DataFrame,
    by_asset: pd.DataFrame,
    by_comitente: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        movimientos.to_excel(writer, sheet_name="Movimientos", index=False)
        resumen_tickers.to_excel(writer, sheet_name="Resumen Tickers", index=False)
        resumen_financiero.to_excel(writer, sheet_name="Resumen Financiero", index=False)
        by_asset.to_excel(writer, sheet_name="Por Activo", index=False)
        by_comitente.to_excel(writer, sheet_name="Por Comitente", index=False)

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
                Lectura consolidada de archivos Activity por comitente, con foco en posición, cash performance
                en USD, carry cobrado y apertura ejecutiva por activo y por cuenta.
            </div>
            <div class="neix-pill-row">
                <div class="neix-pill">Activity Ready</div>
                <div class="neix-pill">Minimalista</div>
                <div class="neix-pill">Ejecutivo</div>
                <div class="neix-pill">Multi-comitente</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.2, 0.8], gap="large")

    with top_left:
        uploaded_files = st.file_uploader(
            "Subí los 3 archivos Activity de comitentes",
            type=["xls", "xlsx"],
            accept_multiple_files=True,
            help="Este módulo ahora está hecho para los Activity crudos, no para el reporte resumido.",
        )

    with top_right:
        st.markdown('<div class="neix-card">', unsafe_allow_html=True)
        st.markdown('<div class="neix-section-title">Cómo interpreta el archivo</div>', unsafe_allow_html=True)
        st.caption(
            "• Lee el bloque Activity crudo\n"
            "\n• Reconstruye compras, ventas, cupones y gastos"
            "\n\n• Calcula posición por ticker"
            "\n\n• Consolida por comitente"
            "\n\n• KPI principal: cash performance en USD"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_files:
        st.info("Subí los archivos para visualizar el dashboard.")
        return

    movimientos_frames: List[pd.DataFrame] = []
    resumen_ticker_frames: List[pd.DataFrame] = []
    resumen_fin_frames: List[pd.DataFrame] = []
    metas: List[Dict[str, str]] = []
    parse_errors: List[str] = []

    for file in uploaded_files:
        try:
            movimientos_df, resumen_ticker_df, resumen_fin_df, meta = _parse_uploaded_activity(file)
            movimientos_frames.append(movimientos_df)
            resumen_ticker_frames.append(resumen_ticker_df)
            resumen_fin_frames.append(resumen_fin_df)
            metas.append(meta)
        except Exception as e:
            parse_errors.append(f"{file.name}: {e}")

    if parse_errors:
        st.error("No pude leer correctamente algunos archivos:")
        for err in parse_errors:
            st.write(f"• {err}")

    if not resumen_ticker_frames:
        st.warning("No se pudo construir información útil con los archivos cargados.")
        return

    movimientos_all = pd.concat(movimientos_frames, ignore_index=True)
    resumen_tickers_all = pd.concat(resumen_ticker_frames, ignore_index=True)
    resumen_fin_all = pd.concat(resumen_fin_frames, ignore_index=True)

    # filtros
    f1, f2, f3 = st.columns([1, 1, 1], gap="medium")
    with f1:
        comitentes = sorted(resumen_tickers_all["Comitente"].dropna().astype(str).unique().tolist())
        selected_comitentes = st.multiselect("Comitentes", options=comitentes, default=comitentes)

    with f2:
        only_open_positions = st.toggle("Solo posiciones abiertas", value=False)

    with f3:
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

    resumen_tickers = resumen_tickers_all[resumen_tickers_all["Comitente"].isin(selected_comitentes)].copy()
    resumen_fin = resumen_fin_all[resumen_fin_all["Comitente"].isin(selected_comitentes)].copy()
    movimientos = movimientos_all[movimientos_all["Comitente"].isin(selected_comitentes)].copy()

    if only_open_positions:
        resumen_tickers = resumen_tickers[resumen_tickers["Saldo_Tenencia"] > 0].copy()

    if resumen_tickers.empty:
        st.warning("Con los filtros aplicados no hay información para mostrar.")
        return

    # KPIs principales
    capital_ingresado_usd = resumen_fin.loc[resumen_fin["Concepto"] == "Capital ingresado (USD)", "Valor"].sum()
    total_compras_usd = resumen_fin.loc[resumen_fin["Concepto"] == "Total compras (USD)", "Valor"].sum()
    total_ventas_usd = resumen_fin.loc[resumen_fin["Concepto"] == "Total ventas (USD)", "Valor"].sum()
    total_cobrado_usd = resumen_fin.loc[resumen_fin["Concepto"] == "Cobrado en RTA/Dividendos (USD)", "Valor"].sum()
    total_gastos_usd = resumen_fin.loc[resumen_fin["Concepto"] == "Gastos PAU$ / Débitos (USD)", "Valor"].sum()
    rendimiento_cash_usd = resumen_fin.loc[resumen_fin["Concepto"] == "Rendimiento neto USD (cash flow)", "Valor"].sum()
    rendimiento_pct = (rendimiento_cash_usd / capital_ingresado_usd * 100) if capital_ingresado_usd else 0.0
    posiciones_abiertas = int((resumen_tickers["Saldo_Tenencia"] > 0).sum())

    k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

    k1.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Capital ingresado</div>
            <div class="neix-kpi-value">{_fmt_usd(capital_ingresado_usd)}</div>
            <div class="{_delta_class(capital_ingresado_usd)}">Base fondeada</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k2.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Compras USD</div>
            <div class="neix-kpi-value">{_fmt_usd(total_compras_usd)}</div>
            <div class="{_delta_class(-total_compras_usd)}">Consumo de capital</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k3.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Ventas USD</div>
            <div class="neix-kpi-value">{_fmt_usd(total_ventas_usd)}</div>
            <div class="{_delta_class(total_ventas_usd)}">Realizado</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k4.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Cobrado RTA / DIV</div>
            <div class="neix-kpi-value">{_fmt_usd(total_cobrado_usd)}</div>
            <div class="{_delta_class(total_cobrado_usd)}">Carry cobrado</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k5.markdown(
        f"""
        <div class="neix-kpi">
            <div class="neix-kpi-label">Rendimiento neto</div>
            <div class="neix-kpi-value">{_fmt_usd(rendimiento_cash_usd)}</div>
            <div class="{_delta_class(rendimiento_cash_usd)}">{_fmt_pct(rendimiento_pct)} · {posiciones_abiertas} abiertas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # tablas consolidadas
    by_asset = (
        resumen_tickers.groupby("Ticker", as_index=False)
        .agg(
            Cant_Comprada=("Cant_Comprada", "sum"),
            Cant_Vendida=("Cant_Vendida", "sum"),
            Saldo_Tenencia=("Saldo_Tenencia", "sum"),
            Costo_Total_USD=("Costo_Total_USD", "sum"),
            Cobrado_RTA_DIV_USD=("Cobrado_RTA_DIV_USD", "sum"),
            Ventas_USD=("Ventas_USD", "sum"),
            Gastos_USD=("Gastos_USD", "sum"),
            Rendimiento_Neto_USD=("Rendimiento_Neto_USD", "sum"),
        )
        .sort_values(
            sort_metric,
            ascending=True if sort_metric in {"Rendimiento_Neto_USD", "Costo_Total_USD"} else False,
        )
        .reset_index(drop=True)
    )

    by_comitente = (
        resumen_tickers.groupby("Comitente", as_index=False)
        .agg(
            Posiciones_Abiertas=("Saldo_Tenencia", lambda s: int((s > 0).sum())),
            Cantidad_Activos=("Ticker", "nunique"),
            Costo_Total_USD=("Costo_Total_USD", "sum"),
            Cobrado_RTA_DIV_USD=("Cobrado_RTA_DIV_USD", "sum"),
            Ventas_USD=("Ventas_USD", "sum"),
            Gastos_USD=("Gastos_USD", "sum"),
            Rendimiento_Neto_USD=("Rendimiento_Neto_USD", "sum"),
        )
        .sort_values("Rendimiento_Neto_USD")
        .reset_index(drop=True)
    )

    current_positions = resumen_tickers[resumen_tickers["Saldo_Tenencia"] > 0].copy().sort_values("Rendimiento_Neto_USD")

    # gráficos
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
        height=max(320, 72 * len(by_comitente)),
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
        position_mix = by_asset.nlargest(min(10, len(by_asset)), "Saldo_Tenencia").copy()
        position_mix["Comitente"] = "Consolidado"

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

    # metadata visual
    if metas:
        periods = []
        for m in metas:
            txt = f"Comitente {m.get('comitente','')}"
            if m.get("fecha_desde") or m.get("fecha_hasta"):
                txt += f" · {m.get('fecha_desde','')} → {m.get('fecha_hasta','')}"
            periods.append(txt)
        st.caption(" | ".join(periods))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Resumen Ejecutivo", "Posición Actual", "Rendimiento", "Por Comitente", "Movimientos", "Resumen Financiero"]
    )

    with tab1:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown('<div class="neix-section-title">Lectura ejecutiva</div>', unsafe_allow_html=True)
            st.markdown('<div class="neix-card">', unsafe_allow_html=True)

            worst_asset = by_asset.nsmallest(1, "Rendimiento_Neto_USD")
            best_asset = by_asset.nlargest(1, "Rendimiento_Neto_USD")
            worst_comitente = by_comitente.nsmallest(1, "Rendimiento_Neto_USD")
            best_comitente = by_comitente.nlargest(1, "Rendimiento_Neto_USD")

            st.markdown(
                f"""
                **Vista consolidada**

                El cash performance consolidado asciende a **{_fmt_usd(rendimiento_cash_usd)}**
                sobre un capital ingresado de **{_fmt_usd(capital_ingresado_usd)}**, equivalente a
                **{_fmt_pct(rendimiento_pct)}** sobre capital.

                **Activo con peor contribución:** {worst_asset.iloc[0]['Ticker']} ({_fmt_usd(worst_asset.iloc[0]['Rendimiento_Neto_USD'])})  
                **Activo con mejor contribución:** {best_asset.iloc[0]['Ticker']} ({_fmt_usd(best_asset.iloc[0]['Rendimiento_Neto_USD'])})

                **Comitente con peor resultado:** {worst_comitente.iloc[0]['Comitente']} ({_fmt_usd(worst_comitente.iloc[0]['Rendimiento_Neto_USD'])})  
                **Comitente con mejor resultado:** {best_comitente.iloc[0]['Comitente']} ({_fmt_usd(best_comitente.iloc[0]['Rendimiento_Neto_USD'])})
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

            resumen_display = pd.DataFrame(
                {
                    "Concepto": [
                        "Capital ingresado",
                        "Compras USD",
                        "Ventas USD",
                        "Cobrado RTA / DIV",
                        "Gastos USD",
                        "Rendimiento neto",
                        "% sobre capital",
                    ],
                    "Valor": [
                        _fmt_usd(capital_ingresado_usd),
                        _fmt_usd(total_compras_usd),
                        _fmt_usd(total_ventas_usd),
                        _fmt_usd(total_cobrado_usd),
                        _fmt_usd(total_gastos_usd),
                        _fmt_usd(rendimiento_cash_usd),
                        _fmt_pct(rendimiento_pct),
                    ],
                }
            )

            st.markdown('<div class="neix-section-title">Resumen consolidado</div>', unsafe_allow_html=True)
            st.dataframe(resumen_display, use_container_width=True, hide_index=True)

        with right:
            st.plotly_chart(chart_by_asset, use_container_width=True)

    with tab2:
        left, right = st.columns([1.02, 0.98], gap="large")

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
                        "Gastos_USD",
                        "Rendimiento_Neto_USD",
                    ]
                ].copy()

                st.dataframe(
                    pos_show.style.format(
                        {
                            "Saldo_Tenencia": "{:,.0f}",
                            "Costo_Total_USD": "USD {:,.2f}",
                            "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                            "Gastos_USD": "USD {:,.2f}",
                            "Rendimiento_Neto_USD": "USD {:,.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        with right:
            st.plotly_chart(chart_positions, use_container_width=True)

    with tab3:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="neix-section-title">Ranking por activo</div>', unsafe_allow_html=True)
            st.dataframe(_styled_asset_table(resumen_tickers), use_container_width=True, hide_index=True)

        with right:
            st.plotly_chart(chart_by_asset, use_container_width=True)

    with tab4:
        left, right = st.columns([0.96, 1.04], gap="large")

        with left:
            st.markdown('<div class="neix-section-title">Resumen por comitente</div>', unsafe_allow_html=True)
            st.dataframe(
                by_comitente.style.format(
                    {
                        "Costo_Total_USD": "USD {:,.2f}",
                        "Cobrado_RTA_DIV_USD": "USD {:,.2f}",
                        "Ventas_USD": "USD {:,.2f}",
                        "Gastos_USD": "USD {:,.2f}",
                        "Rendimiento_Neto_USD": "USD {:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.plotly_chart(chart_by_comitente, use_container_width=True)

    with tab5:
        st.markdown('<div class="neix-section-title">Movimientos crudos</div>', unsafe_allow_html=True)

        move_show = movimientos[
            [
                "Comitente",
                "Fecha de Emisión",
                "Fecha Liquidación",
                "Comprobante",
                "Nro. de Comprobante",
                "Ticker",
                "Cantidad",
                "Precio",
                "Moneda",
                "Importe Pesos",
                "Importe En Moneda",
            ]
        ].copy()

        st.dataframe(move_show, use_container_width=True, hide_index=True)

    with tab6:
        st.markdown('<div class="neix-section-title">Resumen financiero por comitente</div>', unsafe_allow_html=True)
        fin_show = resumen_fin.copy()
        fin_show["Valor"] = fin_show.apply(
            lambda row: _fmt_pct(row["Valor"]) if row["Concepto"] == "% sobre capital USD" else f"{row['Valor']:,.2f}",
            axis=1,
        )
        st.dataframe(fin_show, use_container_width=True, hide_index=True)

    export_bytes = _build_export_excel(
        movimientos=movimientos,
        resumen_tickers=resumen_tickers,
        resumen_financiero=resumen_fin,
        by_asset=by_asset,
        by_comitente=by_comitente,
    )

    st.markdown("")
    c1, c2 = st.columns([0.72, 0.28], gap="large")
    with c1:
        st.caption(
            "Este módulo está preparado para leer Activity crudo y reconstruir una vista ejecutiva consolidada. "
            "La posición y el rendimiento se estiman a partir de compras, ventas, rentas/amortizaciones y gastos en USD."
        )
    with c2:
        st.download_button(
            "Descargar consolidado Excel",
            data=export_bytes,
            file_name="consolidado_rendimiento_cartera.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# =========================================================
# STANDALONE
# =========================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="NEIX · Rendimiento de Cartera",
        page_icon="📊",
        layout="wide",
    )
    render()
