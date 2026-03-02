# tools/comerciales/transactions_analyzer.py
# Versión simplificada: SOLO CASH_MOVEMENTS + TRADE
# Enfocada en entender flujo de caja correctamente.

from __future__ import annotations
import pandas as pd
import streamlit as st
import re
import datetime as dt


# =========================
# Helpers
# =========================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _find_header_row(df_raw: pd.DataFrame):
    """
    Busca la fila donde aparece 'Process Date'
    porque el Excel exportado NO empieza en la fila 1.
    """
    for i in range(min(len(df_raw), 200)):
        row = df_raw.iloc[i].astype(str).tolist()
        normed = [_norm(x) for x in row]
        if "processdate" in normed:
            return i
    return None


def _load_transactions(file) -> pd.DataFrame:
    df_raw = pd.read_excel(file, header=None)

    header_row = _find_header_row(df_raw)
    if header_row is None:
        st.error("No encontré la fila donde empieza la tabla (Process Date).")
        st.stop()

    headers = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row+1:].copy()
    df.columns = headers
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)

    return df


def _parse_dates(df: pd.DataFrame):
    for col in ["Process Date", "Settlement Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _parse_numbers(df: pd.DataFrame):
    if "Net Amount (Base Currency)" in df.columns:
        df["Net Amount (Base Currency)"] = (
            df["Net Amount (Base Currency)"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["Net Amount (Base Currency)"] = pd.to_numeric(
            df["Net Amount (Base Currency)"], errors="coerce"
        )
    return df


# =========================
# MAIN
# =========================
def render_transactions_analyzer():

    st.markdown("## 🧾 Movimientos CV — Cash & Trades")
    st.caption("Primero entendemos flujo de caja (Cash). Después vemos Trades.")

    file = st.file_uploader("Subí el Excel exportado (Transactions)", type=["xlsx","xls"])

    if not file:
        st.info("Subí el Excel para comenzar.")
        return

    df = _load_transactions(file)
    df = _parse_dates(df)
    df = _parse_numbers(df)

    tabs = st.tabs(["CASH_MOVEMENTS", "TRADE"])

    # =========================================
    # TAB 1: CASH_MOVEMENTS
    # =========================================
    with tabs[0]:

        st.subheader("Ingresos y egresos de dinero")

        # Excluir activity within
        cash_df = df.copy()
        cash_df = cash_df[
            ~cash_df["Transaction Type"].str.upper().eq("ACTIVITY WITHIN YOUR ACCT")
        ]

        # Solo movimientos en currency
        cash_df = cash_df[
            cash_df["Security Description"].str.upper().str.contains("CURRENCY", na=False)
        ]

        cash_df = cash_df.sort_values("Settlement Date")

        # Cuadrito limpio
        show_cols = [
            "Process Date",
            "Settlement Date",
            "Net Amount (Base Currency)",
            "Transaction Type",
            "Security Description",
        ]

        cash_df = cash_df[show_cols].reset_index(drop=True)

        # KPIs
        total = cash_df["Net Amount (Base Currency)"].sum()

        col1, col2 = st.columns(2)
        col1.metric("Movimientos netos (Cash)", f"{total:,.2f}")
        col2.metric("Cantidad movimientos", len(cash_df))

        st.divider()

        st.dataframe(
            cash_df,
            use_container_width=True,
            hide_index=True
        )

    # =========================================
    # TAB 2: TRADE
    # =========================================
    with tabs[1]:

        st.subheader("Compras y ventas de activos")

        trade_df = df.copy()

        # Detectamos trades por BUY / SELL
        trade_df = trade_df[
            trade_df["Buy/Sell"].isin(["BUY", "SELL"])
        ]

        trade_df = trade_df.sort_values("Settlement Date")

        st.metric("Cantidad trades", len(trade_df))

        st.dataframe(
            trade_df,
            use_container_width=True,
            hide_index=True
        )


# Wrapper obligatorio del Workbench
def render(_ctx=None):
    render_transactions_analyzer()
