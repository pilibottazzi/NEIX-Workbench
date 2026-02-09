from __future__ import annotations

import io
import re
import datetime as dt
import pandas as pd
import streamlit as st


# =========================
# Utils
# =========================
def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    try:
        s = s.normalize("NFD")
    except Exception:
        pass
    s = re.sub(r"[\u0300-\u036f]", "", s)
    s = s.replace("ï¿½", "o")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def find_col(headers, candidates):
    H = [normalize(h) for h in headers]

    for cand in candidates:
        c = normalize(cand)
        if c in H:
            return H.index(c)

    for i, h in enumerate(H):
        for cand in candidates:
            c = normalize(cand)
            if c and (h.startswith(c) or c.startswith(h) or c in h or h in c):
                return i
    return -1


def parse_number_smart(v):
    if v is None:
        return None
    s = str(v).strip().replace(" ", "")
    if not s:
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") >= 2:
            s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return None


def format_dates(df, colnames):
    for col in df.columns:
        for target in colnames:
            if normalize(target) in normalize(col):
                df[col] = pd.to_datetime(
                    df[col].astype(str).str.split(" ").str[0],
                    errors="coerce",
                    dayfirst=True
                ).dt.date
    return df


# =========================
# UI
# =========================
def render(back_to_home=None):
    st.markdown("## Acreditación MAV")
    st.caption("CPD / PAGARES / FCE MAV")

    col1, col2, col3 = st.columns(3)

    with col1:
        nasdaq_file = st.file_uploader("Cheques NASDAQ (CSV)", type=["csv"])

    with col2:
        cpd_file = st.file_uploader("CPD Instrumentos Listado (CSV)", type=["csv"])

    with col3:
        st.markdown("&nbsp;")
        process = st.button("Procesar")

    if not process:
        return

    if not nasdaq_file:
        st.error("Por favor seleccioná Cheques NASDAQ (CSV)")
        return

    if not cpd_file:
        st.error("Por favor seleccioná CPD Instrumentos Listado (CSV)")
        return

    with st.spinner("Procesando archivos..."):
        # === NASDAQ ===
        nasdaq = pd.read_csv(nasdaq_file, sep=None, engine="python", encoding="latin1")
        headers = list(nasdaq.columns)

        qty_idx = find_col(headers, [
            "Cantidad/nominal", "CantidadNominal", "Cantidad"
        ])
        monto_idx = find_col(headers, [
            "Monto de liquidación", "Monto de liquidacion", "montodeliquidacion"
        ])

        if qty_idx == -1:
            st.error("No encontré columna Cantidad/nominal en NASDAQ")
            return

        if monto_idx == -1:
            st.error("No encontré columna Monto de liquidación en NASDAQ")
            return

        kept_rows = []
        for _, r in nasdaq.iterrows():
            n = parse_number_smart(r.iloc[qty_idx])
            if n is None:
                continue
            if n <= 0:
                kept_rows.append(r)

        df_cheques = pd.DataFrame(kept_rows)
        df_cheques.iloc[:, qty_idx] = df_cheques.iloc[:, qty_idx].apply(parse_number_smart)
        df_cheques.iloc[:, monto_idx] = df_cheques.iloc[:, monto_idx].apply(parse_number_smart)

        df_cheques = format_dates(
            df_cheques,
            ["Fecha de liquidación prevista", "Fecha efectiva de liquidación"]
        )

        # === PARA GALLO ===
        gallo_cols = [
            "Tipo Instrumento",
            "COD.INSTRUMENTO",
            "Referencia",
            "Cuenta de valores negociables",
            "Monto de liquidacion",
            "Cantidad/nominal",
            "Moneda",
            "Fecha efectiva de liquidacion"
        ]

        gallo_rows = []

        base_idxs = {
            "instrumento": find_col(headers, ["Instrumento"]),
            "cuenta": find_col(headers, ["Cuenta de valores negociables"]),
            "cantidad": find_col(headers, ["Cantidad/nominal"]),
            "monto": find_col(headers, ["Monto de liquidacion"]),
            "moneda": find_col(headers, ["Moneda"]),
            "fecha": find_col(headers, ["Fecha efectiva de liquidacion"]),
        }

        for r in kept_rows:
            instrumento = str(r.iloc[base_idxs["instrumento"]]).strip()
            cuenta = str(r.iloc[base_idxs["cuenta"]]).replace("5992/", "").strip()
            gallo_rows.append([
                "",
                "",
                f"ACRED {instrumento}",
                cuenta,
                r.iloc[base_idxs["monto"]],
                r.iloc[base_idxs["cantidad"]],
                r.iloc[base_idxs["moneda"]],
                r.iloc[base_idxs["fecha"]],
            ])

        df_gallo = pd.DataFrame(gallo_rows, columns=gallo_cols)

        # Cheque Nro
        idx_ref = df_gallo.columns.get_loc("Referencia")
        df_gallo.insert(
            idx_ref + 1,
            "Cheque Nro",
            df_gallo["Referencia"].str.extract(r"(\d{5})$", expand=False).fillna("")
        )

        # === CPD MERGE ===
        cpd = pd.read_csv(cpd_file, sep=None, engine="python", encoding="latin1")
        cpd_map = dict(
            zip(cpd.iloc[:, 1].astype(str).str.strip(),
                cpd.iloc[:, 18].astype(str).str.strip())
        )

        for i, r in df_gallo.iterrows():
            ref = r["Referencia"].replace("ACRED ", "").strip()
            if ref in cpd_map:
                df_gallo.at[i, "COD.INSTRUMENTO"] = cpd_map[ref]

        # === PARA TEAMS ===
        teams_cols = [
            "Instrumento",
            "Cuenta de valores negociables",
            "Monto de liquidación",
            "Moneda",
            "Fecha efectiva de liquidación"
        ]

        teams_rows = []

        teams_idxs = {
            "instrumento": find_col(headers, ["Instrumento"]),
            "cuenta": find_col(headers, ["Cuenta de valores negociables"]),
            "monto": find_col(headers, ["Monto de liquidacion"]),
            "moneda": find_col(headers, ["Moneda"]),
            "fecha": find_col(headers, ["Fecha efectiva de liquidacion"]),
        }

        for r in kept_rows:
            monto = parse_number_smart(r.iloc[teams_idxs["monto"]])
            if monto and monto != 0:
                teams_rows.append([
                    r.iloc[teams_idxs["instrumento"]],
                    str(r.iloc[teams_idxs["cuenta"]]).replace("5992/", ""),
                    monto,
                    r.iloc[teams_idxs["moneda"]],
                    r.iloc[teams_idxs["fecha"]],
                ])

        df_teams = pd.DataFrame(teams_rows, columns=teams_cols)
        df_teams = df_teams.sort_values("Cuenta de valores negociables")

        # === EXPORT ===
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_cheques.to_excel(writer, sheet_name="Cheques", index=False)
            df_gallo.to_excel(writer, sheet_name="PARA GALLO", index=False)
            df_teams.to_excel(writer, sheet_name="PARA TEAMS", index=False)

        output.seek(0)

    today = dt.date.today().strftime("%d-%m-%Y")
    st.success("✓ Procesamiento completado exitosamente")

    st.download_button(
        "📥 Descargar Excel",
        data=output,
        file_name=f"Cheques vencidos acreditados por CV {today}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

