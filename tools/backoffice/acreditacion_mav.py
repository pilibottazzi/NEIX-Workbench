from __future__ import annotations

import io
import re
import datetime as dt

import pandas as pd
import streamlit as st


# =========================
# Normalizadores
# =========================
def _norm_header(s: str) -> str:
    s = str(s or "").strip().lower()
    try:
        s = s.normalize("NFD")
    except Exception:
        pass
    s = re.sub(r"[\u0300-\u036f]", "", s)  # saca acentos
    s = s.replace("ï¿½", "o")              # encoding raro
    s = re.sub(r"[^a-z0-9]", "", s)        # deja alfanum
    return s


def _safe_str(v) -> str:
    """String safe: NaN/None/float -> '' / str(value)."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    if pd.isna(v):
        return ""
    return str(v)


def find_col(headers, candidates) -> int:
    H = [_norm_header(h) for h in headers]

    # match exacto normalizado
    for cand in candidates:
        c = _norm_header(cand)
        if c in H:
            return H.index(c)

    # match “parecido”
    for i, h in enumerate(H):
        for cand in candidates:
            c = _norm_header(cand)
            if not c:
                continue
            if h.startswith(c) or c.startswith(h) or (c in h) or (h in c):
                return i

    return -1


# =========================
# Parsers NUM (clave)
# =========================
def parse_qty_int(v) -> int | None:
    """
    Cantidad/nominal (CHEQUES):
    En tu CSV viene como miles con coma: -12,500 => -12500
    Regla: para qty, coma y punto son separadores -> se eliminan.
    """
    s = _safe_str(v).strip()
    if not s or s == "-":
        return None

    sign = -1 if s.startswith("-") else 1

    s = re.sub(r"[^\d\.,\-]", "", s)
    s = s.replace("-", "")
    s = s.replace(".", "").replace(",", "").strip()
    if not s:
        return None

    try:
        return sign * int(s)
    except Exception:
        return None


def parse_monto_float(v) -> float | None:
    """
    Monto de liquidación:
    Puede venir 1.807.030,50 o 1807030.50 o 1,807,030.50.
    Regla: detecta decimal por el separador final.
    """
    s = _safe_str(v).strip()
    if not s or s == "-":
        return None

    s = s.replace(" ", "")

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") >= 2:
            s = s.replace(".", "")

    s = re.sub(r"[^\d\.\-]", "", s)

    try:
        return float(s)
    except Exception:
        return None


def parse_date_cell(v):
    """
    Soporta:
    - dd/mm/yyyy
    - dd-mm-yyyy
    - yyyy-mm-dd
    - con hora al final
    Devuelve datetime.date o None
    """
    s = _safe_str(v).strip()
    if not s:
        return None

    s = s.split(" ")[0]

    m1 = re.match(r"^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})$", s)
    m2 = re.match(r"^(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})$", s)

    try:
        if m1:
            d = int(m1.group(1))
            m = int(m1.group(2))
            y = m1.group(3)
            y = int(("20" + y) if len(y) == 2 else y)
            return dt.date(y, m, d)
        if m2:
            y = int(m2.group(1))
            m = int(m2.group(2))
            d = int(m2.group(3))
            return dt.date(y, m, d)
    except Exception:
        return None

    return None


# =========================
# CSV reader (latin1 + sep auto)
# =========================
def read_csv_auto(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("latin1", errors="replace")

    first = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first.count(";") > first.count(",") else ","

    return pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str)


# =========================
# UI
# =========================
def render(back_to_home=None):
    st.markdown("## Acreditación MAV")
    st.caption("CPD / PAGARES / FCE MAV")

    c1, c2, c3 = st.columns([0.38, 0.38, 0.24])
    with c1:
        nasdaq_file = st.file_uploader("Cheques NASDAQ (CSV)", type=["csv"])
    with c2:
        cpd_file = st.file_uploader("CPD Instrumentos Listado (CSV)", type=["csv"])
    with c3:
        st.markdown("&nbsp;")
        do = st.button("Procesar", use_container_width=True)

    if not do:
        return

    if not nasdaq_file:
        st.error("Por favor seleccioná Cheques NASDAQ (CSV)")
        return
    if not cpd_file:
        st.error("Por favor seleccioná CPD Instrumentos Listado (CSV)")
        return

    with st.spinner("Procesando archivos..."):
        # ======================
        # 1) NASDAQ
        # ======================
        df_nasdaq = read_csv_auto(nasdaq_file)
        headers = list(df_nasdaq.columns)

        qty_idx = find_col(headers, ["Cantidad/nominal", "CantidadNominal", "Cantidad"])
        monto_idx = find_col(headers, [
            "Monto de liquidación",
            "Monto de liquidacion",
            "Monto de liquidaci",
            "montodeliquidac",
            "Monto de liquidaciï¿½n",
            "montodeliquidaciï¿½n",
        ])

        if qty_idx == -1:
            st.error("✗ No encontré la columna Cantidad/nominal en Cheques NASDAQ.")
            return
        if monto_idx == -1:
            st.error("✗ No pude identificar la columna 'Monto de liquidación' (encoding raro).")
            return

        qty_col = headers[qty_idx]
        monto_col = headers[monto_idx]

        kept_mask = []
        invalid = 0
        dropped = 0

        for v in df_nasdaq[qty_col].tolist():
            n = parse_qty_int(v)
            if n is None:
                invalid += 1
                kept_mask.append(False)
                continue
            if n <= 0:
                kept_mask.append(True)
            else:
                kept_mask.append(False)
                dropped += 1

        df_cheques = df_nasdaq.loc[kept_mask].copy()

        df_cheques[qty_col] = df_cheques[qty_col].apply(parse_qty_int)
        df_cheques[monto_col] = df_cheques[monto_col].apply(parse_monto_float)

        # formateo fechas si existen (por nombre)
        for col in df_cheques.columns:
            if "fecha" in _norm_header(col):
                df_cheques[col] = df_cheques[col].apply(parse_date_cell)

        # ======================
        # 2) PARA GALLO (idéntico)
        # ======================
        baseCols = [
            "Instrumento",
            "Cuenta de valores negociables",
            "Cantidad/nominal",
            "Monto de liquidacion",
            "Moneda",
            "Fecha efectiva de liquidacion",
        ]
        baseIdxs = [find_col(headers, [c]) for c in baseCols]

        galloCols = [
            "Tipo Instrumento",
            "COD.INSTRUMENTO",
            "Referencia",
            "Cuenta de valores negociables",
            "Monto de liquidacion",
            "Cantidad/nominal",
            "Moneda",
            "Fecha efectiva de liquidacion",
        ]

        gallo_rows = []
        for _, row in df_cheques.iterrows():
            instrumento = _safe_str(row.iloc[baseIdxs[0]] if baseIdxs[0] != -1 else "").strip()
            cuenta = _safe_str(row.iloc[baseIdxs[1]] if baseIdxs[1] != -1 else "").strip().replace("5992/", "")
            cantidad = row.iloc[baseIdxs[2]] if baseIdxs[2] != -1 else ""
            monto = row.iloc[baseIdxs[3]] if baseIdxs[3] != -1 else ""
            moneda = _safe_str(row.iloc[baseIdxs[4]] if baseIdxs[4] != -1 else "").strip()
            fecha = row.iloc[baseIdxs[5]] if baseIdxs[5] != -1 else ""

            gallo_rows.append([
                "",
                "",
                f"ACRED {instrumento}",
                cuenta,
                monto,
                cantidad,
                moneda,
                fecha,
            ])

        df_gallo = pd.DataFrame(gallo_rows, columns=galloCols)

        if "Cantidad/nominal" in df_gallo.columns:
            df_gallo["Cantidad/nominal"] = df_gallo["Cantidad/nominal"].apply(parse_qty_int)
        if "Monto de liquidacion" in df_gallo.columns:
            df_gallo["Monto de liquidacion"] = df_gallo["Monto de liquidacion"].apply(parse_monto_float)
        if "Fecha efectiva de liquidacion" in df_gallo.columns:
            df_gallo["Fecha efectiva de liquidacion"] = df_gallo["Fecha efectiva de liquidacion"].apply(parse_date_cell)

        # Agregar "Cheque Nro" después de Referencia
        if "Referencia" in df_gallo.columns:
            idx_ref = df_gallo.columns.get_loc("Referencia")
            cheque_num = (
                df_gallo["Referencia"]
                .astype(str)
                .str.extract(r"(\d{5})\D*$", expand=False)
                .fillna("")
            )
            df_gallo.insert(idx_ref + 1, "Cheque Nro", cheque_num)

        # ======================
        # 3) Merge con CPD (FIX NaN/float)
        # ======================
        df_cpd = read_csv_auto(cpd_file)
        if len(df_cpd) > 0 and df_cpd.shape[1] >= 19:
            # JS: codInstr = fila[1], nroCheque = fila[18]
            cpd_map = {}
            for _, fila in df_cpd.iterrows():
                codInstr = _safe_str(fila.iloc[1] if df_cpd.shape[1] > 1 else "").strip()
                nroCheque = _safe_str(fila.iloc[18] if df_cpd.shape[1] > 18 else "").strip()
                if codInstr:
                    cpd_map[codInstr] = nroCheque

            # rel_map: instrumento en cheques -> nroCheque
            instr_idx_cheques = find_col(headers, ["Instrumento"])
            rel_map = {}
            if instr_idx_cheques != -1:
                for _, rr in df_cheques.iterrows():
                    instr = _safe_str(rr.iloc[instr_idx_cheques]).strip()
                    if instr and instr in cpd_map:
                        rel_map[instr] = cpd_map[instr]

            # set COD.INSTRUMENTO según Referencia sin "ACRED "
            if "COD.INSTRUMENTO" in df_gallo.columns and "Referencia" in df_gallo.columns:
                for i in range(len(df_gallo)):
                    ref_val = _safe_str(df_gallo.at[i, "Referencia"])
                    ref_clean = re.sub(r"^ACRED\s+", "", ref_val, flags=re.IGNORECASE).strip()
                    if ref_clean in rel_map:
                        df_gallo.at[i, "COD.INSTRUMENTO"] = rel_map[ref_clean]

        # ======================
        # 4) PARA TEAMS (FIX)
        # ======================
        teamsColsNames = [
            "Instrumento",
            "Cuenta de valores negociables",
            "Monto de liquidación",
            "Moneda",
            "Fecha efectiva de liquidación",
        ]
        teamsIdxs = [find_col(headers, [c]) for c in teamsColsNames]

        if any(i == -1 for i in teamsIdxs):
            faltan = [teamsColsNames[i] for i, idx in enumerate(teamsIdxs) if idx == -1]
            st.warning(f"PARA TEAMS: no encontré columnas: {faltan}. Reviso headers del CSV.")
            df_teams = pd.DataFrame(columns=teamsColsNames)
        else:
            teams_rows = []
            for _, rr in df_cheques.iterrows():
                monto_raw = rr.iloc[teamsIdxs[2]]
                monto_num = parse_monto_float(monto_raw)

                # JS: incluir si monto != 0
                if monto_num is None or monto_num == 0:
                    continue

                instrumento = _safe_str(rr.iloc[teamsIdxs[0]]).strip()
                cuenta = _safe_str(rr.iloc[teamsIdxs[1]]).strip().replace("5992/", "")
                moneda = _safe_str(rr.iloc[teamsIdxs[3]]).strip()
                fecha = parse_date_cell(rr.iloc[teamsIdxs[4]])

                teams_rows.append([instrumento, cuenta, monto_num, moneda, fecha])

            df_teams = pd.DataFrame(teams_rows, columns=teamsColsNames)
            if len(df_teams) > 0:
                df_teams = df_teams.sort_values("Cuenta de valores negociables", kind="stable")

        # ======================
        # 5) Export XLSX
        # ======================
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_cheques.to_excel(writer, sheet_name="Cheques", index=False)
            df_gallo.to_excel(writer, sheet_name="PARA GALLO", index=False)
            df_teams.to_excel(writer, sheet_name="PARA TEAMS", index=False)
        out.seek(0)

    today = dt.date.today().strftime("%d-%m-%Y")

    st.success("✓ Procesamiento completado exitosamente")
    st.info(
        f"Filas NASDAQ luego de filtro (qty<=0): **{len(df_cheques)}** · "
        f"Descartadas qty>0: **{dropped}** · Invalid qty: **{invalid}**"
    )

    st.download_button(
        "📥 Descargar Excel",
        data=out,
        file_name=f"Cheques vencidos acreditados por CV {today}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
