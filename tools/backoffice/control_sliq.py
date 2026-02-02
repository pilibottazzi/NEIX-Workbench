# tools/control_sliq.py
import io
import re
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


# =========================
# Helpers
# =========================
def _clean_str(x) -> str:
    return "" if x is None else str(x).strip()


def _to_num_es(v) -> Optional[float]:
    """
    Convierte número estilo ES:
      "1.234,56" -> 1234.56
      "1000" -> 1000.0
    """
    s = _clean_str(v)
    if not s:
        return None

    # dejar dígitos, coma, punto, signo
    s = re.sub(r"[^\d,.\-]", "", s)

    if "," in s and "." in s:
        # miles con '.', decimal con ','
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        # decimal con coma
        s = s.replace(",", ".")

    try:
        n = float(s)
        return n
    except Exception:
        return None


def _read_csv_bytes(file, sep: str) -> pd.DataFrame:
    """
    Lee CSV desde st.file_uploader (bytes). Prueba utf-8 y fallback cp1252.
    """
    raw = file.getvalue()

    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, encoding=enc)
        except Exception:
            continue

    # último intento: deja que pandas infiera
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# =========================
# Core logic
# =========================
def _build_nasdaq_detalle(df_nas: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Replica tu lógica JS:
    - Filtra ref que NO empieza con "SLIQ-"
    - Cuenta == "7142/10000"
    - Estado != "CANCELADO"
    Devuelve:
      - detalle (df) con columnas:
        Instrumento, Referencia instrucción, Cantidad/nominal, Cuenta..., Estado...
      - sumByInst: {instrumento_int: suma_cantidad_int}
    """
    df = _normalize_cols(df_nas)

    # columnas exactas esperadas
    wanted = [
        "Instrumento",
        "Referencia instrucción",
        "Cantidad/nominal",
        "Cuenta de valores negociables",
        "Estado de instrucción",
    ]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"NASDAQ: faltan columnas {missing}. Columnas disponibles: {list(df.columns)}")

    # preparar
    ref = df["Referencia instrucción"].fillna("").astype(str).str.strip().str.upper()
    cuenta = df["Cuenta de valores negociables"].fillna("").astype(str).str.strip()
    estado = df["Estado de instrucción"].fillna("").astype(str).str.strip().str.upper()

    mask = (~ref.str.startswith("SLIQ-")) & (cuenta == "7142/10000") & (estado != "CANCELADO")
    df_f = df.loc[mask, wanted].copy()

    # Instrumento y Cantidad como num ES
    df_f["Instrumento_num"] = df_f["Instrumento"].apply(_to_num_es)
    df_f["Cantidad_num"] = df_f["Cantidad/nominal"].apply(_to_num_es)

    # JS hacía round() a enteros
    df_f["Instrumento_int"] = df_f["Instrumento_num"].apply(lambda x: int(round(x)) if x is not None else None)
    df_f["Cantidad_int"] = df_f["Cantidad_num"].apply(lambda x: int(round(x)) if x is not None else None)

    # detalle output
    out = pd.DataFrame({
        "Instrumento": df_f["Instrumento_int"].fillna("").astype(object),
        "Referencia instrucción": df_f["Referencia instrucción"].fillna("").astype(str),
        "Cantidad/nominal": df_f["Cantidad_int"].fillna("").astype(object),
        "Cuenta de valores negociables": df_f["Cuenta de valores negociables"].fillna("").astype(str),
        "Estado de instrucción": df_f["Estado de instrucción"].fillna("").astype(str),
    })

    # sumByInst
    sum_by = {}
    for _, r in df_f.iterrows():
        inst = r["Instrumento_int"]
        qty = r["Cantidad_int"]
        if inst is None or qty is None:
            continue
        sum_by[inst] = sum_by.get(inst, 0) + int(qty)

    return out, sum_by


def _build_sliq_raw(df_sliq: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Replica tu lógica JS SLIQ:
    - Toma columnas por posición: 0 Código, 1 Especie, 2 Denominacion, 3 Neto a Liquidar
    - Suma neto por código
    """
    df = _normalize_cols(df_sliq)

    # A veces viene sin headers, a veces con headers. Tu JS usa "row[0..3]" igual.
    # Entonces acá: si detectamos que tiene al menos 4 columnas, usamos por posición.
    if df.shape[1] < 4:
        raise ValueError("SLIQ: el CSV debe tener al menos 4 columnas (Código, Especie, Denominación, Neto).")

    col0, col1, col2, col3 = df.columns[:4]

    df_out = pd.DataFrame({
        "Código": df[col0],
        "Especie": df[col1],
        "Denominacion": df[col2],
        "Neto a Liquidar": df[col3],
    }).copy()

    df_out["Código_num"] = df_out["Código"].apply(_to_num_es)
    df_out["Neto_num"] = df_out["Neto a Liquidar"].apply(_to_num_es)

    # output raw (Código y Neto como num si se puede)
    raw = pd.DataFrame({
        "Código": df_out["Código_num"].apply(lambda x: int(round(x)) if x is not None else ""),
        "Especie": df_out["Especie"].fillna("").astype(str),
        "Denominacion": df_out["Denominacion"].fillna("").astype(str),
        "Neto a Liquidar": df_out["Neto_num"].apply(lambda x: float(x) if x is not None else ""),
    })

    # agrupado por código (no cero)
    by_code = {}
    for _, r in df_out.iterrows():
        cod = r["Código_num"]
        if cod is None:
            continue
        key = int(round(cod))
        especie = _clean_str(r["Especie"])
        denom = _clean_str(r["Denominacion"])
        neto = r["Neto_num"] if r["Neto_num"] is not None else 0.0

        prev = by_code.get(key, {"especie": "", "denom": "", "neto": 0.0})
        by_code[key] = {
            "especie": prev["especie"] or especie,
            "denom": prev["denom"] or denom,
            "neto": prev["neto"] + float(neto),
        }

    return raw, by_code


def _build_control(sum_by_inst: dict, sliq_by_code: dict) -> pd.DataFrame:
    """
    Hoja "Control SLIQ tarde":
    Instrumento/Código | Especie | Denominación | Q NASDAQ | Neto a Liquidar | Q SLIQ (fórmula) | Observación (fórmula)
    Excluye claves con neto=0 y qnas=0 como en tu JS.
    """
    nz_inst = {k: v for k, v in sum_by_inst.items() if v != 0}
    nz_sliq = {k: v for k, v in sliq_by_code.items() if float(v.get("neto", 0.0)) != 0.0}

    keys = set(nz_inst.keys()) | set(nz_sliq.keys())
    data = []
    for k in keys:
        qnas = nz_inst.get(k, 0)
        s = nz_sliq.get(k, {"especie": "", "denom": "", "neto": 0.0})
        data.append({
            "Instrumento/Código": k,
            "Especie": s.get("especie", ""),
            "Denominación": s.get("denom", ""),
            "Q NASDAQ": qnas,
            "Neto a Liquidar": float(s.get("neto", 0.0)),
            "Q SLIQ": "",         # la llenamos con fórmula al exportar
            "Observación": "",    # fórmula al exportar
        })

    # ordenar: primero los que revisar (qnas+neto != 0), luego por instrumento
    def revisar(row):
        return (row["Q NASDAQ"] + row["Neto a Liquidar"]) != 0

    data.sort(key=lambda r: (not revisar(r), r["Instrumento/Código"]))
    return pd.DataFrame(data)


def _export_excel(df_nasdaq: pd.DataFrame, df_control: pd.DataFrame, df_sliq: pd.DataFrame) -> bytes:
    """
    Exporta a Excel con fórmulas en Q SLIQ y Observación.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # hojas
        df_nasdaq.to_excel(writer, sheet_name="Nasdaq", index=False)
        df_control.to_excel(writer, sheet_name="Control SLIQ tarde", index=False)
        df_sliq.to_excel(writer, sheet_name="SLIQ", index=False)

        wb = writer.book
        ws_control = writer.sheets["Control SLIQ tarde"]

        # Formatos numéricos (similar a tu JS)
        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_2d  = wb.add_format({"num_format": "#,##0.00"})

        # Column widths (un poco más prolijo)
        ws_control.set_column("A:A", 18, fmt_int)
        ws_control.set_column("B:B", 14)
        ws_control.set_column("C:C", 30)
        ws_control.set_column("D:D", 12, fmt_int)
        ws_control.set_column("E:E", 16, fmt_2d)
        ws_control.set_column("F:F", 12, fmt_2d)
        ws_control.set_column("G:G", 14)

        # Fórmulas fila por fila (como tu JS: F = D+E ; G = IF(F=0,"OK","REVISAR"))
        # Excel: fila 2 es la primera de datos (fila 1 headers)
        nrows = len(df_control)
        for i in range(nrows):
            excel_row = i + 2
            ws_control.write_formula(i + 1, 5, f"=D{excel_row}+E{excel_row}")               # col F
            ws_control.write_formula(i + 1, 6, f'=IF(F{excel_row}=0,"OK","REVISAR")')       # col G

        # Nasdaq formats (Instrumento y Cantidad)
        ws_nas = writer.sheets["Nasdaq"]
        ws_nas.set_column("A:A", 14, fmt_int)
        ws_nas.set_column("B:B", 24)
        ws_nas.set_column("C:C", 16, fmt_int)
        ws_nas.set_column("D:D", 26)
        ws_nas.set_column("E:E", 20)

        # SLIQ formats (Código int y neto 2d)
        ws_sliq = writer.sheets["SLIQ"]
        ws_sliq.set_column("A:A", 12, fmt_int)
        ws_sliq.set_column("B:B", 14)
        ws_sliq.set_column("C:C", 30)
        ws_sliq.set_column("D:D", 16, fmt_2d)

    return output.getvalue()


# =========================
# Render (Workbench)
# =========================
def render(back_to_home=None):
    # IMPORTANTE: no llamar back_to_home() acá
    st.markdown("## ⚠️ Control SLIQ")
    st.caption("Subí NASDAQ (CSV con coma) y SLIQ (CSV con ;). Genera 'Control SLIQ tarde.xlsx'.")

    col1, col2 = st.columns(2)
    with col1:
        f_nasdaq = st.file_uploader("Instr. de Liquidación NASDAQ (.csv)", type=["csv"], key="sliq_nasdaq")
    with col2:
        f_sliq = st.file_uploader("Especies para un Participante (.csv)", type=["csv"], key="sliq_sliq")

    if not f_nasdaq or not f_sliq:
        st.info("Cargá ambos archivos para habilitar el botón.")
        return

    if not st.button('Generar "Control SLIQ"', type="primary", key="sliq_run"):
        return

    try:
        with st.spinner("Leyendo archivos…"):
            df_nas = _read_csv_bytes(f_nasdaq, sep=",")
            df_slq = _read_csv_bytes(f_sliq, sep=";")

        with st.spinner("Procesando NASDAQ…"):
            df_nas_out, sum_by_inst = _build_nasdaq_detalle(df_nas)

        with st.spinner("Procesando SLIQ…"):
            df_sliq_out, sliq_by_code = _build_sliq_raw(df_slq)

        with st.spinner("Armando Control…"):
            df_control = _build_control(sum_by_inst, sliq_by_code)

        xlsx_bytes = _export_excel(df_nas_out, df_control, df_sliq_out)

        st.success("Listo ✅ Se generó el Excel.")
        st.download_button(
            "Descargar Control SLIQ tarde.xlsx",
            data=xlsx_bytes,
            file_name="Control SLIQ tarde.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sliq_download",
        )

        # Preview chica
        st.divider()
        st.subheader("Preview (Control SLIQ tarde)")
        st.dataframe(df_control.head(50), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Error generando el Control SLIQ.")
        st.exception(e)

