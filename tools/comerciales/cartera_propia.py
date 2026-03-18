from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONSTANTES DE ESTILO
# =========================================================
PRIMARY = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.08)"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
INFO = "#2563eb"


# =========================================================
# UI
# =========================================================
def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }}

          .cp-kpi {{
            background: #f9fafb;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 112px;
          }}

          .cp-kpi-label {{
            color: {MUTED};
            font-size: 0.82rem;
            margin-bottom: 0.3rem;
            letter-spacing: 0.02em;
          }}

          .cp-kpi-value {{
            color: {PRIMARY};
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.1;
          }}

          .cp-kpi-delta {{
            margin-top: 0.3rem;
            font-size: 0.83rem;
            font-weight: 600;
          }}

          .cp-section {{
            font-size: 0.92rem;
            font-weight: 700;
            color: {MUTED};
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 1.4rem 0 0.6rem;
          }}

          div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
          }}

          .stDownloadButton button {{
            border-radius: 10px;
            font-weight: 600;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS GENERALES
# =========================================================
def _fmt_money(v: float, decimals: int = 2) -> str:
    if pd.isna(v):
        return "—"
    return f"$ {v:,.{decimals}f}"


def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:+,.2f}%"


def _pct_change(fin: float, ini: float) -> float:
    if pd.isna(ini) or ini == 0:
        return np.nan
    return (fin / ini - 1) * 100


def _norm(x) -> str:
    return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x).strip()


def _to_ts(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).normalize()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    ts = pd.to_datetime(str(value).strip(), dayfirst=True, errors="coerce")
    return None if pd.isna(ts) else ts.normalize()


def _safe_num(d: Dict, key: str) -> float:
    return pd.to_numeric(d.get(key), errors="coerce")


def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    mask = (~values.isna()) & (~weights.isna()) & (weights != 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


# =========================================================
# DETECCIONES DE NEGOCIO
# =========================================================
def _detect_moneda(categoria: str) -> str:
    c = _norm(categoria).upper()
    if "U$S EXTERIOR" in c or "USD EXTERIOR" in c or "EXTERIOR" in c or "EXT" in c:
        return "USD Exterior"
    if "DOLAR LOCAL" in c or "DÓLAR LOCAL" in c or "USD LOCAL" in c:
        return "USD Local"
    return "ARS"


def _detect_tipo(categoria: str) -> str:
    c = _norm(categoria).upper()
    mapping = {
        "CUENTA CORRIENTE": "Cta. Corriente",
        "FONDOS COMUNES": "FCI",
        "LETRAS": "Letras",
        "TITULOS PUBLICOS": "T. Públicos",
        "TÍTULOS PUBLICOS": "T. Públicos",
        "OBLIGACIONES NEGOCIABLES": "ON",
        "ACCIONES": "Acciones",
    }
    for key, val in mapping.items():
        if key in c:
            return val
    return "Otros"


# =========================================================
# NORMALIZACIÓN NUMÉRICA HTML/TEXTO
# =========================================================
def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "-": np.nan,
            "—": np.nan,
        }
    )

    # elimina separadores de miles y cambia coma decimal por punto
    s = s.str.replace(r"\.", "", regex=True)
    s = s.str.replace(",", ".", regex=False)

    # elimina símbolos
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("U$S", "", regex=False)
    s = s.str.replace("USD", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def _clean_html_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # aplana multicolumns si existen
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([_norm(x) for x in tup if _norm(x) != ""]).strip()
            for tup in out.columns
        ]

    # fuerza nombres de columnas simples
    out.columns = [str(c) for c in out.columns]

    # limpia espacios raros
    for col in out.columns:
        out[col] = out[col].apply(lambda x: _norm(x).replace("\xa0", " "))

    return out


# =========================================================
# PARSEO HEADER
# =========================================================
def _parse_header(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {
        "usuario": "",
        "comitente": "",
        "fecha": None,
    }

    # intento directo original
    try:
        out["usuario"] = _norm(df.iloc[3, 0])
        out["comitente"] = _norm(df.iloc[3, 1])
        out["fecha"] = _to_ts(df.iloc[3, 2])
    except Exception:
        pass

    label_map = {
        "Total Posición": "total_posicion",
        "Portafolio Disponible": "portafolio_disponible",
        "Cuenta Corriente $": "cc_ars",
        "Cuenta Corriente U$S Exterior": "cc_usd_ext",
        "Cuenta Corriente Dolar Local": "cc_usd_local",
        "Cuenta Corriente Dólar Local": "cc_usd_local",
    }

    max_rows = min(len(df), 60)
    max_cols = min(df.shape[1], 12)

    # búsqueda flexible en todo el bloque superior
    for i in range(max_rows):
        for j in range(max_cols):
            val = _norm(df.iloc[i, j])

            if out["fecha"] is None:
                ts = _to_ts(val)
                if ts is not None:
                    out["fecha"] = ts

            if out["usuario"] == "" and ("usuario" in val.lower() or "cliente" in val.lower()):
                if j + 1 < df.shape[1]:
                    out["usuario"] = _norm(df.iloc[i, j + 1])

            if out["comitente"] == "" and "comitente" in val.lower():
                if j + 1 < df.shape[1]:
                    out["comitente"] = _norm(df.iloc[i, j + 1])

            if val in label_map:
                for k in range(j + 1, min(j + 7, df.shape[1])):
                    num = pd.to_numeric(
                        str(df.iloc[i, k]).replace(".", "").replace(",", "."),
                        errors="coerce",
                    )
                    if pd.notna(num):
                        out[label_map[val]] = num
                        break

    return out


# =========================================================
# PARSEO DETALLE
# =========================================================
def _empty_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "Especie", "Estado", "Cantidad", "Precio", "Importe",
        "PctTotal", "Costo", "PctVar", "Resultado",
        "Categoria", "Moneda", "Tipo"
    ])


def _parse_detail(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        "Especie", "Estado", "Cantidad", "Precio", "Importe",
        "PctTotal", "Costo", "PctVar", "Resultado"
    ]

    if df.empty:
        return _empty_detail_df()

    raw = None

    # intento estructura original exacta
    try:
        tmp = df.iloc[16:, 1:10].copy()
        if tmp.shape[1] == 9:
            tmp.columns = expected_cols
            raw = tmp.reset_index(drop=True)
    except Exception:
        pass

    # fallback flexible: buscar bloque de 9 columnas que tenga pinta de detalle
    if raw is None:
        for start_col in range(max(1, df.shape[1] - 8)):
            tmp = df.iloc[:, start_col:start_col + 9].copy()
            if tmp.shape[1] != 9:
                continue

            tmp.columns = expected_cols

            especie_nonempty = tmp["Especie"].astype(str).str.strip().ne("").sum()
            importe_num = _coerce_numeric_series(tmp["Importe"]).notna().sum()
            cantidad_num = _coerce_numeric_series(tmp["Cantidad"]).notna().sum()

            if especie_nonempty > 5 and (importe_num > 3 or cantidad_num > 3):
                raw = tmp.reset_index(drop=True)
                break

    if raw is None:
        return _empty_detail_df()

    raw["Especie"] = raw["Especie"].apply(_norm)
    raw["Estado"] = raw["Estado"].apply(_norm)

    for col in ["Cantidad", "Precio", "Importe", "PctTotal", "Costo", "PctVar", "Resultado"]:
        raw[col] = _coerce_numeric_series(raw[col])

    def _is_subtotal(row) -> bool:
        return pd.isna(row["Cantidad"]) and pd.isna(row["Precio"]) and row["Estado"] != ""

    rows: List[dict] = []
    current_cat = ""

    for _, row in raw.iterrows():
        especie = row["Especie"]

        if especie == "" and pd.isna(row["Cantidad"]) and pd.isna(row["Importe"]):
            continue

        if especie.startswith("* "):
            continue

        if _is_subtotal(row):
            current_cat = row["Estado"]
            continue

        rec = row.to_dict()
        rec["Categoria"] = current_cat
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_detail_df()

    out["Moneda"] = out["Categoria"].apply(_detect_moneda)
    out["Tipo"] = out["Categoria"].apply(_detect_tipo)
    out = out[out["Categoria"] != ""].copy()

    return out


# =========================================================
# LECTURA ARCHIVOS
# =========================================================
def _read_as_html(file) -> pd.DataFrame:
    file.seek(0)
    content = file.read()

    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="ignore")
    else:
        text = str(content)

    if "<html" not in text.lower() and "<table" not in text.lower() and "<pre" not in text.lower():
        raise ValueError("El archivo no contiene estructura HTML reconocible.")

    # intento 1: tablas html clásicas
    try:
        tables = pd.read_html(text)
        if tables:
            tables = sorted(tables, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            return _clean_html_dataframe(tables[0])
    except Exception:
        pass

    # intento 2: bloque <pre> o texto plano dentro del html
    txt = (
        text.replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("<br />", "\n")
            .replace("&nbsp;", " ")
    )

    # sacamos tags html de forma simple
    import re
    txt = re.sub(r"<script.*?</script>", "", txt, flags=re.S | re.I)
    txt = re.sub(r"<style.*?</style>", "", txt, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n+", "\n", txt).strip()

    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No tables found")

    # armamos una pseudo-grilla separando por 2 o más espacios
    rows = []
    for line in lines:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) > 1:
            rows.append(parts)

    if not rows:
        raise ValueError("No tables found")

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    return pd.DataFrame(rows)
    


def _read_as_html(file) -> pd.DataFrame:
    file.seek(0)
    content = file.read()

    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="ignore")
    else:
        text = str(content)

    if "<html" not in text.lower() and "<table" not in text.lower():
        raise ValueError("El archivo no contiene estructura HTML reconocible.")

    tables = pd.read_html(text)
    if not tables:
        raise ValueError("El HTML no contiene tablas.")

    # Busca la tabla más grande, suele ser la correcta
    tables = sorted(tables, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    df_html = _clean_html_dataframe(tables[0])

    # algunas exportaciones HTML pierden filas/columnas vacías; igual devolvemos la mayor
    return df_html


def load_portfolio(file) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Intenta leer:
    1) Excel real
    2) HTML disfrazado de .xls
    """
    name = getattr(file, "name", "archivo").lower()
    last_error = None

    # 1) intento como Excel real
    try:
        df_raw = _read_as_excel(file, name)
        return _parse_header(df_raw), _parse_detail(df_raw)
    except Exception as e:
        last_error = e

    # 2) intento como HTML
    try:
        df_raw = _read_as_html(file)
        return _parse_header(df_raw), _parse_detail(df_raw)
    except Exception as e:
        last_error = e

    raise ValueError(
        f"Error procesando {getattr(file, 'name', 'archivo')}: {last_error}"
    )


# =========================================================
# AGREGADOS / CONSOLIDACIÓN
# =========================================================
def _agg_by_especie(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=[
            "Especie", "Categoria", "Tipo", "Moneda",
            "Cantidad", "Precio", "Importe", "Costo", "Resultado"
        ])

    keys = ["Especie", "Categoria", "Tipo", "Moneda"]

    grouped = (
        detail.groupby(keys, as_index=False)
        .agg(
            Cantidad=("Cantidad", "sum"),
            Importe=("Importe", "sum"),
            Resultado=("Resultado", "sum"),
        )
    )

    precio_df = (
        detail.groupby(keys)
        .apply(lambda g: _weighted_avg(g["Precio"], g["Cantidad"]), include_groups=False)
        .reset_index(name="Precio")
    )

    costo_df = (
        detail.groupby(keys)
        .apply(lambda g: _weighted_avg(g["Costo"], g["Cantidad"]), include_groups=False)
        .reset_index(name="Costo")
    )

    grouped = grouped.merge(precio_df, on=keys, how="left")
    grouped = grouped.merge(costo_df, on=keys, how="left")

    return grouped.sort_values("Importe", ascending=False).reset_index(drop=True)


def consolidate_same_date(files: List) -> Dict[pd.Timestamp, Dict[str, object]]:
    grouped: Dict[pd.Timestamp, Dict[str, object]] = {}

    for file in files:
        header, detail = load_portfolio(file)
        fecha = header.get("fecha")

        if fecha is None:
            raise ValueError(
                f"No se pudo identificar la fecha en {getattr(file, 'name', 'archivo')}"
            )

        if fecha not in grouped:
            grouped[fecha] = {
                "fecha": fecha,
                "headers_raw": [],
                "details_raw": [],
                "file_names": [],
                "comitentes": [],
                "usuarios": [],
            }

        grouped[fecha]["headers_raw"].append(header)
        grouped[fecha]["details_raw"].append(detail.copy())
        grouped[fecha]["file_names"].append(getattr(file, "name", "archivo"))
        grouped[fecha]["comitentes"].append(_norm(header.get("comitente")))
        grouped[fecha]["usuarios"].append(_norm(header.get("usuario")))

    consolidated: Dict[pd.Timestamp, Dict[str, object]] = {}

    for fecha, pack in grouped.items():
        headers_raw = pack["headers_raw"]
        details_raw = pack["details_raw"]

        header_cons = {
            "fecha": fecha,
            "usuario": "Consolidado",
            "comitente": "Consolidado",
            "cantidad_archivos": len(pack["file_names"]),
            "cantidad_comitentes": len({c for c in pack["comitentes"] if c}),
            "archivos": " | ".join(pack["file_names"]),
            "comitentes_lista": " | ".join(sorted({c for c in pack["comitentes"] if c})),
            "total_posicion": np.nansum([_safe_num(h, "total_posicion") for h in headers_raw]),
            "portafolio_disponible": np.nansum([_safe_num(h, "portafolio_disponible") for h in headers_raw]),
            "cc_ars": np.nansum([_safe_num(h, "cc_ars") for h in headers_raw]),
            "cc_usd_ext": np.nansum([_safe_num(h, "cc_usd_ext") for h in headers_raw]),
            "cc_usd_local": np.nansum([_safe_num(h, "cc_usd_local") for h in headers_raw]),
        }

        detail_all = pd.concat(details_raw, ignore_index=True) if details_raw else _empty_detail_df()
        detail_cons = _agg_by_especie(detail_all)

        consolidated[fecha] = {
            "header": header_cons,
            "detail": detail_cons,
            "detail_raw_concat": detail_all,
            "file_names": pack["file_names"],
            "comitentes": sorted({c for c in pack["comitentes"] if c}),
        }

    return dict(sorted(consolidated.items(), key=lambda x: x[0]))


# =========================================================
# COMPARACIONES
# =========================================================
def compare_headers(h_ini: Dict, h_fin: Dict) -> pd.DataFrame:
    rows = [
        ("Total Posición", h_ini.get("total_posicion"), h_fin.get("total_posicion")),
        ("Portafolio Disponible", h_ini.get("portafolio_disponible"), h_fin.get("portafolio_disponible")),
        ("Cuenta Corriente ARS", h_ini.get("cc_ars"), h_fin.get("cc_ars")),
        ("CC USD Exterior", h_ini.get("cc_usd_ext"), h_fin.get("cc_usd_ext")),
        ("CC USD Local", h_ini.get("cc_usd_local"), h_fin.get("cc_usd_local")),
    ]
    df = pd.DataFrame(rows, columns=["Indicador", "Inicio", "Fin"])
    df["Variación $"] = df["Fin"] - df["Inicio"]
    df["Variación %"] = df.apply(lambda r: _pct_change(r["Fin"], r["Inicio"]), axis=1)
    return df


def compare_species(d_ini: pd.DataFrame, d_fin: pd.DataFrame) -> pd.DataFrame:
    keys = ["Especie", "Categoria", "Tipo", "Moneda"]

    a = d_ini.rename(columns={
        "Cantidad": "Cant_Ini",
        "Precio": "Precio_Ini",
        "Importe": "Imp_Ini",
        "Resultado": "Res_Ini",
        "Costo": "Costo_Ini",
    }).copy()

    b = d_fin.rename(columns={
        "Cantidad": "Cant_Fin",
        "Precio": "Precio_Fin",
        "Importe": "Imp_Fin",
        "Resultado": "Res_Fin",
        "Costo": "Costo_Fin",
    }).copy()

    comp = pd.merge(a, b, on=keys, how="outer").fillna(0)

    comp["Var_Cant"] = comp["Cant_Fin"] - comp["Cant_Ini"]
    comp["Var_Importe"] = comp["Imp_Fin"] - comp["Imp_Ini"]
    comp["Var_Pct"] = comp.apply(lambda r: _pct_change(r["Imp_Fin"], r["Imp_Ini"]), axis=1)
    comp["Var_Resultado"] = comp["Res_Fin"] - comp["Res_Ini"]

    comp["Movimiento"] = np.select(
        [
            (comp["Cant_Ini"] > 0) & (comp["Cant_Fin"] > 0) & (comp["Var_Cant"] == 0),
            (comp["Imp_Ini"] == 0) & (comp["Imp_Fin"] != 0),
            (comp["Cant_Ini"] > 0) & (comp["Var_Cant"] > 0),
            (comp["Var_Cant"] < 0) & (comp["Cant_Fin"] == 0),
            (comp["Var_Cant"] < 0) & (comp["Cant_Fin"] > 0),
        ],
        [
            "Mantenida",
            "Posición nueva",
            "Aumento de posición",
            "Cierre de posición",
            "Disminución",
        ],
        default="Mantenida",
    )

    return comp.sort_values("Var_Importe", ascending=False).reset_index(drop=True)


# =========================================================
# EXPORT
# =========================================================
def build_export(
    h_ini: Dict,
    h_fin: Dict,
    d_ini: pd.DataFrame,
    d_fin: pd.DataFrame,
    df_header: pd.DataFrame,
    df_species: pd.DataFrame,
    all_groups: Dict[pd.Timestamp, Dict[str, object]],
) -> bytes:
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([h_ini]).to_excel(writer, sheet_name="Header_Inicio", index=False)
        pd.DataFrame([h_fin]).to_excel(writer, sheet_name="Header_Fin", index=False)
        d_ini.to_excel(writer, sheet_name="Detalle_Inicio", index=False)
        d_fin.to_excel(writer, sheet_name="Detalle_Fin", index=False)
        df_header.to_excel(writer, sheet_name="Resumen", index=False)
        df_species.to_excel(writer, sheet_name="Especies", index=False)

        resumen_fechas = []
        for fecha, pack in all_groups.items():
            resumen_fechas.append({
                "Fecha": fecha,
                "Cantidad archivos": pack["header"].get("cantidad_archivos"),
                "Cantidad comitentes": pack["header"].get("cantidad_comitentes"),
                "Archivos": " | ".join(pack["file_names"]),
                "Comitentes": pack["header"].get("comitentes_lista"),
                "Total Posición": pack["header"].get("total_posicion"),
                "Portafolio Disponible": pack["header"].get("portafolio_disponible"),
            })

        pd.DataFrame(resumen_fechas).to_excel(writer, sheet_name="Fechas_Consolidadas", index=False)

    bio.seek(0)
    return bio.read()


# =========================================================
# RENDER AUXILIARES
# =========================================================
def _kpi_html(label: str, value: str, delta: str, delta_color: str) -> str:
    return f"""
    <div class="cp-kpi">
        <div class="cp-kpi-label">{label}</div>
        <div class="cp-kpi-value">{value}</div>
        <div class="cp-kpi-delta" style="color:{delta_color};">{delta}</div>
    </div>
    """


def _styled(df: pd.DataFrame, fmts: Dict[str, str]):
    if df.empty:
        return df.style
    return df.reset_index(drop=True).style.format(fmts, na_rep="—").hide(axis="index")


# =========================================================
# RENDER PRINCIPAL
# =========================================================
def render() -> None:
    _inject_css()

    st.title("Cartera Propia")
    st.markdown(
        f"""
        <p style="color:{MUTED};font-size:0.95rem;margin-top:-0.3rem;margin-bottom:1.2rem;">
        Subí múltiples archivos de distintas comitentes. El sistema agrupa automáticamente por fecha,
        consolida las tenencias de la misma fecha y compara la fecha inicial vs. la final.
        También soporta archivos .xls que en realidad son HTML exportado por el sistema.
        </p>
        """,
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Subí los Excel de cartera",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="cp_multi",
    )

    run = st.button("Analizar cartera consolidada", use_container_width=True)

    if not run:
        return

    if not files:
        st.warning("Subí al menos dos archivos para continuar.")
        return

    try:
        groups = consolidate_same_date(files)
    except ValueError as e:
        st.error(str(e))
        st.caption(
            "Si algún archivo sigue fallando, probablemente venga con una estructura HTML distinta "
            "a la habitual y haya que ajustar el parser."
        )
        return
    except Exception as e:
        st.error(f"Error general al procesar los archivos: {e}")
        return

    unique_dates = list(groups.keys())

    if len(unique_dates) < 2:
        st.warning("Necesitás archivos de al menos dos fechas distintas para comparar.")
        return

    st.markdown('<div class="cp-section">Fechas detectadas</div>', unsafe_allow_html=True)

    df_detectadas = pd.DataFrame([
        {
            "Fecha": fecha.strftime("%d/%m/%Y"),
            "Cantidad de archivos": groups[fecha]["header"]["cantidad_archivos"],
            "Cantidad de comitentes": groups[fecha]["header"]["cantidad_comitentes"],
            "Total Posición": groups[fecha]["header"]["total_posicion"],
            "Portafolio Disponible": groups[fecha]["header"]["portafolio_disponible"],
            "Comitentes": groups[fecha]["header"]["comitentes_lista"],
        }
        for fecha in unique_dates
    ])

    st.dataframe(
        _styled(df_detectadas, {
            "Total Posición": "$ {:,.2f}",
            "Portafolio Disponible": "$ {:,.2f}",
        }),
        use_container_width=True,
        height=min(280, 70 + 35 * len(df_detectadas)),
    )

    default_ini = 0
    default_fin = len(unique_dates) - 1

    c1, c2 = st.columns(2)
    fecha_ini = c1.selectbox(
        "Fecha inicial",
        options=unique_dates,
        format_func=lambda x: x.strftime("%d/%m/%Y"),
        index=default_ini,
    )
    fecha_fin = c2.selectbox(
        "Fecha final",
        options=unique_dates,
        format_func=lambda x: x.strftime("%d/%m/%Y"),
        index=default_fin,
    )

    if fecha_ini >= fecha_fin:
        st.error("La fecha inicial debe ser anterior a la fecha final.")
        return

    h_ini = groups[fecha_ini]["header"]
    d_ini = groups[fecha_ini]["detail"]
    h_fin = groups[fecha_fin]["header"]
    d_fin = groups[fecha_fin]["detail"]

    st.info(
        f"**Inicial:** {fecha_ini.strftime('%d/%m/%Y')} · "
        f"{h_ini['cantidad_archivos']} archivo(s) · {h_ini['cantidad_comitentes']} comitente(s)\n\n"
        f"**Final:** {fecha_fin.strftime('%d/%m/%Y')} · "
        f"{h_fin['cantidad_archivos']} archivo(s) · {h_fin['cantidad_comitentes']} comitente(s)"
    )

    df_header = compare_headers(h_ini, h_fin)
    df_species = compare_species(d_ini, d_fin)

    total_ini = pd.to_numeric(h_ini.get("total_posicion"), errors="coerce")
    total_fin = pd.to_numeric(h_fin.get("total_posicion"), errors="coerce")
    disp_ini = pd.to_numeric(h_ini.get("portafolio_disponible"), errors="coerce")
    disp_fin = pd.to_numeric(h_fin.get("portafolio_disponible"), errors="coerce")

    total_var = total_fin - total_ini
    total_pct = _pct_change(total_fin, total_ini)
    disp_var = disp_fin - disp_ini

    res_fin = pd.to_numeric(d_fin["Resultado"], errors="coerce").fillna(0).sum() if not d_fin.empty else 0
    mov_counts = df_species["Movimiento"].value_counts().to_dict() if not df_species.empty else {}

    st.markdown('<div class="cp-section">Resumen ejecutivo</div>', unsafe_allow_html=True)

    def _delta_color(v):
        return SUCCESS if pd.notna(v) and v >= 0 else DANGER

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(_kpi_html(
            "Total posición final",
            _fmt_money(total_fin),
            f"{_fmt_money(total_var)} · {_fmt_pct(total_pct)}",
            _delta_color(total_var),
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(_kpi_html(
            "Portafolio disponible final",
            _fmt_money(disp_fin),
            f"var: {_fmt_money(disp_var)}",
            _delta_color(disp_var),
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(_kpi_html(
            "Resultado acumulado (fin)",
            _fmt_money(res_fin),
            "suma de resultados individuales",
            MUTED,
        ), unsafe_allow_html=True)

    with k4:
        nuevas = mov_counts.get("Posición nueva", 0)
        aumentos = mov_counts.get("Aumento de posición", 0)
        disminuc = mov_counts.get("Disminución", 0)
        cierres = mov_counts.get("Cierre de posición", 0)
        mantenidas = mov_counts.get("Mantenida", 0)

        st.markdown(_kpi_html(
            "Movimientos",
            f"{len(df_species)} especies",
            f"Nuevas {nuevas} · Aumentos {aumentos} · Disminuc. {disminuc} · Cierres {cierres} · Mant. {mantenidas}",
            MUTED,
        ), unsafe_allow_html=True)

    st.markdown('<div class="cp-section">Comparación inicio vs. fin</div>', unsafe_allow_html=True)
    st.dataframe(
        _styled(df_header, {
            "Inicio": "$ {:,.2f}",
            "Fin": "$ {:,.2f}",
            "Variación $": "$ {:+,.2f}",
            "Variación %": "{:+,.2f}%",
        }),
        use_container_width=True,
        height=240,
    )

    st.markdown('<div class="cp-section">Detalle general por especie</div>', unsafe_allow_html=True)

    cols_general = [
        "Especie", "Tipo", "Moneda",
        "Imp_Ini", "Imp_Fin", "Var_Importe", "Var_Pct",
        "Var_Resultado", "Movimiento",
    ]
    df_general = df_species[cols_general].copy() if not df_species.empty else pd.DataFrame(columns=cols_general)

    st.dataframe(
        _styled(df_general, {
            "Imp_Ini": "$ {:,.2f}",
            "Imp_Fin": "$ {:,.2f}",
            "Var_Importe": "$ {:+,.2f}",
            "Var_Pct": "{:+,.2f}%",
            "Var_Resultado": "$ {:+,.2f}",
        }),
        use_container_width=True,
        height=500,
    )

    st.markdown('<div class="cp-section">Por tipo de movimiento</div>', unsafe_allow_html=True)

    tab_mant, tab_nueva, tab_aumento, tab_dism, tab_cierre = st.tabs(
        ["Mantenida", "Posición nueva", "Aumento de posición", "Disminución", "Cierre de posición"]
    )

    cols_mov = [
        "Especie", "Tipo", "Moneda",
        "Imp_Ini", "Imp_Fin", "Var_Importe", "Var_Pct", "Var_Resultado",
    ]
    fmt_mov = {
        "Imp_Ini": "$ {:,.2f}",
        "Imp_Fin": "$ {:,.2f}",
        "Var_Importe": "$ {:+,.2f}",
        "Var_Pct": "{:+,.2f}%",
        "Var_Resultado": "$ {:+,.2f}",
    }

    def _render_mov_tab(mov: str, ascending: bool = False) -> None:
        sub = (
            df_species[df_species["Movimiento"] == mov][cols_mov]
            .sort_values("Var_Importe", ascending=ascending)
            if not df_species.empty else pd.DataFrame(columns=cols_mov)
        )

        if sub.empty:
            st.info(f"No hay posiciones en '{mov}'.")
        else:
            st.dataframe(_styled(sub, fmt_mov), use_container_width=True, height=340)

    with tab_mant:
        _render_mov_tab("Mantenida")
    with tab_nueva:
        _render_mov_tab("Posición nueva")
    with tab_aumento:
        _render_mov_tab("Aumento de posición")
    with tab_dism:
        _render_mov_tab("Disminución", ascending=True)
    with tab_cierre:
        _render_mov_tab("Cierre de posición", ascending=True)

    st.markdown('<div class="cp-section">Exportar</div>', unsafe_allow_html=True)
    export_bytes = build_export(h_ini, h_fin, d_ini, d_fin, df_header, df_species, groups)

    st.download_button(
        label="Descargar Excel comparativo consolidado",
        data=export_bytes,
        file_name="cartera_propia_consolidada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
