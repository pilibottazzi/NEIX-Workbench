from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

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
WARNING = "#d97706"
INFO = "#2563eb"
BG_SOFT = "#f9fafb"


# =========================================================
# CONFIG NEGOCIO
# =========================================================
NON_RENTABLE_KEYS = {
    "PESOS",
    "DOLAR LOCAL",
    "DÓLAR LOCAL",
    "USD LOCAL",
    "DOLAR EXTERIOR",
    "DÓLAR EXTERIOR",
    "USD EXTERIOR",
    "CUENTA CORRIENTE",
    "CTA. CORRIENTE",
    "CTA CORRIENTE",
}

PORTFOLIO_REQUIRED_FINAL = [
    "Especie", "Categoria", "Tipo", "Moneda",
    "Cantidad", "Precio", "Importe", "Costo", "Resultado",
    "activo_match", "Cartera Neix", "Archivo", "Fecha"
]

RENTAL_BASE_COLUMNS = [
    "id", "F.Inicio", "F.Vto", "Cliente", "Neix", "Cartera Neix", "VN",
    "Activo", "Especie", "Dias", "Precio", "Tasa Cliente", "Pagar Cliente",
    "Tasa Manager", "Pagar Manager", "Moneda", "Obs", "Estado"
]


# =========================================================
# UI
# =========================================================
def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            max-width: 1400px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }}

          .cp-kpi {{
            background: {BG_SOFT};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 118px;
          }}

          .cp-kpi-label {{
            color: {MUTED};
            font-size: 0.82rem;
            margin-bottom: 0.3rem;
            letter-spacing: 0.02em;
          }}

          .cp-kpi-value {{
            color: {PRIMARY};
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.1;
          }}

          .cp-kpi-delta {{
            margin-top: 0.35rem;
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.3;
          }}

          .cp-section {{
            font-size: 0.92rem;
            font-weight: 700;
            color: {MUTED};
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 1.45rem 0 0.65rem;
          }}

          div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {BORDER};
          }}

          .stDownloadButton button, .stButton button {{
            border-radius: 10px;
            font-weight: 600;
          }}

          .cp-note {{
            background: #fff;
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            color: {MUTED};
            font-size: 0.92rem;
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


def _fmt_num(v: float, decimals: int = 0) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.{decimals}f}"


def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:+,.2f}%"


def _pct_change(fin: float, ini: float) -> float:
    if pd.isna(ini) or ini == 0:
        return np.nan
    return (fin / ini - 1) * 100


def _norm(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).replace("\xa0", " ").strip()


def _norm_upper(x) -> str:
    return _norm(x).upper().strip()


def _strip_extra_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", _norm(s)).strip()


def _normalize_text_key(x) -> str:
    s = _norm_upper(x)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^\w\s./]", "", s)
    return s.strip()


def _normalize_col_name(c: str) -> str:
    s = _normalize_text_key(c)
    mapping = {
        "F INICIO": "F.Inicio",
        "F.INICIO": "F.Inicio",
        "F VTO": "F.Vto",
        "F.VTO": "F.Vto",
        "CARTERA NEIX": "Cartera Neix",
        "ACTIVO": "Activo",
        "ESPECIE": "Especie",
        "VN": "VN",
        "MONEDA": "Moneda",
        "ESTADO": "Estado",
        "ID": "id",
        "CLIENTE": "Cliente",
        "NEIX": "Neix",
        "DIAS": "Dias",
        "PRECIO": "Precio",
        "TASA CLIENTE": "Tasa Cliente",
        "PAGAR CLIENTE": "Pagar Cliente",
        "TASA MANAGER": "Tasa Manager",
        "PAGAR MANAGER": "Pagar Manager",
        "OBS": "Obs",
    }
    return mapping.get(s, _strip_extra_spaces(c))


def _to_ts(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        try:
            return pd.Timestamp(value).normalize()
        except Exception:
            return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    txt = str(value).strip()
    if not txt:
        return None

    candidates = [
        pd.to_datetime(txt, dayfirst=True, errors="coerce"),
        pd.to_datetime(txt, dayfirst=False, errors="coerce"),
    ]
    for ts in candidates:
        if pd.notna(ts):
            return pd.Timestamp(ts).normalize()
    return None


def _safe_num(d: Dict, key: str) -> float:
    return pd.to_numeric(d.get(key), errors="coerce")


def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    mask = (~values.isna()) & (~weights.isna()) & (weights != 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def _coerce_numeric_scalar(x):
    s = _norm(x)
    if s == "":
        return np.nan

    s_low = s.lower()
    if s_low in {"nan", "none", "-", "—"}:
        return np.nan

    s = s.replace("%", "")
    s = s.replace("$", "")
    s = s.replace("u$s", "").replace("U$S", "")
    s = s.replace("usd", "").replace("USD", "")
    s = s.replace("ars", "").replace("ARS", "")
    s = s.replace("(", "-").replace(")", "")
    s = s.replace(" ", "")

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        if s.count(".") > 1:
            s = s.replace(".", "")

    return pd.to_numeric(s, errors="coerce")


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    return series.apply(_coerce_numeric_scalar)


def _looks_like_html(text: str) -> bool:
    low = text.lower()
    return (
        "<html" in low
        or "<table" in low
        or "<tr" in low
        or "<td" in low
        or "<pre" in low
        or "<body" in low
    )


def _first_token(text: str) -> str:
    s = _normalize_text_key(text)
    if not s:
        return ""
    parts = re.split(r"\s+", s)
    token = parts[0].strip()
    token = re.sub(r"[^\w./-]", "", token)
    return token


def _normalize_activo_match_from_portfolio(especie: str) -> str:
    token = _first_token(especie)
    return token.upper()


def _normalize_activo_match_from_rental(activo: str) -> str:
    token = _normalize_text_key(activo)
    token = re.sub(r"\s+", " ", token).strip()
    return token.upper()


def _normalize_cartera_neix(x: str) -> str:
    return _normalize_text_key(x)


def _is_non_rentable_position(row: pd.Series) -> bool:
    especie = _normalize_text_key(row.get("Especie", ""))
    activo = _normalize_text_key(row.get("activo_match", ""))
    tipo = _normalize_text_key(row.get("Tipo", ""))
    moneda = _normalize_text_key(row.get("Moneda", ""))
    categoria = _normalize_text_key(row.get("Categoria", ""))

    all_text = " | ".join([especie, activo, tipo, moneda, categoria])

    if tipo in {"CTA. CORRIENTE", "CUENTA CORRIENTE"}:
        return True

    for key in NON_RENTABLE_KEYS:
        if key in all_text:
            return True

    return False


# =========================================================
# DETECCIONES DE NEGOCIO
# =========================================================
def _detect_moneda(categoria: str) -> str:
    c = _norm(categoria).upper()
    if "U$S EXTERIOR" in c or "USD EXTERIOR" in c or "EXTERIOR" in c:
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
# LECTURA DE ARCHIVOS
# =========================================================
def _read_as_excel(file, name: str) -> pd.DataFrame:
    lower_name = name.lower()

    if lower_name.endswith(".xlsx") or lower_name.endswith(".xlsm"):
        engines = ["openpyxl", "calamine", "xlrd"]
    elif lower_name.endswith(".xls"):
        engines = ["xlrd", "calamine", "openpyxl"]
    else:
        engines = ["openpyxl", "xlrd", "calamine"]

    last_error = None
    for engine in engines:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None, engine=engine)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            last_error = e

    raise last_error if last_error else ValueError("No se pudo leer como Excel.")


def _read_excel_best_effort(file) -> pd.DataFrame:
    name = getattr(file, "name", "archivo")
    lower_name = name.lower()

    if lower_name.endswith(".xlsx") or lower_name.endswith(".xlsm"):
        engines = ["openpyxl", "calamine", "xlrd"]
    elif lower_name.endswith(".xls"):
        engines = ["xlrd", "calamine", "openpyxl"]
    else:
        engines = ["openpyxl", "xlrd", "calamine"]

    last_error = None
    for engine in engines:
        try:
            file.seek(0)
            return pd.read_excel(file, engine=engine)
        except Exception as e:
            last_error = e

    raise last_error if last_error else ValueError(f"No se pudo leer {name}.")


def _clean_html_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([_norm(x) for x in tup if _norm(x) != ""]).strip()
            for tup in out.columns
        ]
    else:
        out.columns = [str(c) for c in out.columns]

    for col in out.columns:
        out[col] = out[col].apply(_norm)

    return out


def _html_to_text_lines(text: str) -> List[str]:
    txt = text
    txt = txt.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    txt = txt.replace("&nbsp;", " ")
    txt = re.sub(r"<script.*?</script>", "", txt, flags=re.I | re.S)
    txt = re.sub(r"<style.*?</style>", "", txt, flags=re.I | re.S)
    txt = re.sub(r"</tr\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"</p\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"</div\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    return lines


def _text_lines_to_dataframe(lines: List[str]) -> pd.DataFrame:
    rows = []

    for line in lines:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) > 1:
            rows.append(parts)

    if not rows:
        for line in lines:
            parts = [p.strip() for p in line.split("\t") if p.strip() != ""]
            if len(parts) > 1:
                rows.append(parts)

    if not rows:
        rows = [[line] for line in lines]

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

    if not _looks_like_html(text):
        raise ValueError("El archivo no contiene estructura HTML reconocible.")

    try:
        tables = pd.read_html(text)
        if tables:
            tables = sorted(tables, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            return _clean_html_dataframe(tables[0])
    except Exception:
        pass

    lines = _html_to_text_lines(text)
    if not lines:
        raise ValueError("No tables found")

    return _text_lines_to_dataframe(lines)


def load_portfolio(file) -> Tuple[Dict[str, object], pd.DataFrame]:
    name = getattr(file, "name", "archivo")
    last_error = None

    try:
        df_raw = _read_as_excel(file, name)
        header = _parse_header(df_raw)
        detail = _parse_detail(df_raw)
        if header.get("fecha") is not None or not detail.empty:
            return header, detail
    except Exception as e:
        last_error = e

    try:
        file.seek(0)
        raw = file.read()
        raw_text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)

        if _looks_like_html(raw_text):
            file.seek(0)
            df_raw = _read_as_html(file)
            header = _parse_header(df_raw)
            detail = _parse_detail(df_raw)
            if header.get("fecha") is not None or not detail.empty:
                return header, detail
    except Exception as e:
        last_error = e

    raise ValueError(f"Error procesando {name}: {last_error}")


# =========================================================
# PARSEO HEADER
# =========================================================
def _parse_header(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {
        "usuario": "",
        "comitente": "",
        "fecha": None,
        "total_posicion": np.nan,
        "portafolio_disponible": np.nan,
        "cc_ars": np.nan,
        "cc_usd_ext": np.nan,
        "cc_usd_local": np.nan,
    }

    try:
        if df.shape[0] > 3 and df.shape[1] > 0:
            out["usuario"] = _norm(df.iloc[3, 0])
        if df.shape[0] > 3 and df.shape[1] > 1:
            out["comitente"] = _norm(df.iloc[3, 1])
        if df.shape[0] > 3 and df.shape[1] > 2:
            out["fecha"] = _to_ts(df.iloc[3, 2])
    except Exception:
        pass

    label_aliases = {
        "total posición": "total_posicion",
        "total posicion": "total_posicion",
        "portafolio disponible": "portafolio_disponible",
        "cuenta corriente $": "cc_ars",
        "cuenta corriente ars": "cc_ars",
        "cuenta corriente u$s exterior": "cc_usd_ext",
        "cuenta corriente usd exterior": "cc_usd_ext",
        "cuenta corriente exterior": "cc_usd_ext",
        "cuenta corriente dolar local": "cc_usd_local",
        "cuenta corriente dólar local": "cc_usd_local",
        "cuenta corriente usd local": "cc_usd_local",
    }

    max_rows = min(len(df), 80)
    max_cols = min(df.shape[1], 20)

    for i in range(max_rows):
        for j in range(max_cols):
            val = _norm(df.iloc[i, j])
            val_low = val.lower()

            if out["fecha"] is None:
                ts = _to_ts(val)
                if ts is not None:
                    out["fecha"] = ts

            if out["usuario"] == "" and any(x in val_low for x in ["usuario", "cliente", "titular"]):
                for k in range(j + 1, min(j + 4, df.shape[1])):
                    candidate = _norm(df.iloc[i, k])
                    if candidate != "":
                        out["usuario"] = candidate
                        break

            if out["comitente"] == "" and "comitente" in val_low:
                for k in range(j + 1, min(j + 4, df.shape[1])):
                    candidate = _norm(df.iloc[i, k])
                    if candidate != "":
                        out["comitente"] = candidate
                        break

            for alias, target in label_aliases.items():
                if alias in val_low:
                    found = False

                    for k in range(j + 1, min(j + 8, df.shape[1])):
                        num = _coerce_numeric_scalar(df.iloc[i, k])
                        if pd.notna(num):
                            out[target] = num
                            found = True
                            break

                    if not found:
                        for r in range(i + 1, min(i + 4, len(df))):
                            num = _coerce_numeric_scalar(df.iloc[r, j])
                            if pd.notna(num):
                                out[target] = num
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


def _score_detail_candidate(tmp: pd.DataFrame) -> int:
    if tmp.shape[1] != 9:
        return -1

    cols = [
        "Especie", "Estado", "Cantidad", "Precio", "Importe",
        "PctTotal", "Costo", "PctVar", "Resultado"
    ]
    cand = tmp.copy()
    cand.columns = cols

    especie_nonempty = cand["Especie"].astype(str).str.strip().ne("").sum()
    cantidad_num = _coerce_numeric_series(cand["Cantidad"]).notna().sum()
    importe_num = _coerce_numeric_series(cand["Importe"]).notna().sum()
    precio_num = _coerce_numeric_series(cand["Precio"]).notna().sum()

    score = int(especie_nonempty + cantidad_num * 2 + importe_num * 2 + precio_num)
    return score


def _find_detail_block(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = [
        "Especie", "Estado", "Cantidad", "Precio", "Importe",
        "PctTotal", "Costo", "PctVar", "Resultado"
    ]

    try:
        tmp = df.iloc[16:, 1:10].copy()
        if tmp.shape[1] == 9:
            tmp.columns = cols
            return tmp.reset_index(drop=True)
    except Exception:
        pass

    best_score = -1
    best_df = None

    ncols = df.shape[1]
    for start_col in range(0, max(1, ncols - 8)):
        tmp = df.iloc[:, start_col:start_col + 9].copy()
        if tmp.shape[1] != 9:
            continue

        score = _score_detail_candidate(tmp)
        if score > best_score:
            best_score = score
            best_df = tmp.copy()

    if best_df is not None and best_score >= 12:
        best_df.columns = cols
        return best_df.reset_index(drop=True)

    return None


def _parse_detail(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_detail_df()

    raw = _find_detail_block(df)
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

        if especie == "" and row["Estado"] != "":
            current_cat = row["Estado"]
            continue

        rec = row.to_dict()
        rec["Categoria"] = current_cat
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_detail_df()

    out["Categoria"] = out["Categoria"].apply(_norm)
    out["Moneda"] = out["Categoria"].apply(_detect_moneda)
    out["Tipo"] = out["Categoria"].apply(_detect_tipo)

    numeric_ok = out[["Cantidad", "Precio", "Importe", "Resultado"]].notna().any(axis=1)
    out = out[numeric_ok].copy()

    return out


# =========================================================
# CONSOLIDACIÓN CARTERA
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

        comitente = _norm(header.get("comitente"))
        usuario = _norm(header.get("usuario"))
        file_name = getattr(file, "name", "archivo")

        detail_ext = detail.copy()
        detail_ext["Cartera Neix"] = comitente
        detail_ext["Usuario"] = usuario
        detail_ext["Archivo"] = file_name
        detail_ext["Fecha"] = fecha
        detail_ext["activo_match"] = detail_ext["Especie"].apply(_normalize_activo_match_from_portfolio)

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
        grouped[fecha]["details_raw"].append(detail_ext.copy())
        grouped[fecha]["file_names"].append(file_name)
        grouped[fecha]["comitentes"].append(comitente)
        grouped[fecha]["usuarios"].append(usuario)

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

        detail_all = pd.concat(details_raw, ignore_index=True) if details_raw else pd.DataFrame(columns=PORTFOLIO_REQUIRED_FINAL)
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
# ALQUILERES
# =========================================================
def _ensure_required_rental_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_normalize_col_name(c) for c in out.columns]

    for col in RENTAL_BASE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[RENTAL_BASE_COLUMNS].copy()


def load_rentals(file, source_name: str) -> pd.DataFrame:
    df = _read_excel_best_effort(file)
    df = _ensure_required_rental_cols(df)

    for c in df.columns:
        if c not in {"VN", "Precio", "Tasa Cliente", "Pagar Cliente", "Tasa Manager", "Pagar Manager"}:
            df[c] = df[c].apply(_norm)

    df["VN"] = _coerce_numeric_series(df["VN"])
    df["Precio"] = _coerce_numeric_series(df["Precio"])
    df["F.Inicio"] = df["F.Inicio"].apply(_to_ts)
    df["F.Vto"] = df["F.Vto"].apply(_to_ts)

    df["Cartera Neix"] = df["Cartera Neix"].apply(_normalize_cartera_neix)
    df["Activo"] = df["Activo"].apply(_normalize_text_key)
    df["Especie"] = df["Especie"].apply(_strip_extra_spaces)
    df["Moneda"] = df["Moneda"].apply(_strip_extra_spaces)
    df["Estado"] = df["Estado"].apply(_strip_extra_spaces)

    df["activo_match"] = df["Activo"].apply(_normalize_activo_match_from_rental)
    df["source_file"] = getattr(file, "name", source_name)
    df["source_type"] = source_name

    df["id_norm"] = df["id"].apply(lambda x: _normalize_text_key(x))
    df["dedup_key"] = np.where(
        df["id_norm"] != "",
        df["id_norm"],
        (
            df["Cartera Neix"].fillna("").astype(str) + "||" +
            df["activo_match"].fillna("").astype(str) + "||" +
            df["VN"].fillna(0).astype(str) + "||" +
            df["F.Inicio"].astype(str) + "||" +
            df["F.Vto"].astype(str)
        )
    )

    return df


def filter_active_rentals(df_active: pd.DataFrame, fecha_final: pd.Timestamp) -> pd.DataFrame:
    if df_active.empty:
        return df_active.copy()
    return df_active[df_active["F.Inicio"].notna() & (df_active["F.Inicio"] <= fecha_final)].copy()


def filter_historical_live_rentals(df_hist: pd.DataFrame, fecha_final: pd.Timestamp) -> pd.DataFrame:
    if df_hist.empty:
        return df_hist.copy()
    return df_hist[
        df_hist["F.Inicio"].notna()
        & (df_hist["F.Inicio"] <= fecha_final)
        & df_hist["F.Vto"].notna()
        & (df_hist["F.Vto"] > fecha_final)
    ].copy()


def consolidate_live_rentals(
    df_active_live: pd.DataFrame,
    df_hist_live: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = df_active_live.copy()
    h = df_hist_live.copy()

    if a.empty and h.empty:
        return (
            pd.DataFrame(columns=RENTAL_BASE_COLUMNS + ["activo_match", "source_type", "dedup_key"]),
            pd.DataFrame(columns=["dedup_key", "source_types", "count"])
        )

    both = pd.concat([a, h], ignore_index=True)

    dup_counts = (
        both.groupby("dedup_key", dropna=False)
        .agg(
            count=("dedup_key", "size"),
            source_types=("source_type", lambda s: " | ".join(sorted(set(s.astype(str)))))
        )
        .reset_index()
    )
    possible_double = dup_counts[dup_counts["count"] > 1].copy()

    both = both.sort_values(by=["source_type"], ascending=True).copy()
    both = both.drop_duplicates(subset=["dedup_key"], keep="first").reset_index(drop=True)

    return both, possible_double


def summarize_rentals_for_match(df_rentals_live: pd.DataFrame) -> pd.DataFrame:
    if df_rentals_live.empty:
        return pd.DataFrame(columns=[
            "activo_match", "Cartera Neix",
            "nominal_alquilado_activo", "nominal_alquilado_historico_vigente",
            "nominal_alquilado_total", "cantidad_registros_alquiler"
        ])

    tmp = df_rentals_live.copy()
    tmp["nominal_alquilado_activo"] = np.where(tmp["source_type"] == "activos", tmp["VN"], 0.0)
    tmp["nominal_alquilado_historico_vigente"] = np.where(tmp["source_type"] == "historicos", tmp["VN"], 0.0)

    out = (
        tmp.groupby(["activo_match", "Cartera Neix"], as_index=False)
        .agg(
            nominal_alquilado_activo=("nominal_alquilado_activo", "sum"),
            nominal_alquilado_historico_vigente=("nominal_alquilado_historico_vigente", "sum"),
            nominal_alquilado_total=("VN", "sum"),
            cantidad_registros_alquiler=("VN", "size"),
        )
    )

    return out


# =========================================================
# COMPARACIONES BASE
# =========================================================
def compare_headers(h_ini: Dict, h_fin: Dict, total_final_ajustado: float) -> pd.DataFrame:
    rows = [
        ("Total Posición", h_ini.get("total_posicion"), h_fin.get("total_posicion"), total_final_ajustado),
        ("Portafolio Disponible", h_ini.get("portafolio_disponible"), h_fin.get("portafolio_disponible"), np.nan),
        ("Cuenta Corriente ARS", h_ini.get("cc_ars"), h_fin.get("cc_ars"), np.nan),
        ("CC USD Exterior", h_ini.get("cc_usd_ext"), h_fin.get("cc_usd_ext"), np.nan),
        ("CC USD Local", h_ini.get("cc_usd_local"), h_fin.get("cc_usd_local"), np.nan),
    ]
    df = pd.DataFrame(rows, columns=["Indicador", "Inicio", "Fin Bruto", "Fin Ajustado"])
    df["Variación Bruta $"] = df["Fin Bruto"] - df["Inicio"]
    df["Variación Bruta %"] = df.apply(lambda r: _pct_change(r["Fin Bruto"], r["Inicio"]), axis=1)
    df["Variación Ajustada $"] = df["Fin Ajustado"] - df["Inicio"]
    df["Variación Ajustada %"] = df.apply(
        lambda r: _pct_change(r["Fin Ajustado"], r["Inicio"]) if pd.notna(r["Fin Ajustado"]) else np.nan,
        axis=1
    )
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
    comp["activo_match"] = comp["Especie"].apply(_normalize_activo_match_from_portfolio)

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
# AJUSTE POR ALQUILERES
# =========================================================
def _build_final_portfolio_for_match(detail_raw_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if detail_raw_final.empty:
        empty = pd.DataFrame(columns=[
            "Especie", "activo_match", "Tipo", "Moneda", "Cartera Neix",
            "nominal_final_bruto", "precio_final", "importe_final_bruto", "es_no_alquilable"
        ])
        return empty, empty

    df = detail_raw_final.copy()

    df["Cartera Neix"] = df["Cartera Neix"].apply(_normalize_cartera_neix)
    df["activo_match"] = df["Especie"].apply(_normalize_activo_match_from_portfolio)
    df["es_no_alquilable"] = df.apply(_is_non_rentable_position, axis=1)

    grouped = (
        df.groupby(
            ["Especie", "activo_match", "Tipo", "Moneda", "Cartera Neix", "es_no_alquilable"],
            as_index=False
        )
        .agg(
            nominal_final_bruto=("Cantidad", "sum"),
            importe_final_bruto=("Importe", "sum"),
        )
    )

    precio_df = (
        df.groupby(["Especie", "activo_match", "Tipo", "Moneda", "Cartera Neix", "es_no_alquilable"])
        .apply(lambda g: _weighted_avg(g["Precio"], g["Cantidad"]), include_groups=False)
        .reset_index(name="precio_final")
    )

    grouped = grouped.merge(
        precio_df,
        on=["Especie", "activo_match", "Tipo", "Moneda", "Cartera Neix", "es_no_alquilable"],
        how="left",
    )

    comparable = grouped[~grouped["es_no_alquilable"]].copy()
    return grouped, comparable


def build_adjustment_tables(
    detail_raw_final: pd.DataFrame,
    rentals_summary: pd.DataFrame,
    rentals_live: pd.DataFrame,
    possible_double: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    portfolio_all, portfolio_matchable = _build_final_portfolio_for_match(detail_raw_final)

    match_df = portfolio_matchable.merge(
        rentals_summary,
        on=["activo_match", "Cartera Neix"],
        how="left",
    )

    for col in [
        "nominal_alquilado_activo",
        "nominal_alquilado_historico_vigente",
        "nominal_alquilado_total",
        "cantidad_registros_alquiler",
    ]:
        if col in match_df.columns:
            match_df[col] = match_df[col].fillna(0.0)

    match_df["nominal_final_real"] = (match_df["nominal_final_bruto"] - match_df["nominal_alquilado_total"]).clip(lower=0)
    match_df["nominal_sobre_stock"] = match_df["nominal_alquilado_total"] - match_df["nominal_final_bruto"]
    match_df["importe_final_real"] = match_df["nominal_final_real"] * match_df["precio_final"]

    match_df["match_status"] = np.select(
        [
            match_df["nominal_alquilado_total"] > 0,
            match_df["nominal_alquilado_total"] == 0,
        ],
        [
            "match_ok",
            "cartera_sin_match",
        ],
        default="cartera_sin_match",
    )

    rentals_no_match = rentals_summary.merge(
        portfolio_matchable[["activo_match", "Cartera Neix"]].drop_duplicates(),
        on=["activo_match", "Cartera Neix"],
        how="left",
        indicator=True,
    )
    rentals_no_match = rentals_no_match[rentals_no_match["_merge"] == "left_only"].drop(columns=["_merge"]).copy()

    cartera_sin_match = match_df[match_df["match_status"] == "cartera_sin_match"].copy()
    match_ok = match_df[match_df["match_status"] == "match_ok"].copy()

    alerts_rows = []

    over_stock = match_df[match_df["nominal_alquilado_total"] > match_df["nominal_final_bruto"]].copy()
    for _, r in over_stock.iterrows():
        alerts_rows.append({
            "tipo_alerta": "alquiler_supera_stock",
            "activo_match": r["activo_match"],
            "Especie": r["Especie"],
            "Cartera Neix": r["Cartera Neix"],
            "detalle": f"Alquiler {r['nominal_alquilado_total']:,.0f} > stock final {r['nominal_final_bruto']:,.0f}",
        })

    if not possible_double.empty:
        for _, r in possible_double.iterrows():
            alerts_rows.append({
                "tipo_alerta": "posible_doble_conteo",
                "activo_match": "",
                "Especie": "",
                "Cartera Neix": "",
                "detalle": f"dedup_key={r['dedup_key']} | count={r['count']} | source={r['source_types']}",
            })

    bad_active = rentals_live[rentals_live["activo_match"].eq("")].copy()
    if not bad_active.empty:
        for _, r in bad_active.iterrows():
            alerts_rows.append({
                "tipo_alerta": "activo_no_normalizable",
                "activo_match": "",
                "Especie": _norm(r.get("Especie")),
                "Cartera Neix": _norm(r.get("Cartera Neix")),
                "detalle": f"No se pudo normalizar Activo: {_norm(r.get('Activo'))}",
            })

    alerts = pd.DataFrame(alerts_rows)
    if alerts.empty:
        alerts = pd.DataFrame(columns=["tipo_alerta", "activo_match", "Especie", "Cartera Neix", "detalle"])

    ajuste = match_df.copy()
    ajuste = ajuste[[
        "Especie", "activo_match", "Cartera Neix",
        "nominal_final_bruto",
        "nominal_alquilado_activo",
        "nominal_alquilado_historico_vigente",
        "nominal_alquilado_total",
        "nominal_final_real",
        "precio_final",
        "importe_final_bruto",
        "importe_final_real",
        "Tipo", "Moneda"
    ]].sort_values(["Cartera Neix", "activo_match", "Especie"]).reset_index(drop=True)

    return {
        "portfolio_all": portfolio_all,
        "portfolio_matchable": portfolio_matchable,
        "ajuste": ajuste,
        "match_ok": match_ok.reset_index(drop=True),
        "cartera_sin_match": cartera_sin_match.reset_index(drop=True),
        "alquileres_sin_match": rentals_no_match.reset_index(drop=True),
        "alertas": alerts.reset_index(drop=True),
    }


def build_nominal_detail(df_species: pd.DataFrame, ajuste_df: pd.DataFrame) -> pd.DataFrame:
    if df_species.empty:
        return pd.DataFrame(columns=[
            "Especie", "activo_match", "Tipo", "Moneda",
            "nominal_inicial", "nominal_final_bruto", "nominal_alquilado",
            "nominal_final_real", "variacion_nominal", "movimiento"
        ])

    alquiler_by_activo = (
        ajuste_df.groupby(["activo_match"], as_index=False)
        .agg(
            nominal_alquilado=("nominal_alquilado_total", "sum"),
            nominal_final_real=("nominal_final_real", "sum"),
            nominal_final_bruto=("nominal_final_bruto", "sum"),
        )
        if not ajuste_df.empty else
        pd.DataFrame(columns=["activo_match", "nominal_alquilado", "nominal_final_real", "nominal_final_bruto"])
    )

    out = df_species.copy()
    out["activo_match"] = out["Especie"].apply(_normalize_activo_match_from_portfolio)
    out = out.merge(alquiler_by_activo, on="activo_match", how="left")

    out["nominal_alquilado"] = out["nominal_alquilado"].fillna(0.0)
    out["nominal_final_real"] = np.where(
        out["nominal_final_real"].notna(),
        out["nominal_final_real"],
        out["Cant_Fin"].fillna(0.0),
    )
    out["nominal_final_bruto"] = np.where(
        out["nominal_final_bruto"].notna(),
        out["nominal_final_bruto"],
        out["Cant_Fin"].fillna(0.0),
    )

    out["nominal_inicial"] = out["Cant_Ini"].fillna(0.0)
    out["variacion_nominal"] = out["nominal_final_real"] - out["nominal_inicial"]

    out = out[[
        "Especie", "activo_match", "Tipo", "Moneda",
        "nominal_inicial", "nominal_final_bruto", "nominal_alquilado",
        "nominal_final_real", "variacion_nominal", "Movimiento"
    ]].rename(columns={"Movimiento": "movimiento"})

    return out.sort_values("variacion_nominal", ascending=False).reset_index(drop=True)


def build_valued_detail(df_species: pd.DataFrame, ajuste_df: pd.DataFrame) -> pd.DataFrame:
    if df_species.empty:
        return pd.DataFrame(columns=[
            "Especie", "activo_match", "Tipo", "Moneda",
            "importe_inicial", "importe_final_bruto", "importe_final_real",
            "variacion_bruta", "variacion_real", "movimiento"
        ])

    ajuste_by_activo = (
        ajuste_df.groupby(["activo_match"], as_index=False)
        .agg(
            importe_final_bruto=("importe_final_bruto", "sum"),
            importe_final_real=("importe_final_real", "sum"),
        )
        if not ajuste_df.empty else
        pd.DataFrame(columns=["activo_match", "importe_final_bruto", "importe_final_real"])
    )

    out = df_species.copy()
    out["activo_match"] = out["Especie"].apply(_normalize_activo_match_from_portfolio)
    out = out.merge(ajuste_by_activo, on="activo_match", how="left")

    out["importe_final_bruto"] = np.where(
        out["importe_final_bruto"].notna(),
        out["importe_final_bruto"],
        out["Imp_Fin"].fillna(0.0),
    )
    out["importe_final_real"] = np.where(
        out["importe_final_real"].notna(),
        out["importe_final_real"],
        out["Imp_Fin"].fillna(0.0),
    )

    out["importe_inicial"] = out["Imp_Ini"].fillna(0.0)
    out["variacion_bruta"] = out["importe_final_bruto"] - out["importe_inicial"]
    out["variacion_real"] = out["importe_final_real"] - out["importe_inicial"]

    out = out[[
        "Especie", "activo_match", "Tipo", "Moneda",
        "importe_inicial", "importe_final_bruto", "importe_final_real",
        "variacion_bruta", "variacion_real", "Movimiento"
    ]].rename(columns={"Movimiento": "movimiento"})

    return out.sort_values("variacion_real", ascending=False).reset_index(drop=True)


# =========================================================
# EXPORT
# =========================================================
def build_export(
    h_ini: Dict,
    h_fin: Dict,
    d_ini: pd.DataFrame,
    d_fin: pd.DataFrame,
    df_general: pd.DataFrame,
    df_val: pd.DataFrame,
    df_nom: pd.DataFrame,
    df_ajuste: pd.DataFrame,
    audit_tables: Dict[str, pd.DataFrame],
    rentals_live: pd.DataFrame,
    rentals_summary: pd.DataFrame,
    all_groups: Dict[pd.Timestamp, Dict[str, object]],
) -> bytes:
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([h_ini]).to_excel(writer, sheet_name="Header_Inicio", index=False)
        pd.DataFrame([h_fin]).to_excel(writer, sheet_name="Header_Fin", index=False)
        d_ini.to_excel(writer, sheet_name="Detalle_Inicio", index=False)
        d_fin.to_excel(writer, sheet_name="Detalle_Fin", index=False)

        df_general.to_excel(writer, sheet_name="Comparacion_General", index=False)
        df_val.to_excel(writer, sheet_name="Detalle_Valuado", index=False)
        df_nom.to_excel(writer, sheet_name="Detalle_Nominales", index=False)
        df_ajuste.to_excel(writer, sheet_name="Ajuste_Alquileres", index=False)

        audit_tables["match_ok"].to_excel(writer, sheet_name="match_ok", index=False)
        audit_tables["cartera_sin_match"].to_excel(writer, sheet_name="cartera_sin_match", index=False)
        audit_tables["alquileres_sin_match"].to_excel(writer, sheet_name="alquileres_sin_match", index=False)
        audit_tables["alertas"].to_excel(writer, sheet_name="alertas", index=False)

        rentals_live.to_excel(writer, sheet_name="Alquileres_Vigentes", index=False)
        rentals_summary.to_excel(writer, sheet_name="Alquileres_Resumen", index=False)

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
        return df.style.hide(axis="index")
    return df.reset_index(drop=True).style.format(fmts, na_rep="—").hide(axis="index")


# =========================================================
# RENDER PRINCIPAL
# =========================================================
def render() -> None:
    _inject_css()

    st.title("Cartera Propia Real Ajustada por Alquileres")
    st.markdown(
        f"""
        <p style="color:{MUTED};font-size:0.95rem;margin-top:-0.25rem;margin-bottom:1.1rem;">
        El módulo consolida múltiples archivos de cartera propia por fecha, compara una fecha inicial vs. final,
        descuenta del stock final los nominales alquilados vigentes y construye una visión económica ajustada,
        con trazabilidad completa del cruce, faltantes y alertas.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cp-section">Archivos de entrada</div>', unsafe_allow_html=True)

    files = st.file_uploader(
        "1) Subí los Excel de cartera propia",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="cp_multi_adj",
    )

    c1, c2 = st.columns(2)
    with c1:
        file_active = st.file_uploader(
            "2) Subí alquileres activos / corrientes",
            type=["xlsx", "xls", "xlsm"],
            accept_multiple_files=False,
            key="rent_active",
        )
    with c2:
        file_hist = st.file_uploader(
            "3) Subí alquileres vencidos / históricos",
            type=["xlsx", "xls", "xlsm"],
            accept_multiple_files=False,
            key="rent_hist",
        )

    st.markdown(
        f"""
        <div class="cp-note">
        En esta primera versión el ajuste por alquileres se aplica únicamente sobre la <b>fecha final</b>.
        La fecha inicial permanece <b>bruta</b>. El cruce se realiza por <b>activo_match + Cartera Neix</b>.
        Las posiciones de caja / cuenta corriente se mantienen en la comparación general, pero se excluyen del universo comparable para el ajuste.
        </div>
        """,
        unsafe_allow_html=True,
    )

    run = st.button("Analizar cartera ajustada", use_container_width=True)

    if not run:
        return

    if not files:
        st.warning("Subí al menos dos archivos de cartera para continuar.")
        return

    if file_active is None or file_hist is None:
        st.warning("Necesitás subir tanto el archivo de alquileres activos como el histórico.")
        return

    try:
        groups = consolidate_same_date(files)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error general al procesar los archivos de cartera: {e}")
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

    c1, c2 = st.columns(2)
    fecha_ini = c1.selectbox(
        "Fecha inicial",
        options=unique_dates,
        format_func=lambda x: x.strftime("%d/%m/%Y"),
        index=0,
    )
    fecha_fin = c2.selectbox(
        "Fecha final",
        options=unique_dates,
        format_func=lambda x: x.strftime("%d/%m/%Y"),
        index=len(unique_dates) - 1,
    )

    if fecha_ini >= fecha_fin:
        st.error("La fecha inicial debe ser anterior a la fecha final.")
        return

    h_ini = groups[fecha_ini]["header"]
    d_ini = groups[fecha_ini]["detail"]
    d_ini_raw = groups[fecha_ini]["detail_raw_concat"]

    h_fin = groups[fecha_fin]["header"]
    d_fin = groups[fecha_fin]["detail"]
    d_fin_raw = groups[fecha_fin]["detail_raw_concat"]

    st.info(
        f"**Inicial:** {fecha_ini.strftime('%d/%m/%Y')} · "
        f"{h_ini['cantidad_archivos']} archivo(s) · {h_ini['cantidad_comitentes']} comitente(s)\n\n"
        f"**Final:** {fecha_fin.strftime('%d/%m/%Y')} · "
        f"{h_fin['cantidad_archivos']} archivo(s) · {h_fin['cantidad_comitentes']} comitente(s)"
    )

    # -------------------------
    # ALQUILERES
    # -------------------------
    try:
        df_active = load_rentals(file_active, "activos")
        df_hist = load_rentals(file_hist, "historicos")
    except Exception as e:
        st.error(f"Error leyendo alquileres: {e}")
        return

    df_active_live = filter_active_rentals(df_active, fecha_fin)
    df_hist_live = filter_historical_live_rentals(df_hist, fecha_fin)
    rentals_live, possible_double = consolidate_live_rentals(df_active_live, df_hist_live)
    rentals_summary = summarize_rentals_for_match(rentals_live)

    # -------------------------
    # TABLAS BASE
    # -------------------------
    total_ini = pd.to_numeric(h_ini.get("total_posicion"), errors="coerce")
    total_fin = pd.to_numeric(h_fin.get("total_posicion"), errors="coerce")

    df_species = compare_species(d_ini, d_fin)
    audit_tables = build_adjustment_tables(
        detail_raw_final=d_fin_raw,
        rentals_summary=rentals_summary,
        rentals_live=rentals_live,
        possible_double=possible_double,
    )

    df_ajuste = audit_tables["ajuste"]
    df_nom = build_nominal_detail(df_species, df_ajuste)
    df_val = build_valued_detail(df_species, df_ajuste)

    total_final_ajustado = pd.to_numeric(df_ajuste["importe_final_real"], errors="coerce").fillna(0).sum()
    df_general = compare_headers(h_ini, h_fin, total_final_ajustado)

    # -------------------------
    # KPIS
    # -------------------------
    total_final_bruto = pd.to_numeric(df_ajuste["importe_final_bruto"], errors="coerce").fillna(0).sum()
    nominal_final_bruto_total = pd.to_numeric(df_ajuste["nominal_final_bruto"], errors="coerce").fillna(0).sum()
    nominal_alquilado_total = pd.to_numeric(df_ajuste["nominal_alquilado_total"], errors="coerce").fillna(0).sum()
    nominal_final_real_total = pd.to_numeric(df_ajuste["nominal_final_real"], errors="coerce").fillna(0).sum()
    diff_bruto_vs_ajustado = total_final_bruto - total_final_ajustado

    qty_match = len(audit_tables["match_ok"])
    qty_cartera_sin_match = len(audit_tables["cartera_sin_match"])
    qty_alquileres_sin_match = len(audit_tables["alquileres_sin_match"])
    qty_alertas = len(audit_tables["alertas"])

    total_var_ajustada = total_final_ajustado - total_ini
    total_pct_ajustada = _pct_change(total_final_ajustado, total_ini)

    st.markdown('<div class="cp-section">Resumen ejecutivo</div>', unsafe_allow_html=True)

    def _delta_color(v):
        return SUCCESS if pd.notna(v) and v >= 0 else DANGER

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(_kpi_html(
            "Posición final bruta",
            _fmt_money(total_final_bruto),
            f"vs inicial: {_fmt_money(total_fin - total_ini)} · {_fmt_pct(_pct_change(total_fin, total_ini))}",
            _delta_color(total_fin - total_ini),
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(_kpi_html(
            "Posición final ajustada",
            _fmt_money(total_final_ajustado),
            f"vs inicial: {_fmt_money(total_var_ajustada)} · {_fmt_pct(total_pct_ajustada)}",
            _delta_color(total_var_ajustada),
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(_kpi_html(
            "Nominal alquilado total",
            _fmt_num(nominal_alquilado_total, 0),
            f"Nominal bruto: {_fmt_num(nominal_final_bruto_total, 0)} · real: {_fmt_num(nominal_final_real_total, 0)}",
            WARNING,
        ), unsafe_allow_html=True)

    with k4:
        st.markdown(_kpi_html(
            "Diferencia bruto vs ajustado",
            _fmt_money(diff_bruto_vs_ajustado),
            f"Match {qty_match} · cartera sin match {qty_cartera_sin_match} · alquileres sin match {qty_alquileres_sin_match} · alertas {qty_alertas}",
            DANGER if diff_bruto_vs_ajustado > 0 else MUTED,
        ), unsafe_allow_html=True)

    # -------------------------
    # COMPARACION GENERAL
    # -------------------------
    st.markdown('<div class="cp-section">1. Comparación general</div>', unsafe_allow_html=True)
    st.dataframe(
        _styled(df_general, {
            "Inicio": "$ {:,.2f}",
            "Fin Bruto": "$ {:,.2f}",
            "Fin Ajustado": "$ {:,.2f}",
            "Variación Bruta $": "$ {:+,.2f}",
            "Variación Bruta %": "{:+,.2f}%",
            "Variación Ajustada $": "$ {:+,.2f}",
            "Variación Ajustada %": "{:+,.2f}%",
        }),
        use_container_width=True,
        height=260,
    )

    # -------------------------
    # DETALLE VALUADO
    # -------------------------
    st.markdown('<div class="cp-section">2. Detalle por especie valuado</div>', unsafe_allow_html=True)
    st.dataframe(
        _styled(df_val, {
            "importe_inicial": "$ {:,.2f}",
            "importe_final_bruto": "$ {:,.2f}",
            "importe_final_real": "$ {:,.2f}",
            "variacion_bruta": "$ {:+,.2f}",
            "variacion_real": "$ {:+,.2f}",
        }),
        use_container_width=True,
        height=500,
    )

    # -------------------------
    # DETALLE NOMINALES
    # -------------------------
    st.markdown('<div class="cp-section">3. Detalle por nominales</div>', unsafe_allow_html=True)
    st.dataframe(
        _styled(df_nom, {
            "nominal_inicial": "{:,.0f}",
            "nominal_final_bruto": "{:,.0f}",
            "nominal_alquilado": "{:,.0f}",
            "nominal_final_real": "{:,.0f}",
            "variacion_nominal": "{:+,.0f}",
        }),
        use_container_width=True,
        height=500,
    )

    # -------------------------
    # AJUSTE POR ALQUILERES
    # -------------------------
    st.markdown('<div class="cp-section">4. Ajuste por alquileres</div>', unsafe_allow_html=True)
    st.dataframe(
        _styled(df_ajuste, {
            "nominal_final_bruto": "{:,.0f}",
            "nominal_alquilado_activo": "{:,.0f}",
            "nominal_alquilado_historico_vigente": "{:,.0f}",
            "nominal_alquilado_total": "{:,.0f}",
            "nominal_final_real": "{:,.0f}",
            "precio_final": "{:,.4f}",
            "importe_final_bruto": "$ {:,.2f}",
            "importe_final_real": "$ {:,.2f}",
        }),
        use_container_width=True,
        height=520,
    )

    # -------------------------
    # AUDITORIA
    # -------------------------
    st.markdown('<div class="cp-section">5. Auditoría de cruce</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "match_ok",
        "cartera_sin_match",
        "alquileres_sin_match",
        "alertas",
    ])

    with tab1:
        if audit_tables["match_ok"].empty:
            st.info("No hay matches registrados.")
        else:
            st.dataframe(
                _styled(
                    audit_tables["match_ok"][[
                        "Especie", "activo_match", "Cartera Neix",
                        "nominal_final_bruto", "nominal_alquilado_total",
                        "nominal_final_real", "precio_final", "importe_final_real"
                    ]],
                    {
                        "nominal_final_bruto": "{:,.0f}",
                        "nominal_alquilado_total": "{:,.0f}",
                        "nominal_final_real": "{:,.0f}",
                        "precio_final": "{:,.4f}",
                        "importe_final_real": "$ {:,.2f}",
                    }
                ),
                use_container_width=True,
                height=360,
            )

    with tab2:
        if audit_tables["cartera_sin_match"].empty:
            st.info("No hay especies de cartera sin match.")
        else:
            st.dataframe(
                _styled(
                    audit_tables["cartera_sin_match"][[
                        "Especie", "activo_match", "Cartera Neix",
                        "nominal_final_bruto", "precio_final", "importe_final_bruto"
                    ]],
                    {
                        "nominal_final_bruto": "{:,.0f}",
                        "precio_final": "{:,.4f}",
                        "importe_final_bruto": "$ {:,.2f}",
                    }
                ),
                use_container_width=True,
                height=360,
            )

    with tab3:
        if audit_tables["alquileres_sin_match"].empty:
            st.info("No hay alquileres sin match.")
        else:
            st.dataframe(
                _styled(
                    audit_tables["alquileres_sin_match"],
                    {
                        "nominal_alquilado_activo": "{:,.0f}",
                        "nominal_alquilado_historico_vigente": "{:,.0f}",
                        "nominal_alquilado_total": "{:,.0f}",
                        "cantidad_registros_alquiler": "{:,.0f}",
                    }
                ),
                use_container_width=True,
                height=360,
            )

    with tab4:
        if audit_tables["alertas"].empty:
            st.success("No se detectaron alertas relevantes.")
        else:
            st.dataframe(
                _styled(audit_tables["alertas"], {}),
                use_container_width=True,
                height=360,
            )

    # -------------------------
    # EXTRA: MOVIMIENTOS
    # -------------------------
    st.markdown('<div class="cp-section">Movimiento por especie</div>', unsafe_allow_html=True)

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

    # -------------------------
    # EXPORT
    # -------------------------
    st.markdown('<div class="cp-section">Exportar</div>', unsafe_allow_html=True)

    export_bytes = build_export(
        h_ini=h_ini,
        h_fin=h_fin,
        d_ini=d_ini_raw,
        d_fin=d_fin_raw,
        df_general=df_general,
        df_val=df_val,
        df_nom=df_nom,
        df_ajuste=df_ajuste,
        audit_tables=audit_tables,
        rentals_live=rentals_live,
        rentals_summary=rentals_summary,
        all_groups=groups,
    )

    st.download_button(
        label="Descargar Excel de cartera ajustada + auditoría",
        data=export_bytes,
        file_name="cartera_propia_real_ajustada_por_alquileres.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
