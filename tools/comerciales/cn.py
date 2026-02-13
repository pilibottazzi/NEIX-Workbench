# tools/comercial/cn.py
from __future__ import annotations

import io
import re
from typing import List, Optional, Dict, Tuple

import pandas as pd
import streamlit as st


# =========================================================
# Config
# =========================================================
SHEETS = ["WSC A", "WSC B", "INSIGNEO"]

OUTPUT_COLS = [
    "Fecha",
    "Cuenta",
    "Producto",
    "Neto Agente",
    "Gross Agente",
    "Id_Off",
    "MANAGER",
    "OFICIAL",
]

NEIX_RED = "#ff3b30"


# =========================================================
# UI
# =========================================================
def _inject_css() -> None:
    st.markdown(
        f"""
<style>
  .block-container {{
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
  }}

  /* Download button full width */
  div[data-testid="stDownloadButton"] {{
    width: 100% !important;
  }}
  div[data-testid="stDownloadButton"] > button {{
    width: 100% !important;
    background: {NEIX_RED} !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    padding: 0.95rem 1rem !important;
    border: 0 !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers
# =========================================================
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _key(s: str) -> str:
    return _norm_col(s).lower().replace("_", " ")


def _clean_text_series(x: pd.Series) -> pd.Series:
    """
    Limpieza mínima y segura:
    - NO cambia coma/punto
    - solo limpia espacios y NBSP
    """
    s = x.astype(str)
    s = s.str.replace("\u00a0", " ", regex=False)  # NBSP
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.replace({"nan": ""})
    return s


def _letters_ratio(s: pd.Series, n: int = 80) -> float:
    """
    Cuánto "texto" tiene una columna (para distinguir Nombre vs Id).
    """
    sample = _clean_text_series(s).dropna().astype(str)
    if sample.empty:
        return 0.0
    sample = sample.head(n)
    joined = " ".join(sample.tolist())
    if not joined:
        return 0.0
    letters = sum(ch.isalpha() for ch in joined)
    return letters / max(1, len(joined))


# =========================================================
# Column resolution (Manager/Oficial: nombre vs id)
# =========================================================
ALIASES: Dict[str, List[str]] = {
    "Fecha": ["fecha", "fec", "date"],
    "Cuenta": ["cuenta", "cta", "account"],
    "Producto": ["producto", "product"],
    "Neto Agente": ["neto agente", "neto", "net agent", "netoagente"],
    "Gross Agente": ["gross agente", "gross", "gross agent", "grossagente"],
    "Id_Off": ["id off", "id_off", "id of", "idoff", "id oficial", "id_oficial", "oficial id"],
    "MANAGER": ["manager", "nombre manager", "manager nombre", "managername", "managernombre"],
    "OFICIAL": ["oficial", "nombre oficial", "oficial nombre", "oficialname", "oficialnombre"],
}

MANAGER_NAME_HINTS = {"managernombre", "manager nombre", "nombre manager", "manager name"}
OFICIAL_NAME_HINTS = {"oficialnombre", "oficial nombre", "nombre oficial", "oficial name"}


def _pick_best_column(df: pd.DataFrame, candidates: List[str], canonical: str) -> Optional[str]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Para MANAGER/OFICIAL: priorizar la más "texto" (nombre) por sobre id numérico
    if canonical in ("MANAGER", "OFICIAL"):
        best = None
        best_score = -1.0
        for c in candidates:
            score = _letters_ratio(df[c])
            if score > best_score:
                best_score = score
                best = c
        return best

    # Para Id_Off: priorizar la más numérica (menos letras)
    if canonical == "Id_Off":
        best = None
        best_score = 10.0
        for c in candidates:
            score = _letters_ratio(df[c])
            if score < best_score:
                best_score = score
                best = c
        return best

    return candidates[0]


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    key_map = {_key(c): c for c in cols}

    rename: Dict[str, str] = {}

    for canonical, aliases in ALIASES.items():
        hits: List[str] = []
        for cand in [canonical] + aliases:
            k = _key(cand)
            if k in key_map:
                hits.append(key_map[k])

        # Extra: detectar nombre manager/oficial aunque el alias no matchee exacto
        if canonical == "MANAGER":
            for k, original in key_map.items():
                if k in MANAGER_NAME_HINTS or ("manager" in k and "nombre" in k):
                    hits.append(original)
        if canonical == "OFICIAL":
            for k, original in key_map.items():
                if k in OFICIAL_NAME_HINTS or ("oficial" in k and "nombre" in k):
                    hits.append(original)

        # dedupe
        seen = set()
        hits = [h for h in hits if not (h in seen or seen.add(h))]

        chosen = _pick_best_column(df, hits, canonical)
        if chosen:
            rename[chosen] = canonical

    if rename:
        df = df.rename(columns=rename)

    return df


# =========================================================
# FECHA: detectar por hoja ARG vs YANKEE + soportar serial Excel
# =========================================================
# dd/mm/yyyy o mm/dd/yyyy (con / - .)
_DMY_RE = re.compile(r"^\s*(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})")
# yyyy-mm-dd ...
_YMD_RE = re.compile(r"^\s*(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})")
# serial excel "45234" o "45234.0"
_NUMERIC_RE = re.compile(r"^\s*\d+(\.\d+)?\s*$")


def _parse_excel_serial_to_datetime(raw: pd.Series) -> pd.Series:
    """
    Excel serial date: días desde 1899-12-30 (convención de pandas).
    """
    num = pd.to_numeric(raw, errors="coerce")
    # descartamos valores muy chicos tipo "1", "2" que no son fechas reales del reporte
    # (pero si llegaran, igual convierte a 1899...)
    dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return dt


def _infer_dayfirst_and_parse(raw: pd.Series) -> Tuple[bool, pd.Series]:
    """
    Decide por hoja si dayfirst True/False y parsea.
    Reglas:
      1) Si el string es yyyy-mm-dd => parse directo (no depende de dayfirst)
      2) Si hay evidencia >12: decide
      3) Si no hay evidencia: elige el parse con menos NaT; si empata, AR (dayfirst=True)
    """
    s = raw.dropna().astype(str)
    if s.empty:
        return True, pd.to_datetime(raw, errors="coerce", dayfirst=True)

    # Si la mayoría está en formato yyyy-mm-dd, no hace falta inferir
    ymd_hits = 0
    checked = 0
    for v in s.head(200).tolist():
        checked += 1
        if _YMD_RE.match(v):
            ymd_hits += 1
    if checked and (ymd_hits / checked) > 0.6:
        dt = pd.to_datetime(raw, errors="coerce")  # ISO safe
        return True, dt

    # Evidencia dd/mm vs mm/dd
    first_gt_12 = 0
    second_gt_12 = 0
    total = 0

    for v in s.head(250).tolist():
        m = _DMY_RE.match(v)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2))
        total += 1
        if a > 12:
            first_gt_12 += 1
        if b > 12:
            second_gt_12 += 1

    if total > 0:
        if first_gt_12 > second_gt_12:
            dayfirst = True
            return dayfirst, pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst)
        if second_gt_12 > first_gt_12:
            dayfirst = False
            return dayfirst, pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst)
        # si empata, caemos al método de menor NaT

    dt_ar = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    dt_us = pd.to_datetime(raw, errors="coerce", dayfirst=False)

    ar_bad = float(dt_ar.isna().mean())
    us_bad = float(dt_us.isna().mean())

    if ar_bad < us_bad:
        return True, dt_ar
    if us_bad < ar_bad:
        return False, dt_us

    # empate => default Argentina
    return True, dt_ar


def _parse_fecha_por_hoja(x: pd.Series) -> pd.Series:
    """
    Parser final por hoja:
    1) limpia texto
    2) si parece serial Excel en proporción alta => convierte como serial
    3) si no => infiere dayfirst por hoja y parsea
    4) normaliza a string YYYY-MM-DD (sin hora) para Looker
    """
    raw = _clean_text_series(x)

    # ¿Parece serial Excel?
    sample = raw.dropna().astype(str)
    if not sample.empty:
        sample = sample.head(200)
        numeric_like = sum(bool(_NUMERIC_RE.match(v)) for v in sample.tolist())
        ratio = numeric_like / len(sample)

        # Si la mayoría son números "puros", asumimos serial Excel
        if ratio > 0.75:
            dt = _parse_excel_serial_to_datetime(raw)
        else:
            _, dt = _infer_dayfirst_and_parse(raw)
    else:
        _, dt = _infer_dayfirst_and_parse(raw)

    # Normalizar a fecha para Looker (YYYY-MM-DD)
    out = pd.Series([""] * len(raw), index=raw.index, dtype="object")
    mask = dt.notna()
    out.loc[mask] = dt.loc[mask].dt.strftime("%Y-%m-%d")
    return out


# =========================================================
# Sheet read
# =========================================================
def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        # dtype=str para NO romper neto/gross ni ids/nombres
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    # si faltan columnas mínimas -> ignorar
    for c in OUTPUT_COLS:
        if c not in df.columns:
            return None

    df = df[OUTPUT_COLS].copy()

    # ✅ Fecha: detectar por HOJA si ARG/YANKEE + soportar serial Excel + unificar a YYYY-MM-DD
    df["Fecha"] = _parse_fecha_por_hoja(df["Fecha"])

    # Texto limpio (sin modificar separadores decimales)
    df["Cuenta"] = _clean_text_series(df["Cuenta"])
    df["Producto"] = _clean_text_series(df["Producto"])
    df["Id_Off"] = _clean_text_series(df["Id_Off"])
    df["MANAGER"] = _clean_text_series(df["MANAGER"])
    df["OFICIAL"] = _clean_text_series(df["OFICIAL"])

    # ✅ Neto/Gross: TAL CUAL VIENEN (solo limpieza mínima)
    df["Neto Agente"] = _clean_text_series(df["Neto Agente"])
    df["Gross Agente"] = _clean_text_series(df["Gross Agente"])

    df.insert(0, "Banco", sheet_name)
    return df


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Consolidado")
    bio.seek(0)
    return bio.read()


# =========================================================
# Entrypoint
# =========================================================
def render(back_to_home=None) -> None:
    _inject_css()

    up = st.file_uploader(
        "CN: Subí el Excel para consolidar bancos",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="cn_bancos_uploader",
    )
    if not up:
        return

    try:
        xls = pd.ExcelFile(io.BytesIO(up.getvalue()))
    except Exception:
        return

    dfs: List[pd.DataFrame] = []
    for s in SHEETS:
        one = _read_one_sheet(xls, s)
        if one is not None:
            dfs.append(one)

    if not dfs:
        return

    df_all = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    st.download_button(
        "Excel",
        data=_to_excel_bytes(df_all),
        file_name="cn_bancos_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("### Consolidado")
    try:
        st.dataframe(df_all, use_container_width=True, height=620, hide_index=True)
    except TypeError:
        st.dataframe(df_all, use_container_width=True, height=620)
