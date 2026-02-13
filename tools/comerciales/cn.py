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
    # ojo: no queremos "None" / "NaT" visibles
    s = s.replace({"nan": "", "None": "", "NaT": ""})
    return s


def _letters_ratio(s: pd.Series, n: int = 80) -> float:
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

    if canonical in ("MANAGER", "OFICIAL"):
        best = None
        best_score = -1.0
        for c in candidates:
            score = _letters_ratio(df[c])
            if score > best_score:
                best_score = score
                best = c
        return best

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

        if canonical == "MANAGER":
            for k, original in key_map.items():
                if k in MANAGER_NAME_HINTS or ("manager" in k and "nombre" in k):
                    hits.append(original)

        if canonical == "OFICIAL":
            for k, original in key_map.items():
                if k in OFICIAL_NAME_HINTS or ("oficial" in k and "nombre" in k):
                    hits.append(original)

        seen = set()
        hits = [h for h in hits if not (h in seen or seen.add(h))]

        chosen = _pick_best_column(df, hits, canonical)
        if chosen:
            rename[chosen] = canonical

    if rename:
        df = df.rename(columns=rename)

    return df


# =========================================================
# FECHA: primero a TEXTO, después parseo por hoja
# =========================================================
_DMY_RE = re.compile(r"^\s*(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\s*$")
_YMD_RE = re.compile(r"^\s*(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})\s*$")
_NUMERIC_RE = re.compile(r"^\s*\d+(\.\d+)?\s*$")


def _parse_excel_serial(raw_text: pd.Series) -> pd.Series:
    num = pd.to_numeric(raw_text, errors="coerce")
    dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return dt


def _choose_dayfirst_from_text(raw_text: pd.Series) -> bool:
    """
    SOLO MIRAMOS TEXTO.
    Decide dayfirst por hoja:
    - evidencia >12 en primer/segundo bloque
    - si no hay evidencia => default AR
    """
    s = raw_text.dropna()
    if s.empty:
        return True

    sample = s.head(250).tolist()

    first_gt_12 = 0
    second_gt_12 = 0
    total = 0

    for v in sample:
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

    if total == 0:
        return True

    if first_gt_12 > second_gt_12:
        return True
    if second_gt_12 > first_gt_12:
        return False

    return True


def _parse_fecha_text_then_parse_per_sheet(x: pd.Series) -> pd.Series:
    """
    1) Convertimos a TEXTO (sin perder nada)
    2) Si parece serial Excel => convertimos serial
    3) Si parece ISO yyyy-mm-dd => parse directo
    4) Si es dd/mm o mm/dd => elegimos dayfirst por hoja y parseamos
    5) Salida final como TEXTO YYYY-MM-DD (sin hora) para Looker
    """
    raw_text = _clean_text_series(x)

    # ------------- Caso A: serial excel (muchos valores numéricos puros)
    sample = raw_text.dropna()
    if not sample.empty:
        sample_head = sample.head(200).tolist()
        numeric_like = sum(bool(_NUMERIC_RE.match(v)) for v in sample_head)
        ratio = numeric_like / len(sample_head)

        if ratio > 0.75:
            dt = _parse_excel_serial(raw_text)
            out = pd.Series([""] * len(raw_text), index=raw_text.index, dtype="object")
            m = dt.notna()
            out.loc[m] = dt.loc[m].dt.strftime("%Y-%m-%d")
            return out

    # ------------- Caso B: ISO (yyyy-mm-dd)
    # si la mayoría calza con yyyy-mm-dd, parse directo
    sample2 = raw_text.dropna()
    if not sample2.empty:
        head = sample2.head(200).tolist()
        iso_hits = sum(bool(_YMD_RE.match(v)) for v in head)
        if (iso_hits / len(head)) > 0.6:
            dt = pd.to_datetime(raw_text, errors="coerce")
            out = pd.Series([""] * len(raw_text), index=raw_text.index, dtype="object")
            m = dt.notna()
            out.loc[m] = dt.loc[m].dt.strftime("%Y-%m-%d")
            return out

    # ------------- Caso C: dd/mm o mm/dd
    dayfirst = _choose_dayfirst_from_text(raw_text)
    dt = pd.to_datetime(raw_text, errors="coerce", dayfirst=dayfirst)

    out = pd.Series([""] * len(raw_text), index=raw_text.index, dtype="object")
    m = dt.notna()
    out.loc[m] = dt.loc[m].dt.strftime("%Y-%m-%d")
    return out


# =========================================================
# Sheet read
# =========================================================
def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    for c in OUTPUT_COLS:
        if c not in df.columns:
            return None

    df = df[OUTPUT_COLS].copy()

    # ✅ Fecha: primero texto, luego parse por hoja, salida final texto YYYY-MM-DD
    df["Fecha"] = _parse_fecha_text_then_parse_per_sheet(df["Fecha"])

    # Texto limpio
    df["Cuenta"] = _clean_text_series(df["Cuenta"])
    df["Producto"] = _clean_text_series(df["Producto"])
    df["Id_Off"] = _clean_text_series(df["Id_Off"])
    df["MANAGER"] = _clean_text_series(df["MANAGER"])
    df["OFICIAL"] = _clean_text_series(df["OFICIAL"])

    # ✅ Neto/Gross raw
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
