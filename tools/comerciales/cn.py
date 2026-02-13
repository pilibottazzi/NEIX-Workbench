# tools/comercial/cn.py
from __future__ import annotations

import io
import re
from typing import List, Optional

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

  /* Botón Excel full ancho */
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
    """Limpieza mínima segura (no toca comas/puntos ni fechas)."""
    s = x.astype(str)
    s = s.str.replace("\u00a0", " ", regex=False)  # NBSP
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.replace({"nan": "", "None": "", "NaT": ""})


def _letters_ratio(s: pd.Series, n: int = 80) -> float:
    """
    % aproximado de letras en el contenido (para elegir Nombre vs Id).
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


def _pick_most_textual(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Elige la columna con más letras (nombre) frente a una numérica (id).
    """
    best = candidates[0]
    best_score = -1.0
    for c in candidates:
        score = _letters_ratio(df[c])
        if score > best_score:
            best_score = score
            best = c
    return best


# =========================================================
# Column resolution (MANAGER/OFICIAL como NOMBRES)
# =========================================================
def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas a los canónicos de OUTPUT_COLS.

    REGLA IMPORTANTE:
    - Para MANAGER y OFICIAL siempre prioriza la columna "Nombre" (más textual)
      si existen ambas (Id y Nombre).
    """
    cols = list(df.columns)
    key_map = {_key(c): c for c in cols}

    def find_candidates(keys_like: List[str]) -> List[str]:
        hits = []
        for k, orig in key_map.items():
            for pat in keys_like:
                if pat in k:
                    hits.append(orig)
                    break
        # dedupe manteniendo orden
        seen = set()
        return [h for h in hits if not (h in seen or seen.add(h))]

    rename = {}

    # Fecha
    for cand in ["fecha", "fec", "date"]:
        if cand in key_map:
            rename[key_map[cand]] = "Fecha"
            break

    # Cuenta
    for cand in ["cuenta", "cta", "account"]:
        if cand in key_map:
            rename[key_map[cand]] = "Cuenta"
            break

    # Producto
    for cand in ["producto", "product"]:
        if cand in key_map:
            rename[key_map[cand]] = "Producto"
            break

    # Neto / Gross
    for cand in ["neto agente", "neto"]:
        if cand in key_map:
            rename[key_map[cand]] = "Neto Agente"
            break

    for cand in ["gross agente", "gross"]:
        if cand in key_map:
            rename[key_map[cand]] = "Gross Agente"
            break

    # Id_Off (id del oficial del sistema) - lo dejamos como id
    idoff_candidates = find_candidates(["id_off", "id off", "idoff", "id oficial", "id_oficial"])
    if idoff_candidates:
        rename[idoff_candidates[0]] = "Id_Off"

    # MANAGER: buscar "manager" y priorizar NOMBRE
    mgr_candidates = find_candidates(["manager"])
    if mgr_candidates:
        chosen = _pick_most_textual(df, mgr_candidates)
        rename[chosen] = "MANAGER"

    # OFICIAL: buscar "oficial" y priorizar NOMBRE
    ofi_candidates = find_candidates(["oficial"])
    if ofi_candidates:
        chosen = _pick_most_textual(df, ofi_candidates)
        rename[chosen] = "OFICIAL"

    return df.rename(columns=rename) if rename else df


# =========================================================
# Read sheet
# =========================================================
def _read_one_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        # dtype=str: NO tocamos decimales ni fechas
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
    except Exception:
        return None

    df.columns = [_norm_col(c) for c in df.columns]
    df = _resolve_columns(df)

    # Validación mínima
    if any(c not in df.columns for c in OUTPUT_COLS):
        return None

    df = df[OUTPUT_COLS].copy()

    # ✅ Fecha: DEJAR TAL CUAL VIENE (solo limpieza)
    df["Fecha"] = _clean_text_series(df["Fecha"])

    # ✅ MANAGER/OFICIAL ya quedaron como NOMBRES por el resolver
    # Limpieza mínima en el resto
    for c in ["Cuenta", "Producto", "Id_Off", "MANAGER", "OFICIAL", "Neto Agente", "Gross Agente"]:
        df[c] = _clean_text_series(df[c])

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
        st.error("No pude leer el archivo. Probá guardarlo como .xlsx y re-subirlo.")
        return

    dfs: List[pd.DataFrame] = []
    for s in SHEETS:
        one = _read_one_sheet(xls, s)
        if one is not None:
            dfs.append(one)

    if not dfs:
        st.warning("No encontré hojas válidas con las columnas esperadas.")
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

