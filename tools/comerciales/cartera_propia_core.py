#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cartera_propia_core.py
# Lógica de conciliación de cartera propia — sin dependencias de Streamlit.

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG BASE
# =============================================================================
CUENTAS_DEFAULT = [904, 905, 906, 907, 908, 909, 910, 992, 997, 999]

DEFAULT_PARES_COMP = [
    (904, 992),
    (997, 999),
]

# =============================================================================
# HELPERS GENERALES
# =============================================================================

def _strip_accents(text: str) -> str:
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    x = _strip_accents(str(x))
    x = re.sub(r"\s+", " ", x.strip())
    return x.upper()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        normalize_text(c).lower().replace(" ", "_").replace(".", "").replace("/", "_")
        for c in out.columns
    ]
    return out


def find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """
    Convierte una serie de strings a float manejando formatos ARS y USD.

    FIX respecto a versión anterior: detecta el formato numérico ANTES de
    normalizar separadores, evitando que '1234.56' se convierta en '123456'
    por el reemplazo ciego de puntos.

    Lógica de detección:
    - Tiene coma Y punto → el último de los dos es el decimal
      · '1.234,56' → europeo  (punto=miles, coma=decimal)
      · '1,234.56' → anglosajón (coma=miles, punto=decimal)
    - Solo coma → si hay ≤ 2 dígitos después de la última coma, es decimal
    - Solo punto → punto como decimal (estándar)
    """
    s = series.astype(str).str.strip()
    s = s.str.replace("\u00a0", "", regex=False)   # espacio no separable
    s = s.str.replace("U$S",   "", regex=False)
    s = s.str.replace("US$",   "", regex=False)
    s = s.str.replace("$",     "", regex=False)
    s = s.str.replace("%",     "", regex=False)
    s = s.str.replace(r"^\((.+)\)$", r"-\1", regex=True)   # (1234) → -1234
    s = s.str.replace(r"^([^-].+)-$", r"-\1", regex=True)  # 1234-  → -1234
    s = s.str.strip()

    def _parse_one(val: str) -> float:
        val = val.strip()
        if val in ("", "nan", "none", "-", "n/a", "nd"):
            return 0.0
        has_dot   = "." in val
        has_comma = "," in val
        try:
            if has_dot and has_comma:
                last_dot   = val.rfind(".")
                last_comma = val.rfind(",")
                if last_comma > last_dot:
                    # europeo: 1.234,56
                    val = val.replace(".", "").replace(",", ".")
                else:
                    # anglosajón: 1,234.56
                    val = val.replace(",", "")
            elif has_comma and not has_dot:
                parts = val.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    val = val.replace(",", ".")   # decimal: 1234,56
                else:
                    val = val.replace(",", "")    # miles: 1,234,567
            # solo punto o sin separadores: ya está bien
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    return pd.Series([_parse_one(v) for v in s], index=series.index, dtype=float)


def format_num(x: Any, decimals: int = 2) -> str:
    try:
        return f"{float(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0"


def _empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cuenta", "especie_h", "moneda",
            "ini", "act", "fin", "comp",
            "ini_ajust", "act_ajust", "fin_ajust", "comp_ajust",
            "dif_cant", "dif_final_original", "dif_final", "dif_importe",
            "precio_ref", "status", "reglas_aplicadas", "match_status",
        ]
    )


def _safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# LECTURA DE ARCHIVOS
# =============================================================================

def read_any_file(file_obj_or_path: Any) -> pd.DataFrame:
    """Lee csv/xlsx/xls/xlsm desde path, string o UploadedFile de Streamlit."""
    if file_obj_or_path is None:
        return pd.DataFrame()

    if isinstance(file_obj_or_path, (str, Path)):
        path = Path(file_obj_or_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_csv(path, sep=";", encoding="latin1")
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Formato no soportado: {suffix}")

    name = getattr(file_obj_or_path, "name", "uploaded_file")
    suffix = Path(name).suffix.lower()
    if hasattr(file_obj_or_path, "seek"):
        file_obj_or_path.seek(0)
    if suffix == ".csv":
        try:
            return pd.read_csv(file_obj_or_path)
        except Exception:
            if hasattr(file_obj_or_path, "seek"):
                file_obj_or_path.seek(0)
            return pd.read_csv(file_obj_or_path, sep=";", encoding="latin1")
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        if hasattr(file_obj_or_path, "seek"):
            file_obj_or_path.seek(0)
        return pd.read_excel(file_obj_or_path)
    raise ValueError(f"Formato no soportado: {suffix}")


# =============================================================================
# PREPARACIÓN DE INPUTS
# =============================================================================

def _prep_df(
    df_raw: pd.DataFrame,
    nombre: str,
    col_cantidad_candidates: list[str] | None = None,
    col_importe_candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    alerts: list[str] = []
    if df_raw is None or df_raw.empty:
        alerts.append(f"Archivo de {nombre} vacío.")
        return pd.DataFrame(columns=["cuenta", "especie_h", "moneda", "cantidad", "importe"]), alerts

    df = normalize_columns(df_raw)

    c_cuenta   = find_first_existing(df, ["cuenta", "comitente", "codigo", "nro_cuenta"])
    c_especie  = find_first_existing(df, ["especie", "instrumento", "ticker", "simbolo", "codigo_especie"])
    c_moneda   = find_first_existing(df, ["moneda", "currency"])
    c_cantidad = find_first_existing(df, col_cantidad_candidates or ["cantidad", "cant", "nominales", "nominal"])
    c_importe  = find_first_existing(df, col_importe_candidates or ["importe", "monto", "valor_mercado", "market_value", "total"])

    if c_cuenta is None or c_especie is None:
        alerts.append(f"No se detectaron columnas clave de {nombre}.")

    out = pd.DataFrame()
    out["cuenta"]    = pd.to_numeric(df[c_cuenta], errors="coerce") if c_cuenta else np.nan
    out["especie_h"] = df[c_especie].astype(str).map(normalize_text) if c_especie else ""
    out["moneda"]    = df[c_moneda].astype(str).map(normalize_text) if c_moneda else "ARS"
    out["cantidad"]  = to_numeric_safe(df[c_cantidad]) if c_cantidad else 0.0
    out["importe"]   = to_numeric_safe(df[c_importe])  if c_importe  else 0.0

    out = out.dropna(subset=["cuenta"])
    out["cuenta"] = out["cuenta"].astype(int)
    out = out[out["especie_h"] != ""].copy()
    out = (
        out.groupby(["cuenta", "especie_h", "moneda"], as_index=False)
        .agg({"cantidad": "sum", "importe": "sum"})
    )
    return out, alerts


def preparar_posiciones(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return _prep_df(df_raw, "posiciones")


def preparar_activity(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return _prep_df(
        df_raw, "activity",
        col_importe_candidates=["importe", "monto", "net_amount", "amount", "total"],
    )


def preparar_portafolio(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return _prep_df(df_raw, "portafolio")


# =============================================================================
# MODELO DE PERÍODO
# =============================================================================

@dataclass
class PeriodoConciliacion:
    fecha_ini: str
    fecha_fin: str
    cuentas: list[int] = field(default_factory=lambda: list(CUENTAS_DEFAULT))
    pares_comp: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_PARES_COMP))

    @property
    def fecha_ini_ts(self) -> pd.Timestamp:
        return pd.to_datetime(self.fecha_ini)

    @property
    def fecha_fin_ts(self) -> pd.Timestamp:
        return pd.to_datetime(self.fecha_fin)


# =============================================================================
# SISTEMA DE REGLAS ENCHUFABLE
# =============================================================================

@dataclass
class Regla:
    """
    Una regla de conciliación vectorizada.

    fn(df, tolerance) → pd.Series[str]
      Recibe el DataFrame de trabajo (con ini, act, fin, pos, dif_final,
      moneda, especie_h, cuenta) y devuelve una Serie de strings:
      tag si aplica, '' si no aplica.

    Ejemplo de regla custom:
        def r_al_series(df, tol):
            mask = df["especie_h"].str.startswith("AL")
            return pd.Series(np.where(mask, "R4_AL_SERIES", ""), index=df.index)

        engine.add(Regla("al_series", "R4_AL_SERIES", r_al_series, "Bonos AL"))
    """
    nombre:      str
    tag:         str
    fn:          Callable[[pd.DataFrame, float], pd.Series]
    descripcion: str = ""


# ── implementaciones de las 4 reglas base ────────────────────────────────────

def _r0_cierre_exacto(df: pd.DataFrame, tol: float) -> pd.Series:
    return pd.Series(np.where(df["dif_final"].abs() <= tol, "R0_CIERRE_EXACTO", ""), index=df.index)


def _r1_solo_posiciones(df: pd.DataFrame, tol: float) -> pd.Series:
    mask = (df["fin"].abs() <= tol) & (df["pos"].abs() > tol)
    return pd.Series(np.where(mask, "R1_SOLO_POSICIONES", ""), index=df.index)


def _r2_sin_activity(df: pd.DataFrame, tol: float) -> pd.Series:
    return pd.Series(np.where(df["act"].abs() <= tol, "R2_SIN_ACTIVITY", ""), index=df.index)


def _r3_sin_inicial(df: pd.DataFrame, tol: float) -> pd.Series:
    return pd.Series(np.where(df["ini"].abs() <= tol, "R3_SIN_INICIAL", ""), index=df.index)


DEFAULT_REGLAS: list[Regla] = [
    Regla("cierre_exacto",   "R0_CIERRE_EXACTO",   _r0_cierre_exacto,   "Diferencia dentro de tolerancia"),
    Regla("solo_posiciones", "R1_SOLO_POSICIONES",  _r1_solo_posiciones, "Sin fin en portafolio pero sí en posiciones"),
    Regla("sin_activity",    "R2_SIN_ACTIVITY",     _r2_sin_activity,    "Sin movimientos en el período"),
    Regla("sin_inicial",     "R3_SIN_INICIAL",      _r3_sin_inicial,     "Especie nueva, sin posición inicial"),
]


class RuleEngine:
    """
    Motor de reglas vectorizado y enchufable.

    Uso básico (reglas por defecto):
        engine = RuleEngine()

    Agregar una regla extra:
        engine.add(Regla("mi_regla", "R4_X", mi_fn, "descripción"))

    Quitar una regla:
        engine.remove("sin_activity")

    Reemplazar una regla:
        engine.replace("sin_activity", Regla(...))
    """

    def __init__(self, reglas: list[Regla] | None = None):
        self._reglas: list[Regla] = list(reglas if reglas is not None else DEFAULT_REGLAS)

    def add(self, regla: Regla) -> None:
        self._reglas.append(regla)

    def remove(self, nombre: str) -> None:
        self._reglas = [r for r in self._reglas if r.nombre != nombre]

    def replace(self, nombre: str, nueva: Regla) -> None:
        self._reglas = [nueva if r.nombre == nombre else r for r in self._reglas]

    @property
    def nombres(self) -> list[str]:
        return [r.nombre for r in self._reglas]

    def aplicar(self, df: pd.DataFrame, tolerance: float) -> tuple[pd.Series, pd.Series]:
        """
        Aplica todas las reglas de forma vectorizada.
        Devuelve (reglas_txt, reglas_conteo) como Series sobre el índice de df.
        """
        resultados: list[pd.Series] = []
        for regla in self._reglas:
            try:
                tags = regla.fn(df, tolerance).astype(str)
            except Exception:
                tags = pd.Series("", index=df.index)
            resultados.append(tags)

        combined = pd.concat(resultados, axis=1)
        combined.columns = [r.nombre for r in self._reglas]

        reglas_txt    = combined.apply(lambda row: " | ".join(v for v in row if v), axis=1)
        reglas_conteo = combined.apply(lambda row: sum(1 for v in row if v), axis=1)
        return reglas_txt, reglas_conteo


# =============================================================================
# MOTOR DE CONCILIACIÓN
# =============================================================================

class ConciliadorMensual:
    def __init__(
        self,
        periodo: PeriodoConciliacion,
        tolerance: float = 0.01,
        rule_engine: RuleEngine | None = None,
    ):
        self.periodo     = periodo
        self.tolerance   = tolerance
        self.rule_engine = rule_engine if rule_engine is not None else RuleEngine()

    @staticmethod
    def _ensure_base(
        df: Optional[pd.DataFrame],
        cuenta: int,
        nombre_cantidad: str,
        nombre_importe: str,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["cuenta", "especie_h", "moneda", nombre_cantidad, nombre_importe])
        out = df.copy()
        keep = [c for c in ["cuenta", "especie_h", "moneda", "cantidad", "importe"] if c in out.columns]
        out = out[keep].copy()
        for col, default in [("cuenta", cuenta), ("especie_h", ""), ("moneda", "ARS"), ("cantidad", 0.0), ("importe", 0.0)]:
            if col not in out.columns:
                out[col] = default
        out = out.rename(columns={"cantidad": nombre_cantidad, "importe": nombre_importe})
        out["cuenta"]    = pd.to_numeric(out["cuenta"], errors="coerce").fillna(cuenta).astype(int)
        out["especie_h"] = out["especie_h"].astype(str).map(normalize_text)
        out["moneda"]    = out["moneda"].astype(str).map(normalize_text)
        for c in [nombre_cantidad, nombre_importe]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out

    def _merge_fuentes(
        self,
        cuenta: int,
        df_posiciones: pd.DataFrame,
        df_activity: pd.DataFrame,
        df_portafolio_ini: pd.DataFrame,
        df_portafolio_fin: pd.DataFrame,
    ) -> pd.DataFrame:
        ini = self._ensure_base(df_portafolio_ini, cuenta, "ini", "importe_ini")
        act = self._ensure_base(df_activity,       cuenta, "act", "importe_act")
        fin = self._ensure_base(df_portafolio_fin, cuenta, "fin", "importe_fin")
        pos = self._ensure_base(df_posiciones,     cuenta, "pos", "importe_pos")

        keys = ["cuenta", "especie_h", "moneda"]
        out = ini.merge(act, on=keys, how="outer")
        out = out.merge(fin, on=keys, how="outer")
        out = out.merge(pos, on=keys, how="outer")

        for col in ["ini", "importe_ini", "act", "importe_act", "fin", "importe_fin", "pos", "importe_pos"]:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        return out

    def conciliar_cuenta(
        self,
        cuenta: int,
        df_posiciones: pd.DataFrame,
        df_activity: pd.DataFrame,
        df_portafolio_ini: pd.DataFrame,
        df_portafolio_fin: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:

        base = self._merge_fuentes(
            cuenta=cuenta,
            df_posiciones=df_posiciones,
            df_activity=df_activity,
            df_portafolio_ini=df_portafolio_ini,
            df_portafolio_fin=df_portafolio_fin,
        )

        if base.empty:
            resumen = {
                "cuenta": cuenta, "especies": 0, "cerradas": 0,
                "pendientes": 0, "reglas_aplicadas": 0,
                "dif_total_abs": 0.0, "dif_total_neto": 0.0,
                "pct_cierre": 0.0, "semaforo": "WARN",
            }
            return _empty_result_df(), resumen, pd.DataFrame(), pd.DataFrame()

        out = base.copy()
        out["comp"]               = 0.0
        out["ini_ajust"]          = out["ini"]
        out["act_ajust"]          = out["act"]
        out["fin_ajust"]          = out["fin"]
        out["comp_ajust"]         = out["comp"]
        out["dif_cant"]           = out["ini"] + out["act"] - out["fin"]
        out["dif_final_original"] = out["dif_cant"]
        out["dif_final"]          = out["ini_ajust"] + out["act_ajust"] + out["comp_ajust"] - out["fin_ajust"]

        # precio de referencia (vectorizado)
        with np.errstate(divide="ignore", invalid="ignore"):
            precio_fin = np.where(out["fin"] != 0, out["importe_fin"] / out["fin"], np.nan)
            precio_ini = np.where(out["ini"] != 0, out["importe_ini"] / out["ini"], np.nan)
            precio_pos = np.where(out["pos"] != 0, out["importe_pos"] / out["pos"], np.nan)

        out["precio_ref"] = (
            pd.Series(precio_fin, index=out.index)
            .fillna(pd.Series(precio_ini, index=out.index))
            .fillna(pd.Series(precio_pos, index=out.index))
            .fillna(0.0)
        )
        out["dif_importe"] = out["dif_final"] * out["precio_ref"]

        # reglas vectorizadas via RuleEngine (reemplaza el iterrows anterior)
        reglas_txt, reglas_conteo = self.rule_engine.aplicar(out, self.tolerance)
        out["reglas_txt"]       = reglas_txt
        out["reglas_aplicadas"] = reglas_conteo

        # status y match (vectorizado)
        cerrado = out["dif_final"].abs() <= self.tolerance
        out["status"]       = np.where(cerrado, "CERRADA",   "PENDIENTE")
        out["match_status"] = np.where(cerrado, "match_ok",  "sin_match")

        df_aud   = out[["cuenta", "especie_h", "moneda", "ini", "act", "fin", "pos",
                         "dif_final", "status", "reglas_txt"]].copy()
        df_match = out[["cuenta", "especie_h", "moneda", "match_status", "status", "dif_final"]].copy()

        total          = len(out)
        cerradas       = int(cerrado.sum())
        pendientes     = total - cerradas
        dif_total_abs  = float(out["dif_final"].abs().sum())
        dif_total_neto = float(out["dif_final"].sum())
        pct_cierre     = (cerradas / total * 100) if total else 0.0
        semaforo       = "OK" if pendientes == 0 else ("WARN" if dif_total_abs <= 1 else "BAD")

        resumen = {
            "cuenta": cuenta, "especies": total, "cerradas": cerradas,
            "pendientes": pendientes, "reglas_aplicadas": int(out["reglas_aplicadas"].sum()),
            "dif_total_abs": dif_total_abs, "dif_total_neto": dif_total_neto,
            "pct_cierre": pct_cierre, "semaforo": semaforo,
        }

        cols_finales = [
            "cuenta", "especie_h", "moneda",
            "ini", "act", "fin", "comp",
            "ini_ajust", "act_ajust", "fin_ajust", "comp_ajust",
            "dif_cant", "dif_final_original", "dif_final", "dif_importe",
            "precio_ref", "status", "reglas_aplicadas", "match_status",
        ]
        return out[cols_finales].copy(), resumen, df_aud, df_match

    def generar_reporte(self, resultados: list[pd.DataFrame]) -> pd.DataFrame:
        if not resultados:
            return _empty_result_df()
        out = _safe_concat(resultados)
        if out.empty:
            return _empty_result_df()
        return out.sort_values(
            by=["cuenta", "status", "dif_final", "especie_h"],
            ascending=[True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)


# =============================================================================
# TABLAS DERIVADAS
# =============================================================================

def build_resumen_cuentas(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=[
            "cuenta", "especies", "cerradas", "pendientes",
            "reglas_aplicadas", "dif_total_abs", "dif_total_neto",
            "pct_cierre", "semaforo",
        ])
    resumen = (
        df_all.groupby("cuenta", as_index=False)
        .agg(
            especies=("especie_h", "count"),
            cerradas=("status", lambda s: int((s == "CERRADA").sum())),
            pendientes=("status", lambda s: int((s == "PENDIENTE").sum())),
            reglas_aplicadas=("reglas_aplicadas", "sum"),
            dif_total_abs=("dif_final", lambda s: float(np.abs(s).sum())),
            dif_total_neto=("dif_final", "sum"),
        )
    )
    resumen["pct_cierre"] = np.where(
        resumen["especies"] > 0,
        resumen["cerradas"] / resumen["especies"] * 100,
        0.0,
    )
    resumen["semaforo"] = np.select(
        [resumen["pendientes"] == 0, resumen["dif_total_abs"] <= 1],
        ["OK", "WARN"],
        default="BAD",
    )
    return resumen.sort_values("cuenta").reset_index(drop=True)


def build_top_pendientes(df_all: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame()
    df = df_all[df_all["status"] == "PENDIENTE"].copy()
    if df.empty:
        return pd.DataFrame(columns=df_all.columns)
    df["abs_dif"] = df["dif_final"].abs()
    df = df.sort_values(["abs_dif", "cuenta", "especie_h"], ascending=[False, True, True])
    return df.drop(columns=["abs_dif"]).head(top_n).reset_index(drop=True)


def build_reglas_aplicadas(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["cuenta", "especie_h", "reglas_aplicadas", "status"])
    out = df_all[["cuenta", "especie_h", "reglas_aplicadas", "status"]].copy()
    return out[out["reglas_aplicadas"] > 0].reset_index(drop=True)


def auditoria_intercuenta(df_all: pd.DataFrame, pares_comp: list[tuple[int, int]]) -> pd.DataFrame:
    if df_all is None or df_all.empty or not pares_comp:
        return pd.DataFrame(columns=[
            "cuenta_origen", "cuenta_destino", "especie_h", "moneda",
            "dif_origen", "dif_destino", "suma_neta",
        ])
    rows = []
    for origen, destino in pares_comp:
        df_o = df_all[df_all["cuenta"] == origen][["especie_h", "moneda", "dif_final"]].rename(columns={"dif_final": "dif_origen"})
        df_d = df_all[df_all["cuenta"] == destino][["especie_h", "moneda", "dif_final"]].rename(columns={"dif_final": "dif_destino"})
        if df_o.empty and df_d.empty:
            continue
        merged = df_o.merge(df_d, on=["especie_h", "moneda"], how="outer").fillna(0.0)
        merged["suma_neta"]      = merged["dif_origen"] + merged["dif_destino"]
        merged["cuenta_origen"]  = origen
        merged["cuenta_destino"] = destino
        rows.append(merged[["cuenta_origen", "cuenta_destino", "especie_h", "moneda",
                             "dif_origen", "dif_destino", "suma_neta"]])
    return _safe_concat(rows)


# =============================================================================
# EXCEL EXPORT
# =============================================================================

def build_excel_bytes(
    df_resumen: pd.DataFrame,
    df_top_pend: pd.DataFrame,
    df_pend: pd.DataFrame,
    df_all: pd.DataFrame,
    df_auditoria: pd.DataFrame,
    df_reglas: pd.DataFrame,
    df_match: pd.DataFrame,
    df_intercuenta: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    sheets = {
        "Resumen":        df_resumen,
        "Top Pendientes": df_top_pend,
        "Pendientes":     df_pend,
        "Detalle":        df_all,
        "Auditoria":      df_auditoria,
        "Reglas":         df_reglas,
        "Match":          df_match,
        "Intercuenta":    df_intercuenta,
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()
