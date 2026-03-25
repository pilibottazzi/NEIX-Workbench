from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

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
    s = series.astype(str).str.strip()

    # normalizaciones comunes
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace("U$S", "", regex=False)
    s = s.str.replace("US$", "", regex=False)
    s = s.str.replace("$", "", regex=False)

    # paréntesis negativos
    s = s.str.replace("(", "-", regex=False)
    s = s.str.replace(")", "", regex=False)

    # si tiene formato 1.234,56
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)

    # guión final negativo
    s = s.str.replace(r"^(.+)-$", r"-\1", regex=True)

    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def format_num(x: Any, decimals: int = 2) -> str:
    try:
        return f"{float(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0"


def _empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cuenta",
            "especie_h",
            "moneda",
            "ini",
            "act",
            "fin",
            "comp",
            "ini_ajust",
            "act_ajust",
            "fin_ajust",
            "comp_ajust",
            "dif_cant",
            "dif_final_original",
            "dif_final",
            "dif_importe",
            "precio_ref",
            "status",
            "reglas_aplicadas",
            "match_status",
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
    """
    Lee csv/xlsx/xls/xlsm desde:
    - path string / Path
    - UploadedFile de Streamlit
    """
    if file_obj_or_path is None:
        return pd.DataFrame()

    # Path / string
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

    # UploadedFile / file-like
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

def preparar_posiciones(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    alerts: list[str] = []

    if df_raw is None or df_raw.empty:
        alerts.append("Archivo de posiciones vacío.")
        return pd.DataFrame(columns=["cuenta", "especie_h", "moneda", "cantidad", "importe"]), alerts

    df = normalize_columns(df_raw)

    c_cuenta = find_first_existing(df, ["cuenta", "comitente", "codigo", "nro_cuenta"])
    c_especie = find_first_existing(df, ["especie", "instrumento", "ticker", "simbolo", "codigo_especie"])
    c_moneda = find_first_existing(df, ["moneda", "currency"])
    c_cantidad = find_first_existing(df, ["cantidad", "cant", "nominales", "nominal"])
    c_importe = find_first_existing(df, ["importe", "monto", "valor_mercado", "market_value", "total"])

    faltantes = [x for x in [c_cuenta, c_especie, c_cantidad] if x is None]
    if faltantes:
        alerts.append("No se detectaron todas las columnas clave de posiciones.")

    out = pd.DataFrame()
    out["cuenta"] = pd.to_numeric(df[c_cuenta], errors="coerce") if c_cuenta else np.nan
    out["especie_h"] = df[c_especie].astype(str).map(normalize_text) if c_especie else ""
    out["moneda"] = df[c_moneda].astype(str).map(normalize_text) if c_moneda else "ARS"
    out["cantidad"] = to_numeric_safe(df[c_cantidad]) if c_cantidad else 0.0
    out["importe"] = to_numeric_safe(df[c_importe]) if c_importe else 0.0

    out = out.dropna(subset=["cuenta"])
    out["cuenta"] = out["cuenta"].astype(int)
    out = out[out["especie_h"] != ""].copy()

    out = (
        out.groupby(["cuenta", "especie_h", "moneda"], as_index=False)
        .agg({"cantidad": "sum", "importe": "sum"})
    )

    return out, alerts


def preparar_activity(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    alerts: list[str] = []

    if df_raw is None or df_raw.empty:
        alerts.append("Archivo de activity vacío.")
        return pd.DataFrame(columns=["cuenta", "especie_h", "moneda", "cantidad", "importe"]), alerts

    df = normalize_columns(df_raw)

    c_cuenta = find_first_existing(df, ["cuenta", "comitente", "codigo", "nro_cuenta"])
    c_especie = find_first_existing(df, ["especie", "instrumento", "ticker", "simbolo", "codigo_especie"])
    c_moneda = find_first_existing(df, ["moneda", "currency"])
    c_cantidad = find_first_existing(df, ["cantidad", "cant", "nominales", "nominal"])
    c_importe = find_first_existing(df, ["importe", "monto", "net_amount", "amount", "total"])

    if c_cuenta is None or c_especie is None:
        alerts.append("No se detectaron columnas clave de activity.")

    out = pd.DataFrame()
    out["cuenta"] = pd.to_numeric(df[c_cuenta], errors="coerce") if c_cuenta else np.nan
    out["especie_h"] = df[c_especie].astype(str).map(normalize_text) if c_especie else ""
    out["moneda"] = df[c_moneda].astype(str).map(normalize_text) if c_moneda else "ARS"
    out["cantidad"] = to_numeric_safe(df[c_cantidad]) if c_cantidad else 0.0
    out["importe"] = to_numeric_safe(df[c_importe]) if c_importe else 0.0

    out = out.dropna(subset=["cuenta"])
    out["cuenta"] = out["cuenta"].astype(int)
    out = out[out["especie_h"] != ""].copy()

    out = (
        out.groupby(["cuenta", "especie_h", "moneda"], as_index=False)
        .agg({"cantidad": "sum", "importe": "sum"})
    )

    return out, alerts


def preparar_portafolio(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    alerts: list[str] = []

    if df_raw is None or df_raw.empty:
        alerts.append("Archivo de portafolio vacío.")
        return pd.DataFrame(columns=["cuenta", "especie_h", "moneda", "cantidad", "importe"]), alerts

    df = normalize_columns(df_raw)

    c_cuenta = find_first_existing(df, ["cuenta", "comitente", "codigo", "nro_cuenta"])
    c_especie = find_first_existing(df, ["especie", "instrumento", "ticker", "simbolo", "codigo_especie"])
    c_moneda = find_first_existing(df, ["moneda", "currency"])
    c_cantidad = find_first_existing(df, ["cantidad", "cant", "nominales", "nominal"])
    c_importe = find_first_existing(df, ["importe", "monto", "valor_mercado", "market_value", "total"])

    if c_cuenta is None or c_especie is None:
        alerts.append("No se detectaron columnas clave de portafolio.")

    out = pd.DataFrame()
    out["cuenta"] = pd.to_numeric(df[c_cuenta], errors="coerce") if c_cuenta else np.nan
    out["especie_h"] = df[c_especie].astype(str).map(normalize_text) if c_especie else ""
    out["moneda"] = df[c_moneda].astype(str).map(normalize_text) if c_moneda else "ARS"
    out["cantidad"] = to_numeric_safe(df[c_cantidad]) if c_cantidad else 0.0
    out["importe"] = to_numeric_safe(df[c_importe]) if c_importe else 0.0

    out = out.dropna(subset=["cuenta"])
    out["cuenta"] = out["cuenta"].astype(int)
    out = out[out["especie_h"] != ""].copy()

    out = (
        out.groupby(["cuenta", "especie_h", "moneda"], as_index=False)
        .agg({"cantidad": "sum", "importe": "sum"})
    )

    return out, alerts


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
# MOTOR DE CONCILIACIÓN
# =============================================================================

class ConciliadorMensual:
    def __init__(self, periodo: PeriodoConciliacion, tolerance: float = 0.01):
        self.periodo = periodo
        self.tolerance = tolerance

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

        if "cuenta" not in out.columns:
            out["cuenta"] = cuenta
        if "especie_h" not in out.columns:
            out["especie_h"] = ""
        if "moneda" not in out.columns:
            out["moneda"] = "ARS"
        if "cantidad" not in out.columns:
            out["cantidad"] = 0.0
        if "importe" not in out.columns:
            out["importe"] = 0.0

        out = out.rename(columns={"cantidad": nombre_cantidad, "importe": nombre_importe})
        out["cuenta"] = pd.to_numeric(out["cuenta"], errors="coerce").fillna(cuenta).astype(int)
        out["especie_h"] = out["especie_h"].astype(str).map(normalize_text)
        out["moneda"] = out["moneda"].astype(str).map(normalize_text)

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
        # prioridad:
        # ini -> portafolio_ini
        # fin -> portafolio_fin
        # act -> activity
        # posiciones queda como info auxiliar si vino
        ini = self._ensure_base(df_portafolio_ini, cuenta, "ini", "importe_ini")
        act = self._ensure_base(df_activity, cuenta, "act", "importe_act")
        fin = self._ensure_base(df_portafolio_fin, cuenta, "fin", "importe_fin")
        pos = self._ensure_base(df_posiciones, cuenta, "pos", "importe_pos")

        keys = ["cuenta", "especie_h", "moneda"]

        out = ini.merge(act, on=keys, how="outer")
        out = out.merge(fin, on=keys, how="outer")
        out = out.merge(pos, on=keys, how="outer")

        for col in [
            "ini", "importe_ini",
            "act", "importe_act",
            "fin", "importe_fin",
            "pos", "importe_pos",
        ]:
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
                "cuenta": cuenta,
                "especies": 0,
                "cerradas": 0,
                "pendientes": 0,
                "reglas_aplicadas": 0,
                "dif_total_abs": 0.0,
                "dif_total_neto": 0.0,
                "pct_cierre": 0.0,
                "semaforo": "WARN",
            }
            return _empty_result_df(), resumen, pd.DataFrame(), pd.DataFrame()

        out = base.copy()

        # conciliación base
        out["comp"] = 0.0
        out["ini_ajust"] = out["ini"]
        out["act_ajust"] = out["act"]
        out["fin_ajust"] = out["fin"]
        out["comp_ajust"] = out["comp"]

        out["dif_cant"] = out["ini"] + out["act"] - out["fin"]
        out["dif_final_original"] = out["dif_cant"]
        out["dif_final"] = out["ini_ajust"] + out["act_ajust"] + out["comp_ajust"] - out["fin_ajust"]

        # precio de referencia
        with np.errstate(divide="ignore", invalid="ignore"):
            precio_ini = np.where(out["ini"] != 0, out["importe_ini"] / out["ini"], np.nan)
            precio_fin = np.where(out["fin"] != 0, out["importe_fin"] / out["fin"], np.nan)
            precio_pos = np.where(out["pos"] != 0, out["importe_pos"] / out["pos"], np.nan)

        out["precio_ref"] = pd.Series(precio_fin).fillna(pd.Series(precio_ini)).fillna(pd.Series(precio_pos)).fillna(0.0)
        out["dif_importe"] = out["dif_final"] * out["precio_ref"]

        # reglas simples
        reglas = []
        match_status = []

        for _, row in out.iterrows():
            applied = 0
            tags = []

            # regla 1: cierre exacto
            if abs(row["dif_final"]) <= self.tolerance:
                tags.append("R0_CIERRE_EXACTO")

            # regla 2: sin fin pero sí posiciones
            if abs(row["fin"]) <= self.tolerance and abs(row["pos"]) > self.tolerance:
                tags.append("R1_SOLO_POSICIONES")

            # regla 3: sin activity
            if abs(row["act"]) <= self.tolerance:
                tags.append("R2_SIN_ACTIVITY")

            # regla 4: sin portafolio inicial
            if abs(row["ini"]) <= self.tolerance:
                tags.append("R3_SIN_INICIAL")

            applied = len(tags)
            reglas.append(" | ".join(tags) if tags else "")
            match_status.append("match_ok" if abs(row["dif_final"]) <= self.tolerance else "sin_match")

        out["reglas_txt"] = reglas
        out["reglas_aplicadas"] = [0 if x == "" else len(x.split(" | ")) for x in reglas]
        out["match_status"] = match_status
        out["status"] = np.where(out["dif_final"].abs() <= self.tolerance, "CERRADA", "PENDIENTE")

        # auditoría
        df_aud = out[[
            "cuenta", "especie_h", "moneda",
            "ini", "act", "fin", "pos",
            "dif_final", "status", "reglas_txt",
        ]].copy()

        # match table
        df_match = out[[
            "cuenta", "especie_h", "moneda",
            "match_status", "status", "dif_final",
        ]].copy()

        # resumen
        total = len(out)
        cerradas = int((out["status"] == "CERRADA").sum())
        pendientes = int((out["status"] == "PENDIENTE").sum())
        reglas_count = int(out["reglas_aplicadas"].sum())
        dif_total_abs = float(out["dif_final"].abs().sum())
        dif_total_neto = float(out["dif_final"].sum())
        pct_cierre = (cerradas / total * 100) if total else 0.0

        if pendientes == 0:
            semaforo = "OK"
        elif dif_total_abs <= 1:
            semaforo = "WARN"
        else:
            semaforo = "BAD"

        resumen = {
            "cuenta": cuenta,
            "especies": total,
            "cerradas": cerradas,
            "pendientes": pendientes,
            "reglas_aplicadas": reglas_count,
            "dif_total_abs": dif_total_abs,
            "dif_total_neto": dif_total_neto,
            "pct_cierre": pct_cierre,
            "semaforo": semaforo,
        }

        cols_finales = [
            "cuenta",
            "especie_h",
            "moneda",
            "ini",
            "act",
            "fin",
            "comp",
            "ini_ajust",
            "act_ajust",
            "fin_ajust",
            "comp_ajust",
            "dif_cant",
            "dif_final_original",
            "dif_final",
            "dif_importe",
            "precio_ref",
            "status",
            "reglas_aplicadas",
            "match_status",
        ]
        out = out[cols_finales].copy()

        return out, resumen, df_aud, df_match

    def generar_reporte(self, resultados: list[pd.DataFrame]) -> pd.DataFrame:
        if not resultados:
            return _empty_result_df()

        out = _safe_concat(resultados)
        if out.empty:
            return _empty_result_df()

        out = out.sort_values(
            by=["cuenta", "status", "dif_final", "especie_h"],
            ascending=[True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)

        return out


# =============================================================================
# TABLAS DERIVADAS
# =============================================================================

def build_resumen_cuentas(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(
            columns=[
                "cuenta", "especies", "cerradas", "pendientes",
                "reglas_aplicadas", "dif_total_abs", "dif_total_neto",
                "pct_cierre", "semaforo",
            ]
        )

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
        [
            resumen["pendientes"] == 0,
            resumen["dif_total_abs"] <= 1,
        ],
        [
            "OK",
            "WARN",
        ],
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
    df = df.drop(columns=["abs_dif"])

    return df.head(top_n).reset_index(drop=True)


def build_reglas_aplicadas(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["cuenta", "especie_h", "reglas_aplicadas", "status"])

    out = df_all[["cuenta", "especie_h", "reglas_aplicadas", "status"]].copy()
    out = out[out["reglas_aplicadas"] > 0].reset_index(drop=True)
    return out


def auditoria_intercuenta(df_all: pd.DataFrame, pares_comp: list[tuple[int, int]]) -> pd.DataFrame:
    if df_all is None or df_all.empty or not pares_comp:
        return pd.DataFrame(
            columns=["cuenta_origen", "cuenta_destino", "especie_h", "moneda", "dif_origen", "dif_destino", "suma_neta"]
        )

    rows = []

    for origen, destino in pares_comp:
        df_o = df_all[df_all["cuenta"] == origen][["especie_h", "moneda", "dif_final"]].copy()
        df_d = df_all[df_all["cuenta"] == destino][["especie_h", "moneda", "dif_final"]].copy()

        if df_o.empty and df_d.empty:
            continue

        df_o = df_o.rename(columns={"dif_final": "dif_origen"})
        df_d = df_d.rename(columns={"dif_final": "dif_destino"})

        merged = df_o.merge(df_d, on=["especie_h", "moneda"], how="outer")
        merged["dif_origen"] = merged["dif_origen"].fillna(0.0)
        merged["dif_destino"] = merged["dif_destino"].fillna(0.0)
        merged["suma_neta"] = merged["dif_origen"] + merged["dif_destino"]
        merged["cuenta_origen"] = origen
        merged["cuenta_destino"] = destino

        rows.append(merged[[
            "cuenta_origen", "cuenta_destino",
            "especie_h", "moneda",
            "dif_origen", "dif_destino", "suma_neta",
        ]])

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

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df_resumen is not None:
            df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
        if df_top_pend is not None:
            df_top_pend.to_excel(writer, sheet_name="Top Pendientes", index=False)
        if df_pend is not None:
            df_pend.to_excel(writer, sheet_name="Pendientes", index=False)
        if df_all is not None:
            df_all.to_excel(writer, sheet_name="Detalle", index=False)
        if df_auditoria is not None:
            df_auditoria.to_excel(writer, sheet_name="Auditoria", index=False)
        if df_reglas is not None:
            df_reglas.to_excel(writer, sheet_name="Reglas", index=False)
        if df_match is not None:
            df_match.to_excel(writer, sheet_name="Match", index=False)
        if df_intercuenta is not None:
            df_intercuenta.to_excel(writer, sheet_name="Intercuenta", index=False)

    output.seek(0)
    return output.getvalue()
