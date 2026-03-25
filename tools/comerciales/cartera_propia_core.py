from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

CUENTAS_DEFAULT = [904, 932, 990, 992, 996, 997, 999, 1000]


@dataclass
class PeriodoConciliacion:
    fecha_ini: str
    fecha_fin: str
    cuentas: List[int]

    pares_comp: Dict[int, int] = field(default_factory=lambda: {
        904: 992,
        992: 904,
        997: 999,
        999: 997,
    })

    cuenta_ib: int = 992

    ratios_adr: Dict[str, float] = field(default_factory=lambda: {
        "AAPL": 20, "AMZN": 150, "ADBE": 44, "AMD": 10, "AAL": 3,
        "ABEV": 0.333, "ARKK": 15, "BABA": 8, "BIDU": 4, "COIN": 5,
        "CSCO": 6, "DIS": 5, "GOOG": 12, "INTC": 10, "JNJ": 2,
        "KO": 5, "META": 6, "MSFT": 10, "NFLX": 5, "NKE": 5,
        "NVDA": 24, "PFE": 10, "PYPL": 5, "QCOM": 5, "SBUX": 5,
        "SQ": 5, "TSLA": 15, "V": 2, "WMT": 5, "XOM": 5,
        "84801": 1, "912797SK4": 1,
    })


# =============================================================================
# HELPERS
# =============================================================================

def normalize_text(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("u$s", "usd")
    s = s.replace("dolar", "usd")
    s = s.replace("dólar", "usd")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^a-zA-Z0-9]+", "_", normalize_text(c)).strip("_")
        for c in out.columns
    ]
    return out


def to_numeric_safe(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\((.*?)\)", r"-\1", regex=True)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("USD", "", regex=False)
    s = s.str.replace("usd", "", regex=False)

    def parse_value(x: str) -> float:
        x = str(x).strip()
        x = re.sub(r"[^0-9,.\-]", "", x)
        if x in {"", "-"}:
            return 0.0

        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        else:
            if x.count(",") == 1 and x.count(".") == 0:
                x = x.replace(",", ".")
            elif x.count(".") > 1 and x.count(",") == 0:
                x = x.replace(".", "")
            elif x.count(",") > 1 and x.count(".") == 0:
                x = x.replace(",", "")

        try:
            return float(x)
        except Exception:
            return 0.0

    return s.apply(parse_value)


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def format_num(x, decimals: int = 2) -> str:
    try:
        return f"{float(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def read_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, sep=";")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep="|")

    if name.endswith((".xls", ".xlsx", ".xlsm")):
        return pd.read_excel(uploaded_file)

    raise ValueError(f"Formato no soportado: {uploaded_file.name}")


def guess_file_role(filename: str) -> Tuple[Optional[int], Optional[str]]:
    name = normalize_text(filename)
    cuenta = None

    for c in CUENTAS_DEFAULT:
        if re.search(rf"(^|[^0-9]){c}([^0-9]|$)", name):
            cuenta = c
            break

    role = None
    if "activity" in name or "movimiento" in name or "movimientos" in name:
        role = "activity"
    elif ("portafolio" in name or "portfolio" in name) and ("inicial" in name or "inicio" in name or "ini" in name):
        role = "portafolio_ini"
    elif ("portafolio" in name or "portfolio" in name) and ("final" in name or "cierre" in name or "fin" in name):
        role = "portafolio_fin"
    elif "posiciones" in name or "consolidado" in name or "conciliacion" in name or "base" in name:
        role = "posiciones"
    elif "inicial" in name and "activity" not in name:
        role = "portafolio_ini"
    elif "final" in name and "activity" not in name:
        role = "portafolio_fin"

    return cuenta, role


# =============================================================================
# HOMOLOGACIÓN INTELIGENTE
# =============================================================================

def especie_base_token(x: str) -> str:
    s = normalize_text(x).upper()
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    # si arranca con ticker clásico
    m = re.match(r"^([A-Z0-9]+)", s)
    if m:
        return m.group(1)

    return s.split(" ")[0]


def homologar_especie(valor: str) -> str:
    s = normalize_text(valor).upper()
    s = re.sub(r"\s+", " ", s).strip()

    if s in {"", "NAN"}:
        return ""

    if "CC PESOS" in s:
        return "CC PESOS"
    if "CC USD LOCAL" in s or "CC DOLAR LOCAL" in s or "CC USD" in s and "LOCAL" in s:
        return "CC USD LOCAL"
    if "CC USD EXTERIOR" in s or "CC U$S EXTERIOR" in s or "EXTERIOR" in s and "CC" in s:
        return "CC USD EXTERIOR"

    token = especie_base_token(s)

    # casos conocidos
    aliases = {
        "GGAL": "GGAL",
        "AL30": "AL30",
        "GD30": "GD30",
        "BBAR": "BBAR",
        "BMA": "BMA",
        "CRES": "CRES",
        "EDN": "EDN",
        "IRSA": "IRSA",
        "TGSU2": "TGSU2",
        "AAPL": "AAPL",
        "AMZN": "AMZN",
        "MSFT": "MSFT",
        "META": "META",
        "NVDA": "NVDA",
        "TSLA": "TSLA",
    }

    return aliases.get(token, token)


def build_homologacion_table(
    df_posiciones: pd.DataFrame,
    df_activity: pd.DataFrame,
) -> pd.DataFrame:
    pos = pd.DataFrame({"origen": "posiciones", "raw": df_posiciones["especie"].astype(str)})
    act = pd.DataFrame({"origen": "activity", "raw": df_activity["ticker"].astype(str)})

    df = pd.concat([pos, act], ignore_index=True)
    df["homologada"] = df["raw"].apply(homologar_especie)
    df["token"] = df["raw"].apply(especie_base_token)
    return df.drop_duplicates().sort_values(["homologada", "origen", "raw"]).reset_index(drop=True)


# =============================================================================
# PREPARACIÓN
# =============================================================================

def preparar_posiciones(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    alerts: List[str] = []
    df = normalize_columns(df)

    mapping = {
        "especie": ["especie", "ticker", "activo", "simbolo", "security"],
        "ini_cant": ["ini_cant", "cantidad_inicial", "ini", "inicial", "saldo_inicial"],
        "act_cant": ["act_cant", "cantidad_activity", "activity", "movimiento_cantidad", "activity_cant"],
        "fin_cant": ["fin_cant", "cantidad_final", "fin", "final", "saldo_final"],
        "ini_imp": ["ini_imp", "importe_inicial", "monto_inicial"],
        "act_imp": ["act_imp", "importe_activity", "monto_activity"],
        "fin_imp": ["fin_imp", "importe_final", "monto_final"],
        "comp": ["comp", "compensacion", "compensaciones"],
        "ini_ib": ["ini_ib"],
        "fin_ib": ["fin_ib"],
        "ini_local": ["ini_local"],
        "fin_local": ["fin_local"],
    }

    out = pd.DataFrame()
    for target, options in mapping.items():
        found = next((c for c in options if c in df.columns), None)
        if found:
            out[target] = df[found]

    if "especie" not in out.columns:
        raise ValueError("El archivo de posiciones debe contener especie/ticker/activo.")

    numeric_cols = [
        "ini_cant", "act_cant", "fin_cant",
        "ini_imp", "act_imp", "fin_imp",
        "comp", "ini_ib", "fin_ib", "ini_local", "fin_local"
    ]
    for c in numeric_cols:
        if c not in out.columns:
            out[c] = 0
            alerts.append(f"Posiciones: se asumió {c}=0")
        out[c] = to_numeric_safe(out[c])

    out["especie"] = out["especie"].astype(str).str.strip()
    out = out[out["especie"] != ""].copy()
    out["especie_h"] = out["especie"].apply(homologar_especie)

    if out["especie_h"].duplicated().any():
        alerts.append("Se detectaron especies duplicadas tras homologación; se consolidaron.")
        agg_map = {c: "sum" for c in numeric_cols}
        agg_map["especie"] = "first"
        out = out.groupby("especie_h", as_index=False).agg(agg_map)

    return out.reset_index(drop=True), alerts


def preparar_activity(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    alerts: List[str] = []
    df = normalize_columns(df)

    mapping = {
        "ticker": ["ticker", "especie", "activo", "simbolo", "security"],
        "cantidad": ["cantidad", "cant", "quantity", "qty"],
        "tipo": ["tipo", "concepto", "movimiento", "side"],
        "nro_comprobante": ["nro_comprobante", "comprobante", "numero_comprobante", "id"],
        "fecha_emision": ["fecha_emision", "fecha", "trade_date", "fecha_operacion"],
        "descripcion": ["descripcion", "detalle"],
    }

    out = pd.DataFrame()
    for target, options in mapping.items():
        found = next((c for c in options if c in df.columns), None)
        if found:
            out[target] = df[found]

    for req in ["ticker", "cantidad"]:
        if req not in out.columns:
            raise ValueError(f"Falta columna obligatoria en Activity: {req}")

    if "tipo" not in out.columns:
        out["tipo"] = ""
        alerts.append("Activity sin tipo; se asumió vacío.")

    if "nro_comprobante" not in out.columns:
        out["nro_comprobante"] = ""
        alerts.append("Activity sin comprobante; deduplicación limitada.")

    if "fecha_emision" not in out.columns:
        out["fecha_emision"] = pd.NaT
        alerts.append("Activity sin fecha.")
    else:
        out["fecha_emision"] = to_datetime_safe(out["fecha_emision"])

    if "descripcion" not in out.columns:
        out["descripcion"] = ""

    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["cantidad"] = to_numeric_safe(out["cantidad"])
    out["tipo"] = out["tipo"].astype(str).str.strip()
    out = out[out["ticker"] != ""].copy()
    out["ticker_h"] = out["ticker"].apply(homologar_especie)

    return out.reset_index(drop=True), alerts


def preparar_portafolio(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    alerts: List[str] = []
    df = normalize_columns(df)

    mapping = {
        "descripcion": ["descripcion", "especie", "detalle", "ticker", "security"],
        "cantidad": ["cantidad", "cant", "quantity", "qty"],
    }

    out = pd.DataFrame()
    for target, options in mapping.items():
        found = next((c for c in options if c in df.columns), None)
        if found:
            out[target] = df[found]

    if "descripcion" not in out.columns:
        out["descripcion"] = ""
        alerts.append("Portafolio sin descripción.")
    if "cantidad" not in out.columns:
        out["cantidad"] = 0
        alerts.append("Portafolio sin cantidad; se asumió 0.")

    out["descripcion"] = out["descripcion"].astype(str).str.strip()
    out["cantidad"] = to_numeric_safe(out["cantidad"])
    out["descripcion_h"] = out["descripcion"].apply(homologar_especie)

    return out.reset_index(drop=True), alerts


# =============================================================================
# REGLAS
# =============================================================================

class ReglasReconciliacion:
    @staticmethod
    def regla_preconcertadas(df_activity: pd.DataFrame, fecha_ini: str, tiene_ini: bool) -> pd.DataFrame:
        if df_activity.empty or "fecha_emision" not in df_activity.columns or not tiene_ini:
            return pd.DataFrame()
        return df_activity[df_activity["fecha_emision"] == pd.to_datetime(fecha_ini)].copy()

    @staticmethod
    def regla_duplicados(df_activity: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        cols_dedup = ["nro_comprobante", "tipo", "ticker_h", "cantidad"]
        cols = [c for c in cols_dedup if c in df_activity.columns]
        if not cols:
            return df_activity.copy(), 0
        before = len(df_activity)
        out = df_activity.drop_duplicates(subset=cols, keep="first").copy()
        return out, before - len(out)

    @staticmethod
    def regla_vto_prestamo(df_activity: pd.DataFrame) -> Tuple[pd.DataFrame, float, int]:
        if "tipo" not in df_activity.columns:
            return df_activity.copy(), 0.0, 0
        mask = df_activity["tipo"].astype(str).str.upper() == "VTO PRESTAMO"
        excl = df_activity.loc[mask].copy()
        total = excl["cantidad"].sum() if "cantidad" in excl.columns else 0.0
        return df_activity.loc[~mask].copy(), total, len(excl)

    @staticmethod
    def regla_decr_desfase(df_activity: pd.DataFrame, ini: float, fin: float, ticker: str) -> Dict:
        if "tipo" not in df_activity.columns or "cantidad" not in df_activity.columns:
            return {"ticker": ticker, "match": False, "ajuste_fin": 0, "exceso": 0}

        decr_mask = df_activity["tipo"].astype(str).str.upper() == "DECR"
        decr_cant = df_activity.loc[decr_mask, "cantidad"].sum()
        trading_cant = df_activity.loc[~decr_mask, "cantidad"].sum()
        pos_change = fin - ini
        decr_en_posicion = pos_change - trading_cant
        exceso = abs(decr_cant) - abs(decr_en_posicion)
        dif = ini + df_activity["cantidad"].sum() - fin

        return {
            "ticker": ticker,
            "decr_activity": decr_cant,
            "decr_posicion": decr_en_posicion,
            "exceso": exceso,
            "dif_calculada": dif,
            "match": abs(abs(exceso) - abs(dif)) < 1,
            "ajuste_fin": -exceso,
        }

    @staticmethod
    def regla_al30_local_exterior(df_portafolio: pd.DataFrame, ticker_h: str = "AL30") -> Dict:
        if df_portafolio is None or df_portafolio.empty:
            return {"tiene_local": False, "tiene_exterior": False, "cant_total": 0.0}

        al30_rows = df_portafolio[df_portafolio["descripcion"].astype(str).str.contains("AL30", case=False, na=False)].copy()
        local = al30_rows[~al30_rows["descripcion"].astype(str).str.contains("EUR", case=False, na=False)]
        ext = al30_rows[al30_rows["descripcion"].astype(str).str.contains("EUR", case=False, na=False)]

        return {
            "tiene_local": not local.empty,
            "tiene_exterior": not ext.empty,
            "cant_local": float(local["cantidad"].sum()) if not local.empty else 0.0,
            "cant_exterior": float(ext["cantidad"].sum()) if not ext.empty else 0.0,
            "cant_total": float(al30_rows["cantidad"].sum()) if not al30_rows.empty else 0.0,
        }

    @staticmethod
    def regla_cc_pesos(especie_h: str) -> bool:
        return especie_h in {"CC PESOS", "CC USD LOCAL", "CC USD EXTERIOR"}

    @staticmethod
    def regla_comp_solo_si_dif(dif_cant: float, comp: float) -> float:
        return 0 if abs(dif_cant) < 1 else comp


class ConversionADR:
    @staticmethod
    def obtener_ratio(especie_h: str, ratios_conocidos: Dict[str, float]) -> float:
        return float(ratios_conocidos.get(especie_h, 1))

    @staticmethod
    def calcular_total_992(local_value: float, ib_value: float, ratio: float) -> float:
        return local_value + (ib_value * ratio)


# =============================================================================
# MOTOR
# =============================================================================

class ConciliadorMensual:
    def __init__(self, periodo: PeriodoConciliacion):
        self.periodo = periodo
        self.reglas = ReglasReconciliacion()

    def conciliar_especie(
        self,
        cuenta: int,
        especie: str,
        especie_h: str,
        ini_cant: float,
        act_cant: float,
        fin_cant: float,
        ini_imp: float = 0,
        act_imp: float = 0,
        fin_imp: float = 0,
        comp: float = 0,
        ratio: float = 1,
        df_activity_especie: Optional[pd.DataFrame] = None,
        df_portafolio_ini: Optional[pd.DataFrame] = None,
        df_portafolio_fin: Optional[pd.DataFrame] = None,
    ) -> Dict:
        ajustes: List[str] = []
        flags: List[str] = []

        ini_original = float(ini_cant)
        act_original = float(act_cant)
        fin_original = float(fin_cant)
        comp_original = float(comp)

        ini_ajust = ini_original
        act_ajust = act_original
        fin_ajust = fin_original
        comp_ajust = comp_original

        if self.reglas.regla_cc_pesos(especie_h):
            return {
                "cuenta": cuenta,
                "especie": especie,
                "especie_h": especie_h,
                "ini": ini_original,
                "act": act_original,
                "fin": fin_original,
                "comp": comp_original,
                "ratio": ratio,
                "ini_imp": ini_imp,
                "act_imp": act_imp,
                "fin_imp": fin_imp,
                "ini_ajust": ini_original,
                "act_ajust": act_original,
                "fin_ajust": fin_original,
                "comp_ajust": 0.0,
                "dif_cant_original": ini_original + act_original - fin_original,
                "dif_final_original": ini_original + act_original - fin_original + comp_original,
                "dif_cant": 0.0,
                "dif_final": 0.0,
                "dif_importe": ini_imp + act_imp - fin_imp,
                "status": "CERRADA (importe)",
                "ajustes": "CC conciliada por importe",
                "flags": "CC_IMPORTE",
                "reglas_aplicadas": 1,
                "imputacion_sugerida": "",
            }

        if especie_h == "AL30" and df_portafolio_ini is not None and df_portafolio_fin is not None:
            al30_ini = self.reglas.regla_al30_local_exterior(df_portafolio_ini)
            al30_fin = self.reglas.regla_al30_local_exterior(df_portafolio_fin)
            if al30_ini["tiene_local"] and al30_ini["tiene_exterior"]:
                ini_ajust = al30_ini["cant_total"]
                fin_ajust = al30_fin["cant_total"]
                ajustes.append("AL30 local + exterior sumado")
                flags.append("AL30_DOBLE_LINEA")

        if df_activity_especie is not None and not df_activity_especie.empty:
            pre = self.reglas.regla_preconcertadas(df_activity_especie, self.periodo.fecha_ini, tiene_ini=abs(ini_original) > 0)
            if not pre.empty:
                pre_total = pre["cantidad"].sum()
                act_ajust -= pre_total
                ajustes.append(f"Pre-concertadas excluidas ({format_num(pre_total)})")
                flags.append("PRECONCERTADA")

            decr_ops = df_activity_especie[df_activity_especie["tipo"].astype(str).str.upper() == "DECR"]
            if not decr_ops.empty:
                analisis = self.reglas.regla_decr_desfase(df_activity_especie, ini_ajust, fin_ajust, especie_h)
                if analisis["match"] and abs(analisis["exceso"]) > 0:
                    fin_ajust += analisis["ajuste_fin"]
                    ajustes.append(f"DECR ajustado en Fin ({format_num(analisis['ajuste_fin'])})")
                    flags.append("DECR_DESFASE")

        dif_cant_original = ini_original + act_original - fin_original
        dif_sin_comp = ini_ajust + act_ajust - fin_ajust
        comp_ajust = self.reglas.regla_comp_solo_si_dif(dif_sin_comp, comp_original)

        if comp_original != 0 and comp_ajust == 0:
            ajustes.append("Compensación ignorada por DifCant=0")
            flags.append("COMP_IGNORADA")

        dif_final_original = dif_cant_original + comp_original
        dif_cant = ini_ajust + act_ajust - fin_ajust
        dif_final = dif_cant + comp_ajust
        status = "CERRADA" if abs(dif_final) < 1 else "PENDIENTE"

        if status == "PENDIENTE":
            flags.append("PENDIENTE")

        # imputación sugerida
        imputacion = ""
        if "PRECONCERTADA" in flags:
            imputacion = "Revisar ops concertadas en fecha inicial"
        elif "DECR_DESFASE" in flags:
            imputacion = "Revisar liquidación DECR vs posición custodio"
        elif abs(dif_final) > 0 and df_activity_especie is None:
            imputacion = "Revisar especie sin activity / posible faltante en base"
        elif abs(dif_final) > 0 and abs(comp_ajust) == 0 and cuenta in self.periodo.pares_comp:
            imputacion = "Revisar compensación intercuenta"
        elif abs(dif_final) > 0:
            imputacion = "Revisar movement / homologación / timing"

        return {
            "cuenta": cuenta,
            "especie": especie,
            "especie_h": especie_h,
            "ini": ini_original,
            "act": act_original,
            "fin": fin_original,
            "comp": comp_original,
            "ratio": ratio,
            "ini_imp": ini_imp,
            "act_imp": act_imp,
            "fin_imp": fin_imp,
            "ini_ajust": ini_ajust,
            "act_ajust": act_ajust,
            "fin_ajust": fin_ajust,
            "comp_ajust": comp_ajust,
            "dif_cant_original": dif_cant_original,
            "dif_final_original": dif_final_original,
            "dif_cant": dif_cant,
            "dif_final": dif_final,
            "dif_importe": ini_imp + act_imp - fin_imp,
            "status": status,
            "ajustes": " | ".join(ajustes),
            "flags": " | ".join(flags),
            "reglas_aplicadas": len(ajustes),
            "imputacion_sugerida": imputacion,
        }

    def conciliar_cuenta(
        self,
        cuenta: int,
        df_posiciones: pd.DataFrame,
        df_activity: pd.DataFrame,
        df_portafolio_ini: Optional[pd.DataFrame] = None,
        df_portafolio_fin: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
        conversor = ConversionADR()
        auditoria: List[Dict] = []

        original_rows = len(df_activity)
        df_activity_dedup, dup_removed = self.reglas.regla_duplicados(df_activity)
        df_activity_clean, vto_total, vto_count = self.reglas.regla_vto_prestamo(df_activity_dedup)

        if dup_removed > 0:
            auditoria.append({
                "cuenta": cuenta,
                "tipo_evento": "ACTIVITY_DUPLICADOS",
                "detalle": f"Se eliminaron {dup_removed} duplicados",
                "impacto": "WARN",
            })

        if vto_count > 0:
            auditoria.append({
                "cuenta": cuenta,
                "tipo_evento": "VTO_PRESTAMO",
                "detalle": f"Se excluyeron {vto_count} ops por {format_num(vto_total)}",
                "impacto": "WARN",
            })

        homologacion = build_homologacion_table(df_posiciones, df_activity_clean)

        resultados: List[Dict] = []

        for _, row in df_posiciones.iterrows():
            especie = row["especie"]
            especie_h = row["especie_h"]

            act_especie = df_activity_clean[df_activity_clean["ticker_h"] == especie_h].copy()

            ini_cant = float(row.get("ini_cant", 0) or 0)
            fin_cant = float(row.get("fin_cant", 0) or 0)
            ratio = 1.0

            if cuenta == self.periodo.cuenta_ib:
                ini_ib = float(row.get("ini_ib", 0) or 0)
                fin_ib = float(row.get("fin_ib", 0) or 0)
                ini_local = float(row.get("ini_local", ini_cant) or 0)
                fin_local = float(row.get("fin_local", fin_cant) or 0)
                ratio = conversor.obtener_ratio(especie_h, self.periodo.ratios_adr)
                ini_cant = conversor.calcular_total_992(ini_local, ini_ib, ratio)
                fin_cant = conversor.calcular_total_992(fin_local, fin_ib, ratio)

            res = self.conciliar_especie(
                cuenta=cuenta,
                especie=especie,
                especie_h=especie_h,
                ini_cant=ini_cant,
                act_cant=float(row.get("act_cant", 0) or 0),
                fin_cant=fin_cant,
                ini_imp=float(row.get("ini_imp", 0) or 0),
                act_imp=float(row.get("act_imp", 0) or 0),
                fin_imp=float(row.get("fin_imp", 0) or 0),
                comp=float(row.get("comp", 0) or 0),
                ratio=ratio,
                df_activity_especie=act_especie if not act_especie.empty else None,
                df_portafolio_ini=df_portafolio_ini,
                df_portafolio_fin=df_portafolio_fin,
            )
            resultados.append(res)

        df_result = pd.DataFrame(resultados)

        pos_set = set(df_posiciones["especie_h"].astype(str))
        act_set = set(df_activity_clean["ticker_h"].astype(str))

        match_ok = sorted(pos_set & act_set)
        pos_sin_match = sorted(pos_set - act_set)
        act_sin_match = sorted(act_set - pos_set)

        match_table = pd.DataFrame(
            [{"grupo": "match_ok", "especie_h": x} for x in match_ok] +
            [{"grupo": "posiciones_sin_match", "especie_h": x} for x in pos_sin_match] +
            [{"grupo": "activity_sin_match", "especie_h": x} for x in act_sin_match]
        )

        for esp in act_sin_match:
            cant = df_activity_clean.loc[df_activity_clean["ticker_h"] == esp, "cantidad"].sum()
            auditoria.append({
                "cuenta": cuenta,
                "tipo_evento": "ACTIVITY_SIN_MATCH",
                "detalle": f"{esp}: {format_num(cant)}",
                "impacto": "BAD",
            })

        for esp in pos_sin_match:
            auditoria.append({
                "cuenta": cuenta,
                "tipo_evento": "POSICION_SIN_MATCH",
                "detalle": f"{esp}: existe en posiciones y no en activity",
                "impacto": "WARN",
            })

        resumen = {
            "cuenta": cuenta,
            "activity_rows_original": original_rows,
            "activity_rows_clean": len(df_activity_clean),
            "duplicados_eliminados": dup_removed,
            "vto_prestamo_excluidos": vto_count,
            "total_especies": len(df_result),
            "cerradas": int(df_result["status"].astype(str).str.contains("CERRADA").sum()),
            "pendientes": int((df_result["status"] == "PENDIENTE").sum()),
            "pct_cierre": float(df_result["status"].astype(str).str.contains("CERRADA").mean() * 100) if len(df_result) else 0,
            "dif_total_abs": float(df_result["dif_final"].abs().sum()),
            "dif_total_neto": float(df_result["dif_final"].sum()),
        }

        return (
            df_result,
            resumen,
            pd.DataFrame(auditoria),
            match_table.merge(homologacion, how="left", left_on="especie_h", right_on="homologada"),
        )

    def generar_reporte(self, resultados: List[pd.DataFrame]) -> pd.DataFrame:
        if not resultados:
            return pd.DataFrame()
        return pd.concat(resultados, ignore_index=True)


# =============================================================================
# INTER-CUENTA
# =============================================================================

def auditoria_intercuenta(
    df_all: pd.DataFrame,
    pares_comp: Dict[int, int],
) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame(columns=["cuenta_origen", "cuenta_destino", "especie_h", "dif_origen", "dif_destino", "suma_neta", "estado"])

    processed = set()
    rows: List[Dict] = []

    for origen, destino in pares_comp.items():
        pair_key = tuple(sorted([origen, destino]))
        if pair_key in processed:
            continue
        processed.add(pair_key)

        df_o = df_all[df_all["cuenta"] == origen][["especie_h", "dif_final"]].copy()
        df_d = df_all[df_all["cuenta"] == destino][["especie_h", "dif_final"]].copy()

        df_o = df_o.groupby("especie_h", as_index=False)["dif_final"].sum().rename(columns={"dif_final": "dif_origen"})
        df_d = df_d.groupby("especie_h", as_index=False)["dif_final"].sum().rename(columns={"dif_final": "dif_destino"})

        merged = df_o.merge(df_d, on="especie_h", how="outer").fillna(0)
        merged["cuenta_origen"] = origen
        merged["cuenta_destino"] = destino
        merged["suma_neta"] = merged["dif_origen"] + merged["dif_destino"]
        merged["estado"] = np.where(merged["suma_neta"].abs() < 1, "OK", "REVISAR")
        rows.extend(merged.to_dict("records"))

    return pd.DataFrame(rows)


# =============================================================================
# TABLAS DERIVADAS
# =============================================================================

def build_resumen_cuentas(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()

    resumen = (
        df_all.groupby("cuenta", dropna=False)
        .agg(
            especies=("especie_h", "count"),
            cerradas=("status", lambda s: (s.astype(str).str.contains("CERRADA")).sum()),
            pendientes=("status", lambda s: (s == "PENDIENTE").sum()),
            reglas_aplicadas=("reglas_aplicadas", "sum"),
            dif_total_abs=("dif_final", lambda s: s.abs().sum()),
            dif_total_neto=("dif_final", "sum"),
        )
        .reset_index()
    )

    resumen["pct_cierre"] = np.where(
        resumen["especies"] > 0,
        resumen["cerradas"] / resumen["especies"] * 100,
        0,
    )

    def classify(row) -> str:
        if row["pendientes"] == 0:
            return "OK"
        if row["pct_cierre"] >= 85:
            return "WARN"
        return "BAD"

    resumen["semaforo"] = resumen.apply(classify, axis=1)
    return resumen.sort_values(["semaforo", "cuenta"]).reset_index(drop=True)


def build_top_pendientes(df_all: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()
    top = df_all[df_all["status"] == "PENDIENTE"].copy()
    if top.empty:
        return top
    top["abs_dif"] = top["dif_final"].abs()
    top = top.sort_values(["abs_dif", "cuenta", "especie_h"], ascending=[False, True, True]).head(top_n)
    cols = [
        "cuenta", "especie", "especie_h", "ini", "act", "fin", "comp",
        "ini_ajust", "act_ajust", "fin_ajust", "comp_ajust",
        "dif_final", "ajustes", "flags", "imputacion_sugerida"
    ]
    return top[cols].reset_index(drop=True)


def build_reglas_aplicadas(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame(columns=["regla", "cantidad"])

    counter: Dict[str, int] = {}
    for val in df_all["flags"].fillna("").astype(str):
        parts = [p.strip() for p in val.split("|") if p.strip()]
        for p in parts:
            counter[p] = counter.get(p, 0) + 1

    df = pd.DataFrame([{"regla": k, "cantidad": v} for k, v in counter.items()])
    if df.empty:
        return pd.DataFrame(columns=["regla", "cantidad"])
    return df.sort_values("cantidad", ascending=False).reset_index(drop=True)


# =============================================================================
# EXCEL
# =============================================================================

def auto_adjust_columns(ws) -> None:
    for col_cells in ws.columns:
        max_len = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            try:
                max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 42)


def style_header(ws) -> None:
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    fill = PatternFill("solid", fgColor="FF3B30")
    font = Font(color="FFFFFF", bold=True)
    border = Border(
        bottom=Side(style="thin", color="DDDDDD"),
        top=Side(style="thin", color="DDDDDD"),
        left=Side(style="thin", color="DDDDDD"),
        right=Side(style="thin", color="DDDDDD"),
    )

    for cell in ws[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border


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
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
        df_top_pend.to_excel(writer, sheet_name="Top Pendientes", index=False)
        df_pend.to_excel(writer, sheet_name="Pendientes", index=False)
        df_all.to_excel(writer, sheet_name="Detalle", index=False)
        df_auditoria.to_excel(writer, sheet_name="Auditoria", index=False)
        df_reglas.to_excel(writer, sheet_name="Reglas", index=False)
        df_match.to_excel(writer, sheet_name="Match", index=False)
        df_intercuenta.to_excel(writer, sheet_name="Intercuenta", index=False)

        wb = writer.book
        for ws_name in wb.sheetnames:
            ws = wb[ws_name]
            style_header(ws)
            auto_adjust_columns(ws)
            ws.freeze_panes = "A2"

    output.seek(0)
    return output.getvalue()
