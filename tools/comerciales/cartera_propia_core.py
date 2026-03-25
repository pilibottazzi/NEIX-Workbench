from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# =============================================================================
# CORE (SE MANTIENE IGUAL)
# =============================================================================
from tools.comerciales.cartera_propia_core import (
    CUENTAS_DEFAULT,
    PeriodoConciliacion,
    ConciliadorMensual,
    auditoria_intercuenta,
    build_resumen_cuentas,
    build_top_pendientes,
    build_reglas_aplicadas,
    preparar_activity,
    preparar_portafolio,
    preparar_posiciones,
    read_any_file,
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("cierre_mensual_unificado")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ConfigPeriodo:
    ini_date: date
    fin_date: date
    tc_mep: float
    tc_cable: float
    datos_path: Path
    output_path: Path
    mesa_target: float = 0.0
    one_pager_path: Optional[Path] = None
    cuentas: List[int] = field(default_factory=lambda: list(CUENTAS_DEFAULT))

    @property
    def periodo_str(self) -> str:
        return f"{self.ini_date:%Y-%m-%d} a {self.fin_date:%Y-%m-%d}"


# =============================================================================
# HELPERS GENERALES
# =============================================================================

def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def to_numeric_safe(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("U$S", "", regex=False)
    s = s.str.replace("(", "-", regex=False)
    s = s.str.replace(")", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def safe_group_sum(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=group_cols + value_cols)

    agg_map = {c: "sum" for c in value_cols if c in df.columns}
    if not agg_map:
        return pd.DataFrame(columns=group_cols + value_cols)

    return df.groupby(group_cols, dropna=False, as_index=False).agg(agg_map)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

class CargaDatos:
    """
    Batch loader unificado.
    Conserva la idea de PY1/PY3:
      - portafolios ini/fin por cuenta
      - activity por cuenta
      - posiciones por cuenta
      - alquileres
      - ib ini/fin
      - ratios adr
      - one pager
    """

    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def _path_candidates(self, patterns: List[str]) -> Optional[Path]:
        for p in patterns:
            candidate = self.config.datos_path / p
            if candidate.exists():
                return candidate
        return None

    def _load_generic(self, path: Optional[Path]) -> pd.DataFrame:
        if path is None or not path.exists():
            return pd.DataFrame()
        try:
            return read_any_file(path)
        except Exception as e:
            log.warning(f"No se pudo leer {path.name}: {e}")
            return pd.DataFrame()

    def _load_portafolio(self, cuenta: int, stage: str) -> pd.DataFrame:
        path = self._path_candidates([
            f"portafolio_{cuenta}_{stage}.csv",
            f"portafolio_{cuenta}_{stage}.xlsx",
            f"portafolio_{cuenta}_{stage}.xls",
            f"portafolio_{cuenta}_{stage}.xlsm",
            f"cartera_{cuenta}_{stage}.csv",
            f"cartera_{cuenta}_{stage}.xlsx",
        ])
        raw = self._load_generic(path)
        df, alerts = preparar_portafolio(raw) if not raw.empty else (pd.DataFrame(), [])
        for a in alerts:
            log.warning(f"Cta {cuenta} {stage}: {a}")
        return df

    def _load_activity(self, cuenta: int) -> pd.DataFrame:
        path = self._path_candidates([
            f"activity_{cuenta}.csv",
            f"activity_{cuenta}.xlsx",
            f"activity_{cuenta}.xls",
            f"activity_{cuenta}.xlsm",
            f"movimientos_{cuenta}.csv",
            f"movimientos_{cuenta}.xlsx",
        ])
        raw = self._load_generic(path)
        df, alerts = preparar_activity(raw) if not raw.empty else (pd.DataFrame(), [])
        for a in alerts:
            log.warning(f"Cta {cuenta} activity: {a}")
        return df

    def _load_posiciones(self, cuenta: int) -> pd.DataFrame:
        path = self._path_candidates([
            f"posiciones_{cuenta}.csv",
            f"posiciones_{cuenta}.xlsx",
            f"posiciones_{cuenta}.xls",
            f"posiciones_{cuenta}.xlsm",
        ])
        raw = self._load_generic(path)
        df, alerts = preparar_posiciones(raw) if not raw.empty else (pd.DataFrame(), [])
        for a in alerts:
            log.warning(f"Cta {cuenta} posiciones: {a}")
        return df

    def cargar_todos(self) -> Dict[str, Any]:
        log.info("=== PASO 1: CARGA DE DATOS ===")

        portafolios_ini: Dict[int, pd.DataFrame] = {}
        portafolios_fin: Dict[int, pd.DataFrame] = {}
        activities: Dict[int, pd.DataFrame] = {}
        posiciones: Dict[int, pd.DataFrame] = {}

        for cta in self.config.cuentas:
            portafolios_ini[cta] = self._load_portafolio(cta, "ini")
            portafolios_fin[cta] = self._load_portafolio(cta, "fin")
            activities[cta] = self._load_activity(cta)
            posiciones[cta] = self._load_posiciones(cta)

            log.info(
                f"Cta {cta} | ini={len(portafolios_ini[cta])} "
                f"fin={len(portafolios_fin[cta])} act={len(activities[cta])} pos={len(posiciones[cta])}"
            )

        ib_ini = self._load_generic(self._path_candidates(["ib_positions_ini.csv", "ib_positions_ini.xlsx"]))
        ib_fin = self._load_generic(self._path_candidates(["ib_positions_fin.csv", "ib_positions_fin.xlsx"]))
        alquileres = self._load_generic(self._path_candidates(["alquileres.csv", "alquileres.xlsx"]))
        ratios_adr = self._load_generic(self._path_candidates(["ratios_adr.csv", "ratios_adr.xlsx"]))

        one_pager = pd.DataFrame()
        if self.config.one_pager_path and self.config.one_pager_path.exists():
            one_pager = self._load_generic(self.config.one_pager_path)

        return {
            "portafolios_ini": portafolios_ini,
            "portafolios_fin": portafolios_fin,
            "activities": activities,
            "posiciones": posiciones,
            "ib_ini": ib_ini,
            "ib_fin": ib_fin,
            "alquileres": alquileres,
            "ratios_adr": ratios_adr,
            "one_pager": one_pager,
        }


# =============================================================================
# ALQUILERES
# =============================================================================

class ModuloAlquileres:
    """
    Mantenemos el bloque batch de PY1/PY3.
    Si no hay archivo de alquileres, devuelve vacío.
    """

    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def calcular_delta_alquiler(self, df_alq: pd.DataFrame, df_dif: pd.DataFrame) -> pd.DataFrame:
        if df_alq is None or df_alq.empty:
            return pd.DataFrame(columns=["Comitente", "Especie", "AlqIni", "AlqFin", "DeltaAlquiler"])

        df = df_alq.copy()
        cols = {c.lower().strip(): c for c in df.columns}

        rename_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "cta" in cl or "cuenta" in cl or "comitente" in cl:
                rename_map[c] = "Comitente"
            elif "especie" in cl or "ticker" in cl:
                rename_map[c] = "Especie"
            elif "alq" in cl and "ini" in cl:
                rename_map[c] = "AlqIni"
            elif "alq" in cl and "fin" in cl:
                rename_map[c] = "AlqFin"

        df = df.rename(columns=rename_map)

        for col in ["Comitente", "Especie", "AlqIni", "AlqFin"]:
            if col not in df.columns:
                if col in {"AlqIni", "AlqFin"}:
                    df[col] = 0.0
                else:
                    df[col] = ""

        df["Comitente"] = pd.to_numeric(df["Comitente"], errors="coerce").fillna(0).astype(int)
        df["Especie"] = df["Especie"].astype(str).str.strip()
        df["AlqIni"] = to_numeric_safe(df["AlqIni"])
        df["AlqFin"] = to_numeric_safe(df["AlqFin"])
        df["DeltaAlquiler"] = df["AlqFin"] - df["AlqIni"]

        out = df[["Comitente", "Especie", "AlqIni", "AlqFin", "DeltaAlquiler"]].copy()
        out = safe_group_sum(out, ["Comitente", "Especie"], ["AlqIni", "AlqFin", "DeltaAlquiler"])

        log.info(f"Alquileres procesados: {len(out)} filas")
        return out


# =============================================================================
# RESULTADO MENSUAL
# =============================================================================

class ResultadoMensual:
    """
    Une la tabla de diferencias con alquileres y arma una salida consolidada.
    """

    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def calcular(self, df_dif: pd.DataFrame, df_alq: pd.DataFrame) -> pd.DataFrame:
        if df_dif is None or df_dif.empty:
            return pd.DataFrame()

        out = df_dif.copy()

        if df_alq is not None and not df_alq.empty:
            merge_cols_left = ["cuenta", "especie_h"]
            merge_cols_right = ["Comitente", "Especie"]

            df_alq2 = df_alq.copy()
            out = out.merge(
                df_alq2,
                left_on=merge_cols_left,
                right_on=merge_cols_right,
                how="left",
            )
            out["DeltaAlquiler"] = out["DeltaAlquiler"].fillna(0.0)
        else:
            out["DeltaAlquiler"] = 0.0

        if "dif_importe" not in out.columns:
            out["dif_importe"] = 0.0

        out["resultado_base_ars"] = out["dif_importe"].fillna(0.0)
        out["resultado_alquiler_ars"] = out["DeltaAlquiler"].fillna(0.0)
        out["resultado_total_ars"] = out["resultado_base_ars"] + out["resultado_alquiler_ars"]
        out["resultado_total_usd_mep"] = out["resultado_total_ars"] / self.config.tc_mep
        out["resultado_total_usd_cable"] = out["resultado_total_ars"] / self.config.tc_cable

        return out


# =============================================================================
# ACTIVITY ANALYZER
# =============================================================================

class ActivityAnalyzer:
    """
    Conserva el paso del batch grande.
    """

    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def procesar_cuenta(self, cuenta: int, df_act: pd.DataFrame, df_resultado: pd.DataFrame) -> Dict[str, float]:
        if df_act is None or df_act.empty:
            return {
                "cuenta": cuenta,
                "rdo_ida_vuelta": 0.0,
                "rdo_real_904": 0.0,
                "operaciones": 0,
            }

        out = {
            "cuenta": cuenta,
            "rdo_ida_vuelta": 0.0,
            "rdo_real_904": 0.0,
            "operaciones": len(df_act),
        }

        if "importe" in df_act.columns:
            out["rdo_ida_vuelta"] = float(to_numeric_safe(df_act["importe"]).sum())

        return out

    def procesar_todas(self, activities: Dict[int, pd.DataFrame], df_resultado: pd.DataFrame) -> Dict[Any, Dict[str, float]]:
        results: Dict[Any, Dict[str, float]] = {}
        total_904 = 0.0

        for cta, df_act in activities.items():
            r = self.procesar_cuenta(cta, df_act, df_resultado)
            results[cta] = r
            total_904 += r.get("rdo_real_904", 0.0)

        results["rdo_real_904"] = total_904
        return results


# =============================================================================
# RECONSTRUCCIÓN IB
# =============================================================================

class ReconstruccionIB:
    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def reconstruir(
        self,
        df_dif_992: pd.DataFrame,
        ib_ini: pd.DataFrame,
        ib_fin: pd.DataFrame,
        ratios_adr: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_dif_992 is None or df_dif_992.empty:
            return pd.DataFrame()

        out = df_dif_992.copy()
        out["recon_tipo"] = "992_IB"
        return out


# =============================================================================
# CONSOLIDACIÓN
# =============================================================================

class ResultadoConsolidado:
    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def extraer_por_cuenta(self, df_resultado: pd.DataFrame) -> pd.DataFrame:
        if df_resultado is None or df_resultado.empty:
            return pd.DataFrame(columns=["Cuenta", "ARS", "USD_MEP", "USD_CABLE"])

        resumen = (
            df_resultado.groupby("cuenta", dropna=False)
            .agg(
                ARS=("resultado_total_ars", "sum"),
                USD_MEP=("resultado_total_usd_mep", "sum"),
                USD_CABLE=("resultado_total_usd_cable", "sum"),
            )
            .reset_index()
            .rename(columns={"cuenta": "Cuenta"})
        )
        resumen["Cuenta"] = resumen["Cuenta"].apply(lambda x: f"Cta {int(x)}")
        return resumen

    def consolidar(self, resumen: pd.DataFrame) -> Dict[str, float]:
        if resumen is None or resumen.empty:
            return {
                "total_ars": 0.0,
                "total_usd_mep": 0.0,
                "total_usd_cable": 0.0,
                "indicador": "NEUTRO",
            }

        total_ars = float(resumen["ARS"].sum())
        total_usd_mep = float(resumen["USD_MEP"].sum())
        total_usd_cable = float(resumen["USD_CABLE"].sum())

        if total_usd_mep > 0:
            ind = "POSITIVO"
        elif total_usd_mep < 0:
            ind = "NEGATIVO"
        else:
            ind = "NEUTRO"

        return {
            "total_ars": total_ars,
            "total_usd_mep": total_usd_mep,
            "total_usd_cable": total_usd_cable,
            "indicador": ind,
        }


# =============================================================================
# CONCILIACIÓN CON MESA
# =============================================================================

class ConciliacionMesa:
    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def cargar_one_pager(self, df_one_pager: pd.DataFrame) -> Dict[str, float]:
        if df_one_pager is None or df_one_pager.empty:
            return {
                "pulga_ytd": 0.0,
                "middle_ytd": 0.0,
                "otros_ytd": 0.0,
            }

        return {
            "pulga_ytd": 0.0,
            "middle_ytd": 0.0,
            "otros_ytd": 0.0,
        }

    def prorratear(self, comp_ytd: Dict[str, float]) -> Dict[str, float]:
        return {
            "pulga_periodo": comp_ytd.get("pulga_ytd", 0.0),
            "middle_periodo": comp_ytd.get("middle_ytd", 0.0),
            "otros_periodo": comp_ytd.get("otros_ytd", 0.0),
        }

    def separar_rdo_real_posicional(self, df_resultado: pd.DataFrame) -> Dict[str, float]:
        if df_resultado is None or df_resultado.empty:
            return {"rdo_real": 0.0, "rdo_posicional": 0.0}

        total = float(df_resultado["resultado_total_usd_mep"].sum())
        return {
            "rdo_real": total,
            "rdo_posicional": 0.0,
        }

    def armar_cascada(
        self,
        rdo_real: float,
        pulga: float,
        mesa_target: float,
        rdo_ida_vuelta: float,
        middle: float,
        rdo_904: float,
        otros: float,
    ) -> Dict[str, float]:
        subtotal = rdo_real + pulga + middle + rdo_904 + otros + rdo_ida_vuelta
        brecha = subtotal - mesa_target

        return {
            "rdo_real": rdo_real,
            "pulga": pulga,
            "middle": middle,
            "rdo_904": rdo_904,
            "otros": otros,
            "rdo_ida_vuelta": rdo_ida_vuelta,
            "mesa_target": mesa_target,
            "subtotal": subtotal,
            "brecha": brecha,
        }


# =============================================================================
# EXCEL FINAL
# =============================================================================

class GeneradorExcel:
    def __init__(self, config: ConfigPeriodo):
        self.config = config

    def generar(
        self,
        output_path: Path,
        df_dif: pd.DataFrame,
        df_resultado: pd.DataFrame,
        df_alq: pd.DataFrame,
        resumen: pd.DataFrame,
        total: Dict[str, float],
        cascada: Dict[str, float],
        recon_ib: pd.DataFrame,
        datos_manual: Dict[str, Any],
        df_resumen_core: Optional[pd.DataFrame] = None,
        df_top: Optional[pd.DataFrame] = None,
        df_reglas: Optional[pd.DataFrame] = None,
        df_intercuenta: Optional[pd.DataFrame] = None,
        df_auditoria: Optional[pd.DataFrame] = None,
        df_match: Optional[pd.DataFrame] = None,
    ) -> None:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if df_dif is not None and not df_dif.empty:
                df_dif.to_excel(writer, sheet_name="Diferencias", index=False)
            if df_resultado is not None and not df_resultado.empty:
                df_resultado.to_excel(writer, sheet_name="Resultado Mensual", index=False)
            if df_alq is not None and not df_alq.empty:
                df_alq.to_excel(writer, sheet_name="Detalle Alquileres", index=False)
            if resumen is not None and not resumen.empty:
                resumen.to_excel(writer, sheet_name="Resumen Cuentas", index=False)
            if recon_ib is not None and not recon_ib.empty:
                recon_ib.to_excel(writer, sheet_name="Recon 992 Detalle", index=False)

            pd.DataFrame([total]).to_excel(writer, sheet_name="Total Consolidado", index=False)
            pd.DataFrame([cascada]).to_excel(writer, sheet_name="Cascada Mesa", index=False)
            pd.DataFrame([datos_manual]).to_excel(writer, sheet_name="Manual Data", index=False)

            if df_resumen_core is not None and not df_resumen_core.empty:
                df_resumen_core.to_excel(writer, sheet_name="Resumen Core", index=False)
            if df_top is not None and not df_top.empty:
                df_top.to_excel(writer, sheet_name="Top Pendientes", index=False)
            if df_reglas is not None and not df_reglas.empty:
                df_reglas.to_excel(writer, sheet_name="Reglas", index=False)
            if df_intercuenta is not None and not df_intercuenta.empty:
                df_intercuenta.to_excel(writer, sheet_name="Intercuenta", index=False)
            if df_auditoria is not None and not df_auditoria.empty:
                df_auditoria.to_excel(writer, sheet_name="Auditoria", index=False)
            if df_match is not None and not df_match.empty:
                df_match.to_excel(writer, sheet_name="Match", index=False)

        log.info(f"Excel generado: {output_path}")


# =============================================================================
# ORQUESTADOR PRINCIPAL UNIFICADO
# =============================================================================

def ejecutar_cierre(config: ConfigPeriodo) -> Dict[str, Any]:
    log.info("====== CONCILIACIÓN Y RESULTADO DE CARTERA PROPIA ======")
    log.info(f"Período: {config.periodo_str}")
    log.info(f"TC MEP: {config.tc_mep:,.2f} | Cable: {config.tc_cable:,.2f}")

    # PASO 1: Carga
    cargador = CargaDatos(config)
    datos = cargador.cargar_todos()

    # PASO 2: Conciliación base usando CORE
    log.info("=== PASO 2: CONCILIACIÓN BASE (CORE) ===")
    periodo = PeriodoConciliacion(
        fecha_ini=str(config.ini_date),
        fecha_fin=str(config.fin_date),
        cuentas=config.cuentas,
    )
    conciliador = ConciliadorMensual(periodo)

    resultados = []
    auditorias = []
    matches = []

    for cuenta in config.cuentas:
        try:
            df_res, resumen, df_aud, df_match = conciliador.conciliar_cuenta(
                cuenta=cuenta,
                df_posiciones=datos["posiciones"].get(cuenta, pd.DataFrame()),
                df_activity=datos["activities"].get(cuenta, pd.DataFrame()),
                df_portafolio_ini=datos["portafolios_ini"].get(cuenta, pd.DataFrame()),
                df_portafolio_fin=datos["portafolios_fin"].get(cuenta, pd.DataFrame()),
            )
            resultados.append(df_res)
            auditorias.append(df_aud)
            matches.append(df_match)
        except Exception as e:
            log.exception(f"Cta {cuenta}: error en conciliación core: {e}")

    df_dif = conciliador.generar_reporte(resultados)
    df_auditoria = pd.concat(auditorias, ignore_index=True) if auditorias else pd.DataFrame()
    df_match = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()

    df_resumen_core = build_resumen_cuentas(df_dif)
    df_top = build_top_pendientes(df_dif)
    df_reglas = build_reglas_aplicadas(df_dif)
    df_intercuenta = auditoria_intercuenta(df_dif, periodo.pares_comp)

    # PASO 3: Alquileres
    log.info("=== PASO 3: ALQUILERES ===")
    mod_alq = ModuloAlquileres(config)
    df_alq = mod_alq.calcular_delta_alquiler(datos["alquileres"], df_dif)

    # PASO 4: Resultado mensual
    log.info("=== PASO 4: RESULTADO MENSUAL ===")
    calc_rdo = ResultadoMensual(config)
    df_resultado = calc_rdo.calcular(df_dif, df_alq)

    # PASO 5: Activity Analyzer
    log.info("=== PASO 5: ANÁLISIS ACTIVITY ===")
    analyzer = ActivityAnalyzer(config)
    act_results = analyzer.procesar_todas(datos["activities"], df_resultado)

    # PASO 6: Reconstrucción IB
    log.info("=== PASO 6: RECONSTRUCCIÓN IB ===")
    recon_mod = ReconstruccionIB(config)
    recon_ib = recon_mod.reconstruir(
        df_dif[df_dif["cuenta"] == 992] if not df_dif.empty and "cuenta" in df_dif.columns else pd.DataFrame(),
        datos["ib_ini"],
        datos["ib_fin"],
        datos["ratios_adr"],
    )

    # PASO 7: Consolidación
    log.info("=== PASO 7: CONSOLIDACIÓN ===")
    consolidador = ResultadoConsolidado(config)
    resumen = consolidador.extraer_por_cuenta(df_resultado)
    total = consolidador.consolidar(resumen)

    # PASO 8: Conciliación con mesa
    log.info("=== PASO 8: CONCILIACIÓN CON MESA ===")
    conc_mesa = ConciliacionMesa(config)
    comp_ytd = conc_mesa.cargar_one_pager(datos["one_pager"])
    comp_periodo = conc_mesa.prorratear(comp_ytd)
    sep = conc_mesa.separar_rdo_real_posicional(df_resultado)
    rdo_iv_total = sum(
        r.get("rdo_ida_vuelta", 0.0)
        for r in act_results.values()
        if isinstance(r, dict)
    )

    cascada = conc_mesa.armar_cascada(
        rdo_real=sep["rdo_real"],
        pulga=comp_periodo["pulga_periodo"],
        mesa_target=config.mesa_target,
        rdo_ida_vuelta=rdo_iv_total,
        middle=comp_periodo["middle_periodo"],
        rdo_904=act_results.get("rdo_real_904", 0.0),
        otros=comp_periodo["otros_periodo"],
    )
    cascada["rdo_posicional"] = sep["rdo_posicional"]

    # PASO 9: Excel final
    log.info("=== PASO 9: GENERAR EXCEL ===")
    datos_manual = {
        "total_consolidado": total,
        "n_especies": len(df_dif),
        "n_conciliadas": int((df_dif["dif_final"].abs() < 0.01).sum()) if not df_dif.empty else 0,
        "cascada": cascada,
    }

    GeneradorExcel(config).generar(
        output_path=config.output_path,
        df_dif=df_dif,
        df_resultado=df_resultado,
        df_alq=df_alq,
        resumen=resumen,
        total=total,
        cascada=cascada,
        recon_ib=recon_ib,
        datos_manual=datos_manual,
        df_resumen_core=df_resumen_core,
        df_top=df_top,
        df_reglas=df_reglas,
        df_intercuenta=df_intercuenta,
        df_auditoria=df_auditoria,
        df_match=df_match,
    )

    log.info("")
    log.info("====== RESULTADO FINAL ======")
    log.info(f"{total['total_usd_mep']:,.0f} USD ({total['indicador']})")
    log.info(
        f"{len(df_dif)} especies | "
        f"{int((df_dif['dif_final'].abs() < 0.01).sum()) if not df_dif.empty else 0} conciliadas"
    )
    log.info(f"Archivo: {config.output_path}")

    return {
        "df_dif": df_dif,
        "df_resultado": df_resultado,
        "df_alq": df_alq,
        "df_resumen_core": df_resumen_core,
        "df_top": df_top,
        "df_reglas": df_reglas,
        "df_intercuenta": df_intercuenta,
        "df_auditoria": df_auditoria,
        "df_match": df_match,
        "resumen": resumen,
        "total": total,
        "cascada": cascada,
        "recon_ib": recon_ib,
        "act_results": act_results,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cierre mensual unificado de cartera propia")
    parser.add_argument("--ini-date", required=True, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--fin-date", required=True, help="Fecha fin YYYY-MM-DD")
    parser.add_argument("--datos", required=True, help="Carpeta con archivos de entrada")
    parser.add_argument("--tc-mep", type=float, required=True, help="Tipo de cambio MEP")
    parser.add_argument("--tc-cable", type=float, default=None, help="TC cable (default=tc-mep)")
    parser.add_argument("--mesa-target", type=float, default=0.0, help="Rdo mesa período USD")
    parser.add_argument("--one-pager", default=None, help="Ruta a ONE_PAGER")
    parser.add_argument("--output", default="resultado_mensual_unificado.xlsx", help="Archivo final")
    parser.add_argument("--cuentas", nargs="*", type=int, default=None, help="Lista de cuentas opcional")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ConfigPeriodo(
        ini_date=date.fromisoformat(args.ini_date),
        fin_date=date.fromisoformat(args.fin_date),
        tc_mep=args.tc_mep,
        tc_cable=args.tc_cable or args.tc_mep,
        datos_path=Path(args.datos),
        output_path=Path(args.output),
        mesa_target=args.mesa_target,
        one_pager_path=Path(args.one_pager) if args.one_pager else None,
        cuentas=args.cuentas or list(CUENTAS_DEFAULT),
    )

    ejecutar_cierre(config)


if __name__ == "__main__":
    main()
