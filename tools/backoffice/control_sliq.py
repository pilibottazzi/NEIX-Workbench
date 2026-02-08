# tools/control_sliq.py
from __future__ import annotations

import io
import re
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import streamlit as st


# =========================================================
# UI (estilo parecido al HTML)
# =========================================================
def _inject_ui_css():
    st.markdown(
        """
        <style>
          .sliq-topbar{
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding: 10px 6px 14px 6px;
          }
          .sliq-brand{
            display:flex;
            align-items:center;
            gap:12px;
          }
          .sliq-logo{
            width:34px; height:34px;
            border-radius:10px;
            border:1px solid rgba(0,0,0,0.10);
            display:flex; align-items:center; justify-content:center;
            font-weight:900;
            color:#111827;
            background:#fff;
            box-shadow:0 2px 10px rgba(0,0,0,0.04);
          }
          .sliq-title{
            margin:0;
            font-size: 1.35rem;
            font-weight: 900;
            color:#111827;
          }
          .sliq-sub{
            margin: 2px 0 0 0;
            color:#6b7280;
            font-size:.92rem;
          }

          .sliq-card{
            border:1px solid rgba(0,0,0,0.08);
            border-radius:16px;
            padding:18px;
            background:#fff;
            box-shadow:0 2px 10px rgba(0,0,0,0.04);
            margin-top: 10px;
          }

          .sliq-logs{
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
            background: rgba(249,250,251,0.9);
            font-size: .92rem;
            color:#111827;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers de parsing / normalización (replica JS)
# =========================================================
def _clean_str(x) -> str:
    return "" if x is None else str(x).strip()


def _strip_accents(s: str) -> str:
    # Normalización simple sin depender de unidecode
    # (quita marcas diacríticas unicode)
    return "".join(ch for ch in s.normalize("NFD") if not ("\u0300" <= ch <= "\u036f"))


def _norm_header(s: Any) -> str:
    s = _clean_str(s).lower()
    # quita tildes/acentos y compacta espacios
    try:
        s = _strip_accents(s)  # type: ignore[attr-defined]
    except Exception:
        # fallback (por si no existe normalize)
        pass
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_num_es(v) -> Optional[float]:
    """
    Convierte número estilo ES:
      "1.234,56" -> 1234.56
      "1000" -> 1000.0
    """
    s = _clean_str(v)
    if not s:
        return None

    s = re.sub(r"[^\d,.\-]", "", s)

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        n = float(s)
        return n
    except Exception:
        return None


def _read_text_with_fallback(file) -> str:
    """
    Lee st.file_uploader (bytes) como texto.
    Prueba utf-8, si aparecen caracteres raros, intenta cp1252.
    """
    raw = file.getvalue()
    for enc in ("utf-8", "cp1252"):
        try:
            txt = raw.decode(enc)
            # Si hay � (replacement char), probamos el otro
            if "\uFFFD" in txt and enc != "cp1252":
                continue
            return txt
        except Exception:
            continue
    # último recurso
    return raw.decode("utf-8", errors="replace")


def _read_csv_as_table(text: str, sep: str) -> pd.DataFrame:
    """
    Lee CSV a DataFrame sin header (header=None) para poder detectar header row como en JS.
    Maneja BOM.
    """
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    # engine="python" suele ser más tolerante con CSV raros
    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=None,
        dtype=str,
        engine="python",
        keep_default_na=False,
    )


def _find_header_row(df0: pd.DataFrame, max_rows: int = 20) -> int:
    """
    Replica findHeaderRow(aoa): elige la fila con mayor "ancho" (celdas no vacías) en las primeras N filas.
    """
    lim = min(max_rows, len(df0))
    best = 0
    best_w = -1

    for r in range(lim):
        row = df0.iloc[r].tolist()
        # contamos celdas NO vacías (similar a "len" útil)
        w = sum(1 for x in row if _clean_str(x) != "")
        if w > best_w:
            best_w = w
            best = r
    return best


def _map_exact_indexes(header_row: List[Any], wanted: List[str]) -> List[int]:
    """
    Replica mapExactIndexes: mapea wanted -> índice en header, usando normHeader.
    """
    header_map = {_norm_header(h): i for i, h in enumerate(header_row)}
    out = []
    for w in wanted:
        key = _norm_header(w)
        out.append(header_map.get(key, -1))
    return out


# =========================================================
# Core logic (idéntica al JS)
# =========================================================
def _build_nasdaq_detalle_from_table(df0: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    NASDAQ:
    - Detecta header row
    - Mapea columnas exactas por nombre normalizado (sin tildes, etc.)
    - Filtra:
        ref NO empieza con "SLIQ-"
        cuenta == "7142/10000"
        estado != "CANCELADO"
    - Detalle: (Instrumento, Referencia, Cantidad, Cuenta, Estado)
    - sumByInst: suma Cantidad por Instrumento (enteros redondeados)
    """
    hdr_idx = _find_header_row(df0)
    header = df0.iloc[hdr_idx].tolist()

    want = [
        "Instrumento",
        "Referencia instrucción",
        "Cantidad/nominal",
        "Cuenta de valores negociables",
        "Estado de instrucción",
    ]
    idx = _map_exact_indexes(header, want)
    if any(i < 0 for i in idx):
        raise ValueError(
            "NASDAQ: falta alguna columna exacta (según encabezados). "
            f"Busco: {want}. Header detectado: {header}"
        )

    detalle_rows: List[List[Any]] = []
    sum_by_inst: Dict[int, int] = {}

    # iterar filas de datos
    for r in range(hdr_idx + 1, len(df0)):
        row = df0.iloc[r].tolist()
        if all(_clean_str(x) == "" for x in row):
            continue

        ref = _clean_str(row[idx[1]]).upper()
        if ref.startswith("SLIQ-"):
            continue

        cuenta = _clean_str(row[idx[3]])
        if cuenta != "7142/10000":
            continue

        estado = _clean_str(row[idx[4]]).upper()
        if estado == "CANCELADO":
            continue

        inst_num = _to_num_es(row[idx[0]])
        q_num = _to_num_es(row[idx[2]])

        inst_int = int(round(inst_num)) if inst_num is not None else None
        q_int = int(round(q_num)) if q_num is not None else None

        detalle_rows.append([
            inst_int if inst_int is not None else "",
            _clean_str(row[idx[1]]),
            q_int if q_int is not None else "",
            cuenta,
            _clean_str(row[idx[4]]),
        ])

        if inst_int is not None and q_int is not None:
            sum_by_inst[inst_int] = sum_by_inst.get(inst_int, 0) + int(q_int)

    out = pd.DataFrame(
        detalle_rows,
        columns=[
            "Instrumento",
            "Referencia instrucción",
            "Cantidad/nominal",
            "Cuenta de valores negociables",
            "Estado de instrucción",
        ],
    )
    return out, sum_by_inst


def _build_sliq_from_table(df0: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    SLIQ (replica JS):
    - Detecta header row (aunque el JS usa la misma lógica)
    - Toma columnas por posición: 0 Código, 1 Especie, 2 Denominacion, 3 Neto a Liquidar
    - Suma neto por código (manteniendo especie/denom como first-non-empty)
    """
    hdr_idx = _find_header_row(df0)
    # data desde hdr_idx+1
    if df0.shape[1] < 4:
        raise ValueError("SLIQ: el CSV debe tener al menos 4 columnas (Código, Especie, Denominación, Neto).")

    out_rows: List[List[Any]] = []
    by_code: Dict[int, Dict[str, Any]] = {}

    for r in range(hdr_idx + 1, len(df0)):
        row = df0.iloc[r].tolist()
        if all(_clean_str(x) == "" for x in row):
            continue

        cod = _to_num_es(row[0])
        especie = _clean_str(row[1])
        denom = _clean_str(row[2])
        neto = _to_num_es(row[3])

        cod_int = int(round(cod)) if cod is not None else None
        neto_val = float(neto) if neto is not None else None

        out_rows.append([
            cod_int if cod_int is not None else "",
            especie,
            denom,
            neto_val if neto_val is not None else "",
        ])

        if cod_int is not None:
            prev = by_code.get(cod_int, {"especie": "", "denom": "", "neto": 0.0})
            by_code[cod_int] = {
                "especie": prev["especie"] or especie,
                "denom": prev["denom"] or denom,
                "neto": float(prev["neto"]) + (float(neto_val) if neto_val is not None else 0.0),
            }

    out = pd.DataFrame(out_rows, columns=["Código", "Especie", "Denominacion", "Neto a Liquidar"])
    return out, by_code


def _build_control(sum_by_inst: Dict[int, int], sliq_by_code: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Control SLIQ tarde (idéntico a JS):
    - Excluye ceros: qnas!=0 o neto!=0
    - Unión de keys
    - Orden: primero REVISAR (qnas+neto != 0), luego por código
    - Q SLIQ y Observación se llenan con fórmulas al exportar
    """
    nz_inst = {k: v for k, v in sum_by_inst.items() if v != 0}
    nz_sliq = {k: v for k, v in sliq_by_code.items() if float(v.get("neto", 0.0)) != 0.0}

    keys = set(nz_inst.keys()) | set(nz_sliq.keys())

    data: List[Dict[str, Any]] = []
    for k in keys:
        qnas = nz_inst.get(k, 0)
        s = nz_sliq.get(k, {"especie": "", "denom": "", "neto": 0.0})
        neto = float(s.get("neto", 0.0))
        revisar = (qnas + neto) != 0

        data.append({
            "_revisar": revisar,
            "_k": k,
            "Instrumento/Código": k,
            "Especie": s.get("especie", ""),
            "Denominación": s.get("denom", ""),
            "Q NASDAQ": qnas,
            "Neto a Liquidar": neto,
            "Q SLIQ": "",
            "Observación": "",
        })

    data.sort(key=lambda d: (not d["_revisar"], d["_k"]))

    for d in data:
        d.pop("_revisar", None)
        d.pop("_k", None)

    return pd.DataFrame(data, columns=[
        "Instrumento/Código", "Especie", "Denominación",
        "Q NASDAQ", "Neto a Liquidar", "Q SLIQ", "Observación"
    ])


def _export_excel(df_nasdaq: pd.DataFrame, df_control: pd.DataFrame, df_sliq: pd.DataFrame) -> bytes:
    """
    Exporta Excel:
      - Nasdaq
      - Control SLIQ tarde (con fórmulas)
      - SLIQ
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_nasdaq.to_excel(writer, sheet_name="Nasdaq", index=False)
        df_control.to_excel(writer, sheet_name="Control SLIQ tarde", index=False)
        df_sliq.to_excel(writer, sheet_name="SLIQ", index=False)

        wb = writer.book

        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_2d = wb.add_format({"num_format": "#,##0.00"})

        # ---- Nasdaq formatting
        ws_n = writer.sheets["Nasdaq"]
        ws_n.set_column("A:A", 14, fmt_int)
        ws_n.set_column("B:B", 28)
        ws_n.set_column("C:C", 16, fmt_int)
        ws_n.set_column("D:D", 28)
        ws_n.set_column("E:E", 20)

        # ---- SLIQ formatting
        ws_s = writer.sheets["SLIQ"]
        ws_s.set_column("A:A", 12, fmt_int)
        ws_s.set_column("B:B", 14)
        ws_s.set_column("C:C", 32)
        ws_s.set_column("D:D", 18, fmt_2d)

        # ---- Control formatting + formulas
        ws_c = writer.sheets["Control SLIQ tarde"]
        ws_c.set_column("A:A", 20, fmt_int)
        ws_c.set_column("B:B", 14)
        ws_c.set_column("C:C", 34)
        ws_c.set_column("D:D", 12, fmt_int)
        ws_c.set_column("E:E", 16, fmt_2d)
        ws_c.set_column("F:F", 12, fmt_2d)
        ws_c.set_column("G:G", 14)

        # Fórmulas: F = D + E ; G = IF(F=0,"OK","REVISAR")
        nrows = len(df_control)
        for i in range(nrows):
            excel_row = i + 2  # fila 2 es primer dato
            ws_c.write_formula(i + 1, 5, f"=D{excel_row}+E{excel_row}")
            ws_c.write_formula(i + 1, 6, f'=IF(F{excel_row}=0,"OK","REVISAR")')

    return output.getvalue()


# =========================================================
# Render (Streamlit tool)
# =========================================================
def render(back_to_home=None):
    _inject_ui_css()

    # Header como tu HTML (logo + título)
    st.markdown(
        """
        <div class="sliq-topbar">
          <div class="sliq-brand">
            <div class="sliq-logo">N</div>
            <div>
              <div class="sliq-title">⚠️ Control SLIQ</div>
              <div class="sliq-sub">Genera el Excel "Control SLIQ tarde.xlsx" a partir de NASDAQ y SLIQ.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sliq-card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        f_nasdaq = st.file_uploader("Instr. de Liquidación NASDAQ", type=["csv"], key="sliq_nasdaq")
    with c2:
        f_sliq = st.file_uploader("Especies para un Participante", type=["csv"], key="sliq_sliq")

    run = st.button('Generar "Control SLIQ"', type="primary", key="sliq_run")

    st.markdown("</div>", unsafe_allow_html=True)  # cierre card

    if not run:
        return

    if not f_nasdaq or not f_sliq:
        st.error("Faltan archivos: cargá NASDAQ y SLIQ (CSV).")
        return

    logs: List[str] = []
    def log(m: str):
        logs.append(m)

    try:
        log("Leyendo NASDAQ (,) y SLIQ (;)…")

        # ---- NASDAQ
        nas_txt = _read_text_with_fallback(f_nasdaq)
        df_n0 = _read_csv_as_table(nas_txt, sep=",")
        if df_n0.empty:
            st.error("NASDAQ: archivo vacío.")
            return

        log("Procesando NASDAQ…")
        df_nas_out, sum_by_inst = _build_nasdaq_detalle_from_table(df_n0)

        # ---- SLIQ
        sliq_txt = _read_text_with_fallback(f_sliq)
        df_s0 = _read_csv_as_table(sliq_txt, sep=";")
        if df_s0.empty:
            st.error("SLIQ: archivo vacío.")
            return

        log("Procesando SLIQ…")
        df_sliq_out, sliq_by_code = _build_sliq_from_table(df_s0)

        # ---- CONTROL
        log("Armando Control SLIQ tarde…")
        df_control = _build_control(sum_by_inst, sliq_by_code)

        # ---- EXCEL
        log("Generando Excel…")
        xlsx_bytes = _export_excel(df_nas_out, df_control, df_sliq_out)

        st.success("Listo ✅ Se generó el archivo.")
        st.download_button(
            "Descargar Control SLIQ tarde.xlsx",
            data=xlsx_bytes,
            file_name="Control SLIQ tarde.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sliq_download",
        )

        # Logs + preview (opcional, pero útil)
        st.markdown('<div class="sliq-logs">', unsafe_allow_html=True)
        st.write("\n".join(f"• {m}" for m in logs))
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Preview — Control SLIQ tarde")
        st.dataframe(df_control.head(80), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Error generando el Control SLIQ.")
        st.exception(e)


    except Exception as e:
        st.error("Error generando el Control SLIQ.")
        st.exception(e)

