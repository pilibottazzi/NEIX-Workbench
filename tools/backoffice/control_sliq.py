# tools/backoffice/control_sliq.py
from __future__ import annotations

import io
import re
import csv
from io import StringIO
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import streamlit as st


# =========================================================
# UI (NEIX — minimal, ejecutivo, ordenado)
# =========================================================
NEIX_RED = "#ff3b30"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "rgba(17,24,39,0.10)"
CARD_BG = "#ffffff"


def _inject_ui_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --text: #111827;
            --muted: #6b7280;
            --border: rgba(17,24,39,0.10);
            --border2: rgba(17,24,39,0.08);
            --bg: #ffffff;
            --soft: #f9fafb;
            --red: {NEIX_RED};
            --radius: 18px;
            --radius2: 14px;
          }}

          /* =============================
             PAGE LAYOUT
             ============================= */
          .block-container {{
            max-width: 1080px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
          }}

          hr {{
            border: 0;
            border-top: 1px solid var(--border2);
            margin: 22px 0;
          }}

          /* =============================
             WORKBENCH HEADER
             ============================= */
          .nw-title {{
            text-align: center;
            margin: 0;
            letter-spacing: .35em;
            font-weight: 700;
            color: var(--text);
            font-size: 1.25rem;
          }}

          .nw-sub {{
            text-align: center;
            margin: 8px 0 0 0;
            color: var(--muted);
            font-size: .95rem;
          }}

          /* =============================
             TOOL HEADER
             ============================= */
          .tool-head {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-top: 26px;
            margin-bottom: 8px;
          }}

          .tool-badge {{
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: var(--text);
            background: #fff;
            border: 1px solid var(--border);
          }}

          .tool-title {{
            margin: 0;
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--text);
            line-height: 1.1;
          }}

          .tool-desc {{
            margin: 6px 0 0 0;
            color: var(--muted);
            font-size: .95rem;
          }}

          /* =============================
             UPLOAD CARDS
             ============================= */
          .u-card {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px 16px 14px 16px;
            background: var(--bg);
          }}

          .u-title {{
            margin: 0 0 6px 0;
            font-weight: 800;
            color: var(--text);
            font-size: 1.02rem;
          }}

          .u-hint {{
            margin: 0 0 12px 0;
            color: var(--muted);
            font-size: .90rem;
          }}

          /* =============================
             FILE UPLOADER (CLEAN)
             ============================= */
          [data-testid="stFileUploader"] {{
            margin-top: 0 !important;
            padding-top: 0 !important;
          }}

          [data-testid="stFileUploader"] label {{
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
          }}

          [data-testid="stFileUploader"] > div {{
            margin-top: 0 !important;
          }}

          [data-testid="stFileUploaderDropzone"] {{
            border-radius: 14px !important;
            border: 1px dashed rgba(17,24,39,0.20) !important;
            background: var(--soft) !important;
          }}

          [data-testid="stFileUploaderDropzone"] > div {{
            padding: .75rem .85rem !important;
          }}

          /* =============================
             PRIMARY BUTTON
             ============================= */
          div.stButton > button[kind="primary"] {{
            width: 100%;
            background: var(--red);
            color: #fff;
            border-radius: 18px;
            padding: 12px 16px;
            font-weight: 800;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: none !important;
          }}

          div.stButton > button[kind="primary"]:hover {{
            filter: brightness(0.98);
            transform: translateY(-1px);
          }}

          /* =============================
             DATAFRAME
             ============================= */
          [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
          }}

          /* =============================
             MISC FIXES
             ============================= */
          label,
          .stMarkdown p {{
            margin-bottom: .35rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers de parsing / normalización (replica HTML “perfecto”)
# =========================================================
def _clean_str(x) -> str:
    return "" if x is None else str(x).strip()


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in s.normalize("NFD") if not ("\u0300" <= ch <= "\u036f"))


def _norm_header(s: Any) -> str:
    s = _clean_str(s).lower()
    try:
        s = _strip_accents(s)
    except Exception:
        pass
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_num_es(v) -> Optional[float]:
    """
    IGUAL al HTML:
    parseFloat(s.replace(/\./g,"").replace(",","."))

    - Quita puntos (miles)
    - Cambia coma a punto (decimales)
    - Tolera signo negativo
    """
    s = _clean_str(v)
    if not s:
        return None

    # deja solo dígitos, puntos, comas y '-'
    s = re.sub(r"[^\d,.\-]", "", s)

    # igual al JS: borra todos los '.' y cambia ',' por '.'
    s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None


def _read_text_with_fallback(file) -> str:
    raw = file.getvalue()
    for enc in ("utf-8", "cp1252"):
        try:
            txt = raw.decode(enc)
            if "\uFFFD" in txt and enc != "cp1252":
                continue
            return txt
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_csv_as_table(text: str, sep: str) -> pd.DataFrame:
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            header=None,
            dtype=str,
            engine="python",
            keep_default_na=False,
        )
    except Exception:
        pass

    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            header=None,
            dtype=str,
            engine="python",
            keep_default_na=False,
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
    except Exception:
        pass

    fixed_lines: List[str] = []
    for line in text.splitlines():
        if line.count('"') % 2 == 1:
            line = line.replace('"', '""')
        fixed_lines.append(line)

    reader = csv.reader(StringIO("\n".join(fixed_lines)), delimiter=sep, quotechar='"')
    data = list(reader)
    max_len = max((len(r) for r in data), default=0)
    padded = [r + [""] * (max_len - len(r)) for r in data]

    return pd.DataFrame(padded, dtype=str)


def _find_header_row_js_like(df0: pd.DataFrame, max_rows: int = 20) -> int:
    """
    Igual al HTML: elige la fila con mayor "length" (cantidad de columnas),
    no por cantidad de celdas no vacías.
    """
    lim = min(max_rows, len(df0))
    best = 0
    best_len = 0
    for r in range(lim):
        row = df0.iloc[r].tolist()
        ln = len(row)
        if ln > best_len:
            best_len = ln
            best = r
    return best


def _map_exact_indexes(header_row: List[Any], wanted: List[str]) -> List[int]:
    header_map = {_norm_header(h): i for i, h in enumerate(header_row)}
    return [header_map.get(_norm_header(w), -1) for w in wanted]


# =========================================================
# Core logic (idéntica al HTML)
# =========================================================
def _build_nasdaq_detalle_from_table(df0: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
    hdr_idx = _find_header_row_js_like(df0)
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
            "NASDAQ: falta alguna columna exacta.\n"
            f"Busco: {want}\n"
            f"Header detectado: {header}"
        )

    detalle_rows: List[List[Any]] = []
    sum_by_inst: Dict[int, int] = {}

    for r in range(hdr_idx + 1, len(df0)):
        row = df0.iloc[r].tolist()
        if len(row) == 0 or all(_clean_str(x) == "" for x in row):
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
    hdr_idx = _find_header_row_js_like(df0)

    if df0.shape[1] < 4:
        raise ValueError("SLIQ: el CSV debe tener al menos 4 columnas.")

    out_rows: List[List[Any]] = []
    by_code: Dict[int, Dict[str, Any]] = {}

    for r in range(hdr_idx + 1, len(df0)):
        row = df0.iloc[r].tolist()
        if len(row) == 0 or all(_clean_str(x) == "" for x in row):
            continue

        cod = _to_num_es(row[0] if len(row) > 0 else "")
        especie = _clean_str(row[1] if len(row) > 1 else "")
        denom = _clean_str(row[2] if len(row) > 2 else "")
        neto = _to_num_es(row[3] if len(row) > 3 else "")

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

    return pd.DataFrame(
        data,
        columns=[
            "Instrumento/Código",
            "Especie",
            "Denominación",
            "Q NASDAQ",
            "Neto a Liquidar",
            "Q SLIQ",
            "Observación",
        ],
    )


def _export_excel(df_nasdaq: pd.DataFrame, df_control: pd.DataFrame, df_sliq: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_nasdaq.to_excel(writer, sheet_name="Nasdaq", index=False)
        df_control.to_excel(writer, sheet_name="Control SLIQ tarde", index=False)
        df_sliq.to_excel(writer, sheet_name="SLIQ", index=False)

        wb = writer.book
        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_2d = wb.add_format({"num_format": "#,##0.00"})

        ws_n = writer.sheets["Nasdaq"]
        ws_n.set_column("A:A", 14, fmt_int)
        ws_n.set_column("B:B", 28)
        ws_n.set_column("C:C", 16, fmt_int)
        ws_n.set_column("D:D", 28)
        ws_n.set_column("E:E", 20)

        ws_s = writer.sheets["SLIQ"]
        ws_s.set_column("A:A", 12, fmt_int)
        ws_s.set_column("B:B", 14)
        ws_s.set_column("C:C", 32)
        ws_s.set_column("D:D", 18, fmt_2d)

        ws_c = writer.sheets["Control SLIQ tarde"]
        ws_c.set_column("A:A", 20, fmt_int)
        ws_c.set_column("B:B", 14)
        ws_c.set_column("C:C", 34)
        ws_c.set_column("D:D", 12, fmt_int)
        ws_c.set_column("E:E", 16, fmt_2d)
        ws_c.set_column("F:F", 12, fmt_2d)
        ws_c.set_column("G:G", 14)

        nrows = len(df_control)
        for i in range(nrows):
            excel_row = i + 2
            ws_c.write_formula(i + 1, 5, f"=D{excel_row}+E{excel_row}")
            ws_c.write_formula(i + 1, 6, f'=IF(F{excel_row}=0,"OK","REVISAR")')

    return output.getvalue()


# =========================================================
# Render
# =========================================================
def render(back_to_home=None):
    _inject_ui_css()

    st.markdown(
        """
          <div>
            <h2 class="tool-title">Control SLIQ</h2>
            <p class="tool-desc">Cargá NASDAQ (,) y SLIQ (;). Genera <b>Control SLIQ tarde.xlsx</b>.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="u-title">Instr. de Liquidación NASDAQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="u-hint">CSV separado por coma (<b>,</b>)</div>', unsafe_allow_html=True)
        f_nasdaq = st.file_uploader("", type=["csv"], key="sliq_nasdaq", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="u-title">Especies para un Participante</div>', unsafe_allow_html=True)
        st.markdown('<div class="u-hint">CSV separado por punto y coma (<b>;</b>)</div>', unsafe_allow_html=True)
        f_sliq = st.file_uploader("", type=["csv"], key="sliq_sliq", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    run = st.button('Generar "Control SLIQ"', type="primary", key="sliq_run")

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

        nas_txt = _read_text_with_fallback(f_nasdaq)
        df_n0 = _read_csv_as_table(nas_txt, sep=",")
        if df_n0.empty:
            st.error("NASDAQ: archivo vacío.")
            return

        log("Procesando NASDAQ…")
        df_nas_out, sum_by_inst = _build_nasdaq_detalle_from_table(df_n0)

        sliq_txt = _read_text_with_fallback(f_sliq)
        bad_quotes = sum(1 for ln in sliq_txt.splitlines() if ln.count('"') % 2 == 1)
        if bad_quotes:
            st.warning(f"SLIQ: detecté {bad_quotes} línea(s) con comillas desbalanceadas. Se corrigieron automáticamente.")

        df_s0 = _read_csv_as_table(sliq_txt, sep=";")
        if df_s0.empty:
            st.error("SLIQ: archivo vacío.")
            return

        log("Procesando SLIQ…")
        df_sliq_out, sliq_by_code = _build_sliq_from_table(df_s0)

        log("Armando Control SLIQ tarde…")
        df_control = _build_control(sum_by_inst, sliq_by_code)

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

        st.markdown("<hr/>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Error generando el Control SLIQ.")
        st.exception(e)
