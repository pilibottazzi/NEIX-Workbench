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
# UI (NEIX — limpio y normal)
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
          .block-container {{
            max-width: 1120px;
            padding-top: 1.25rem;
            padding-bottom: 2.25rem;
          }}

          .sliq-head {{ margin: 0 0 10px 0; }}
          .sliq-kicker {{
            display:flex; align-items:center; gap:10px;
            margin-bottom: 6px;
          }}
          .sliq-badge {{
            width:38px; height:38px;
            border-radius: 12px;
            border: 1px solid {BORDER};
            display:flex; align-items:center; justify-content:center;
            font-weight: 800;
            letter-spacing: .05em;
            background:#fff;
          }}
          .sliq-title {{
            margin:0;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.15;
            color:{TEXT};
          }}
          .sliq-sub {{
            margin: 0 0 0 48px;
            color:{MUTED};
            font-size: .95rem;
          }}

          .sliq-card {{
            border:1px solid {BORDER};
            border-radius:16px;
            padding:14px 14px 12px 14px;
            background:{CARD_BG};
            box-shadow: 0 2px 12px rgba(0,0,0,0.03);
          }}
          .sliq-card-title {{
            margin:0 0 2px 0;
            font-weight: 700;
            font-size: 1.02rem;
            color:{TEXT};
          }}
          .sliq-card-hint {{
            margin:0 0 10px 0;
            color:{MUTED};
            font-size: .88rem;
          }}

          [data-testid="stFileUploaderDropzone"] {{
            border-radius: 14px !important;
            border: 1px dashed rgba(17,24,39,0.22) !important;
            background: rgba(249,250,251,0.75) !important;
          }}
          [data-testid="stFileUploaderDropzone"] > div {{
            padding: 0.65rem 0.85rem !important;
          }}

          div.stButton > button[kind="primary"] {{
            width: 100%;
            background: {NEIX_RED};
            border: 1px solid rgba(0,0,0,0.06);
            color: #fff;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 800;
            box-shadow: 0 10px 20px rgba(255,59,48,0.16);
            transition: transform .06s ease, box-shadow .12s ease, filter .12s ease;
          }}
          div.stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            filter: brightness(0.98);
            box-shadow: 0 14px 26px rgba(255,59,48,0.20);
          }}

          hr {{ margin: 1.1rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_open() -> None:
    st.markdown('<div class="sliq-card">', unsafe_allow_html=True)


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Helpers de parsing / normalización (replica JS)
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
    s = _clean_str(v)
    if not s:
        return None

    s = re.sub(r"[^\d,.\-]", "", s)

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

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


def _find_header_row(df0: pd.DataFrame, max_rows: int = 20) -> int:
    lim = min(max_rows, len(df0))
    best = 0
    best_w = -1
    for r in range(lim):
        row = df0.iloc[r].tolist()
        w = sum(1 for x in row if _clean_str(x) != "")
        if w > best_w:
            best_w = w
            best = r
    return best


def _map_exact_indexes(header_row: List[Any], wanted: List[str]) -> List[int]:
    header_map = {_norm_header(h): i for i, h in enumerate(header_row)}
    return [header_map.get(_norm_header(w), -1) for w in wanted]


# =========================================================
# Core logic (idéntica al JS)
# =========================================================
def _build_nasdaq_detalle_from_table(df0: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
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
            "NASDAQ: falta alguna columna exacta.\n"
            f"Busco: {want}\n"
            f"Header detectado: {header}"
        )

    detalle_rows: List[List[Any]] = []
    sum_by_inst: Dict[int, int] = {}

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
    hdr_idx = _find_header_row(df0)

    if df0.shape[1] < 4:
        raise ValueError("SLIQ: el CSV debe tener al menos 4 columnas.")

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
        <div class="sliq-head">
          <div class="sliq-kicker">
            <div class="sliq-badge">N</div>
            <div class="sliq-title">Control SLIQ</div>
          </div>
          <div class="sliq-sub">Cargá NASDAQ (,) y SLIQ (;). Genera <b>Control SLIQ tarde.xlsx</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        _card_open()
        st.markdown('<div class="sliq-card-title">Instr. de Liquidación NASDAQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="sliq-card-hint">CSV separado por coma (<b>,</b>)</div>', unsafe_allow_html=True)
        f_nasdaq = st.file_uploader("", type=["csv"], key="sliq_nasdaq", label_visibility="collapsed")
        _card_close()

    with c2:
        _card_open()
        st.markdown('<div class="sliq-card-title">Especies para un Participante</div>', unsafe_allow_html=True)
        st.markdown('<div class="sliq-card-hint">CSV separado por punto y coma (<b>;</b>)</div>', unsafe_allow_html=True)
        f_sliq = st.file_uploader("", type=["csv"], key="sliq_sliq", label_visibility="collapsed")
        _card_close()

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

        st.divider()
        st.subheader("Preview — Control SLIQ tarde")
        st.dataframe(df_control.head(80), use_container_width=True, hide_index=True)

        with st.expander("Ver logs", expanded=False):
            st.write("\n".join(f"• {m}" for m in logs))

    except Exception as e:
        st.error("Error generando el Control SLIQ.")
        st.exception(e)

