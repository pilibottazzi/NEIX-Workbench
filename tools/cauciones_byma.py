# tools/cauciones_byma.py
import os
import streamlit as st
import pandas as pd


# =========================
# Config
# =========================
DATA_PATH = os.path.join("data", "Garantia Byma.xlsx")
REQUIRED_COLS = ["ESPECIE", "AFORO", "MARGEN", "MÁXIMO POR ESPECIE", "LISTA"]


# =========================
# Helpers
# =========================
def _to_float_amount(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _fmt_ars(x) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "-"


def _fmt_pct(x) -> str:
    try:
        return f"{int(round(float(x) * 100))}%"
    except Exception:
        return "-"


def _is_divide_by_100(tipo_activo: str) -> bool:
    """
    Regla BYMA de tu script:
    - Divide por 100 si es: TÍTULO/OBLIGACIÓN/BONO/LETRA/LECAP
    - NO divide si es CEDEAR o ACCIONES
    """
    t = (tipo_activo or "").upper().strip()

    # Excluir explícitamente
    if ("CEDEAR" in t) or ("ACCIÓN" in t) or ("ACCIONES" in t) or ("ACCION" in t):
        return False

    activos_base_100 = ["TÍTULO", "TITULO", "OBLIGACIÓN", "OBLIGACION", "BONO", "LETRA", "LECAP"]
    return any(a in t for a in activos_base_100)


@st.cache_data(show_spinner=False)
def cargar_aforos_byma() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No existe el archivo '{DATA_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(DATA_PATH)

    # Normalizar columnas a MAYÚSCULAS para que no dependa de cómo venga el Excel
    df.columns = df.columns.astype(str).str.strip().str.upper()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas {missing}. Columnas disponibles: {list(df.columns)}"
        )

    # Normalizaciones
    df["ESPECIE"] = df["ESPECIE"].astype(str).str.upper().str.strip()
    df["AFORO"] = pd.to_numeric(df["AFORO"], errors="coerce")
    df["MARGEN"] = pd.to_numeric(df["MARGEN"], errors="coerce")
    df["MÁXIMO POR ESPECIE"] = pd.to_numeric(df["MÁXIMO POR ESPECIE"], errors="coerce")
    df["LISTA"] = df["LISTA"].astype(str)

    df = df.dropna(subset=["ESPECIE", "AFORO"])
    return df


# =========================
# Main render
# =========================
def render(back_to_home=None):
    if callable(back_to_home):
        back_to_home()

    st.markdown("## 🧾 Calculadora de Garantías BYMA")
    st.caption("Calculá garantía admitida por especie según aforos BYMA (Excel pre-cargado en el repo).")

    # Cargar datos
    try:
        df_aforos = cargar_aforos_byma()
    except Exception as e:
        st.error("No pude cargar el Excel de aforos BYMA.")
        st.exception(e)
        st.stop()

    # Estado
    if "byma_operaciones" not in st.session_state:
        st.session_state.byma_operaciones = []

    metodo = st.radio(
        "¿Cómo querés ingresar el valor?",
        ["Por monto", "Por precio y nominales"],
        horizontal=True
    )

    with st.form("form_byma_operacion", clear_on_submit=True):
        especies = [""] + sorted(df_aforos["ESPECIE"].unique().tolist())
        especie = st.selectbox("Seleccioná la especie", options=especies, index=0)

        monto = None
        tipo_activo = ""
        dividir_por_100 = True

        if especie:
            row = df_aforos[df_aforos["ESPECIE"] == especie]
            if not row.empty:
                tipo_activo = str(row.iloc[0]["LISTA"])
                dividir_por_100 = _is_divide_by_100(tipo_activo)

        if metodo == "Por monto":
            monto_txt = st.text_input("Monto (AR$)", placeholder="Ej: 1.000.000")
            monto = _to_float_amount(monto_txt)
        else:
            c1, c2 = st.columns(2)
            precio_txt = c1.text_input("Precio", placeholder="Ej: 68,75")
            nominales_txt = c2.text_input("Nominales", placeholder="Ej: 100.000")

            precio = _to_float_amount(precio_txt)
            nominales = _to_float_amount(nominales_txt)

            if precio is not None and nominales is not None:
                monto = precio * nominales
                if dividir_por_100:
                    monto = monto / 100.0

        if especie:
            st.caption(
                f"Tipo: **{tipo_activo or '-'}** · "
                f"{'Se divide' if dividir_por_100 else 'No se divide'} por 100 para valor de mercado (regla BYMA)."
            )

        submitted = st.form_submit_button("Agregar", type="primary")

        if submitted:
            if not especie:
                st.warning("Seleccioná una especie válida.")
            elif monto is None or monto <= 0:
                st.warning("Ingresá valores válidos.")
            else:
                row = df_aforos[df_aforos["ESPECIE"] == especie]
                if row.empty:
                    st.warning("La especie seleccionada no se encontró en el Excel.")
                else:
                    datos = row.iloc[0]
                    aforo = float(datos["AFORO"])
                    margen = float(datos["MARGEN"]) if pd.notna(datos["MARGEN"]) else None
                    maximo = float(datos["MÁXIMO POR ESPECIE"]) if pd.notna(datos["MÁXIMO POR ESPECIE"]) else None
                    garantia = monto * aforo

                    st.session_state.byma_operaciones.append({
                        "Especie": especie,
                        "Tipo de activo": str(datos["LISTA"]),
                        "Método": metodo,
                        "Monto": float(monto),
                        "Aforo": aforo,
                        "Margen": margen,
                        "Máximo permitido": maximo,
                        "Garantía admitida": float(garantia),
                    })

    st.divider()

    ops = st.session_state.byma_operaciones
    if ops:
        st.subheader("Resultado del cálculo")

        df_res = pd.DataFrame(ops)

        # tabla formateada simple (cloud-friendly)
        show = df_res.copy()
        show["Monto"] = show["Monto"].map(_fmt_ars)
        show["Aforo"] = show["Aforo"].map(_fmt_pct)
        if "Margen" in show.columns:
            show["Margen"] = show["Margen"].map(_fmt_pct)
        if "Máximo permitido" in show.columns:
            show["Máximo permitido"] = show["Máximo permitido"].map(_fmt_ars)
        show["Garantía admitida"] = show["Garantía admitida"].map(_fmt_ars)

        st.dataframe(show, use_container_width=True, hide_index=True)

        total = df_res["Garantía admitida"].sum()
        st.markdown(f"### Garantía total admitida: **AR$ {_fmt_ars(total)}**")

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar operaciones (CSV)",
            csv,
            file_name="garantias_byma_operaciones.csv",
            mime="text/csv"
        )
    else:
        st.info("Todavía no agregaste operaciones.")

    # Reiniciar (solo esto, como pediste)
    if st.button("Reiniciar cálculo"):
        st.session_state.byma_operaciones = []
        st.rerun()

