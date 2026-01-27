# tools/cauciones_mae.py
import os
import streamlit as st
import pandas as pd


# =========================
# Config
# =========================
DATA_PATH = os.path.join("data", "Garantia MAE.xlsx")
REQUIRED_COLS = ["ESPECIE", "AFORO", "CONCENTRACIÓN (EN PESOS)", "ACTIVO"]


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


def _normalize_colname(c: str) -> str:
    return str(c).strip().upper()


def _find_col(df_cols, target: str):
    target_n = _normalize_colname(target)
    for c in df_cols:
        if _normalize_colname(c) == target_n:
            return c

    base = target_n.replace("Ó", "O").replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ú", "U")
    for c in df_cols:
        cn = _normalize_colname(c)
        cn2 = cn.replace("Ó", "O").replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ú", "U")
        if base in cn2:
            return c
    return None


def _is_divide_by_100(tipo_activo: str) -> bool:
    t = (tipo_activo or "").upper().strip()
    if ("CEDEAR" in t) or ("ACCIÓN" in t) or ("ACCIONES" in t) or ("ACCION" in t):
        return False
    return True


@st.cache_data(show_spinner=False)
def cargar_aforos_mae() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No existe el archivo '{DATA_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    col_especie = _find_col(df.columns, "ESPECIE")
    col_aforo = _find_col(df.columns, "AFORO")
    col_conc = _find_col(df.columns, "CONCENTRACIÓN (EN PESOS)")
    col_activo = _find_col(df.columns, "ACTIVO")

    missing = []
    if col_especie is None: missing.append("ESPECIE")
    if col_aforo is None: missing.append("AFORO")
    if col_conc is None: missing.append("CONCENTRACIÓN (EN PESOS)")
    if col_activo is None: missing.append("ACTIVO")

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas {missing}. Columnas disponibles: {list(df.columns)}"
        )

    df = df.rename(columns={
        col_especie: "ESPECIE",
        col_aforo: "AFORO",
        col_conc: "CONCENTRACIÓN (EN PESOS)",
        col_activo: "ACTIVO",
    })

    df["ESPECIE"] = df["ESPECIE"].astype(str).str.upper().str.strip()
    df["AFORO"] = pd.to_numeric(df["AFORO"], errors="coerce")
    df["CONCENTRACIÓN (EN PESOS)"] = pd.to_numeric(df["CONCENTRACIÓN (EN PESOS)"], errors="coerce")
    df["ACTIVO"] = df["ACTIVO"].astype(str)

    df = df.dropna(subset=["ESPECIE", "AFORO"])
    return df


# =========================
# Main render
# =========================
def render(back_to_home=None):
    # IMPORTANTE: NO llamar back_to_home() acá.
    # El botón "Volver al Workbench" ya lo renderiza app.py
    st.markdown("## 🧾 Calculadora de Garantías MAE")
    st.caption("Calculá garantía admitida por especie según aforos MAE (Excel pre-cargado en el repo).")

    try:
        df_aforos = cargar_aforos_mae()
    except Exception as e:
        st.error("No pude cargar el Excel de aforos MAE.")
        st.exception(e)
        st.stop()

    if "mae_operaciones" not in st.session_state:
        st.session_state.mae_operaciones = []

    metodo = st.radio(
        "¿Cómo querés ingresar el valor?",
        ["Por monto", "Por precio y nominales"],
        horizontal=True,
        key="mae_metodo"
    )

    with st.form("form_mae_operacion", clear_on_submit=True):
        especies = [""] + sorted(df_aforos["ESPECIE"].dropna().unique().tolist())
        especie = st.selectbox("Seleccioná la especie", options=especies, index=0, key="mae_especie")

        tipo_activo = ""
        dividir_por_100 = True
        monto = None

        if especie:
            row = df_aforos[df_aforos["ESPECIE"] == especie]
            if not row.empty:
                tipo_activo = str(row.iloc[0]["ACTIVO"])
                dividir_por_100 = _is_divide_by_100(tipo_activo)

        if metodo == "Por monto":
            monto_txt = st.text_input("Monto (AR$)", placeholder="Ej: 1.000.000", key="mae_monto")
            monto = _to_float_amount(monto_txt)
        else:
            c1, c2 = st.columns(2)
            precio_txt = c1.text_input("Precio", placeholder="Ej: 68,75", key="mae_precio")
            nominales_txt = c2.text_input("Nominales", placeholder="Ej: 100.000", key="mae_nominales")

            precio = _to_float_amount(precio_txt)
            nominales = _to_float_amount(nominales_txt)

            if precio is not None and nominales is not None:
                monto = precio * nominales
                if dividir_por_100:
                    monto = monto / 100.0

        if especie:
            st.caption(
                f"Tipo: **{tipo_activo or '-'}** · "
                f"{'Se divide' if dividir_por_100 else 'No se divide'} por 100 para valor de mercado (regla MAE)."
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
                    limite = (
                        float(datos["CONCENTRACIÓN (EN PESOS)"])
                        if pd.notna(datos["CONCENTRACIÓN (EN PESOS)"])
                        else None
                    )
                    garantia = float(monto) * aforo

                    st.session_state.mae_operaciones.append({
                        "Especie": especie,
                        "Tipo de activo": str(datos["ACTIVO"]),
                        "Método": metodo,
                        "Monto": float(monto),
                        "Aforo": aforo,
                        "Límite por especie": limite,
                        "Garantía admitida": float(garantia),
                    })

    st.divider()

    ops = st.session_state.mae_operaciones
    if ops:
        st.subheader("Resultado del cálculo")
        df_res = pd.DataFrame(ops)

        show = df_res.copy()
        show["Monto"] = show["Monto"].map(_fmt_ars)
        show["Aforo"] = show["Aforo"].map(_fmt_pct)
        if "Límite por especie" in show.columns:
            show["Límite por especie"] = show["Límite por especie"].map(_fmt_ars)
        show["Garantía admitida"] = show["Garantía admitida"].map(_fmt_ars)

        st.dataframe(show, use_container_width=True, hide_index=True)

        total = df_res["Garantía admitida"].sum()
        st.markdown(f"### Garantía total admitida: **AR$ {_fmt_ars(total)}**")

    if st.button("Reiniciar cálculo", key="mae_reset"):
        st.session_state.mae_operaciones = []
        st.rerun()
