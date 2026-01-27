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
    """
    Convierte strings típicos AR:
      '1.000.000' -> 1000000
      '68,75' -> 68.75
      '100.000' -> 100000
    """
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
    Regla que venías usando:
    - CEDEAR / ACCIONES: NO dividir por 100
    - resto: dividir por 100
    """
    t = (tipo_activo or "").upper().strip()
    if ("CEDEAR" in t) or ("ACCIÓN" in t) or ("ACCIONES" in t) or ("ACCION" in t):
        return False
    return True


@st.cache_data(show_spinner=False)
def cargar_aforos_mae() -> pd.DataFrame:
    """
    Lee el excel desde el repo: data/Garantia MAE.xlsx
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No existe el archivo '{DATA_PATH}'. Subilo al repo dentro de la carpeta 'data/'."
        )

    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip().str.upper()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas {missing}. Columnas disponibles: {list(df.columns)}")

    # Normalizaciones básicas
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
    # Botón volver (si viene del Workbench)
    if callable(back_to_home):
        back_to_home()

    st.markdown("## 🧾 Calculadora de Garantías MAE")
    st.caption("Calculá garantía admitida por especie según aforos MAE (Excel pre-cargado en el repo).")

    # Cargar datos
    try:
        df_aforos = cargar_aforos_mae()
    except Exception as e:
        st.error("No pude cargar el Excel de aforos.")
        st.exception(e)
        st.stop()

    # Estado
    if "mae_operaciones" not in st.session_state:
        st.session_state.mae_operaciones = []

    # UI
    metodo = st.radio(
        "¿Cómo querés ingresar el valor?",
        ["Por monto", "Por precio y nominales"],
        horizontal=True
    )

    with st.form("form_mae_operacion", clear_on_submit=True):
        especies = [""] + sorted(df_aforos["ESPECIE"].unique().tolist())
        especie = st.selectbox("Seleccioná la especie", options=especies, index=0)

        tipo_activo = ""
        dividir_por_100 = True
        monto = None

        if especie:
            row = df_aforos[df_aforos["ESPECIE"] == especie]
            if not row.empty:
                tipo_activo = str(row.iloc[0]["ACTIVO"])
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
                    limite = float(datos["CONCENTRACIÓN (EN PESOS)"]) if pd.notna(datos["CONCENTRACIÓN (EN PESOS)"]) else None
                    garantia = monto * aforo

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

    # Resultados
    ops = st.session_state.mae_operaciones
    if ops:
        st.subheader("Resultado del cálculo")

        df_res = pd.DataFrame(ops)

        # Mostrar tabla formateada (sin Styler para cloud)
        show = df_res.copy()
        show["Monto"] = show["Monto"].map(_fmt_ars)
        show["Aforo"] = show["Aforo"].map(_fmt_pct)
        if "Límite por especie" in show.columns:
            show["Límite por especie"] = show["Límite por especie"].map(_fmt_ars)
        show["Garantía admitida"] = show["Garantía admitida"].map(_fmt_ars)

        st.dataframe(show, use_container_width=True, hide_index=True)

        total = df_res["Garantía admitida"].sum()
        st.markdown(f"### Garantía total admitida: **AR$ {_fmt_ars(total)}**")

        # Descargar CSV (datos originales numéricos)
        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar operaciones (CSV)",
            csv,
            file_name="garantias_mae_operaciones.csv",
            mime="text/csv"
        )

        # Eliminar una fila
        st.markdown("#### Administrar operaciones")
        idx = st.selectbox(
            "Eliminar operación (por índice)",
            options=list(range(len(df_res))),
            format_func=lambda i: f"{i} — {df_res.loc[i,'Especie']} — AR$ {_fmt_ars(df_res.loc[i,'Monto'])}"
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Eliminar seleccionada"):
                try:
                    st.session_state.mae_operaciones.pop(int(idx))
                    st.rerun()
                except Exception:
                    st.warning("No pude eliminar esa fila.")
        with col2:
            st.caption("Tip: si querés borrar todo, usá “Reiniciar cálculo”.")

    else:
        st.info("Todavía no agregaste operaciones.")

    # Reiniciar
    if st.button("Reiniciar cálculo"):
        st.session_state.mae_operaciones = []
        st.rerun()

