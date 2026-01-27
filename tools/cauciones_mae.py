
import streamlit as st 
import pandas as pd

# --- Cargar datos de MAE ---
@st.cache_data
def cargar_aforos_mae():
    ruta = "C:/Users/pbottazzi/OneDrive - NEIX S.A/Documentos/pyNeix/GARANTIA/Garantia MAE.xlsx"
    df = pd.read_excel(ruta)
    df.columns = df.columns.str.strip().str.upper()
    columnas_requeridas = ["ESPECIE", "AFORO", "CONCENTRACIÓN (EN PESOS)", "ACTIVO"]
    for col in columnas_requeridas:
        if col not in df.columns:
            st.error(f"Falta la columna requerida: {col}")
            st.stop()
    return df

df_aforos = cargar_aforos_mae()

# --- UI ---
st.title("Calculadora de Garantías MAE")
st.caption("Ingresá una o más especies para calcular la garantía admitida según aforos MAE.")

if "operaciones" not in st.session_state:
    st.session_state.operaciones = []

metodo = st.radio("¿Cómo querés ingresar el valor?", ["Por monto", "Por precio y nominales"])

# --- Formulario ---
with st.form("formulario_operacion"):
    especie = st.selectbox(
        "Seleccioná la especie", 
        options=[""] + sorted(df_aforos["ESPECIE"].unique()),
        index=0,
        placeholder="Ej: AL30"
    )

    monto = None
    tipo_activo = ""
    dividir_por_100 = True

    if especie and especie in df_aforos["ESPECIE"].values:
        tipo_activo = str(df_aforos[df_aforos["ESPECIE"] == especie]["ACTIVO"].values[0]).upper().strip()

        if "CEDEAR" in tipo_activo or "ACCIÓN" in tipo_activo or "ACCIONES" in tipo_activo:
            dividir_por_100 = False
        else:
            dividir_por_100 = True

    if metodo == "Por monto":
        monto_texto = st.text_input("Monto", value="", placeholder="Ej: 1.000.000")
        try:
            monto = float(monto_texto.replace(".", "").replace(",", "."))
        except ValueError:
            monto = None
    else:
        col1, col2 = st.columns(2)
        precio_texto = col1.text_input("Precio", placeholder="Ej: 68.75")
        nominales_texto = col2.text_input("Nominales", placeholder="Ej: 100.000")
        try:
            precio = float(precio_texto.replace(",", "."))
            nominales = float(nominales_texto.replace(".", "").replace(",", "."))
            monto = precio * nominales
            if dividir_por_100:
                monto = monto / 100
        except ValueError:
            monto = None

    if especie != "":
        st.markdown(
            f"<div style='font-size: 12px; color: grey; margin-top: -10px;'>"
            f"*Esta especie es un <strong>{tipo_activo}</strong>. "
            f"{'Se divide' if dividir_por_100 else 'No se divide'} por 100 para obtener el valor de mercado según MAE.*"
            f"</div>",
            unsafe_allow_html=True
        )

    submitted = st.form_submit_button("Agregar")

    if submitted:
        if especie == "":
            st.warning("Seleccioná una especie válida.")
        elif monto is None:
            st.warning("Ingresá valores válidos.")
        else:
            try:
                datos = df_aforos[df_aforos["ESPECIE"] == especie].iloc[0]
                st.session_state.operaciones.append({
                    "Especie": especie,
                    "Monto": monto,
                    "Aforo": datos["AFORO"],
                    "Límite por especie": datos["CONCENTRACIÓN (EN PESOS)"],
                    "Tipo de activo": datos["ACTIVO"],
                    "Método": metodo,
                    "Garantía admitida": monto * datos["AFORO"]
                })
            except IndexError:
                st.warning("La especie seleccionada no se encontró en el archivo.")

# --- Mostrar resultados ---
if st.session_state.operaciones:
    st.markdown("<h4 style='margin-top: 30px;'>Resultado del cálculo</h4>", unsafe_allow_html=True)

    df_resultado = pd.DataFrame(st.session_state.operaciones)

    def estilo_tabla(df):
        return df.style.format({
            "Monto": lambda x: f"{int(round(x)):,}".replace(",", "."),
            "Aforo": lambda x: f"{int(round(x * 100))}%",
            "Límite por especie": lambda x: f"{int(round(x)):,}".replace(",", "."),
            "Garantía admitida": lambda x: f"{int(round(x)):,}".replace(",", ".")
        }).set_properties(**{
            "text-align": "center",
            "border": "1px solid #ccc",
            "padding": "6px"
        })

    st.dataframe(estilo_tabla(df_resultado), use_container_width=True, hide_index=True)

    total_garantía = int(round(df_resultado["Garantía admitida"].sum()))
    total_garantía_str = f"{total_garantía:,}".replace(",", ".")

    st.markdown(
        f"<div style='font-size: 18px; margin-top: 24px;'>"
        f"<strong>Garantía total admitida:</strong> AR$ {total_garantía_str}"
        f"</div>", unsafe_allow_html=True
    )

if st.button("Reiniciar cálculo"):
    st.session_state.operaciones = []

