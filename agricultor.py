import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Establecer el estilo de la página
st.set_page_config(
    page_title="Predicción de tipo de cultivo",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color:#00d7e5;
        padding: 20px;
    }
    .main .block-container {
        padding: 20px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMarkdown h2 {
        color: #4CAF50;  /* Color del título h2 */
    }
    .stMarkdown h3 {
        color:#1E90FF;  /* Color del título h3 */
    }
    </style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("Predicción de tipo de cultivo")

# Cargar el conjunto de datos
datos = pd.read_csv("data/agricola.csv")

# Convertir la columna de fecha a características numéricas adicionales
datos['Fecha_Siembra'] = pd.to_datetime(datos['Fecha_Siembra'])
datos['Dia_Siembra'] = datos['Fecha_Siembra'].dt.day
datos['Mes_Siembra'] = datos['Fecha_Siembra'].dt.month
datos['Año_Siembra'] = datos['Fecha_Siembra'].dt.year

# Eliminar la columna de fecha original
datos.drop('Fecha_Siembra', axis=1, inplace=True)

# Dividir los datos en características (X) y objetivo (Y)
X = datos.drop("Tipo_Producto", axis=1)
Y = datos["Tipo_Producto"]

# Crear una variable de objetivo para las plagas
datos["Tiene_Plagas"] = datos["Presencia_Plagas_Enfermedades"].apply(lambda x: 1 if x == "Sí" else 0)
Y_plagas = datos["Tiene_Plagas"]

# Dividir los datos en entrenamiento y prueba
X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(X, Y, test_size=0.35, random_state=42)
X_entrenamiento_plagas, X_prueba_plagas, Y_entrenamiento_plagas, Y_prueba_plagas = train_test_split(X, Y_plagas, test_size=0.35, random_state=42)

# Definir columnas categóricas y numéricas
columnas_categoricas = ['Tipo_Suelo', 'Tipo_Irrigacion', 'Uso_Fertilizantes']
columnas_numericas = ['Temperatura', 'Humedad', 'Precipitacion', 'Altitud', 'pH_Suelo', 'Luz_Solar', 'Dia_Siembra', 'Mes_Siembra', 'Año_Siembra']

# Definir transformador para preprocesamiento de datos
transformador = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), columnas_categoricas),
    ('num', SimpleImputer(strategy='mean'), columnas_numericas)
])

# Construir pipeline con preprocesamiento y modelo
pipeline_cultivo = Pipeline(steps=[
    ('preprocesamiento', transformador),
    ('modelo', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

pipeline_plagas = Pipeline(steps=[
    ('preprocesamiento', transformador),
    ('modelo', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

# Entrenar los modelos
pipeline_cultivo.fit(X_entrenamiento, Y_entrenamiento)
pipeline_plagas.fit(X_entrenamiento_plagas, Y_entrenamiento_plagas)

# Obtener la entrada del usuario y realizar la predicción
def obtener_entrada_usuario():
    st.sidebar.header("Datos del cultivo")
    temperatura = st.sidebar.text_input("Temperatura", value="25")
    humedad = st.sidebar.text_input("Humedad", value="50")
    tipo_suelo = st.sidebar.selectbox("Tipo de suelo", ["Arenoso", "Arcilloso", "Mixto"])
    precipitacion = st.sidebar.text_input("Precipitación", value="50")
    altitud = st.sidebar.text_input("Altitud", value="500")
    tipo_irrigacion = st.sidebar.selectbox("Tipo de irrigación", ["Goteo", "Gravedad", "Aspersión"])
    ph_suelo = st.sidebar.text_input("pH del suelo", value="7.0")
    luz_solar = st.sidebar.text_input("Horas de luz solar", value="12")
    uso_fertilizantes = st.sidebar.selectbox("Uso de fertilizantes", ["Orgánicos", "Químicos"])
    fecha_siembra = st.sidebar.text_input("Fecha de siembra (YYYY-MM-DD)", value=str(datetime.today().date()))

    # Obtener fecha de siembra
    fecha_siembra = pd.to_datetime(fecha_siembra)
    año_siembra = fecha_siembra.year
    mes_siembra = fecha_siembra.month
    dia_siembra = fecha_siembra.day

    return pd.DataFrame({
        "Temperatura": [float(temperatura)],
        "Humedad": [float(humedad)],
        "Tipo_Suelo": [tipo_suelo],
        "Precipitacion": [float(precipitacion)],
        "Altitud": [float(altitud)],
        "Tipo_Irrigacion": [tipo_irrigacion],
        "pH_Suelo": [float(ph_suelo)],
        "Luz_Solar": [float(luz_solar)],
        "Uso_Fertilizantes": [uso_fertilizantes],
        "Año_Siembra": [año_siembra],
        "Mes_Siembra": [mes_siembra],
        "Dia_Siembra": [dia_siembra]
    })

def predecir_tipo_cultivo(entrada):
    prediccion_cultivo = pipeline_cultivo.predict(entrada)
    tipo_producto = prediccion_cultivo[0].lower()  # Convertir a minúsculas para coincidir con los nombres de archivo
    imagen_path = os.path.join("img", f"{tipo_producto}.jpeg")  # Ruta de la imagen correspondiente al producto

    prediccion_plagas = pipeline_plagas.predict_proba(entrada)[0][1]  # Probabilidad de que tenga plagas

    # Verificar si la imagen existe
    if not os.path.isfile(imagen_path):
        st.warning(f"No se encontró la imagen para el producto: {tipo_producto}")
        imagen_path = None

    return tipo_producto, imagen_path, prediccion_plagas

# Interfaz de usuario
entrada_usuario = obtener_entrada_usuario()
tipo_cultivo, imagen_producto, probabilidad_plagas = predecir_tipo_cultivo(entrada_usuario)

# Mostrar el resultado de la predicción
st.markdown("## Resultado de la predicción")
st.markdown(f"### El tipo de cultivo recomendado es: **{tipo_cultivo.capitalize()}**")
if imagen_producto:
    st.image(imagen_producto, caption=f"Imagen del producto: {tipo_cultivo.capitalize()}", width=400)
else:
    st.markdown(f"### No se encontró la imagen para el producto predicho.")

st.markdown("## Probabilidad de plagas")
st.markdown(f"### La probabilidad de que el cultivo tenga plagas es: **{probabilidad_plagas:.2f}**")

st.markdown("## Métricas del modelo")
st.markdown(f"### Precisión del modelo de cultivo: **{pipeline_cultivo.score(X_prueba, Y_prueba):.2f}**")
st.markdown(f"### Precisión del modelo de plagas: **{pipeline_plagas.score(X_prueba_plagas, Y_prueba_plagas):.2f}**")

# Visualización de gráficos
st.markdown("## Comparación de características del cultivo")
fig, ax = plt.subplots()
sns.boxplot(data=datos, x='Tipo_Producto', y='Temperatura', ax=ax)
ax.axhline(y=entrada_usuario['Temperatura'].values[0], color='r', linestyle='--')

# Rotar etiquetas del eje x y ajustar su alineación
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

st.pyplot(fig)


# Definir la ruta donde quieres guardar el archivo
ruta_guardado = "predicciones"

# Descargar los resultados
st.sidebar.header("Descargar resultados")
if st.sidebar.button("Exportar a CSV"):
    entrada_usuario['Tipo_Cultivo_Predicho'] = tipo_cultivo.capitalize()  # Nombre del cultivo predicho
    entrada_usuario['Probabilidad_Plagas'] = probabilidad_plagas
    nombre_archivo = f"resultados_prediccion_{tipo_cultivo.lower()}.csv"  # Nombre del archivo con el cultivo predicho
    
    # Crear el nombre del archivo con la ruta completa
    ruta_completa_archivo = os.path.join(ruta_guardado, nombre_archivo)
    
    # Guardar el archivo en la ruta especificada
    entrada_usuario.to_csv(ruta_completa_archivo, index=False)
    st.sidebar.success(f"Archivo exportado como `{ruta_completa_archivo}`")
