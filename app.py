import streamlit as st
import requests
import io
from PIL import Image, ImageDraw, ImageFont
import gitlab
import os
from urllib.parse import quote
import gdown
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import traceback
import random
import json
from datetime import datetime
from matplotlib.colors import Normalize

def init_gitlab_connection():
    """Inicializa la conexión con GitLab"""
    gl = gitlab.Gitlab('https://gitlab.com')
    return gl
def get_project():
    """Obtiene el proyecto de GitLab"""
    gl = init_gitlab_connection()
    project = gl.projects.get('tulcanjose1/zonacem-ai')
    return project
def get_image_data(file_path):
    """Obtiene los datos de la imagen desde GitLab"""
    try:
        base_url = "https://gitlab.com/tulcanjose1/zonacem-ai/-/raw/main/"
        url = base_url + quote(file_path)
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error al cargar la imagen: {str(e)}")
        return None
def download_and_load_model_from_gitlab(url, model_name):
    """Descarga y carga un modelo desde GitLab con barra de progreso"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
            # Convertir URL del formato web a la ruta de archivo en el repositorio
            # Extraer la ruta del archivo después de "main/"
            file_path = url.split('/blob/main/')[1]
            # Crear contenedores para la barra de progreso y mensajes
            progress_text = st.empty()
            progress_bar = st.progress(0)
            # Mensaje inicial
            progress_text.text(f"Descargando modelo {model_name} desde GitLab...")
            progress_bar.progress(0.3)
            # Obtener proyecto
            project = get_project()
            
            # Usar directamente el método alternativo sin intentar el método que falla
            progress_text.text("Descargando modelo mediante URL directa...")
            
            # Ajustar la ruta para que sea compatible con get_image_data
            # Necesitamos convertir de "blob/main/" a "raw/main/"
            raw_url = url.replace("/-/blob/", "/-/raw/")
            
            # Descargar usando get_image_data con la URL completa
            file_content = requests.get(raw_url).content
            
            with open(temp_file.name, 'wb') as f:
                f.write(file_content)
            
            progress_bar.progress(0.7)
            progress_text.text("Descarga completada. Cargando modelo en memoria...")
            
            # Cargar modelo
            model = tf.keras.models.load_model(temp_file.name)
            
            # Completar barra de progreso
            progress_bar.progress(1.0)
            progress_text.text("¡Modelo cargado con éxito!")
            
            # Esperar un momento para que el usuario vea el mensaje
            import time
            time.sleep(1)
            
            # Limpiar mensajes temporales
            progress_text.empty()
            progress_bar.empty()
            
            os.unlink(temp_file.name)
            return model
            
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        traceback.print_exc()
        return None

def show_model_info(model):  
    """Muestra información reducida sobre el modelo"""
    st.subheader("Información del Modelo")
    # Mostrar solo información básica
    st.write("Forma de entrada:", model.input_shape)
    st.write("Forma de salida:", model.output_shape)
    #st.write(f"Número total de parámetros: {model.count_params():,}")

def evaluate_model(model, dataset_path, num_predictions=5, random_selection=True, selected_indices=None):
    try:
        # Constantes para normalización/desnormalización
        STRUCT_MIN, STRUCT_MAX = 0, 20    # metros
        ANTENNA_MIN, ANTENNA_MAX = 0, 40  # metros
        POWER_MIN = -100   # dBm (Piso de ruido térmico)
        POWER_MAX = 40     # dBm (Potencia máxima teórica)
        ACTUAL_POWER_MAX = -10  # dBm (Máximo real)
        
        # Establecer el número total estimado de imágenes (sabemos que hay aproximadamente 10,000)
        total_images = 10000
        st.info(f"Tamaño del dataset: {total_images} imágenes")
        
        # Seleccionar índices de imágenes a procesar
        if random_selection:
            if num_predictions > total_images:
                num_predictions = total_images
                st.warning(f"Solo hay {total_images} imágenes disponibles")
            # Selección aleatoria de índices
            import random
            indices = random.sample(range(total_images), num_predictions)
            indices.sort()  # Ordenar para mejor seguimiento
            st.write(f"Evaluando escenarios con índices aleatorios: {indices}")
        else:
            # Usar índices proporcionados
            if selected_indices is None or len(selected_indices) == 0:
                st.error("No se proporcionaron índices para evaluación")
                return
            # Validar que los índices estén dentro del rango
            valid_indices = [idx for idx in selected_indices if 0 <= idx < total_images]
            if len(valid_indices) != len(selected_indices):
                st.warning(f"Algunos índices están fuera del rango válido (0-{total_images-1})")
            indices = valid_indices
            if not indices:
                st.error("No hay índices válidos para evaluar")
                return
            st.write(f"Evaluando imágenes con índices: {indices}")
        
        # Obtener las imágenes y procesarlas directamente por índice
        project = get_project()
        
        # Procesar las imágenes seleccionadas
        for i, idx in enumerate(indices):
            try:
                with st.spinner(f"Cargando escenario #{idx+1}..."):
                    # Construir nombres de archivo basados en el índice
                    # Asumiendo que los archivos siguen un patrón como "image_XXXX.png" donde XXXX es el índice con ceros a la izquierda
                    file_format = f"image_{idx:03d}.png"  # Ajustar según el formato real de tus archivos
                    
                    # Construir rutas completas
                    struct_path = f"{dataset_path}/structures/{file_format}"
                    pixel_path = f"{dataset_path}/selected_pixels/{file_format}"
                    ant_path = f"{dataset_path}/antenna_position/{file_format}"
                    power_path = f"{dataset_path}/combined_power/{file_format}"
                    
                    # Cargar las imágenes directamente por ruta
                    struct_img = np.array(Image.open(io.BytesIO(get_image_data(struct_path))).convert('L'), dtype=np.float32)
                    pixel_img = np.array(Image.open(io.BytesIO(get_image_data(pixel_path))).convert('L'), dtype=np.float32)
                    ant_img = np.array(Image.open(io.BytesIO(get_image_data(ant_path))).convert('L'), dtype=np.float32)
                    power_img = np.array(Image.open(io.BytesIO(get_image_data(power_path))).convert('L'), dtype=np.float32)
                
                # El resto del código de procesamiento permanece igual
                # Normalizar imágenes
                struct_norm = struct_img / 255.0
                pixel_norm = (pixel_img / 255.0) * (ACTUAL_POWER_MAX - POWER_MIN) / (POWER_MAX - POWER_MIN)
                ant_norm = ant_img / 255.0
                power_norm = (power_img / 255.0) * (ACTUAL_POWER_MAX - POWER_MIN) / (POWER_MAX - POWER_MIN)
                
                # Preparar entrada para el modelo
                combined_input = np.stack([struct_norm, pixel_norm, ant_norm], axis=-1)
                combined_input = np.expand_dims(combined_input, axis=0)
                # Realizar predicción
                prediction_norm = model.predict(combined_input)[0]
                # Desnormalizar predicción y valores reales
                prediction_denorm = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                power_denorm = power_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                # Mostrar índice real de la imagen y número de predicción
                st.write(f"### Predicción {i+1} (Escenario #{idx} del dataset)")
                # Extraer nombre del archivo para mostrar información adicional
                #filename = os.path.basename(structures[idx])
                filename = os.path.basename(struct_path)
                
                st.write(f"Archivo: {filename}")
                # Convertir tensores a arrays de NumPy
                power_np = power_denorm.numpy() if hasattr(power_denorm, 'numpy') else np.array(power_denorm)
                pred_np = prediction_denorm.numpy() if hasattr(prediction_denorm, 'numpy') else np.array(prediction_denorm)
                # Si hay dimensiones adicionales, tomar solo la primera capa
                if power_np.ndim > 2:
                    power_np = power_np[:, :, 0]
                if pred_np.ndim > 2:
                    pred_np = pred_np[:, :, 0]
                # Crear una máscara para ignorar valores de fondo (cercanos al mínimo)
                mask_np = power_np > (POWER_MIN + 1)
                # Aplicar la máscara a los valores
                power_masked = power_np[mask_np]
                prediction_masked = pred_np[mask_np]
                # Contar píxeles válidos
                valid_pixels = np.sum(mask_np)
                # Calcular métricas solo con píxeles válidos
                if valid_pixels > 0:
                    # Error absoluto medio
                    mae = np.mean(np.abs(prediction_masked - power_masked))
                    # Error cuadrático medio
                    mse = np.mean(np.square(prediction_masked - power_masked))
                    # Raíz del error cuadrático medio
                    rmse = np.sqrt(mse)
                    # Calcular R² (coeficiente de determinación)
                    power_mean = np.mean(power_masked)
                    ss_res = np.sum(np.square(power_masked - prediction_masked))
                    ss_tot = np.sum(np.square(power_masked - power_mean))
                    
                    # Evitar división por cero
                    if ss_tot > 0:
                        r2 = 1 - (ss_res / ss_tot)
                    else:
                        r2 = 0.0
                else:
                    # Si no hay píxeles válidos, establecer valores predeterminados
                    mae = 0.0
                    rmse = 0.0
                    r2 = 0.0
                # Mostrar métricas
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("MAE (dB)", f"{mae:.2f}")
                with metrics_col2:
                    st.metric("RMSE (dB)", f"{rmse:.2f}")
                with metrics_col3:
                    st.metric("R²", f"{r2:.4f}")
                # Visualización de las capas de entrada
                st.write("### Capas de Entrada")
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                # Desnormalizar cada capa
                layer1 = struct_norm * STRUCT_MAX
                layer2 = pixel_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                layer3 = ant_norm * ANTENNA_MAX
                # Configurar visualizaciones
                im1 = ax1.imshow(layer1, extent=[0, 50, 0, 50], cmap='viridis')
                ax1.set_title('Estructura (m)', pad=20)
                plt.colorbar(im1, ax=ax1)
                im2 = ax2.imshow(layer2, extent=[0, 50, 0, 50], cmap='viridis')
                ax2.set_title('Píxeles (dBm)', pad=20)
                plt.colorbar(im2, ax=ax2)
                im3 = ax3.imshow(layer3, extent=[0, 50, 0, 50], cmap='viridis')
                ax3.set_title('Antena (m)', pad=20)
                plt.colorbar(im3, ax=ax3)
                for ax in [ax1, ax2, ax3]:
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_xlabel('Distancia (m)')
                    ax.set_ylabel('Distancia (m)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                # Visualización de resultados
                st.write("### Resultados")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                im1 = ax1.imshow(power_np, extent=[0, 50, 0, 50], cmap='viridis')
                ax1.set_title('Power Real (dBm)', pad=20)
                plt.colorbar(im1, ax=ax1)
                im2 = ax2.imshow(pred_np, extent=[0, 50, 0, 50], cmap='viridis')
                ax2.set_title('Predicción (dBm)', pad=20)
                plt.colorbar(im2, ax=ax2)
                for ax in [ax1, ax2]:
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_xlabel('Distancia (m)')
                    ax.set_ylabel('Distancia (m)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                # Calcular y visualizar diferencias 
                fig, ax = plt.subplots(figsize=(8, 7))
                difference = pred_np - power_np
                im = ax.imshow(difference, extent=[0, 50, 0, 50], cmap='RdBu_r', 
                            vmin=-10, vmax=10)  # Ajusta vmin/vmax según el rango esperado
                ax.set_title('Diferencia (Predicción - Real) (dB)', pad=20)
                plt.colorbar(im, ax=ax)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_xlabel('Distancia (m)')
                ax.set_ylabel('Distancia (m)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                # Histogramas si hay suficientes datos válidos
                if valid_pixels > 10:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(power_masked, bins=50, alpha=0.5, label='Real')
                    ax.hist(prediction_masked, bins=50, alpha=0.5, label='Predicción')
                    ax.set_xlabel('Potencia (dBm)')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Distribución de valores reales vs predichos')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    # Diagrama de dispersión (valores reales vs predichos)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(power_masked, prediction_masked, alpha=0.3)
                    min_val = min(np.min(power_masked), np.min(prediction_masked))
                    max_val = max(np.max(power_masked), np.max(prediction_masked))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')  # Línea diagonal de referencia
                    ax.set_xlabel('Valores reales (dBm)')
                    ax.set_ylabel('Valores predichos (dBm)')
                    ax.set_title('Valores reales vs predichos')
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                st.markdown("---")  # Separador entre predicciones

            except Exception as e:
                st.error(f"Error al procesar el escenario #{idx+1}: {str(e)}")
                st.warning("Intentando con el siguiente escenario...")
                continue
                
    except Exception as e:
        st.error(f"Error durante la evaluación: {str(e)}")
        st.error(f"Stacktrace: {str(traceback.format_exc())}")

def main():
    st.set_page_config(page_title="ZonaCEM AI", page_icon="📡", layout="wide")
    st.title("ZonaCEM AI")
    st.markdown("Esta aplicación utiliza modelos de IA para predecir la potencia recibida en entornos de estaciones base celulares.")

    # Constants
    DATASET_URLS = {
        "Modelo 1.95GHz": {
            "base": "datasets/dataset_1.95_30muestras_exclusion",
            "folders": ["structures", "selected_pixels", "antenna_position", "combined_power"],
            "folder_names": ["Estructuras", "Puntos de Medición", "Posición de Antena", "Mapa de Potencia"]
        },
        "Modelo 2.13GHz": {
            "base": "datasets/dataset_2.13_30muestras_exclusión",
            "folders": ["structures", "selected_pixels", "antenna_position", "combined_power"],
            "folder_names": ["Estructuras", "Puntos de Medición", "Posición de Antena", "Mapa de Potencia"]
        },
        "Modelo 2.65GHz": {
            "base": "datasets/dataset_2.65_30muestras_exclusion",
            "folders": ["structures", "selected_pixels", "antenna_position", "combined_power"],
            "folder_names": ["Estructuras", "Puntos de Medición", "Posición de Antena", "Mapa de Potencia"]
        }
    }
    
    MODELOS = {
        "Modelo 1.95GHz": "https://gitlab.com/tulcanjose1/zonacem-ai/-/blob/main/modelos/Modelo%201.95_Optimizado.keras",
        "Modelo 2.13GHz": "https://gitlab.com/tulcanjose1/zonacem-ai/-/blob/main/modelos/Modelo%202.13_Optimizado.keras",
        "Modelo 2.65GHz": "https://gitlab.com/tulcanjose1/zonacem-ai/-/blob/main/modelos/Modelo%202.65_Optimizado.keras"
    }
    
    # Initialize session state
    for key in ['modelo_actual', 'nombre_modelo_actual']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Sidebar
    # Sidebar
    with st.sidebar:
        st.header("OPCIONES")
        vista_seleccionada = st.radio(
            "Seleccione una opción:",
            ["Manual de Usuario", "Datasets de Imágenes", "Modelos y Evaluación", "Evaluar nuevos escenarios"]
        )
        
        st.markdown("---")
        st.markdown("""
        ### Sobre las opciones 
        
        Lo que puede hacer en esta interfaz: 
        1. **Manual de Usuario**: Guía de uso de la aplicación.
        2. **Datasets**: Muestra de los datasets de cada frecuencia.
        3. **Modelos y Evaluación**: Evaluar cada modelo con cada dataset y ver las métricas de evaluación.
        4. **Evaluar nuevos escenarios**: Evaluar cada modelo con escenarios nuevos sin etiqueta.
        
        ### Sobre el Modelo
        
        El modelo utiliza tres capas de entrada:
        1. **Estructura**: Representación de paredes y obstáculos
        2. **Píxeles Seleccionados**: Puntos de medición conocidos
        3. **Posición de Antena**: Ubicación y altura de la antena
        
        Y predice:
        - **Mapa de Potencia**: Distribución de potencia de señal en dBm
        """)

    # Nueva sección: Manual de Usuario
    if vista_seleccionada == "Manual de Usuario":
        st.header("Manual de Usuario")
        st.markdown("""
        Bienvenido a **ZonaCEM AI**. Esta sección le servirá de guía para utilizar la aplicación.

        **Pasos y Descripción:**
        
        1. **Datasets de Imágenes:**  
           Aquí podrá visualizar los conjuntos de datos que contienen imágenes asociadas a cada modelo. Seleccione el modelo y la carpeta de interés para ver las imágenes disponibles.

        2. **Modelos y Evaluación:**  
           En esta sección puede cargar un modelo preentrenado, evaluarlo utilizando el dataset correspondiente y ver las métricas de evaluación.  
           - **Cargar Modelo:** Permite cargar el modelo en memoria.
           - **Descargar Modelo:** Opción para descargar el modelo.
           - **Evaluar Modelo:** Ejecute la evaluación con los datos del dataset seleccionado.
        
        3. **Evaluar nuevos escenarios:**  
           Esta opción permite aplicar los modelos en escenarios nuevos sin etiquetas para predecir la distribución de potencia.

        **Recomendaciones:**
        
        - Antes de evaluar un modelo, asegúrese de cargarlo.
        - Si decide evaluar un modelo con un dataset diferente, tome en cuenta que los resultados podrían no ser óptimos.
        - Consulte la documentación adjunta o contacte al soporte para más detalles.
        """)
    
    # Sección: Datasets de Imágenes
    elif vista_seleccionada == "Datasets de Imágenes":
        modelo_seleccionado = st.selectbox(
            "Selecciona un modelo", list(DATASET_URLS.keys())
        )
        
        # Crear un diccionario que mapee los nombres en español a los nombres originales
        folder_mapping = {DATASET_URLS[modelo_seleccionado]["folder_names"][i]: DATASET_URLS[modelo_seleccionado]["folders"][i] 
                        for i in range(len(DATASET_URLS[modelo_seleccionado]["folders"]))}
        
        # Mostrar los nombres en español en el selectbox
        carpeta_nombre = st.selectbox(
            "Selecciona una carpeta", DATASET_URLS[modelo_seleccionado]["folder_names"]
        )
        
        # Obtener el nombre original de la carpeta para usar en la ruta
        carpeta_seleccionada = folder_mapping[carpeta_nombre]
        
        ruta_base = DATASET_URLS[modelo_seleccionado]["base"]
        ruta_completa = f"{ruta_base}/{carpeta_seleccionada}"
        
        # Download dataset button
        #download_dataset(ruta_base)
        
        try:
            project = get_project()
            items = project.repository_tree(path=ruta_completa, ref='main')
            imagenes = [item['path'] for item in items if 
                    item['type'] == 'blob' and 
                    item['path'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            if imagenes:
                # Seleccionar 20 imágenes aleatorias si hay más de 20
                import random
                if len(imagenes) > 20:
                    imagenes = random.sample(imagenes, 20)
                
                cols = st.columns(3)
                for idx, imagen_path in enumerate(imagenes):
                    with cols[idx % 3]:
                        image_data = get_image_data(imagen_path)
                        if image_data:
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption=os.path.basename(imagen_path))
            else:
                st.warning("No se encontraron imágenes en esta carpeta")
        except Exception as e:
            st.error(f"Error al acceder al repositorio: {str(e)}")

    # Modelos y Evaluación view
    elif vista_seleccionada == "Modelos y Evaluación":
        modelo_seleccionado = st.selectbox(
            "Selecciona un modelo", list(MODELOS.keys())
        )
        
        # Display currently loaded model
        if st.session_state['modelo_actual'] and st.session_state['nombre_modelo_actual']:
            st.info(f"📊 Modelo actualmente cargado en memoria: **{st.session_state['nombre_modelo_actual']}**")
        else:
            st.warning("⚠️ No hay ningún modelo cargado actualmente. Por favor, carga un modelo antes de evaluarlo.")
        
        # Model button (solo queda un botón, así que no necesitamos columnas)
        if st.button("Cargar Modelo"):
            # Clear previous model if exists
            if st.session_state['modelo_actual']:
                st.session_state['modelo_actual'] = None
                import gc
                gc.collect()  # Help free memory
                
            # Load new model with progress bar
            modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado)
            if modelo:
                show_model_info(modelo)
                st.success(f"✅ Modelo {modelo_seleccionado} cargado exitosamente")
                st.session_state['modelo_actual'] = modelo
                st.session_state['nombre_modelo_actual'] = modelo_seleccionado
        
        # Evaluation section
        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.subheader("Evaluación del Modelo")
            st.write("Presiona el botón para evaluar el modelo con los datos seleccionados.")
        
        with eval_col2:
            is_model_loaded = st.session_state['modelo_actual'] is not None
            evaluar_clicked = st.button("Evaluar Modelo", disabled=not is_model_loaded)
        
        # Run evaluation
        if evaluar_clicked and is_model_loaded:
            dataset_path = DATASET_URLS[modelo_seleccionado]["base"]
            
            # Warning if dataset doesn't match model
            if modelo_seleccionado != st.session_state['nombre_modelo_actual']:
                st.warning(f"⚠️ Estás evaluando el modelo **{st.session_state['nombre_modelo_actual']}** con el dataset de **{modelo_seleccionado}**. Los resultados pueden no ser óptimos.")
            
            evaluate_model(st.session_state['modelo_actual'], dataset_path)

    else:
            def calculate_intensity(value, min_val, max_val):
                """Calcula la intensidad del color basado en el valor"""
                return int(((value - min_val) / (max_val - min_val)) * 255)

            def draw_grid_with_labels(draw, size=256):
                """Dibuja una cuadrícula con etiquetas específicas en metros"""
                pixels_per_meter = 256/50
                meter_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                
                for meters in meter_values: 
                    pixels = int(meters * pixels_per_meter)
                    draw.text((pixels, size + 5), str(meters), fill=128)  # Etiquetas eje X
                    draw.text((-20, pixels), str(meters), fill=128)       # Etiquetas eje Y
                
                # Etiquetas de unidades
                draw.text((size + 5, size + 5), 'm', fill=128)  # Unidad eje X
                draw.text((-20, -10), 'm', fill=128)            # Unidad eje Y

            def save_image_and_data(img, prefix, data):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if not os.path.exists("resultados"):
                    os.makedirs("resultados")
                    
                # Crear imagen limpia sin cuadrícula ni etiquetas
                clean_img = Image.new('L', (256, 256), 0)
                clean_draw = ImageDraw.Draw(clean_img)
                
                # Dibujar elementos según el tipo de datos
                if prefix == "estructuras":
                    for struct in data:
                        intensity = calculate_intensity(struct['altura'], 3, 20)
                        if struct['tipo'] == 'linea':
                            clean_draw.line(struct['coords'], fill=intensity, width=2)
                        elif struct['tipo'] == 'rectangulo':
                            x0, y0 = struct['coords'][0]
                            x1, y1 = struct['coords'][1]
                            rect_coords = [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))]
                            clean_draw.rectangle(rect_coords, outline=intensity, width=2)
                elif prefix == "posicion_antena" and data:
                    x, y, h = data
                    intensity = calculate_intensity(h, 10, 40)
                    clean_draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=intensity)

                elif prefix == "mediciones":
                    for point in data:
                        x, y, dbm = point
                        intensity = calculate_intensity(dbm, -100, 40)
                        #clean_draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=intensity)
                        clean_draw.point((x-2, y-2), fill=intensity)
                        #clean_draw.rectangle([(x, y), (x+1, y+1)], fill=intensity)

                        
                # Guardar imagen y datos
                img_path = f"resultados/{prefix}_{timestamp}.png"
                data_path = f"resultados/{prefix}_{timestamp}.json"
                clean_img.save(img_path)
                
                with open(data_path, 'w') as f:
                    json.dump(data, f)
                return img_path, data_path

            def delete_saved_file(img_path):
                """Borra la imagen y su archivo JSON asociado"""
                try:
                    os.remove(img_path)
                    json_path = img_path.replace('.png', '.json')
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    return True
                except Exception as e:
                    st.error(f"Error al borrar el archivo: {str(e)}")
                    return False

            def show_saved_images():
                if not os.path.exists("resultados"):
                    return
                    
                st.sidebar.header("Imágenes Guardadas")
                image_files = [f for f in os.listdir("resultados") if f.endswith(".png")]
                image_files.sort(key=lambda x: os.path.getmtime(os.path.join("resultados", x)), reverse=True)
                
                for filename in image_files:
                    img_path = os.path.join("resultados", filename)
                    with st.sidebar.container():
                        st.sidebar.image(img_path, caption=filename, width=256)
                        col1, col2 = st.sidebar.columns(2)
                        
                        with col1:
                            with open(img_path, "rb") as file:
                                st.download_button(
                                    label="Descargar",
                                    data=file,
                                    file_name=filename,
                                    mime="image/png",
                                    key=f"download_{filename}"
                                )
                        
                        with col2:
                            if st.button("Borrar", key=f"delete_{filename}"):
                                if delete_saved_file(img_path):
                                    st.success(f"Imagen {filename} borrada correctamente")
                                    st.rerun()
                        
                        st.sidebar.markdown("---")

            def process_uploaded_image(uploaded_file, image_type):
                """Procesa una imagen cargada y la convierte al formato correcto"""
                if uploaded_file is not None:
                    image_bytes = uploaded_file.getvalue()
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convertir a escala de grises y redimensionar si es necesario
                    image = image.convert('L')
                    if image.size != (256, 256):
                        image = image.resize((256, 256))
                        
                    # Guardar temporalmente
                    temp_path = f"temp_{image_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    image.save(temp_path)
                    
                    # Mostrar la imagen cargada
                    st.image(image, caption=f"Imagen cargada: {image_type}")
                    
                    return image, temp_path
                return None, None

            def display_image_with_matplotlib(pil_image, title, data_type=None):
                """Muestra una imagen con matplotlib con valores reales"""
                img_array = np.array(pil_image)
                
                # Configurar vmin, vmax y etiqueta según tipo de datos
                config = {
                    "estructuras": (3, 20, "Altura (m)"),
                    "posicion_antena": (10, 40, "Altura (m)"),
                    "mediciones": (-100, 40, "Señal (dBm)")
                }
                
                vmin, vmax, cbar_label = config.get(data_type, (0, 255, "Intensidad"))
                
                # Crear figura y mostrar imagen
                fig, ax = plt.subplots(figsize=(8, 6))
                norm = Normalize(0, 255)
                im = ax.imshow(img_array, cmap='viridis', extent=[0, 50, 0, 50], norm=norm, origin='upper')
                
                # Configurar ejes y título
                ax.set_xlabel('Distancia (m)')
                ax.set_ylabel('Distancia (m)')
                ax.set_title(title, pad=20)
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Añadir colorbar con valores reales
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(cbar_label)
                
                # Configurar ticks de colorbar para mostrar valores reales
                tick_positions = np.linspace(0, 255, 6)
                tick_labels = np.linspace(vmin, vmax, 6)
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels([f"{label:.1f}" for label in tick_labels])
                
                plt.tight_layout()
                return fig

            def create_common_editor_components(data_type, edit_function):
                """Componentes comunes para todos los editores"""
                st.subheader(f"Cargar imagen de {data_type}")
                uploaded_file = st.file_uploader(f"Seleccionar imagen de {data_type}", 
                                            type=["png", "jpg", "jpeg"], 
                                            key=f"upload_{data_type}")
                
                if uploaded_file:
                    img, temp_path = process_uploaded_image(uploaded_file, data_type)
                    if img:
                        setattr(st.session_state, f"uploaded_{data_type}", img)
                        setattr(st.session_state, f"uploaded_{data_type}_path", temp_path)

            def create_building_interface():
                st.subheader("Editor de estructuras")
                if 'structures' not in st.session_state:
                    st.session_state.structures = []
                if 'drawing_tool' not in st.session_state:
                    st.session_state.drawing_tool = "linea"
                
                # Crear y mostrar imagen
                img = Image.new('L', (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw_grid_with_labels(draw)
                
                # Dibujar estructuras existentes
                pixels_per_meter = 256/50
                for struct in st.session_state.structures:
                    intensity = calculate_intensity(struct['altura'], 3, 20)
                    if struct['tipo'] == 'linea':
                        draw.line(struct['coords'], fill=intensity, width=2)
                    elif struct['tipo'] == 'rectangulo':
                        x0, y0 = struct['coords'][0]
                        x1, y1 = struct['coords'][1]
                        rect_coords = [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))]
                        draw.rectangle(rect_coords, outline=intensity, width=2)
                
                # Mostrar imagen
                fig = display_image_with_matplotlib(img, "Estructuras", data_type="estructuras")
                st.pyplot(fig)
                plt.close(fig)
                
                # Cargar imagen
                create_common_editor_components("estructuras", None)
                
                # Controles de dibujo
                st.session_state.drawing_tool = st.radio(
                    "Herramienta",
                    options=["linea", "rectangulo"],
                    format_func=lambda x: "Línea" if x == "linea" else "Rectángulo"
                )
                
                # Coordenadas
                col1, col2 = st.columns(2)
                with col1:
                    start_x_m = st.number_input("X inicial (m)", 0, 50, 25)
                    start_y_m = st.number_input("Y inicial (m)", 0, 50, 25)
                    start_x = int(start_x_m * pixels_per_meter)
                    start_y = int((50 - start_y_m) * pixels_per_meter)
                with col2:
                    end_x_m = st.number_input("X final (m)", 0, 50, 35)
                    end_y_m = st.number_input("Y final (m)", 0, 50, 35)
                    end_x = int(end_x_m * pixels_per_meter)
                    end_y = int((50 - end_y_m) * pixels_per_meter)
                
                altura = st.slider("Altura (metros)", 3, 20, 10)
                
                if st.button("Añadir Estructura", key="add_structure"):
                    new_structure = {
                        'tipo': st.session_state.drawing_tool,
                        'coords': [(start_x, start_y), (end_x, end_y)],
                        'altura': altura
                    }
                    st.session_state.structures.append(new_structure)
                    st.rerun()
                
                # Botones de acción
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Borrar Todo", key="clear_structures"):
                        st.session_state.structures = []
                        st.rerun()
                with col2:
                    if st.button("Deshacer", key="undo_structure"):
                        if st.session_state.structures:
                            st.session_state.structures.pop()
                            st.rerun()
                with col3:
                    if st.button("Guardar", key="save_structures"):
                        save_image_and_data(img, "estructuras", st.session_state.structures)

            def create_point_interface():
                st.subheader("Editor de posición de estación base")
                if 'reference_point' not in st.session_state:
                    st.session_state.reference_point = None
                
                # Crear y mostrar imagen
                img = Image.new('L', (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw_grid_with_labels(draw)
                
                # Dibujar punto existente
                pixels_per_meter = 256/50
                if st.session_state.reference_point:
                    x, y, h = st.session_state.reference_point
                    intensity = calculate_intensity(h, 10, 40)
                    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=intensity)
                
                # Mostrar imagen
                fig = display_image_with_matplotlib(img, "Posición de estación base", data_type="trasmisor")
                st.pyplot(fig)
                plt.close(fig)
                
                # Cargar imagen
                create_common_editor_components("trasmisor", None)
                
                # Controles de posición
                col1, col2 = st.columns(2)
                with col1:
                    point_x_m = st.number_input("Posición X (m)", 0, 50, 25)
                    point_x = int(point_x_m * pixels_per_meter)
                with col2:
                    point_y_m = st.number_input("Posición Y (m)", 0, 50, 25)
                    point_y = int((50 - point_y_m) * pixels_per_meter)
                
                altura = st.slider("Altura (metros)", 10, 40, 25)
                
                if st.button("Colocar Punto", key="add_point"):
                    st.session_state.reference_point = (point_x, point_y, altura)
                    st.rerun()
                
                # Botones de acción
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Borrar", key="clear_point"):
                        st.session_state.reference_point = None
                        st.rerun()
                with col2:
                    if st.button("Guardar Punto", key="save_point"):
                        save_image_and_data(img, "trasmisor", st.session_state.reference_point)

            def create_pixel_selector():
                st.subheader("Selector de puntos de medición")
                if 'measurement_points' not in st.session_state:
                    st.session_state.measurement_points = []
                
                # Crear y mostrar imagen
                img = Image.new('L', (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw_grid_with_labels(draw)
                
                # Dibujar puntos existentes
                for point in st.session_state.measurement_points:
                    x, y, dbm = point
                    intensity = calculate_intensity(dbm, -100, 40)
                    #draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=intensity)
                    draw.point((x, y), fill=intensity)
                    #draw.point([(x, y), (x+1, y+1)], fill=intensity)

                # Mostrar imagen
                fig = display_image_with_matplotlib(img, "Mediciones", data_type="mediciones")
                st.pyplot(fig)
                plt.close(fig)
                
                # Cargar imagen
                create_common_editor_components("mediciones", None)
                
                # Controles de posición y valor
                pixels_per_meter = 256/50
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_m = st.number_input("X (m)", 0, 50, 25, key="measurement_x")
                    x = int(x_m * pixels_per_meter)
                with col2:
                    y_m = st.number_input("Y (m)", 0, 50, 25, key="measurement_y")
                    y = int((50 - y_m) * pixels_per_meter)
                with col3:
                    dbm = st.number_input("Valor (dBm)", -100, 40, -50)
                
                if st.button("Añadir Medición", key="add_measurement"):
                    st.session_state.measurement_points.append((x, y, dbm))
                    st.rerun()
                
                # Botones de acción
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Borrar Todo", key="clear_measurements"):
                        st.session_state.measurement_points = []
                        st.rerun()
                with col2:
                    if st.button("Deshacer", key="undo_measurement"):
                        if st.session_state.measurement_points:
                            st.session_state.measurement_points.pop()
                            st.rerun()
                with col3:
                    if st.button("Guardar Mediciones", key="save_measurements"):
                        save_image_and_data(img, "mediciones", st.session_state.measurement_points)

            def add_model_evaluation_section():
                """Sección para evaluar escenarios con modelos y mostrar resultados detallados"""
                st.header("Evaluar con el modelo")
                
                # Constantes para normalización/desnormalización
                STRUCT_MIN, STRUCT_MAX = 0, 20        # metros (altura de estructuras)
                ANTENNA_MIN, ANTENNA_MAX = 10, 40     # metros (altura de antena)
                POWER_MIN = -100                       # dBm (Piso de ruido térmico)
                POWER_MAX = 40                         # dBm (Potencia máxima teórica)
                ACTUAL_POWER_MAX = -10                 # dBm (Máximo real esperado)
                
                # Seleccionar modelo
                modelo_seleccionado = st.selectbox(
                    "Selecciona un modelo para la evaluación",
                    list(MODELOS.keys()),
                    key="modelo_selector"
                )
                
                # Verificar si las imágenes necesarias están cargadas
                required_images = ['estructuras', 'trasmisor', 'mediciones']
                images_loaded = all(hasattr(st.session_state, f'uploaded_{img}') for img in required_images)
                
                # Funciones auxiliares para la segmentación
                def create_segmentation_mask(power_map, antenna_pos):
                    """
                    Crea una máscara de segmentación basada en niveles de potencia
                    Retorna la máscara y los radios calculados para las zonas
                    """
                    height, width = power_map.shape
                    mask = np.zeros((height, width, 3))  # RGB mask
                    
                    # Crear matrices de coordenadas
                    y, x = np.ogrid[:height, :width]
                    
                    # Encontrar puntos máximos para cada zona
                    high_power_mask = power_map > 10
                    mid_power_mask = (power_map <= 10) & (power_map > 0)
                    
                    # Encontrar el radio para la zona roja (>10 dBm)
                    if np.any(high_power_mask):
                        high_power_points = np.where(high_power_mask)
                        distances = np.sqrt((high_power_points[0] - antenna_pos[0])**2 + 
                                        (high_power_points[1] - antenna_pos[1])**2)
                        red_radius = np.max(distances)
                    else:
                        red_radius = 0
                        
                    # Encontrar el radio para la zona amarilla (0-10 dBm)
                    if np.any(mid_power_mask):
                        mid_power_points = np.where(mid_power_mask)
                        distances = np.sqrt((mid_power_points[0] - antenna_pos[0])**2 + 
                                        (mid_power_points[1] - antenna_pos[1])**2)
                        yellow_radius = np.max(distances)
                    else:
                        yellow_radius = red_radius
                    
                    # Crear las zonas
                    distances = np.sqrt((y - antenna_pos[0])**2 + (x - antenna_pos[1])**2)
                    
                    # Zona roja (>10 dBm)
                    red_zone = distances <= red_radius
                    mask[red_zone] = [1, 0, 0]  # Rojo
                    
                    # Zona amarilla (0-10 dBm)
                    yellow_zone = (distances > red_radius) & (distances <= yellow_radius)
                    mask[yellow_zone] = [1, 1, 0]  # Amarillo
                    
                    # Zona verde (<0 dBm)
                    green_zone = distances > yellow_radius
                    mask[green_zone] = [0, 1, 0]  # Verde
                    
                    return mask, red_radius, yellow_radius
                
                def find_antenna_position(ant_img):
                    """
                    Detecta la posición de la antena en la imagen de antenas.
                    
                    Args:
                        ant_img: array numpy con la imagen de la antena
                    
                    Returns:
                        tuple: (y, x) coordenadas de la posición de la antena
                    """
                    # Encontrar el valor máximo en la imagen de la antena
                    max_value = np.max(ant_img)
                    
                    # Encontrar las coordenadas del pixel que marca la posición de la antena
                    antenna_pos = np.where(ant_img == max_value)
                    
                    if len(antenna_pos[0]) == 0:
                        raise ValueError("No se encontró la posición de la antena")
                    
                    # Tomar la primera posición encontrada
                    return antenna_pos[0][0], antenna_pos[1][0]
                
                # Botón para ejecutar evaluación
                if st.button("Ejecutar evaluación", disabled=not images_loaded, key="execute_evaluation_button"):
                    if not images_loaded:
                        st.warning("Debes cargar las tres imágenes antes de evaluar.")
                        return
                    
                    with st.spinner("Cargando modelo y evaluando..."):
                        try:
                            # Cargar modelo
                            modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado)
                            
                            if modelo:
                                # Preparar y normalizar las entradas
                                struct_img = np.array(st.session_state.uploaded_estructuras) / 255.0
                                pixel_img = np.array(st.session_state.uploaded_mediciones) / 255.0
                                ant_img = np.array(st.session_state.uploaded_trasmisor) / 255.0
                                
                                # Combinar en un tensor de entrada
                                combined_input = np.stack([struct_img, pixel_img, ant_img], axis=-1)
                                combined_input = np.expand_dims(combined_input, axis=0)
                                
                                # Ejecutar predicción
                                st.info("Ejecutando predicción...")
                                prediction_norm = modelo.predict(combined_input)[0]
                                
                                # Desnormalizar predicción
                                prediction_denorm = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                                
                                # Convertir a arrays de NumPy si son tensores
                                pred_np = prediction_denorm.numpy() if hasattr(prediction_denorm, 'numpy') else np.array(prediction_denorm)
                                
                                # Si hay dimensiones adicionales, tomar solo la primera capa
                                if pred_np.ndim > 2:
                                    pred_np = pred_np[:, :, 0]
                                
                                # Crear imagen de resultado
                                img_resultado = Image.fromarray((np.clip((pred_np - POWER_MIN) / (ACTUAL_POWER_MAX - POWER_MIN), 0, 1) * 255).astype(np.uint8))
                                
                                st.success("¡Predicción completada!")
                                
                                # Mostrar métricas básicas de la predicción
                                st.subheader("Métricas de la predicción")
                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric("Potencia máxima", f"{np.max(pred_np):.2f} dBm")
                                with metrics_col2:
                                    st.metric("Potencia mínima", f"{np.min(pred_np):.2f} dBm")
                                
                                # Visualización de las capas de entrada
                                st.write("### Capas de Entrada")
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                
                                # Desnormalizar cada capa para visualización
                                layer1 = struct_img * STRUCT_MAX
                                layer2 = pixel_img * (POWER_MAX - POWER_MIN) + POWER_MIN
                                layer3 = ant_img * ANTENNA_MAX
                                
                                # Configurar visualizaciones
                                im1 = ax1.imshow(layer1, extent=[0, 50, 0, 50], cmap='viridis')
                                ax1.set_title('Estructura', pad=20)
                                plt.colorbar(im1, ax=ax1)
                                
                                im2 = ax2.imshow(layer2, extent=[0, 50, 0, 50], cmap='viridis')
                                ax2.set_title('Mediciones', pad=20)
                                plt.colorbar(im2, ax=ax2)
                                
                                im3 = ax3.imshow(layer3, extent=[0, 50, 0, 50], cmap='viridis')
                                ax3.set_title('Posición estación base', pad=20)
                                plt.colorbar(im3, ax=ax3)
                                
                                for ax in [ax1, ax2, ax3]:
                                    ax.grid(True, linestyle='--', alpha=0.3)
                                    ax.set_xlabel('Distancia (m)')
                                    ax.set_ylabel('Distancia (m)')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Mostrar resultado
                                st.subheader("Resultado de la predicción")
                                fig, ax = plt.subplots(figsize=(8, 7))
                                max_pred = np.max(pred_np)
                                adjusted_vmax = np.ceil(max_pred / 5) * 5  # Redondea hacia arriba en múltiplos de 5
                                im = ax.imshow(pred_np, extent=[0, 50, 0, 50], cmap='viridis',
                                            vmin=POWER_MIN, vmax=adjusted_vmax)

                                ax.set_title('Mapa de Potencia Predicho', pad=20)
                                plt.colorbar(im, ax=ax)
                                ax.grid(True, linestyle='--', alpha=0.3)
                                ax.set_xlabel('Distancia (m)')
                                ax.set_ylabel('Distancia (m)')
                                plt.tight_layout()
                                
                                # Guardar la figura para descargarla
                                fig_result = fig
                                
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Histograma de valores predichos
                                st.subheader("Distribución de potencia predicha")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Crear máscara para ignorar valores de fondo (cercanos al mínimo)
                                mask_np = pred_np > (POWER_MIN + 1)
                                prediction_masked = pred_np[mask_np]
                                
                                if len(prediction_masked) > 10:  # Solo si hay suficientes datos válidos
                                    ax.hist(prediction_masked, bins=50, alpha=0.7, color='blue', label='Predicción')
                                    ax.set_xlabel('Potencia (dBm)')
                                    ax.set_ylabel('Frecuencia')
                                    ax.set_title('Distribución de valores de potencia predichos')
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.info("No hay suficientes datos válidos para generar un histograma")
                                plt.close(fig)
                                
                                # ------------ SECCIÓN MEJORADA: SEGMENTACIÓN DE EXPOSICIÓN A CEM --------------
                                st.subheader("Segmentación de exposición a CEM")

                                try:
                                    # Obtener la posición de la antena
                                    antenna_pos = find_antenna_position(ant_img)
                                    
                                    # Crear máscara de segmentación
                                    seg_mask, red_radius, yellow_radius = create_segmentation_mask(pred_np, antenna_pos)
                                    
                                    # Convertir las coordenadas para visualización
                                    height, width = pred_np.shape
                                    antenna_plot_pos = (antenna_pos[1] * 50/width, antenna_pos[0] * 50/height)
                                    
                                    # Radio en metros
                                    red_radius_meters = red_radius * 50/width
                                    yellow_radius_meters = yellow_radius * 50/width
                                    
                                    # Crear figura para la segmentación
                                    fig, ax = plt.subplots(figsize=(8, 7))
                                    ax.imshow(seg_mask, extent=[0, 50, 0, 50])
                                    
                                    # Marcar la posición de la antena
                                    ax.plot(antenna_plot_pos[0], antenna_plot_pos[1], 'b+', markersize=10, label='Antena')
                                    
                                    # Dibujar círculos para los radios
                                    red_circle = plt.Circle(antenna_plot_pos, red_radius_meters, 
                                                        fill=False, color='white', linestyle='-')
                                    yellow_circle = plt.Circle(antenna_plot_pos, yellow_radius_meters, 
                                                            fill=False, color='white', linestyle='-')
                                    ax.add_artist(red_circle)
                                    ax.add_artist(yellow_circle)
                                    
                                    # Configurar aspecto de la gráfica
                                    ax.set_title('Segmentación de exposición a CEM', pad=20)
                                    ax.grid(True, linestyle='--', alpha=0.3)
                                    ax.set_xlabel('Distancia (m)')
                                    ax.set_ylabel('Distancia (m)')
                                    
                                    # Añadir leyenda de colores mejorada
                                    from matplotlib.patches import Patch
                                    legend_elements = [
                                        Patch(facecolor='red', label='Zona de rebasamiento (>10 dBm)'),
                                        Patch(facecolor='yellow', label='Zona ocupacional (0-10 dBm)'),
                                        Patch(facecolor='green', label='Zona poblacional / conformidad (<0 dBm)')
                                    ]
                                    ax.legend(handles=legend_elements, loc='upper right')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Mostrar información de los radios y métricas adicionales de CEM
                                    st.write("### Métricas de exposición a CEM")
                                    
                                    # Calcular áreas de exposición
                                    area_rebasamiento = np.pi * red_radius_meters**2
                                    area_ocupacional = np.pi * (yellow_radius_meters**2 - red_radius_meters**2)
                                    
                                    # Calcular porcentajes de la zona total
                                    total_area = 50 * 50  # área total en m²
                                    porcentaje_rebasamiento = (area_rebasamiento / total_area) * 100
                                    porcentaje_ocupacional = (area_ocupacional / total_area) * 100
                                    
                                    # Métricas en columnas
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Radio zona de rebasamiento", f"{red_radius_meters:.2f} m")
                                        st.metric("Área zona de rebasamiento", f"{area_rebasamiento:.2f} m²")
                                        st.metric("% del área total (rebasamiento)", f"{porcentaje_rebasamiento:.2f}%")
                                        
                                    with col2:
                                        st.metric("Radio zona ocupacional", f"{yellow_radius_meters:.2f} m")
                                        st.metric("Área zona ocupacional", f"{area_ocupacional:.2f} m²")
                                        st.metric("% del área total (ocupacional)", f"{porcentaje_ocupacional:.2f}%")
                                    
                                    # Recomendaciones de seguridad basadas en los resultados
                                    st.write("### Recomendaciones de seguridad")
                                    
                                    if red_radius_meters > 5:
                                        st.warning(f"""
                                        ⚠️ **Alerta**: La zona de rebasamiento tiene un radio de {red_radius_meters:.2f} m, superior a 5 metros. 
                                        Se recomienda establecer barreras físicas y señalización clara para evitar el acceso a esta área.
                                        """)
                                    else:
                                        st.info(f"""
                                        ℹ️ La zona de rebasamiento tiene un radio de {red_radius_meters:.2f} m, con un área controlable.
                                        Se recomienda establecer señalización adecuada alrededor de la antena.
                                        """)
                                    
                                    if yellow_radius_meters > 15:
                                        st.warning(f"""
                                        ⚠️ **Precaución**: La zona ocupacional se extiende hasta {yellow_radius_meters:.2f} m.
                                        Personal técnico debe limitar su tiempo de exposición en esta área y usar equipo de protección adecuado.
                                        """)
                                    
                                    # Análisis de conformidad normativa
                                    cumple_normas = True
                                    if porcentaje_rebasamiento > 5:
                                        cumple_normas = False
                                        st.error("❌ La zona de rebasamiento excede el 5% del área total, posible incumplimiento normativo.")
                                    
                                    if cumple_normas:
                                        st.success("✅ Los niveles de exposición a CEM cumplen con los límites normativos recomendados.")
                                    
                                    # Guardar la figura para descargar
                                    buffer_seg = io.BytesIO()
                                    fig.savefig(buffer_seg, format='png', bbox_inches='tight', dpi=300)
                                    buffer_seg.seek(0)
                                    
                                    # Generar nombre de archivo con timestamp para segmentación
                                    timestamp_seg = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename_seg = f"segmentacion_exposicion_cem_{timestamp_seg}.png"
                                    
                                    # Botón de descarga para segmentación
                                    st.download_button(
                                        label="Descargar imagen de segmentación",
                                        data=buffer_seg,
                                        file_name=filename_seg,
                                        mime="image/png",
                                        key="download_segmentation_button"
                                    )
                                    
                                    # Generar informe de exposición CEM como texto para descargar
                                    informe_text = f"""
                                    INFORME DE EXPOSICIÓN A CAMPOS ELECTROMAGNÉTICOS (CEM)
                                    Fecha y hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                                    
                                    MÉTRICAS PRINCIPALES:
                                    - Radio zona de rebasamiento: {red_radius_meters:.2f} m
                                    - Radio zona ocupacional: {yellow_radius_meters:.2f} m
                                    - Área zona de rebasamiento: {area_rebasamiento:.2f} m²
                                    - Área zona ocupacional: {area_ocupacional:.2f} m²
                                    - Porcentaje del área total (rebasamiento): {porcentaje_rebasamiento:.2f}%
                                    - Porcentaje del área total (ocupacional): {porcentaje_ocupacional:.2f}%
                                    
                                    ANÁLISIS DE CONFORMIDAD:
                                    - Zona de rebasamiento: {'EXCEDE LÍMITES RECOMENDADOS' if porcentaje_rebasamiento > 5 else 'DENTRO DE LÍMITES ACEPTABLES'}
                                    - Zona ocupacional: {'AMPLIA' if yellow_radius_meters > 15 else 'ACEPTABLE'}
                                    
                                    RECOMENDACIONES:
                                    1. {
                                        "Establecer barreras físicas y señalización clara para evitar el acceso a la zona de rebasamiento." 
                                        if red_radius_meters > 5 
                                        else "Establecer señalización adecuada alrededor de la antena."
                                    }
                                    2. Personal técnico debe limitar su tiempo de exposición en la zona ocupacional.
                                    3. Realizar mediciones periódicas para verificar que los niveles se mantienen estables.
                                    4. Utilizar equipo de protección personal adecuado al trabajar en la zona ocupacional.
                                    
                                    NIVELES DE POTENCIA:
                                    - Valor máximo: {np.max(pred_np):.2f} dBm
                                    - Valor mínimo: {np.min(pred_np):.2f} dBm
                                    
                                    Este informe es generado automáticamente y debe ser validado por un ingeniero especializado en telecomunicaciones.
                                    """
                                    
                                    # Botón para descargar el informe de texto
                                    st.download_button(
                                        label="Descargar informe de exposición CEM",
                                        data=informe_text,
                                        file_name=f"informe_exposicion_cem_{timestamp_seg}.txt",
                                        mime="text/plain",
                                        key="download_report_button"
                                    )
                                    
                                    plt.close(fig)
                                    
                                except Exception as e:
                                    st.error(f"Error al generar la segmentación: {str(e)}")
                                    st.exception(e)
                                # ----------------- FIN SECCIÓN MEJORADA ------------------
                                
                                # Preparar imagen para descargar
                                # Convertir la figura del resultado a una imagen para descargar
                                buffer = io.BytesIO()
                                fig_result.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
                                buffer.seek(0)
                                
                                # Generar nombre de archivo con timestamp
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"mapa_potencia_{timestamp}.png"
                                
                                # Botón de descarga
                                st.download_button(
                                    label="Descargar imagen",
                                    data=buffer,
                                    file_name=filename,
                                    mime="image/png",
                                    key="download_button"
                                )
                            else:
                                st.error("No se pudo cargar el modelo seleccionado")
                        except Exception as e:
                            st.error(f"Error durante la evaluación: {str(e)}")
                            st.exception(e)

            def main():
                st.title("Editor de Imágenes para Análisis de Cobertura")
                tab1, tab2, tab3, tab4 = st.tabs(["Estructuras", "Posición estación base", "Mediciones", "Evaluación"])
                with tab1:
                    create_building_interface()
                with tab2:
                    create_point_interface()
                with tab3:
                    create_pixel_selector()
                with tab4:
                    add_model_evaluation_section()
                
                # Mostrar imágenes guardadas en la barra lateral
                show_saved_images()

            if __name__ == "__main__":
                main()  
                
if __name__ == "__main__":
# Importar módulos adicionales necesarios
           import time
           main()
