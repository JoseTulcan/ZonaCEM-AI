
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
import zipfile
from io import BytesIO
import time
import math
from scipy import stats
from PIL import Image

def init_gitlab_connection(): 
    """Inicializa la conexión con GitLab"""
    gl = gitlab.Gitlab('https://gitlab.com')
    return gl
def get_project():
    """Obtiene el proyecto de GitLab"""
    gl = init_gitlab_connection()
    #project = gl.projects.get('tulcanjose1/zonacem-ai')
    project = gl.projects.get('tulcanjose0/zonacem-ai')
    return project
def get_image_data(file_path):
    """Obtiene los datos de la imagen desde GitLab"""
    try:
        #base_url = "https://gitlab.com/tulcanjose1/zonacem-ai/-/raw/main/"
        base_url = "https://gitlab.com/tulcanjose0/zonacem-ai/-/raw/main/" 
        url = base_url + quote(file_path)
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error al cargar la imagen: {str(e)}")
        return None
    
# FUNCIONES AUXILIARES PARA CALIBRACIÓN ADAPTATIVA - PARTE 1
def extract_measurement_points_from_image(pixels_img, num_expected_points=30, 
                                        img_size=(256, 256), area_size=(50, 50)):
    """Extrae puntos de medición de la imagen de píxeles con rango adaptativo"""
    # Convertir PIL Image a numpy array
    pixels_array = np.array(pixels_img) / 255.0
    
    # Convertir a 2D si es necesario
    if len(pixels_array.shape) == 3:
        pixels_array = pixels_array[:, :, 0]
        
    # Filtrar solo píxeles con valores significativos (> 0.1)
    significant_pixels = pixels_array[pixels_array > 0.1]
    
    if len(significant_pixels) > 0:
        # Usar percentiles para establecer rango más robusto
        min_normalized = np.percentile(significant_pixels, 5)   # Percentil 5
        max_normalized = np.percentile(significant_pixels, 95)  # Percentil 95
        
        # Establecer rango de potencias basado en los valores de la imagen
        POWER_MIN_ADAPTIVE = -60.0 + (min_normalized * 20.0)  # Rango: -60 a -40 dBm
        POWER_MAX_ADAPTIVE = -40.0 + (max_normalized * 20.0)  # Rango: -40 a -20 dBm
        
        # Asegurar que el rango sea lógico
        if POWER_MAX_ADAPTIVE <= POWER_MIN_ADAPTIVE:
            POWER_MAX_ADAPTIVE = POWER_MIN_ADAPTIVE + 15.0
            
    else:
        # Valores por defecto si no hay píxeles significativos
        POWER_MIN_ADAPTIVE = -50.0
        POWER_MAX_ADAPTIVE = -25.0
        min_normalized = pixels_array.min()
        max_normalized = pixels_array.max()
    
    # ALGORITMO PARA EXTRAER EXACTAMENTE 30 PUNTOS
    extracted_points = {}
    extracted_power_values = {}
    
    threshold_percentiles = [99.5, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    
    for percentile in threshold_percentiles:
        threshold = np.percentile(pixels_array, percentile)
        y_coords, x_coords = np.where(pixels_array > threshold)
        if len(x_coords) >= num_expected_points:
            if len(x_coords) > num_expected_points:
                brightness_values = pixels_array[y_coords, x_coords]
                sorted_indices = np.argsort(brightness_values)[::-1][:num_expected_points]
                x_coords = x_coords[sorted_indices]
                y_coords = y_coords[sorted_indices]
            
            # Crear diccionario con los puntos exactos encontrados
            for i, (x_px, y_px) in enumerate(zip(x_coords, y_coords)):
                # Convertir píxeles a metros
                x_m, y_m = pixels_to_meters(x_px, y_px, img_size, area_size)
                
                # Obtener valor de potencia normalizado
                power_normalized = pixels_array[y_px, x_px]
                
                # CONVERSIÓN ADAPTATIVA: usar el rango calculado dinámicamente
                power_real = power_normalized * (POWER_MAX_ADAPTIVE - POWER_MIN_ADAPTIVE) + POWER_MIN_ADAPTIVE
                
                point_id = f"P{i+1:02d}"
                extracted_points[point_id] = (x_m, y_m)
                extracted_power_values[point_id] = power_real
            
            break
    
    return extracted_points, extracted_power_values, {
        'POWER_MIN_ADAPTIVE': POWER_MIN_ADAPTIVE,
        'POWER_MAX_ADAPTIVE': POWER_MAX_ADAPTIVE,
        'min_normalized': min_normalized,
        'max_normalized': max_normalized
    }

def calculate_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_antenna_position(antenna_img, threshold=0.1):  # Umbral más bajo
    """Encuentra la posición de la antena en la imagen - VERSIÓN MEJORADA"""
    
    # Asegurar que sea 2D
    if len(antenna_img.shape) == 3:
        if antenna_img.shape[2] == 1:
            antenna_img = antenna_img[:, :, 0]
        else:
            antenna_img = antenna_img.mean(axis=2)
    
    # Normalizar si es necesario
    if antenna_img.max() > 1.0:
        antenna_img = antenna_img / 255.0
    
    print(f"DEBUG: Rango de valores en antena: {antenna_img.min():.3f} - {antenna_img.max():.3f}")
    
    # Buscar píxeles con valor > threshold
    y_indices, x_indices = np.where(antenna_img > threshold)
    
    print(f"DEBUG: Píxeles encontrados con umbral {threshold}: {len(x_indices)}")
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        antenna_x = np.mean(x_indices)
        antenna_y = np.mean(y_indices)
        
        print(f"DEBUG: Antena detectada en píxeles: ({antenna_x:.1f}, {antenna_y:.1f})")
        
        # Convertir a metros para verificación
        center_x_m = antenna_x * 50.0 / 256.0
        center_y_m = (256 - antenna_y) * 50.0 / 256.0
        print(f"DEBUG: Antena en metros: ({center_x_m:.1f}, {center_y_m:.1f})")
        
        return (antenna_x, antenna_y)
    else:
        print(f"DEBUG: No se encontró antena, usando centro por defecto")
        return (antenna_img.shape[1] // 2, antenna_img.shape[0] // 2)

def meters_to_pixels(coords, img_size=(256, 256), area_size=(50, 50)):
    """Convierte coordenadas en metros a píxeles"""
    scale_x = img_size[0] / area_size[0]
    scale_y = img_size[1] / area_size[1]
    
    center_x, center_y = img_size[0] // 2, img_size[1] // 2
    
    px = int(center_x + coords[0] * scale_x)
    py = int(center_y - coords[1] * scale_y)
    
    return (px, py)

def pixels_to_meters(px, py, img_size=(256, 256), area_size=(50, 50)):
    """Convierte coordenadas en píxeles a metros"""
    scale_x = img_size[0] / area_size[0]
    scale_y = img_size[1] / area_size[1]
    
    center_x, center_y = img_size[0] // 2, img_size[1] // 2
    
    x = (px - center_x) / scale_x
    y = (center_y - py) / scale_y
    
    return (x, y)

# FUNCIONES AUXILIARES PARA CALIBRACIÓN ADAPTATIVA - PARTE 2
def apply_log_distance_model(x, y, antenna_pos, P0, n):
    """Aplica el modelo log-distancia: P(d) = P0 - 10*n*log10(d)"""
    d = calculate_distance((x, y), antenna_pos)
    if d < 0.1:  # Evitar log(0)
        d = 0.1
    
    return P0 - 10 * n * math.log10(d)

def create_log_distance_map(img_size=(256, 256), area_size=(50, 50), 
                            antenna_pos=None, P0=-30, n=2):
    """Crea un mapa de potencia basado en el modelo log-distancia"""
    if antenna_pos is None:
        antenna_pos = (0, 0)
        
    power_map = np.zeros(img_size)
    
    for py in range(img_size[0]):
        for px in range(img_size[1]):
            pos_m = pixels_to_meters(px, py, img_size, area_size)
            power_map[py, px] = apply_log_distance_model(pos_m[0], pos_m[1], antenna_pos, P0, n)
    
    return power_map

def calculate_calibration_parameters_log_distance(model, input_image, measurement_points, 
                                                real_power_values, antenna_pos_m):
    """Calcula parámetros de calibración usando modelo log-distancia"""
    POWER_MIN, POWER_MAX = -100, 40
    
    # 1. Obtener predicción inicial del modelo
    initial_prediction_norm = model.predict(np.expand_dims(input_image, axis=0))[0]
    if initial_prediction_norm.shape[-1] > 1:
        initial_prediction_norm = initial_prediction_norm[:, :, 0]
    else:
        initial_prediction_norm = initial_prediction_norm.squeeze()
    
    initial_prediction = initial_prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
    
    # 2. Calcular distancias y realizar regresión log-distancia con mediciones reales
    distances = []
    real_powers = []
    pred_powers = []
    
    for point_id, (x, y) in measurement_points.items():
        # Distancia desde la antena
        distance = calculate_distance((x, y), antenna_pos_m)
        distances.append(distance)
        
        # Potencia real medida
        real_powers.append(real_power_values[point_id])
        
        # Potencia predicha por el modelo en ese punto
        px, py = meters_to_pixels((x, y))
        if 0 <= px < initial_prediction.shape[1] and 0 <= py < initial_prediction.shape[0]:
            pred_powers.append(initial_prediction[py, px])
    
    # 3. Regresión log-distancia con mediciones reales
    log_distances = np.log10(distances)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_distances, real_powers)
    
    # Parámetros del modelo log-distancia
    n_path_loss = -slope / 10  # Exponente de pérdida de trayecto
    P0_reference = intercept    # Potencia de referencia a 1 metro

    # 4. Calibrar predicción del modelo usando el modelo log-distancia
    log_distance_map = create_log_distance_map(
        antenna_pos=antenna_pos_m, P0=P0_reference, n=n_path_loss
    )
    
    # 5. Calcular transformación lineal: predicción_modelo -> log_distancia
    pred_values_all = []
    log_values_all = []
    
    for point_id, (x, y) in measurement_points.items():
        px, py = meters_to_pixels((x, y))
        if 0 <= px < initial_prediction.shape[1] and 0 <= py < initial_prediction.shape[0]:
            pred_values_all.append(initial_prediction[py, px])
            log_values_all.append(log_distance_map[py, px])
    
    # Regresión lineal para calibración
    calib_slope, calib_intercept, _, _, _ = stats.linregress(pred_values_all, log_values_all)
    
    # 6. Calcular rango objetivo basado en mediciones reales
    target_min = min(real_power_values.values())
    target_max = max(real_power_values.values())
    
    # Aplicar transformación completa para calcular ajuste de rango
    calibrated_full = calib_slope * initial_prediction + calib_intercept
    current_min = np.min(calibrated_full)
    current_max = np.max(calibrated_full)
    
    scale = (target_max - target_min) / (current_max - current_min)
    offset = target_min - scale * current_min
    
    return {
        'slope': calib_slope,
        'intercept': calib_intercept,
        'scale': scale,
        'offset': offset,
        'POWER_MIN': POWER_MIN,
        'POWER_MAX': POWER_MAX,
        'n_path_loss': n_path_loss,
        'P0_reference': P0_reference,
        'r_squared': r_value**2,
        'target_min': target_min,
        'target_max': target_max
    }

def calculate_metrics(real_values, predicted_values):
    """Calcula métricas MAE y RMSE"""
    real_array = np.array(real_values)
    pred_array = np.array(predicted_values)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(real_array - pred_array))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((real_array - pred_array)**2))
    
    return {
        'MAE': mae,
        'RMSE': rmse
    }

def evaluate_calibrated_model_adaptive(model, input_image, points_dict, power_dict, POWER_MIN=-100, POWER_MAX=40):
    """Evalúa el modelo calibrado usando los puntos extraídos"""
    prediction_norm = model.predict(np.expand_dims(input_image, axis=0))[0]
    if prediction_norm.shape[-1] > 1:
        prediction_norm = prediction_norm[:, :, 0]
    else:
        prediction_norm = prediction_norm.squeeze()
    
    prediction = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
    
    errors = []
    results = []
    real_values = []
    predicted_values = []
    
    for point_id, (x, y) in points_dict.items():
        px, py = meters_to_pixels((x, y))
        if 0 <= px < prediction.shape[1] and 0 <= py < prediction.shape[0]:
            real_power = power_dict[point_id]
            predicted_power = prediction[py, px]
            error = abs(predicted_power - real_power)
            
            errors.append(error)
            real_values.append(real_power)
            predicted_values.append(predicted_power)
            results.append({
                'point': point_id,
                'real': real_power,
                'predicted': predicted_power,
                'error': error
            })
    
    # Calcular métricas MAE y RMSE
    metrics = calculate_metrics(real_values, predicted_values)
    
    return {
        'avg_error': np.mean(errors),
        'max_error': np.max(errors),
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'results': results,
        'prediction': prediction
    }

# FUNCIONES AUXILIARES PARA CALIBRACIÓN ADAPTATIVA - PARTE 3
def create_calibrated_model_single_file(base_model, calibration_params, input_sample):
    """Crea un modelo calibrado que aplica la transformación log-distancia"""
    
    slope = float(calibration_params['slope'])
    intercept = float(calibration_params['intercept'])
    scale = float(calibration_params['scale'])
    offset = float(calibration_params['offset'])
    
    POWER_MIN = float(calibration_params['POWER_MIN'])
    POWER_MAX = float(calibration_params['POWER_MAX'])
    
    inputs = tf.keras.Input(shape=base_model.input_shape[1:])
    base_output = base_model(inputs)
    
    if isinstance(base_output, list):
        base_output = base_output[0]
    
    # Aplicar transformación completa basada en log-distancia
    denormalized = base_output * (POWER_MAX - POWER_MIN) + POWER_MIN
    calibrated = denormalized * slope + intercept
    adjusted = calibrated * scale + offset
    normalized_output = (adjusted - POWER_MIN) / (POWER_MAX - POWER_MIN)
    
    calibrated_model = tf.keras.Model(inputs=inputs, outputs=normalized_output)
    
    return calibrated_model

def extract_measurement_points_from_streamlit_image(pixels_array, num_expected_points=30, 
                                                  img_size=(256, 256), area_size=(50, 50)):
    """
    Extrae puntos de medición de la imagen de píxeles (adaptado para Streamlit)
    """
    # Si viene con dimensión de canal, tomar solo el primer canal
    if len(pixels_array.shape) == 3:
        pixels_array = pixels_array[:, :, 0]
    
    # Convertir a escala 0-255 si está normalizada
    if pixels_array.max() <= 1.0:
        pixels_array = pixels_array * 255.0
    
    # DETECTAR PÍXELES CON VALOR > 0
    non_zero_mask = pixels_array > 0
    y_coords, x_coords = np.where(non_zero_mask)
    
    if len(x_coords) == 0:
        return {}, {}
    
    extracted_points = {}
    extracted_power_values = {}
    
    for i, (x_px, y_px) in enumerate(zip(x_coords, y_coords)):
        # Convertir píxeles a metros
        x_m, y_m = pixels_to_meters(x_px, y_px, img_size, area_size)
        
        # Obtener valor del píxel
        pixel_value = pixels_array[y_px, x_px]
        
        # CONVERSIÓN: 0 = -100 dBm, 255 = 40 dBm
        power_real = (pixel_value / 255.0) * 140.0 - 100.0
        
        point_id = f"P{i+1:02d}"
        extracted_points[point_id] = (x_m, y_m)
        extracted_power_values[point_id] = power_real
    
    return extracted_points, extracted_power_values

def find_antenna_position_corrected(antenna_img, threshold=0.5):
    """Encuentra la posición de la antena - ORDEN CORRECTO (x, y)"""
    if len(antenna_img.shape) == 3:
        antenna_img = antenna_img[:, :, 0]
    
    if antenna_img.max() > 1.0:
        antenna_img = antenna_img / 255.0
    
    y_indices, x_indices = np.where(antenna_img > threshold)
    if len(x_indices) > 0 and len(y_indices) > 0:
        antenna_x = np.mean(x_indices)  # Promedio de coordenadas X (columnas)
        antenna_y = np.mean(y_indices)  # Promedio de coordenadas Y (filas)
        return (antenna_x, antenna_y)  # Retornar como (x, y)
    else:
        return (antenna_img.shape[1] // 2, antenna_img.shape[0] // 2)  # (x, y) = (ancho/2, alto/2)
    
def perform_adaptive_calibration(modelo, struct_img, pixel_img, ant_img):
    """
    Realiza calibración adaptativa usando la lógica del código de referencia
    """
    try:
        # Convertir imágenes PIL a arrays numpy si es necesario
        if hasattr(struct_img, 'convert'):
            struct_array = np.array(struct_img.convert('L')) / 255.0
        else:
            struct_array = np.array(struct_img) / 255.0
            
        if hasattr(pixel_img, 'convert'):
            pixels_array = np.array(pixel_img.convert('L')) / 255.0
        else:
            pixels_array = np.array(pixel_img) / 255.0
            
        if hasattr(ant_img, 'convert'):
            antenna_array = np.array(ant_img.convert('L')) / 255.0
        else:
            antenna_array = np.array(ant_img) / 255.0
        
        # Asegurar que tengan la forma correcta
        if len(struct_array.shape) == 3:
            struct_array = struct_array[:, :, 0] if struct_array.shape[2] == 1 else struct_array.mean(axis=2)
        if len(pixels_array.shape) == 3:
            pixels_array = pixels_array[:, :, 0] if pixels_array.shape[2] == 1 else pixels_array.mean(axis=2)
        if len(antenna_array.shape) == 3:
            antenna_array = antenna_array[:, :, 0] if antenna_array.shape[2] == 1 else antenna_array.mean(axis=2)
        
        # Redimensionar si es necesario
        target_size = (256, 256)
        if struct_array.shape != target_size:
            struct_array = np.array(Image.fromarray(struct_array).resize(target_size)) / 255.0
        if pixels_array.shape != target_size:
            pixels_array = np.array(Image.fromarray(pixels_array).resize(target_size)) / 255.0
        if antenna_array.shape != target_size:
            antenna_array = np.array(Image.fromarray(antenna_array).resize(target_size)) / 255.0
        
        # Añadir dimensión de canal si es necesario
        if len(struct_array.shape) == 2:
            struct_array = np.expand_dims(struct_array, axis=-1)
        if len(pixels_array.shape) == 2:
            pixels_array = np.expand_dims(pixels_array, axis=-1)
        if len(antenna_array.shape) == 2:
            antenna_array = np.expand_dims(antenna_array, axis=-1)
        
        # Crear imagen de entrada concatenada (como en el código de referencia)
        input_image = np.concatenate([struct_array, pixels_array, antenna_array], axis=-1)
        
        # EXTRAER PUNTOS DE MEDICIÓN (lógica del primer código)
        extracted_points, extracted_power_values = extract_measurement_points_from_streamlit_image(pixels_array)
        
        if not extracted_points:
            return {'success': False, 'error': 'No se pudieron extraer puntos de medición'}
        
        # ENCONTRAR POSICIÓN DE LA ANTENA 
        antenna_pos_px = find_antenna_position_corrected(antenna_array)
        antenna_pos_m = pixels_to_meters(antenna_pos_px[0], antenna_pos_px[1])
        
        # CALIBRACIÓN CON LOG-DISTANCIA (usando la función del primer código)
        calibration_params = calculate_calibration_parameters_log_distance(
            modelo, input_image, extracted_points, extracted_power_values, antenna_pos_m
        )
        
        # CREAR MODELO CALIBRADO
        calibrated_model = create_calibrated_model_single_file(
            modelo, calibration_params, input_image
        )
        
        calibrated_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # EVALUACIÓN FINAL
        evaluation = evaluate_calibrated_model_adaptive(
            calibrated_model, input_image, extracted_points, extracted_power_values
        )
        
        return {
            'success': True,
            'calibrated_model': calibrated_model,
            'evaluation': evaluation,
            'calibration_params': calibration_params,
            'input_image': input_image,
            'extracted_points': extracted_points,
            'extracted_power_values': extracted_power_values,
            'antenna_pos_m': antenna_pos_m
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
    
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
                    #file_format = f"image_{idx:03d}.png"  # Ajustar según el formato real de tus archivos
                    file_format = f"image_{idx+1:03d}.PNG" 

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

def get_all_files_from_folder(proyecto, ruta_carpeta, max_retries=3):
    """Obtiene TODOS los archivos de una carpeta, manejando paginación"""
    all_items = []
    page = 1
    per_page = 100  # GitLab permite hasta 100 items por página
    
    while True:
        try:
            # Intentar obtener archivos con paginación
            items = proyecto.repository_tree(
                path=ruta_carpeta, 
                ref='main', 
                recursive=True, 
                all=True,
                per_page=per_page,
                page=page
            )
            
            if not items:
                break
                
            all_items.extend(items)
            
            # Si obtuvimos menos items que per_page, hemos llegado al final
            if len(items) < per_page:
                break
                
            page += 1
            time.sleep(0.1)  # Pequeña pausa para no sobrecargar la API
            
        except Exception as e:
            st.warning(f"Error obteniendo página {page}: {str(e)}")
            if max_retries > 0:
                time.sleep(1)
                return get_all_files_from_folder(proyecto, ruta_carpeta, max_retries - 1)
            else:
                break
    
    return all_items

def get_preview_images(proyecto, ruta_carpeta, num_images=50):
    """Obtiene una muestra de imágenes para vista previa rápida"""
    try:
        all_items = []
        page = 1
        per_page = 100  # Máximo por página
        
        # Obtener suficientes elementos para garantizar 50 imágenes
        while len([item for item in all_items if 
                  item.get('type') == 'blob' and 
                  item.get('path', '').lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]) < num_images:
            
            items = proyecto.repository_tree(
                path=ruta_carpeta, 
                ref='main', 
                recursive=True,
                per_page=per_page,
                page=page
            )
            
            if not items:
                break
                
            all_items.extend(items)
            
            # Si obtuvimos menos items que per_page, hemos llegado al final
            if len(items) < per_page:
                break
                
            page += 1
            
            # Límite de seguridad para evitar bucles infinitos
            if page > 10:  # Máximo 1000 elementos
                break
        
        # Filtrar solo imágenes
        imagenes = [item['path'] for item in all_items if
                   item.get('type') == 'blob' and
                   item.get('path', '').lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
        
        return imagenes
        
    except Exception as e:
        st.error(f"Error obteniendo vista previa: {str(e)}")
        return []

def generate_gitlab_download_url(base_url, path, filename_prefix=""):
    """
    Genera URL de descarga directa de GitLab para una carpeta específica
    """
    # Extraer información de la URL base
    # Ejemplo: https://gitlab.com/tulcanjose1/zonacem-ai
    if "gitlab.com" in base_url:
        # Construir URL de descarga directa
        project_path = base_url.replace("https://gitlab.com/", "")
        download_url = f"https://gitlab.com/{project_path}/-/archive/main/{project_path.split('/')[-1]}-main.zip?path={path}"
        return download_url
    return None

def create_direct_download_link(ruta_carpeta, carpeta_nombre, modelo_nombre):
    """Crea enlace de descarga directa desde GitLab"""
    try:
        # URL base del proyecto GitLab
        #base_url = "https://gitlab.com/tulcanjose1/zonacem-ai"
        base_url = "https://gitlab.com/tulcanjose0/zonacem-ai"
            
        # Generar URL de descarga directa
        download_url = generate_gitlab_download_url(base_url, ruta_carpeta, f"{modelo_nombre}_{carpeta_nombre}")
        
        if download_url:
            return download_url
        else:
            return None
            
    except Exception as e:
        st.error(f"Error generando enlace de descarga: {str(e)}")
        return None

def create_direct_dataset_download_link(ruta_base, modelo_nombre):
    """Crea enlace de descarga directa para todo el dataset"""
    try:
        # URL base del proyecto GitLab
        #base_url = "https://gitlab.com/tulcanjose1/zonacem-ai"
        base_url = "https://gitlab.com/tulcanjose0/zonacem-ai"

        # Generar URL de descarga directa para toda la carpeta base
        download_url = generate_gitlab_download_url(base_url, ruta_base, f"{modelo_nombre}_dataset_completo")
        
        if download_url:
            return download_url
        else:
            return None
            
    except Exception as e:
        st.error(f"Error generando enlace de dataset: {str(e)}")
        return None

def download_from_gitlab(url, filename):
    """
    Descarga archivo directamente desde GitLab y lo devuelve como bytes
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error descargando: Status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error en descarga: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="ZonaCEM AI", page_icon="📡", layout="wide")
    st.title("ZonaCEM AI")

    # Constants
    DATASET_URLS = {
        "Modelo 1.95GHz": {
            "base": "datasets/1.95_GHz_dataset", 
            "folders": ["structures", "selected_pixels", "antenna_position", "combined_power"],
            "folder_names": ["Estructuras", "Puntos de medición", "Posición de antena", "Mapa de potencia"]
        },
        "Modelo 2.13GHz": {
            "base": "datasets/2.13_GHz_dataset",
            "folders": ["structures", "selected_pixels", "antenna_position", "combined_power"],
            "folder_names": ["Estructuras", "Puntos de medición", "Posición de antena", "Mapa de potencia"]
        },
        "Modelo 2.65GHz": {
            "base": "datasets/2.65_GHz_dataset",
            "folders": ["structures", "selected_pixels", "antenna_position", "combined_power"],
            "folder_names": ["Estructuras", "Puntos de medición", "Posición de antena", "Mapa de potencia"]
        }
    } 
    
    MODELOS = {
        "Modelo 1.95GHz": "https://gitlab.com/tulcanjose0/zonacem-ai/-/blob/main/modelos/Modelo_1.95_GHz.keras", 
        "Modelo 2.13GHz": "https://gitlab.com/tulcanjose0/zonacem-ai/-/blob/main/modelos/Modelo_2.13_GHz.keras",
        "Modelo 2.65GHz": "https://gitlab.com/tulcanjose0/zonacem-ai/-/blob/main/modelos/Modelo_2.65_GHz.keras"
    }
     
    # Initialize session state
    for key in ['modelo_actual', 'nombre_modelo_actual']:
        if key not in st.session_state:
            st.session_state[key] = None

    # AGREGAR ESTA LÍNEA con las demás inicializaciones
    if 'show_citation' not in st.session_state:
        st.session_state.show_citation = False

    def create_segmentation_mask(power_map, antenna_pos):
        """
        Crea máscara de segmentación por zonas de potencia con umbrales específicos:
        - Zona ROJA: Potencia > 10 dBm (alta exposición)
        - Zona AMARILLA: Potencia entre 0 y 10 dBm (exposición media)
        - Zona VERDE: Potencia < 0 dBm (baja exposición)
        
        Los radios se calculan basándose en el punto más lejano que cumple cada condición
        """
        height, width = power_map.shape
        mask = np.zeros((height, width, 3))  # máscara RGB
        
        # antenna_pos viene como (x, y) en píxeles
        antenna_x, antenna_y = antenna_pos  # x es columna, y es fila
        
        print(f"Antena ubicada en: ({antenna_x:.1f}, {antenna_y:.1f}) píxeles")
        
        # Crear matrices de coordenadas
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Definir umbrales de potencia específicos
        HIGH_POWER_THRESHOLD = 19.0  # dBm - Zona roja - rebasamiento
        MID_POWER_THRESHOLD = 12.0    # dBm - Zona amarilla - ocupacional
        
        # Encontrar puntos que superan cada umbral
        high_power_mask = power_map > HIGH_POWER_THRESHOLD
        mid_power_mask = power_map > MID_POWER_THRESHOLD
        
        print(f"Puntos con potencia > {HIGH_POWER_THRESHOLD} dBm: {np.sum(high_power_mask)}")
        print(f"Puntos con potencia > {MID_POWER_THRESHOLD} dBm: {np.sum(mid_power_mask)}")
        
        # Calcular el radio para la zona roja (potencia > 10 dBm)
        if np.any(high_power_mask):
            high_power_points = np.where(high_power_mask)
            # Calcular distancias desde la antena a todos los puntos con alta potencia
            distances_high = np.sqrt((high_power_points[1] - antenna_x)**2 + 
                                    (high_power_points[0] - antenna_y)**2)
            red_radius = np.max(distances_high) if len(distances_high) > 0 else 0
            print(f"Radio zona ROJA (>10 dBm): {red_radius:.1f} píxeles")
            
            # Mostrar información adicional
            max_power_in_red = np.max(power_map[high_power_mask])
            print(f"Potencia máxima en zona roja: {max_power_in_red:.2f} dBm")
        else:
            red_radius = 0
            print("No se encontraron puntos con potencia > 10 dBm")
        
        # Calcular el radio para la zona amarilla (potencia > 0 dBm)
        if np.any(mid_power_mask):
            mid_power_points = np.where(mid_power_mask)
            # Calcular distancias desde la antena a todos los puntos con potencia media
            distances_mid = np.sqrt((mid_power_points[1] - antenna_x)**2 + 
                                (mid_power_points[0] - antenna_y)**2)
            yellow_radius = np.max(distances_mid) if len(distances_mid) > 0 else red_radius
            print(f"Radio zona AMARILLA (>0 dBm): {yellow_radius:.1f} píxeles")
            
            # Mostrar información adicional
            power_in_yellow_zone = power_map[(mid_power_mask) & (~high_power_mask)]
            if len(power_in_yellow_zone) > 0:
                avg_power_yellow = np.mean(power_in_yellow_zone)
                print(f"Potencia promedio en zona amarilla: {avg_power_yellow:.2f} dBm")
        else:
            yellow_radius = red_radius
            print("No se encontraron puntos con potencia > 0 dBm")
        
        # Asegurar que el radio amarillo sea al menos igual al rojo
        if yellow_radius < red_radius:
            yellow_radius = red_radius
            print(f"Radio amarillo ajustado a: {yellow_radius:.1f} píxeles")
        
        # Crear las zonas circulares con centro en la antena
        distances_from_antenna = np.sqrt((x_coords - antenna_x)**2 + (y_coords - antenna_y)**2)
        
        # Zona roja (alta potencia > 10 dBm)
        if red_radius > 0:
            red_zone = distances_from_antenna <= red_radius
            mask[red_zone] = [1, 0, 0]  # Rojo
            red_area = np.sum(red_zone)
            print(f"Área zona roja: {red_area} píxeles")
        
        # Zona amarilla (potencia media: 0-10 dBm)
        if yellow_radius > red_radius:
            yellow_zone = (distances_from_antenna > red_radius) & (distances_from_antenna <= yellow_radius)
            mask[yellow_zone] = [1, 1, 0]  # Amarillo
            yellow_area = np.sum(yellow_zone)
            print(f"Área zona amarilla: {yellow_area} píxeles")
        
        # Zona verde (baja potencia < 0 dBm)
        green_zone = distances_from_antenna > yellow_radius
        mask[green_zone] = [0, 1, 0]  # Verde
        green_area = np.sum(green_zone)
        print(f"Área zona verde: {green_area} píxeles")
        
        return mask, red_radius, yellow_radius                                                                                                                                                       
    
    # Función helper para crear headers estilizados
    def crear_header_seccion(titulo, subtitulo, icono="📱"):
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">
                {icono} {titulo}
            </h1>
            <h3 style="color: #e8f4fd; text-align: center; margin-top: 10px; font-weight: 300;">
                {subtitulo}
            </h3>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("SECCIONES")
        vista_seleccionada = st.radio(
            "Seleccione una opción:",
            ["Manual de usuario", "Datasets de imágenes", "Modelos y evaluación", "Evaluar nuevos escenarios", "Artículo y publicaciones"]
        )

        
    if vista_seleccionada == "Manual de usuario":
        crear_header_seccion(
            titulo="Manual de Usuario",
            subtitulo="ZonaCEM AI - Estimación de Zonas de Exposición a CEM",
            icono="📱"
        ) 
         
        # Introducción general  
        st.markdown("""
        ### 🎯 Bienvenido a ZonaCEM AI
        
        **ZonaCEM AI** es una aplicación de inteligencia artificial diseñada para predecir la distribución de potencia 
        de estaciones base celulares en diferentes frecuencias, facilitando la planificación, vigilancia y control 
        de redes móviles. 

        La predicción mapas de potencia recibida en estaciones base celulares es una tarea importante en la planificación de redes móviles, incluyendo su posterior vigilancia y control.
        
        En esta aplicación, se utilizan modelos de inteligencia artificial para predecir la distribución de potencia de estaciones base celulares que funcionan en diferentes frecuencias.
        Se generaron tres datasets de imágenes que simulan estaciones base celulares para las frecuencias de 1.95GHz, 2.13GHz y 2.65GHz, cada uno con las siguientes capas: 
        
        - **Estructuras**: Representación de edificaciones y obstáculos(paredes). 
        - **Puntos de medición**: Ubicación de las mediciones dispersas de potencia realizadas.
        - **Posición de antena**: Ubicación de la estación base.  
        - **Mapa de potencia**: Distribución de potencia recibida.
        
        Las tres primeras capas corresponden a las entradas de los modelos, por lo que esta aplicación predice la potencia recibida a partir de las estructuras, medidas dispersas y posición de la estación base y su altura.
        
        La altura de las estructuras y la antena se mide en metros mientras que el mapa de potencia recibida se mide en dBm al igual que las mediciones dispersas.
        Estas magnitudes se representan con diferentes intensidades de color en las imágenes en escala de grises, siendo el color blanco el valor más alto.
                    
        """)
        
        # Guía de secciones de la aplicación
        st.subheader("🗺️ Guía de Secciones")
         
        # Sección 1: Datasets
        with st.expander("1️⃣ **Datasets de Imágenes** - Explorar datos de entrenamiento", expanded=False):
            col_ds1, col_ds2 = st.columns([2, 1])
            
            with col_ds1:
                st.markdown("""
                **🎯 Propósito:** Visualizar y explorar los conjuntos de datos utilizados para entrenar los modelos.
                
                **📋 Pasos para usar:**
                1. **Seleccionar frecuencia** (1.95 GHz, 2.13 GHz, 2.65 GHz)
                2. **Elegir capa** (Estructuras, Mediciones, Antena, Potencia)
                3. **Explorar imágenes** disponibles en el dataset
                
                **💡 Casos de uso:**
                - Entender la estructura de los datos
                - Comparar datasets entre frecuencias
                - Analizar patrones de propagación
                """)
            
            with col_ds2:
                st.info("""
                **📊 Datasets disponibles:**
                - Uno por cada una de las 3 frecuencias
                - 4 capas cada uno
                - 30 muestras por dataset
                - Imágenes de 256 x 256 píxeles
                """)
        
        # Sección 2: Modelos y Evaluación
        with st.expander("2️⃣ **Modelos y Evaluación** - Validar rendimiento de IA", expanded=False):
            col_eval1, col_eval2 = st.columns([2, 1])
            
            with col_eval1:
                st.markdown("""
                **🎯 Propósito:** Cargar, evaluar y validar el rendimiento de los modelos entrenados.
                
                **📋 Flujo de trabajo:**
                1. **Seleccionar modelo** → Elegir frecuencia objetivo
                2. **Cargar modelo** → Botón "Cargar Modelo" 
                3. **Evaluar** → Botón "Evaluar Modelo"
                4. **Analizar métricas** → Revisar MAE, RMSE
                
                **🔄 Evaluación cruzada:**
                - Evaluar modelo con datasets de otras frecuencias
                - Analizar generalización del modelo
                - Comparar rendimiento entre frecuencias
                """)
            
            with col_eval2:
                st.success("""
                **📈 Métricas incluidas:**
                - MAE (Error Absoluto Medio)
                - RMSE (Raíz del Error Cuadrático Medio)
                - Visualizaciones comparativas
                """)
        
        # Sección 3: Nuevos Escenarios
        with st.expander("3️⃣ **Evaluar Nuevos Escenarios** - Predicciones personalizadas", expanded=False):
            st.markdown("""
            **🎯 Propósito:** Crear escenarios personalizados y obtener predicciones de distribución de potencia.
                        
            Esta opción permite aplicar los modelos entrenados en las tres frecuencias en escenarios nuevos sin etiquetas para predecir la distribución de potencia.

            """)
            
            st.markdown("""

            Todas las variables tienen un rango de valores específico, que se deben seguir para obtener una predicción adecuada. 
            
            Las imágenes guardadas se mostrarán en la barra lateral, donde se podrá descargar o borrar las imágenes guardadas. Con las imágenes descargadas se podrá evaluar el modelo seleccionando Evaluar Escenarios. No se puede evaluar los modelos simplemente con guardar las imágenes es necesario descargarlas para subirlas en el lugar señalado para cada capa.
                        
            """)

            # Subsecciones detalladas
            st.markdown("### 🛠️ **Herramientas de Construcción de Escenarios**")
            
            tab_struct, tab_antenna, tab_measure = st.tabs(["🏗️ Estructuras", "📡 Antenas", "📊 Mediciones"])
            
            with tab_struct:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("""
                    **🏢 Construcción de Estructuras**
                    
                    **Elementos disponibles:**
                    - **Líneas:** Paredes
                    - **Rectángulos:** Edificios completos
                    
                    **Parámetros configurables:**
                    - Coordenadas X, Y
                    - Altura en metros
                    - Dimensiones (para rectángulos)
                    """)

                st.markdown("""El usuario puede construir líneas y rectángulos para representar las estructuras y obstáculos, indicando la ubicación y altura. Una vez definidos los parámetros se selecciona la opción Añadir estructura, lo que hará visible la estructura en la imagen para poder continuar con la siguiente estructura. Si ingresó un dato que no es correcto, puede presionar el boton Deshacer para borrar la última estructura añadida, o en su defecto Borrar Todo, si lo que desea es empezar de cero. Al terminar de definir todas las estructuras se selecciona Guardar para guardar la imagen en memoria de la aplicación.
                            
                """)

                with col_s2:
                    st.markdown("""
                    **⚙️ Controles de Edición**
                    
                    - 🟢 **Añadir estructura** → Confirmar elemento
                    - ↩️ **Deshacer** → Eliminar último elemento  
                    - 🗑️ **Borrar Todo** → Reiniciar canvas
                    - 💾 **Guardar** → Finalizar capa de estructuras
                    
                    """)
            
            with tab_antenna:
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("""
                    **📡 Posicionamiento de Antena**
                    
                    **Parámetros:**
                    - **Coordenada X** (posición horizontal)
                    - **Coordenada Y** (posición vertical)  
                    - **Altura** (en metros)
                    
                    **Restricciones:**
                    - Solo una estación base por escenario

                    """)
                
                st.markdown("""El usuario puede seleccionar la ubicación y altura de la antena, y seleccionar Colocar Punto para mostrar la antena en la imagen. 
                             Puede borrar la antena seleccionada presionando Borrar, o simplemente definir una posición y seleccionar nuevamente Colocar Punto. Finalmente debe seleccionar Guardar punto para guardar la imagen.""")

                with col_a2:
                    st.markdown("""
                    **⚙️ Controles**
                    
                    - 📍 **Colocar Punto** → Posicionar antena
                    - 🗑️ **Borrar** → Eliminar antena actual
                    - 💾 **Guardar punto** → Confirmar posición
                    
                    """)
            
            with tab_measure:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("""
                    **📊 Puntos de Medición**
                    
                    **Datos requeridos:**
                    - **Posición X, Y** en el mapa
                    - **Valor de potencia** en dBm
                    - **Mediciones completas** 30 para tener el rendimiento adecuado 
                                
                    """)
                
                st.markdown(""" Se debe seleccionar Añadir Medición para agregar la medición en la imagen y continuar con la siguiente hasta completar todas las mediciones. 
                        Si se ingresó un dato incorrecto, se puede borrar la última medición añadida presionando Deshacer, o en su defecto Borrar Todo para empezar de cero. Al finalizar se selecciona Guardar para guardar la imagen en memoria de la aplicación.""")

                with col_m2:
                    st.markdown("""
                    **⚙️ Gestión de Mediciones**
                    
                    - ➕ **Añadir Medición** → Confirmar punto
                    - ↩️ **Deshacer** → Eliminar último punto
                    - 🗑️ **Borrar Todo** → Limpiar todas las mediciones
                    - 💾 **Guardar** → Finalizar capa y guardarla
                    
                    """)
            
            # Evaluación del escenario
            st.markdown("### 🚀 **Proceso de Evaluación**")
            col_proc1, col_proc2 = st.columns(2)
            
            with col_proc1:
                st.markdown("""
                **Si se generan las capas en esta aplicaciónión:**
    
                1. Completar la construcción las 3 capas de entrada y guardarlas
                2. Descargar imágenes desde el la barra lateral
                3. Cargar cada imagen en la pestaña correspondiente desde el botón Buscar archivos
                4. Seleccionar modelo de evaluación en la pestaña Evaluación
                5. Activar la calibración adaptativa para obtener mejores resultados, o desactivar si desea obtener la predicción del modelo sin calibrar
                6. Presionar el botón Ejecutar Evaluación para realizar la predicción
                7. Descargar las imágenes predichas desde el botón Descargar imagen en la parte inferior de cada imagen (Opcional)

                **Si las imágenes ya están generadas:**
                    Omitir los pasos 1 y 2.
                        
                """)
        
        # Sección 4: Artículo y Publicaciones
        with st.expander("4️⃣ **Artículo y Publicaciones** - Documentación académica", expanded=False):
            col_art1, col_art2 = st.columns(2)
            
            with col_art1:
                st.markdown("""
                **📄 Contenido disponible:**
                - Artículo científico completo
                
                **🔗 Recursos:**
                - Repositorio Zenodo (DOI)
                - Código fuente (GitLab)
                """)
            
            with col_art2:
                st.markdown("""
                **📖 Visualización:**
                - Vista previa integrada del PDF
                - Acceso directo al documento
                - Descarga para uso offline
                
                **🎓 Uso académico:**
                - DOI para referencia permanente
                - Acceso abierto y gratuito
                """)
        
        # Mejores prácticas y recomendaciones
        st.subheader("💡 Mejores Prácticas y Recomendaciones")
        
        col_tips1, col_tips2 = st.columns(2)
        
        with col_tips1:
            with st.expander("🎯 **Para Mejores Resultados**", expanded=False):
                st.markdown("""
                **✅ Recomendaciones:**
                
                - **Carga de modelos:** Siempre cargar antes de evaluar
                - **Datos realistas:** Usar valores dentro de rangos válidos
                - **Suficientes mediciones:** 30 puntos distribuidos
                
                **📊 Rangos válidos:**
                - **Coordenadas:** 0-255 pixeles
                - **Alturas:** 3-40 metros  
                - **Potencia:** -100 a 40 dBm
                """)
        
    elif vista_seleccionada == "Datasets de imágenes":
        crear_header_seccion(
            titulo="Datasets de Imágenes",
            subtitulo="ZonaCEM AI - Exploración de conjuntos de datos de entrenamiento",
            icono="📊"
        )
        modelo_seleccionado = st.selectbox(
            "Selecciona un modelo", list(DATASET_URLS.keys())
        )
        
        # Crear un diccionario que mapee los nombres en español a los nombres originales
        folder_mapping = {DATASET_URLS[modelo_seleccionado]["folder_names"][i]: DATASET_URLS[modelo_seleccionado]["folders"][i]
                         for i in range(len(DATASET_URLS[modelo_seleccionado]["folders"]))}
        
        # Mostrar los nombres en español en el selectbox
        carpeta_nombre = st.selectbox(
            "Selecciona una carpeta (capa)", DATASET_URLS[modelo_seleccionado]["folder_names"]
        )
        
        # Obtener el nombre original de la carpeta para usar en la ruta
        carpeta_seleccionada = folder_mapping[carpeta_nombre]
        
        ruta_base = DATASET_URLS[modelo_seleccionado]["base"]
        ruta_completa = f"{ruta_base}/{carpeta_seleccionada}"
        
        # Botones de descarga con URLs directas
        st.subheader("📥 Descargar datasets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**📁 Carpeta: {carpeta_nombre}**")
            
            # Generar enlace de descarga directa para la carpeta
            folder_download_url = create_direct_download_link(ruta_completa, carpeta_nombre, modelo_seleccionado)
            
            if folder_download_url:
                st.markdown(f"""
                <a href="{folder_download_url}" target="_blank">
                    <button style="
                        background-color: #ff4b4b;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 16px;
                        text-decoration: none;
                        display: inline-block;
                        margin: 5px 0;
                    ">
                        🔗 Descargar Carpeta
                    </button>
                </a>
                """, unsafe_allow_html=True)
                
                #st.info("💡 Haz clic en el botón para descargar ")
            else:
                st.error("No se pudo generar el enlace de descarga")
        
        with col2:
            st.write(f"**📦 Dataset Completo: {modelo_seleccionado}**")
            
            # Generar enlace de descarga directa para todo el dataset
            dataset_download_url = create_direct_dataset_download_link(ruta_base, modelo_seleccionado)
            
            if dataset_download_url:
                st.markdown(f"""
                <a href="{dataset_download_url}" target="_blank">
                    <button style="
                        background-color: #1f77b4;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 16px;
                        text-decoration: none;
                        display: inline-block;
                        margin: 5px 0;
                    ">
                        🔗 Descargar Dataset Completo
                    </button>
                </a>
                """, unsafe_allow_html=True)
                
                st.warning("⚠️ El dataset completo puede ser un archivo grande")
            else:
                st.error("No se pudo generar el enlace del dataset")
        
        # Opción alternativa (botones tradicionales como fallback)
        st.divider()

        col3, col4 = st.columns(2)
                 
        # Vista previa de imágenes
        st.subheader(f"🖼️ Vista previa - Carpeta: {carpeta_nombre}")
        
        try:
            project = get_project()
            
            # Usar la función para obtener más imágenes para vista previa
            #st.info("Cargando vista previa de 50 imágenes...")
            imagenes_preview = get_preview_images(project, ruta_completa, num_images=50)
            
            if imagenes_preview: 
                # Mostrar información básica
                total_encontradas = len(imagenes_preview)
                st.info(f"Vista previa: Mostrando **{min(50, total_encontradas)}** imágenes de esta carpeta.")
                
                # Seleccionar hasta 50 imágenes aleatorias para la vista previa
                num_mostrar = min(50, total_encontradas)
                if total_encontradas > 50:
                    imagenes_muestra = random.sample(imagenes_preview, 50)
                else:
                    imagenes_muestra = imagenes_preview
                
                # Mostrar en filas de 5 columnas para mejor visualización
                cols = st.columns(5)
                imagenes_cargadas = 0
                
                for idx, imagen_path in enumerate(imagenes_muestra):
                    with cols[idx % 5]:
                        try:
                            image_data = get_image_data(imagen_path)
                            if image_data:
                                image = Image.open(io.BytesIO(image_data))
                                st.image(image, caption=os.path.basename(imagen_path))
                                imagenes_cargadas += 1
                        except Exception as e:
                            st.error(f"Error: {os.path.basename(imagen_path)}")
                
                st.success(f"✅ Se cargaron {imagenes_cargadas}/{num_mostrar} imágenes en la vista previa")
                            
            else:
                st.warning("❌ No se encontraron imágenes en esta carpeta")
                
        except Exception as e:
            st.error(f"❌ Error al acceder al repositorio: {str(e)}")
        
    elif vista_seleccionada == "Modelos y evaluación":
        crear_header_seccion(
            titulo="Modelos y Evaluación",
            subtitulo="ZonaCEM AI - Validación y métricas de rendimiento de los modelos",
            icono="🤖"
        )

        st.subheader("Selección de modelo y dataset")

        # Crear dos columnas para los selectores
        col1, col2 = st.columns(2) 
 
        with col1:
            st.write("**Modelo a cargar:**")
            modelo_seleccionado = st.selectbox(
                "Selecciona un modelo para cargar", 
                list(MODELOS.keys()),
                key="modelo_selector"
            )

        with col2:
            st.write("**Dataset para evaluación:**")
            dataset_seleccionado = st.selectbox(
                "Selecciona dataset para evaluar", 
                list(DATASET_URLS.keys()),
                key="dataset_selector"
            )

        # Botón de descarga del modelo
        st.subheader("📥 Descargar modelo")
        st.write(f"**🤖 Modelo seleccionado: {modelo_seleccionado}**")

        # Generar enlace de descarga directa para el modelo
        model_download_url = MODELOS[modelo_seleccionado]

        # Convertir URL de GitLab a enlace de descarga directa
        if "/-/blob/" in model_download_url:
            model_download_url = model_download_url.replace("/-/blob/", "/-/raw/")

        st.markdown(f"""
        <a href="{model_download_url}" target="_blank">
            <button style="
                background-color: #28a745;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                text-decoration: none;
                display: inline-block;
                margin: 5px 0;
            ">
                🔗 Descargar Modelo {modelo_seleccionado}
            </button>
        </a>
        """, unsafe_allow_html=True)

        st.info("💡 El archivo del modelo puede ser grande (varios MB). La descarga comenzará automáticamente.")
        st.divider()

        # Display currently loaded model
        if st.session_state['modelo_actual'] and st.session_state['nombre_modelo_actual']:
            st.info(f"📊 Modelo actualmente cargado en memoria: **{st.session_state['nombre_modelo_actual']}**")
        else:
            st.warning("⚠️ No hay ningún modelo cargado actualmente. Por favor, carga un modelo antes de evaluarlo.")
        
        # Model button (solo queda un botón, así que no necesitamos columnas)
        if st.button("Cargar modelo para evaluarlo"):
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
                st.rerun()
        
        # Evaluation section
        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.subheader("Evaluación del modelo")
            st.write("Presiona el botón para evaluar el modelo con los datos seleccionados.")
        
        with eval_col2:
            is_model_loaded = st.session_state['modelo_actual'] is not None
            evaluar_clicked = st.button("Evaluar modelo", disabled=not is_model_loaded)
    
        if evaluar_clicked and is_model_loaded:
            dataset_path = DATASET_URLS[dataset_seleccionado]["base"]
            
            # Warning if dataset doesn't match model
            if dataset_seleccionado != st.session_state['nombre_modelo_actual']:
                st.warning(f"⚠️ Estás evaluando el modelo **{st.session_state['nombre_modelo_actual']}** con el dataset de **{dataset_seleccionado}**. Los resultados pueden no ser óptimos.")
            
            evaluate_model(st.session_state['modelo_actual'], dataset_path)

    # Evaluar nuevos escenarios
    elif vista_seleccionada == "Evaluar nuevos escenarios":
            crear_header_seccion(
                titulo="Evaluación de Nuevos Escenarios",
                subtitulo="ZonaCEM AI - Predicciones personalizadas con modelos entrenados",
                icono="🎯"
            )

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

                # elif prefix == "posicion_antena" and data:
                #     x, y, h = data
                #     intensity = calculate_intensity(h, 10, 40)
                #     clean_draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=intensity) 

                elif prefix in ["posicion_antena", "trasmisor"] and data:
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
                    
                st.sidebar.header("Imágenes guardadas")
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
                    "trasmisor": (10, 40, "Altura (m)"),
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
                
                # CSS corregido para personalizar completamente el file uploader
                st.markdown("""
                <style>
                /* Ocultar completamente TODO el contenido del botón original */
                div[data-testid="stFileUploader"] button {
                    font-size: 0 !important;
                    position: relative;
                }
                
                div[data-testid="stFileUploader"] button * {
                    display: none !important;
                }
                
                /* Reemplazar con texto en español para el botón */
                div[data-testid="stFileUploader"] button::before {
                    content: "Buscar archivos";
                    font-size: 14px;
                    font-weight: 500;
                    color: white;
                    display: block !important;
                }
                
                /* Ocultar COMPLETAMENTE todo el contenido original del área de arrastre */
                div[data-testid="stFileUploader"] section > div {
                    display: none !important;
                }
                
                /* Ocultar cualquier elemento small */
                div[data-testid="stFileUploader"] small {
                    display: none !important;
                }
                
                /* Ocultar cualquier SVG o icono original */
                div[data-testid="stFileUploader"] svg {
                    display: none !important;
                }
                
                /* Personalizar el área de arrastre */
                div[data-testid="stFileUploader"] section {
                    position: relative;
                    min-height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }
                
                /* Crear ÚNICAMENTE nuestro icono de nube personalizado */
                div[data-testid="stFileUploader"] section::before {
                    content: "☁️";
                    font-size: 32px;
                    margin-bottom: 12px;
                    display: block;
                }
                
                /* Añadir el texto principal en español */
                div[data-testid="stFileUploader"] section::after {
                    content: "Sube tu imagen aquí";
                    display: block;
                    text-align: center;
                    color: #666;
                    font-size: 16px;
                    font-weight: 500;
                    margin-bottom: 0;
                }
                
                /* Crear un elemento adicional para el límite de archivo */
                div[data-testid="stFileUploader"]::after {
                    content: "Límite 200MB por archivo • PNG, JPG, JPEG";
                    display: block;
                    color: #888;
                    font-size: 12px;
                    text-align: center;
                    margin-top: 8px;
                    padding-top: 4px;
                }
                
                /* Ocultar el botón adicional que aparece después de cargar archivo */
                div[data-testid="stFileUploader"] div[data-testid="fileUploader"] button:not(:first-child) {
                    display: none !important;
                }
                
                /* Ocultar botones adicionales en el área de archivos cargados */
                div[data-testid="stFileUploader"] .uploadedFiles button {
                    display: none !important;
                }
                
                /* Ocultar todos los botones que aparecen después del archivo cargado */
                div[data-testid="stFileUploader"] section + * button {
                    display: none !important;
                }
                
                /* Asegurar que no se muestren otros elementos de texto */
                div[data-testid="stFileUploader"] section div {
                    font-size: 0 !important;
                }
                
                div[data-testid="stFileUploader"] section div * {
                    font-size: 0 !important;
                }
                
                /* Mejorar el espaciado general */
                div[data-testid="stFileUploader"] {
                    margin-bottom: 10px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.subheader(f"Cargar imagen de {data_type}")
                uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key=f"upload_{data_type}")
                
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
                
                if st.button("Añadir estructura", key="add_structure"):
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
                    if st.button("Borrar todo", key="clear_structures"):
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
                
                if st.button("Colocar punto", key="add_point"):
                    st.session_state.reference_point = (point_x, point_y, altura)
                    st.rerun()
                
                # Botones de acción
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Borrar", key="clear_point"):
                        st.session_state.reference_point = None
                        st.rerun()
                with col2:
                    if st.button("Guardar punto", key="save_point"):
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
                
                if st.button("Añadir medición", key="add_measurement"):
                    st.session_state.measurement_points.append((x, y, dbm))
                    st.rerun()
                
                # Botones de acción
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Borrar todo", key="clear_measurements"):
                        st.session_state.measurement_points = []
                        st.rerun()
                with col2:
                    if st.button("Deshacer", key="undo_measurement"):
                        if st.session_state.measurement_points:
                            st.session_state.measurement_points.pop()
                            st.rerun()
                with col3:
                    if st.button("Guardar mediciones", key="save_measurements"):
                        save_image_and_data(img, "mediciones", st.session_state.measurement_points)

            def add_model_evaluation_section():
                """Sección para evaluar escenarios con modelos, calibración adaptativa y resultados detallados"""
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
                
                # Checkbox para activar calibración adaptativa
                enable_calibration = st.checkbox(
                    "Activar calibración adaptativa con modelo log-distancia", 
                    value=True,
                    help="Calibra el modelo usando los puntos de medición para mejorar la precisión"
                )
                
                # Botón para ejecutar evaluación
                if st.button("Ejecutar evaluación", disabled=not images_loaded, key="execute_evaluation_button"):
                    if not images_loaded:
                        st.warning("Debes cargar las tres imágenes antes de evaluar.")
                        return
                    
                    with st.spinner("Cargando modelo y evaluando..."):
                        try:
                            # Cargar modelo base
                            modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado)
                            
                            if modelo:
                                # Preparar imágenes
                                struct_img = st.session_state.uploaded_estructuras
                                pixel_img = st.session_state.uploaded_mediciones
                                ant_img = st.session_state.uploaded_trasmisor
                                
                                # Verificar si se debe realizar calibración adaptativa
                                if enable_calibration:
                                    st.info("Realizando calibración adaptativa...")
                                    
                                    # Realizar calibración adaptativa
                                    calibration_result = perform_adaptive_calibration(
                                        modelo, struct_img, pixel_img, ant_img
                                    )
                                    
                                    if calibration_result['success']:
                                        # Usar modelo calibrado
                                        modelo_final = calibration_result['calibrated_model']
                                        evaluation = calibration_result['evaluation']
                                        calib_params = calibration_result['calibration_params']
                                        input_image = calibration_result['input_image']
                                        
                                        # Obtener predicción del modelo calibrado
                                        prediction_norm = modelo_final.predict(np.expand_dims(input_image, axis=0))[0]
                                        if prediction_norm.shape[-1] > 1:
                                            prediction_norm = prediction_norm[:, :, 0]
                                        else:
                                            prediction_norm = prediction_norm.squeeze()
                                        
                                        prediction_denorm = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                                        pred_np = prediction_denorm
                                        
                                        st.success("¡Calibración y predicción completadas!")
                                        
                                        # Mostrar métricas de calibración
                                        st.subheader("Métricas de calibración adaptativa")
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("MAE", f"{evaluation['MAE']:.2f} dB")
                                        with col2:
                                            st.metric("RMSE", f"{evaluation['RMSE']:.2f} dB")
                                        with col3:
                                            st.metric("Error máximo", f"{evaluation['max_error']:.2f} dB")
                                    else:
                                        st.error(f"Error en calibración: {calibration_result['error']}")
                                        return
                                else:
                                    # Evaluación estándar sin calibración
                                    st.info("Ejecutando predicción estándar...")

                                    # Preparar entrada IGUAL que en el código de referencia
                                    struct_array = np.array(st.session_state.uploaded_estructuras) / 255.0
                                    pixels_array = np.array(st.session_state.uploaded_mediciones) / 255.0
                                    antenna_array = np.array(st.session_state.uploaded_trasmisor) / 255.0

                                    # Asegurar dimensiones correctas
                                    if len(struct_array.shape) == 3:
                                        struct_array = struct_array[:, :, 0] if struct_array.shape[2] == 1 else struct_array.mean(axis=2)
                                    if len(pixels_array.shape) == 3:
                                        pixels_array = pixels_array[:, :, 0] if pixels_array.shape[2] == 1 else pixels_array.mean(axis=2)
                                    if len(antenna_array.shape) == 3:
                                        antenna_array = antenna_array[:, :, 0] if antenna_array.shape[2] == 1 else antenna_array.mean(axis=2)

                                    # Añadir dimensión de canal
                                    struct_array = np.expand_dims(struct_array, axis=-1)
                                    pixels_array = np.expand_dims(pixels_array, axis=-1)
                                    antenna_array = np.expand_dims(antenna_array, axis=-1)

                                    # Combinar en tensor de entrada (ORDEN CORRECTO: struct, pixels, antenna)
                                    input_image = np.concatenate([struct_array, pixels_array, antenna_array], axis=-1)
                                    input_batch = np.expand_dims(input_image, axis=0)

                                    # Ejecutar predicción
                                    prediction_norm = modelo.predict(input_batch)[0]

                                    # Manejar dimensiones de salida
                                    if len(prediction_norm.shape) > 2:
                                        prediction_norm = prediction_norm[:, :, 0]

                                    # Desnormalizar CORRECTAMENTE
                                    prediction_denorm = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN

                                    # Convertir a numpy
                                    pred_np = prediction_denorm.numpy() if hasattr(prediction_denorm, 'numpy') else np.array(prediction_denorm)

                                    st.success("¡Predicción completada!")

                                    # Calcular métricas usando los puntos de medición
                                    pixel_array = np.array(st.session_state.uploaded_mediciones)
                                    measurement_coords = np.where(pixel_array > 0)

                                    if len(measurement_coords[0]) > 0:
                                        # Extraer valores reales de las mediciones
                                        measured_values = []
                                        predicted_values = []
                                        
                                        for i in range(len(measurement_coords[0])):
                                            y, x = measurement_coords[0][i], measurement_coords[1][i]
                                            # Valor real desnormalizado
                                            real_value = (pixel_array[y, x] / 255.0) * (POWER_MAX - POWER_MIN) + POWER_MIN
                                            # Valor predicho
                                            pred_value = pred_np[y, x]
                                            
                                            measured_values.append(real_value)
                                            predicted_values.append(pred_value)
                                        
                                        # Convertir a arrays de NumPy
                                        measured_values = np.array(measured_values)
                                        predicted_values = np.array(predicted_values)
                                        
                                        # Calcular métricas
                                        errors = predicted_values - measured_values
                                        mae = np.mean(np.abs(errors))
                                        rmse = np.sqrt(np.mean(errors**2))
                                        max_error = np.max(np.abs(errors))
                                        avg_error = np.mean(errors)
                                        
                                        # Mostrar métricas
                                        st.subheader("Métricas de evaluación")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("MAE", f"{mae:.2f} dB")
                                        with col2:
                                            st.metric("RMSE", f"{rmse:.2f} dB")
                                        with col3:
                                            st.metric("Error máximo", f"{max_error:.2f} dB")
                                    else:
                                        st.warning("No se encontraron puntos de medición para calcular métricas")    
                                
                                # Mostrar métricas básicas de la predicción
                                st.subheader("Métricas de la predicción")
                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric("Potencia máxima", f"{np.max(pred_np):.2f} dBm")
                                with metrics_col2:
                                    st.metric("Potencia mínima", f"{np.min(pred_np):.2f} dBm")
 
                                # Visualización de las capas de entrada
                                st.write("### Capas de entrada")
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                                # Obtener las capas de entrada para visualización
                                if enable_calibration and calibration_result['success']:
                                    # Usar las capas del proceso de calibración
                                    layer1 = (input_image[:, :, 0] * STRUCT_MAX)
                                    layer2 = (input_image[:, :, 1] * (POWER_MAX - POWER_MIN) + POWER_MIN)
                                    layer3 = (input_image[:, :, 2] * ANTENNA_MAX)
                                else:
                                    # Convertir imágenes PIL a arrays NumPy primero
                                    struct_array_viz = np.array(struct_img) / 255.0
                                    pixel_array_viz = np.array(pixel_img) / 255.0
                                    antenna_array_viz = np.array(ant_img) / 255.0
                                    
                                    # Asegurar que sean 2D
                                    if len(struct_array_viz.shape) == 3:
                                        struct_array_viz = struct_array_viz[:, :, 0] if struct_array_viz.shape[2] == 1 else struct_array_viz.mean(axis=2)
                                    if len(pixel_array_viz.shape) == 3:
                                        pixel_array_viz = pixel_array_viz[:, :, 0] if pixel_array_viz.shape[2] == 1 else pixel_array_viz.mean(axis=2)
                                    if len(antenna_array_viz.shape) == 3:
                                        antenna_array_viz = antenna_array_viz[:, :, 0] if antenna_array_viz.shape[2] == 1 else antenna_array_viz.mean(axis=2)
                                    
                                    # Aplicar desnormalización
                                    layer1 = struct_array_viz * STRUCT_MAX
                                    layer2 = pixel_array_viz * (POWER_MAX - POWER_MIN) + POWER_MIN
                                    layer3 = antenna_array_viz * ANTENNA_MAX

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

                                # Calcular rango dinámico real de los datos
                                pred_min = np.min(pred_np)
                                pred_max = np.max(pred_np)

                                # Solo usar valores que no sean del fondo (mayor que POWER_MIN + umbral)
                                threshold = POWER_MIN + 10  # Umbral para filtrar fondo
                                mask = pred_np > threshold
                                valid_data = pred_np[mask]

                                if len(valid_data) > 100:  # Si hay suficientes datos válidos
                                    data_min = np.min(valid_data)
                                    data_max = np.max(valid_data)
                                    
                                    # Usar el rango de datos válidos con un pequeño margen
                                    buffer_range = (data_max - data_min) * 0.05  # 5% de buffer
                                    vmin_plot = max(data_min - buffer_range, POWER_MIN)
                                    vmax_plot = data_max + buffer_range
                                else:
                                    # Fallback: usar todo el rango de datos
                                    vmin_plot = pred_min
                                    vmax_plot = pred_max

                                # Crear la visualización
                                fig, ax = plt.subplots(figsize=(8, 7))
                                im = ax.imshow(pred_np, extent=[0, 50, 0, 50], cmap='viridis',
                                            vmin=vmin_plot, vmax=vmax_plot)

                                ax.set_title('Mapa de potencia predicho', pad=20)
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('Potencia (dBm)')
                                ax.grid(True, linestyle='--', alpha=0.3)
                                ax.set_xlabel('Distancia (m)')
                                ax.set_ylabel('Distancia (m)')

                                # Mostrar información del rango
                                st.info(f"Rango de visualización: {vmin_plot:.1f} a {vmax_plot:.1f} dBm")

                                plt.tight_layout()
                                
                                # Guardar la figura para descargarla
                                fig_result = fig
                                 
                                #st.pyplot(fig)
                                st.pyplot(fig, use_container_width=False)
                                                               
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
                                    label="Descargar imagen de predicción",
                                    data=buffer,
                                    file_name=filename,
                                    mime="image/png",
                                    key="download_button"
                                )
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

                                # Botón de descarga para la imagen de distribución
                                buffer_dist = io.BytesIO()
                                fig.savefig(buffer_dist, format='png', bbox_inches='tight', dpi=300)
                                buffer_dist.seek(0)

                                timestamp_dist = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename_dist = f"distribucion_potencia_{timestamp_dist}.png"

                                st.download_button(
                                    label="Descargar imagen de distribución",
                                    data=buffer_dist,
                                    file_name=filename_dist,
                                    mime="image/png",
                                    key="download_dist_button"
                                )

                                # Generar segmentación
                                st.subheader("Segmentación por zonas de potencia")
                                try:
                                    # Detectar posición de la antena usando la función exacta del primer código
                                    if enable_calibration and calibration_result['success']:
                                        antenna_pos_px = find_antenna_position(input_image[:, :, 2])
                                    else:
                                        antenna_pos_px = find_antenna_position(antenna_array)
                                    
                                    # Crear máscara de segmentación usando la función exacta
                                    pred_mask, pred_r1, pred_r2 = create_segmentation_mask(pred_np, antenna_pos_px)
                                    
                                    # Crear visualización EXACTA
                                    fig_seg, ax_seg = plt.subplots(figsize=(8, 7))
                                    ax_seg.imshow(pred_mask)
                                    
                                    # Marcar la antena
                                    ax_seg.plot(antenna_pos_px[0], antenna_pos_px[1], 'w+', markersize=15, markeredgewidth=3)
                                    
                                    # Círculos concéntricos con el mismo centro
                                    if pred_r1 > 0:
                                        circle1 = plt.Circle(antenna_pos_px, pred_r1, fill=False, color='white', linewidth=2, linestyle='-')
                                        ax_seg.add_artist(circle1)
                                    
                                    if pred_r2 > pred_r1:
                                        circle2 = plt.Circle(antenna_pos_px, pred_r2, fill=False, color='white', linewidth=2, linestyle='--')
                                        ax_seg.add_artist(circle2)
                                    
                                    ax_seg.set_xlabel('Píxeles (X)')
                                    ax_seg.set_ylabel('Píxeles (Y)')
                                    ax_seg.grid(True, linestyle='--', alpha=0.3)
                                    ax_seg.set_title('Segmentación por zonas de potencia', pad=20)
                                    
                                    # Agregar leyenda EXACTA
                                    from matplotlib.patches import Patch
                                    legend_elements = [
                                        Patch(facecolor='red', label='Zona de rabasamiento (> 19 dBm)'),
                                        Patch(facecolor='yellow', label='Zona ocupacional (12-19 dBm)'),
                                        Patch(facecolor='green', label='Zona de conformidad (< 12 dBm)')
                                    ]
                                    ax_seg.legend(handles=legend_elements, loc='upper right')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_seg, use_container_width=False)
                                    
                                    # Información EXACTA
                                    antenna_x_meters = antenna_pos_px[1] * 50/256
                                    antenna_y_meters = (256 - antenna_pos_px[0]) * 50/256
                                    
                                    st.info(f"""
                                    **Información de segmentación:**
                                    - Posición de antena: ({antenna_x_meters:.2f}, {antenna_y_meters:.2f}) metros
                                    - Posición píxeles: ({antenna_pos_px[0]:.1f}, {antenna_pos_px[1]:.1f})
                                    - Radio zona roja: {pred_r1 * 50/256:.2f} metros
                                    - Radio zona amarilla: {pred_r2 * 50/256:.2f} metros
                                    """)
                                    
                                    # Botón de descarga
                                    buffer_seg = io.BytesIO()
                                    fig_seg.savefig(buffer_seg, format='png', bbox_inches='tight', dpi=300)
                                    buffer_seg.seek(0)
                                    
                                    timestamp_seg = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename_seg = f"segmentacion_potencia_{timestamp_seg}.png"
                                    
                                    st.download_button(
                                        label="Descargar imagen de segmentación",
                                        data=buffer_seg,
                                        file_name=filename_seg,
                                        mime="image/png",
                                        key="download_seg_button"
                                    )
                                    
                                    plt.close(fig_seg)
                                    
                                except Exception as e:
                                    st.error(f"Error generando segmentación: {str(e)}")
                                    st.exception(e)
                                                                
                            else:
                                st.error("No se pudo cargar el modelo seleccionado")
                        except Exception as e:
                            st.error(f"Error durante la evaluación: {str(e)}")
                            st.exception(e)

            def main():
                #st.title("Editor de Imágenes para Análisis de Cobertura")
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

    elif vista_seleccionada == "Artículo y publicaciones":
        crear_header_seccion(
            titulo="Artículo y Publicaciones",
            subtitulo="ZonaCEM AI - Documentación académica y referencias científicas",
            icono="📄"
        )
        
        st.markdown(""" 
        Esta sección contiene los enlaces y referencias a las publicaciones científicas derivadas de esta investigación.
        """)
        
        # Información del artículo principal
        st.subheader("Artículo: Estimación de las Zonas de Exposición a Campos Electromagnéticos en Estaciones Base Celulares Usando Inteligencia Artificial")
        
        # Información básica del artículo
        st.markdown("""

            **Autores:** José Luis Mera Tulcán y Giovanni Javier Pantoja Mora
            
            **Institución:** Universidad de Nariño
            
            **DOI:** `10.5281/zenodo.17110951`
            
            **Resumen:** Se presenta una metodología que permite la estimación de mapas de potencia recibida en estaciones base celulares a partir de medidas dispersas empleando redes neuronales convolucionales.
            
            **Palabras clave:** Inteligencia artificial, deep learning, U-Net, predicción de potencia, estación base celular, campos electromagnéticos.
            """)
        
        # Enlaces de acceso principales
        st.subheader("📂 Enlaces de Acceso")   

        # URLs actualizadas - Versión 5
        ZENODO_RECORD_URL = "https://doi.org/10.5281/zenodo.17110951"
        ZENODO_PDF_URL = "https://zenodo.org/records/17247692/files/Art%C3%ADculo.pdf"
        ZENODO_PDF_PREVIEW_URL = "https://zenodo.org/records/17247692/preview/Art%C3%ADculo.pdf?include_deleted=0"
        ZENODO_PDF_DOWNLOAD_URL = "https://zenodo.org/records/17247692/files/Art%C3%ADculo.pdf?download=1"
        GITLAB_REPO_URL = "https://gitlab.com/tulcanjose0/zonacem-ai"

        col1, col2, col3, col4 = st.columns(4)
        
        with col1: 
            st.markdown(f"""
            <a href="{ZENODO_RECORD_URL}" target="_blank"> 
                <button style="
                    background-color: #0066cc;
                    color: white;
                    padding: 15px 20px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    text-decoration: none;
                    display: inline-block;
                    margin: 5px 0;
                    width: 100%;
                    text-align: center;
                ">
                    🏛️ Zenodo<br>
                    <small>(Repositorio)</small>
                </button>
            </a>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <a href="{ZENODO_PDF_DOWNLOAD_URL}" target="_blank">
                <button style="
                    background-color: #28a745;
                    color: white;
                    padding: 15px 20px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    text-decoration: none;
                    display: inline-block;
                    margin: 5px 0;
                    width: 100%;
                    text-align: center;
                ">
                    💾 Descargar<br>
                    <small>(10.5 MB)</small>
                </button>
            </a>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <a href="{GITLAB_REPO_URL}" target="_blank">
                <button style="
                    background-color: #fc6d26;
                    color: white;
                    padding: 15px 20px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    text-decoration: none;
                    display: inline-block;
                    margin: 5px 0;
                    width: 100%;
                    text-align: center;
                ">
                    💻 Código<br>
                    <small>(GitLab)</small>
                </button>
            </a>
            """, unsafe_allow_html=True)

        # Vista previa del PDF usando Google Docs Viewer
        st.subheader("📖 Vista Previa del Artículo")
        
        pdf_url = "https://zenodo.org/records/17247692/files/Art%C3%ADculo.pdf"
        google_viewer_url = f"https://docs.google.com/gview?url={pdf_url}&embedded=true"
        
        st.markdown(f"""
        <iframe src="{google_viewer_url}" 
                width="100%" 
                height="800" 
                style="border: 1px solid #ddd; border-radius: 10px;">
            <p>No se pudo cargar el visualizador. 
            <a href="{pdf_url}" target="_blank">Haz clic aquí para abrir el PDF</a>
            </p>
        </iframe>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    import time
    main()