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

# ===== IMPORTAR SISTEMA DE TRADUCCIONES =====
from translations import get_text, TRANSLATIONS

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
    
def download_and_load_model_from_gitlab(url, model_name, language='es'):
    """Descarga y carga un modelo desde GitLab con barra de progreso"""
    from translations import get_text
    
    def t(key, **kwargs):
        return get_text(key, language, **kwargs)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
            # Convertir URL del formato web a la ruta de archivo en el repositorio
            # Extraer la ruta del archivo después de "main/"
            file_path = url.split('/blob/main/')[1]
            # Crear contenedores para la barra de progreso y mensajes
            progress_text = st.empty()
            progress_bar = st.progress(0)
            # Mensaje inicial
            progress_text.text(t("downloading_model", name=model_name))
            progress_bar.progress(0.3)
            # Obtener proyecto
            project = get_project()
            
            # Usar directamente el método alternativo sin intentar el método que falla
            progress_text.text(t("downloading_model_url"))
            
            # Ajustar la ruta para que sea compatible con get_image_data
            # Necesitamos convertir de "blob/main/" a "raw/main/"
            raw_url = url.replace("/-/blob/", "/-/raw/")
            
            # Descargar usando get_image_data con la URL completa
            file_content = requests.get(raw_url).content
            
            with open(temp_file.name, 'wb') as f:
                f.write(file_content)
            
            progress_bar.progress(0.7)
            progress_text.text(t("download_complete_loading"))
            
            # Cargar modelo
            model = tf.keras.models.load_model(temp_file.name)
            
            # Completar barra de progreso
            progress_bar.progress(1.0)
            progress_text.text(t("model_loaded_successfully"))
            
            # Esperar un momento para que el usuario vea el mensaje
            import time
            time.sleep(1)
            
            # Limpiar mensajes temporales
            progress_text.empty()
            progress_bar.empty()
            
            os.unlink(temp_file.name)
            return model
            
    except Exception as e:
        st.error(f"{t('model_load_error')}: {str(e)}")
        traceback.print_exc()
        return None

def show_model_info(model):  
    """Muestra información reducida sobre el modelo"""
    # Acceder a la función t desde session_state o pasarla como parámetro
    st.subheader("Model Information" if st.session_state.get('language', 'en') == 'en' else "Información del Modelo")
    st.write("Input shape:" if st.session_state.get('language', 'en') == 'en' else "Forma de entrada:", model.input_shape)
    st.write("Output shape:" if st.session_state.get('language', 'en') == 'en' else "Forma de salida:", model.output_shape)

def evaluate_model(model, dataset_path, num_predictions=5, random_selection=True, selected_indices=None, language='en'):
    # AGREGAR: Función helper para traducciones
    def t(key, **kwargs):
        return get_text(key, language, **kwargs)
    
    try:
        # Constantes para normalización/desnormalización (sin cambios)
        STRUCT_MIN, STRUCT_MAX = 0, 20    # metros
        ANTENNA_MIN, ANTENNA_MAX = 0, 40  # metros
        POWER_MIN = -100   # dBm (Piso de ruido térmico)
        POWER_MAX = 40     # dBm (Potencia máxima teórica)
        ACTUAL_POWER_MAX = -10  # dBm (Máximo real)
        
        # Establecer el número total estimado de imágenes (sabemos que hay aproximadamente 10,000)
        total_images = 10000
        # CAMBIO: Traducir mensaje
        st.info(t('dataset_size', size=total_images))
        
        # Seleccionar índices de imágenes a procesar
        if random_selection:
            if num_predictions > total_images:
                num_predictions = total_images
                # CAMBIO: Traducir warning
                st.warning(t('only_images_available', total=total_images))
            # Selección aleatoria de índices
            import random
            indices = random.sample(range(total_images), num_predictions)
            indices.sort()  # Ordenar para mejor seguimiento
            # CAMBIO: Traducir mensaje
            st.write(f"{t('evaluating_random_scenarios')}: {indices}")
        else:
            # Usar índices proporcionados
            if selected_indices is None or len(selected_indices) == 0:
                # CAMBIO: Traducir error
                st.error(t('no_indices_provided'))
                return
            # Validar que los índices estén dentro del rango
            valid_indices = [idx for idx in selected_indices if 0 <= idx < total_images]
            if len(valid_indices) != len(selected_indices):
                # CAMBIO: Traducir warning
                st.warning(t('indices_out_of_range', max=total_images-1))
            indices = valid_indices
            if not indices:
                # CAMBIO: Traducir error
                st.error(t('no_valid_indices'))
                return
            # CAMBIO: Traducir mensaje
            st.write(f"{t('evaluating_images_indices')}: {indices}")
        
        # Obtener las imágenes y procesarlas directamente por índice
        project = get_project()
        
        # Procesar las imágenes seleccionadas
        for i, idx in enumerate(indices):
            try:
                # CAMBIO: Traducir spinner
                with st.spinner(t('loading_scenario', num=idx+1)):
                    # Construir nombres de archivo basados en el índice
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
                
                # CAMBIO: Traducir título
                st.write(f"### {t('prediction')} {i+1} ({t('scenario')} #{idx} {t('from_dataset')})")
                
                # Extraer nombre del archivo para mostrar información adicional
                filename = os.path.basename(struct_path)
                
                # CAMBIO: Traducir label
                st.write(f"{t('file')}: {filename}")
                
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
                
                # Mostrar métricas (sin cambios)
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("MAE (dB)", f"{mae:.2f}")
                with metrics_col2:
                    st.metric("RMSE (dB)", f"{rmse:.2f}")
                with metrics_col3:
                    st.metric("R²", f"{r2:.4f}")
                
                # CAMBIO: Traducir título
                st.write(f"### {t('input_layers')}")
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Desnormalizar cada capa
                layer1 = struct_norm * STRUCT_MAX
                layer2 = pixel_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                layer3 = ant_norm * ANTENNA_MAX
                
                # CAMBIO: Traducir títulos de los gráficos
                im1 = ax1.imshow(layer1, extent=[0, 50, 0, 50], cmap='viridis')
                ax1.set_title(f"{t('structure_m')}", pad=20)
                plt.colorbar(im1, ax=ax1)
                
                im2 = ax2.imshow(layer2, extent=[0, 50, 0, 50], cmap='viridis')
                ax2.set_title(f"{t('pixels_dbm')}", pad=20)
                plt.colorbar(im2, ax=ax2)
                
                im3 = ax3.imshow(layer3, extent=[0, 50, 0, 50], cmap='viridis')
                ax3.set_title(f"{t('antenna_m')}", pad=20)
                plt.colorbar(im3, ax=ax3)
                
                # CAMBIO: Traducir labels de ejes
                for ax in [ax1, ax2, ax3]:
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_xlabel(t('distance_m'))
                    ax.set_ylabel(t('distance_m'))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # CAMBIO: Traducir título
                st.write(f"### {t('results')}")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                
                # CAMBIO: Traducir títulos
                im1 = ax1.imshow(power_np, extent=[0, 50, 0, 50], cmap='viridis')
                ax1.set_title(f"{t('real_power_dbm')}", pad=20)
                plt.colorbar(im1, ax=ax1)
                
                im2 = ax2.imshow(pred_np, extent=[0, 50, 0, 50], cmap='viridis')
                ax2.set_title(f"{t('prediction_dbm')}", pad=20)
                plt.colorbar(im2, ax=ax2)
                
                # CAMBIO: Traducir labels
                for ax in [ax1, ax2]:
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_xlabel(t('distance_m'))
                    ax.set_ylabel(t('distance_m'))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Calcular y visualizar diferencias 
                fig, ax = plt.subplots(figsize=(8, 7))
                difference = pred_np - power_np
                im = ax.imshow(difference, extent=[0, 50, 0, 50], cmap='RdBu_r', 
                            vmin=-10, vmax=10)
                # CAMBIO: Traducir título
                ax.set_title(f"{t('difference_prediction_real')}", pad=20)
                plt.colorbar(im, ax=ax)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_xlabel(t('distance_m'))
                ax.set_ylabel(t('distance_m'))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Histogramas si hay suficientes datos válidos
                if valid_pixels > 10:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # CAMBIO: Traducir labels de leyenda
                    ax.hist(power_masked, bins=50, alpha=0.5, label=t('real'))
                    ax.hist(prediction_masked, bins=50, alpha=0.5, label=t('prediction'))
                    ax.set_xlabel(t('power_dbm'))
                    ax.set_ylabel(t('frequency'))
                    ax.set_title(t('distribution_real_vs_predicted'))
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Diagrama de dispersión (valores reales vs predichos)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(power_masked, prediction_masked, alpha=0.3)
                    min_val = min(np.min(power_masked), np.min(prediction_masked))
                    max_val = max(np.max(power_masked), np.max(prediction_masked))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                    # CAMBIO: Traducir labels
                    ax.set_xlabel(t('real_values_dbm'))
                    ax.set_ylabel(t('predicted_values_dbm'))
                    ax.set_title(t('real_vs_predicted_values'))
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                st.markdown("---")  # Separador entre predicciones

            except Exception as e:
                # CAMBIO: Traducir mensajes de error
                st.error(t('error_processing_scenario', num=idx+1, error=str(e)))
                st.warning(t('trying_next_scenario'))
                continue
                
    except Exception as e:
        # CAMBIO: Traducir mensajes de error
        st.error(f"{t('evaluation_error')}: {str(e)}")
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
    # ===== INICIALIZAR IDIOMA =====
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  # INGLÉS POR DEFECTO
    
    # Función helper para obtener texto
    def t(key, **kwargs):
        return get_text(key, st.session_state.language, **kwargs)
    
    # ===== CONFIGURACIÓN DE PÁGINA =====
    st.set_page_config(page_title=t("page_title"), page_icon="📡", layout="wide")
    st.title(t("page_title"))


    # Constants (sin cambios)
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

    if 'show_citation' not in st.session_state:
        st.session_state.show_citation = False

    # NUEVO: Inicializar tab activo para la sección scenarios
    if 'active_scenario_tab' not in st.session_state:
        st.session_state.active_scenario_tab = 0

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
    
    # ===== FUNCIÓN HELPER PARA HEADERS (actualizada) =====
    def crear_header_seccion(titulo_key, subtitulo_key, icono="📱"):
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">
                {icono} {t(titulo_key)}
            </h1>
            <h3 style="color: #e8f4fd; text-align: center; margin-top: 10px; font-weight: 300;">
                {t(subtitulo_key)}
            </h3>
        </div>
        """, unsafe_allow_html=True)

    # ===== BARRA LATERAL =====
# REEMPLAZA TODO EL SIDEBAR (busca "with st.sidebar:" y reemplaza hasta el siguiente elif/if principal)

    with st.sidebar:
        st.markdown("---")
        
        # ===== INICIALIZAR VISTA SELECCIONADA =====
        if 'current_view' not in st.session_state:
            st.session_state.current_view = "manual"
        
        # ===== SELECTOR DE IDIOMA =====
        st.markdown(
            "<h4 style='margin-bottom: 15px;'><b>Language / Idioma:</b></h4>",
            unsafe_allow_html=True
        )
        
        # Crear dos columnas para los botones de idioma
        lang_col1, lang_col2 = st.columns(2)
        
        with lang_col1:
            if st.session_state.language == "en":
                st.markdown("""
                <div style="background-color: rgba(255, 107, 53, 0.2); border: 2px solid #FF6B35; border-radius: 0.5rem; padding: 8px; text-align: center;">
                    <span style="color: #FF6B35; font-weight: 600;">🇺🇸 English</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("🇺🇸 English", key="lang_en_btn", use_container_width=True):
                    st.session_state.language = "en"
                    st.rerun()
        
        with lang_col2:
            if st.session_state.language == "es":
                st.markdown("""
                <div style="background-color: rgba(255, 107, 53, 0.2); border: 2px solid #FF6B35; border-radius: 0.5rem; padding: 8px; text-align: center;">
                    <span style="color: #FF6B35; font-weight: 600;">🇪🇸 Español</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("🇪🇸 Español", key="lang_es_btn", use_container_width=True):
                    st.session_state.language = "es"
                    st.rerun()
        
        st.markdown("---")
        
        # ===== MENÚ PRINCIPAL =====
        st.markdown(
            "<h4 style='margin-bottom: 15px;'><b>" + t("sidebar_header") + "</b></h4>",
            unsafe_allow_html=True
        )
        
        # Crear opciones de menú traducidas
        menu_options = {
            t("menu_manual"): "manual",
            t("menu_datasets"): "datasets", 
            t("menu_models"): "models",
            t("menu_scenarios"): "scenarios",
        }
        
        menu_labels = list(menu_options.keys())
        menu_values = list(menu_options.values())
        
        # Crear botones verticales para el menú
        for idx, (label, value) in enumerate(menu_options.items()):
            if st.session_state.current_view == value:
                # Mostrar como caja naranja si está activo
                st.markdown(f"""
                <div style="background-color: rgba(255, 107, 53, 0.2); border: 2px solid #FF6B35; border-radius: 0.5rem; padding: 12px; text-align: center; margin-bottom: 8px;">
                    <span style="color: #FF6B35; font-weight: 600;">{label}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Mostrar como botón normal si no está activo
                if st.button(label, key=f"menu_btn_{idx}", use_container_width=True):
                    st.session_state.current_view = value
                    st.rerun()
        
        st.markdown("---")
        
        # Obtener la vista actual para usarla después
        vista_seleccionada = st.session_state.current_view
        
        # CAMBIO: Actualizar la vista actual en session_state
        st.session_state.current_view = vista_seleccionada


    # ===== SECCIÓN: MANUAL DE USUARIO =====
    if vista_seleccionada == "manual":
        crear_header_seccion(
            titulo_key="header_manual_title",
            subtitulo_key="header_manual_subtitle",
            icono="📱"
        )
        
        # CAMBIO COMPLETO: Todo el contenido usando funciones de traducción
        st.markdown(f"### 🎯 {t('welcome_title')}")
        
        st.markdown(t("welcome_intro"))
        
        #st.markdown(t("app_description"))
        
        # Lista de capas con formato
        st.markdown(t('datasets_intro'))
        st.markdown(f"""
    - **{t('layer_structures')}**: {t('structures_description')}
    - **{t('layer_measurements')}**: {t('measurements_description')}
    - **{t('layer_antenna')}**: {t('antenna_description')}
    - **{t('layer_power')}**: {t('power_description')}
    """)
        
        st.markdown(t('prediction_description'))
        st.markdown(t('units_description'))
        
        # Guía de secciones de la aplicación
        st.subheader(f"🗺️ {t('sections_guide')}")
        
        # Sección 1: Datasets
        with st.expander(f"1️⃣ **{t('section_datasets')}** - {t('section_datasets_subtitle')}", expanded=False):
            col_ds1, col_ds2 = st.columns([2, 1])
            
            with col_ds1:
                st.markdown(f"""
    **🎯 {t('purpose')}:** {t('datasets_purpose')}

    **📋 {t('steps_to_use')}:**
    1. **{t('select_frequency')}** (1.95 GHz, 2.13 GHz, 2.65 GHz)
    2. **{t('choose_layer')}** ({t('layer_structures')}, {t('layer_measurements')}, {t('layer_antenna')}, {t('layer_power')})
    3. **{t('explore_images')}** {t('available_in_dataset')}

    **💡 {t('use_cases')}:**
    - {t('understand_data_structure')}
    - {t('compare_datasets')}
    - {t('analyze_patterns')}
    """)
            
            with col_ds2:
                st.info(f"""
    **📊 {t('available_datasets')}:**
    - {t('one_per_frequency')}
    - {t('four_layers_each')}
    - {t('thirty_samples')}
    - {t('image_size')}
    """)
        
        # Sección 2: Modelos y Evaluación
        with st.expander(f"2️⃣ **{t('section_models')}** - {t('section_models_subtitle')}", expanded=False):
            col_eval1, col_eval2 = st.columns([2, 1])
            
            with col_eval1:
                st.markdown(f"""
    **🎯 {t('purpose')}:** {t('models_purpose')}

    **📋 {t('workflow')}:**
    1. **{t('select_model')}** → {t('choose_target_frequency')}
    2. **{t('load_model')}** → {t('load_model_button')}
    3. **{t('evaluate')}** → {t('evaluate_model_button')}
    4. **{t('analyze_metrics')}** → {t('review_mae_rmse')}

    **🔄 {t('cross_evaluation')}:**
    - {t('evaluate_other_datasets')}
    - {t('analyze_generalization')}
    - {t('compare_performance')}
    """)
            
            with col_eval2:
                st.success(f"""
    **📈 {t('included_metrics')}:**
    - MAE ({t('mae_description')})
    - RMSE ({t('rmse_description')})
    - {t('comparative_visualizations')}
    """)
        
        # Sección 3: Nuevos Escenarios
        with st.expander(f"3️⃣ **{t('section_scenarios')}** - {t('section_scenarios_subtitle')}", expanded=False):
            st.markdown(f"""
    **🎯 {t('purpose')}:** {t('scenarios_purpose')}

    {t('scenarios_intro')}

    {t('value_ranges_info')}

    {t('saved_images_info')}
    """)

            # Subsecciones detalladas
            st.markdown(f"### 🛠️ **{t('scenario_tools')}**")
            
            tab_struct, tab_antenna, tab_measure = st.tabs([
                f"🏗️ {t('tab_structures_full')}", 
                f"📡 {t('tab_antenna_full')}", 
                f"📊 {t('tab_measurements_full')}"
            ])
            
            with tab_struct:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"""
    **🏢 {t('building_structures')}**

    **{t('available_elements')}:**
    - **{t('lines')}:** {t('walls')}
    - **{t('rectangles')}:** {t('complete_buildings')}

    **{t('configurable_parameters')}:**
    - {t('coordinates_xy')}
    - {t('height_meters')}
    - {t('dimensions_rectangles')}
    """)

                st.markdown(t('structures_instructions'))

                with col_s2:
                    st.markdown(f"""
    **⚙️ {t('editing_controls')}**

    - 🟢 **{t('add_structure')}** → {t('confirm_element')}
    - ↩️ **{t('undo')}** → {t('remove_last_element')}
    - 🗑️ **{t('clear_all')}** → {t('restart_canvas')}
    - 💾 **{t('save')}** → {t('finalize_layer')}
    """)
            
            with tab_antenna:
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown(f"""
    **📡 {t('antenna_positioning')}**

    **{t('parameters')}:**
    - **{t('coordinate')} X** ({t('horizontal_position')})
    - **{t('coordinate')} Y** ({t('vertical_position')})
    - **{t('height')}** ({t('in_meters')})

    **{t('restrictions')}:**
    - {t('one_antenna_per_scenario')}
    """)
                
                st.markdown(t('antenna_instructions'))

                with col_a2:
                    st.markdown(f"""
    **⚙️ {t('controls')}**

    - 📍 **{t('place_point')}** → {t('position_antenna')}
    - 🗑️ **{t('delete')}** → {t('remove_current_antenna')}
    - 💾 **{t('save_point')}** → {t('confirm_position')}
    """)
            
            with tab_measure:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
    **📊 {t('measurement_points')}**

    **{t('required_data')}:**
    - **{t('position')} X, Y** {t('on_map')}
    - **{t('power_value')}** {t('in_dbm')}
    - **{t('complete_measurements')}** {t('thirty_for_performance')}
    """)
                
                st.markdown(t('measurements_instructions'))

                with col_m2:
                    st.markdown(f"""
    **⚙️ {t('measurement_management')}**

    - ➕ **{t('add_measurement')}** → {t('confirm_point')}
    - ↩️ **{t('undo')}** → {t('remove_last_point')}
    - 🗑️ **{t('clear_all')}** → {t('clear_all_measurements')}
    - 💾 **{t('save')}** → {t('finalize_and_save')}
    """)
            
            # Evaluación del escenario
            st.markdown(f"### 🚀 **{t('evaluation_process')}**")
            col_proc1, col_proc2 = st.columns(2)
            
            with col_proc1:
                st.markdown(t('evaluation_steps'))
    
        
        # Mejores prácticas y recomendaciones
        st.subheader(f"💡 {t('best_practices')}")
        
        col_tips1, col_tips2 = st.columns(2)
        
        with col_tips1:
            with st.expander(f"🎯 **{t('for_best_results')}**", expanded=False):
                st.markdown(f"""
    **✅ {t('recommendations')}:**

    - **{t('model_loading')}:** {t('always_load_before_eval')}
    - **{t('realistic_data')}:** {t('use_valid_ranges')}
    - **{t('sufficient_measurements')}:** {t('thirty_distributed_points')}

    **📊 {t('valid_ranges')}:**
    - **{t('coordinates')}:** {t('coordinates_range')}
    - **{t('heights')}:** {t('heights_range')}
    - **{t('power')}:** {t('power_range')}
    """)

    # ===== SECCIÓN: DATASETS =====
    elif vista_seleccionada == "datasets":
        crear_header_seccion(
            titulo_key="header_datasets_title",
            subtitulo_key="header_datasets_subtitle",
            icono="📊"
        )
        
        # CAMBIO: Crear nombres de modelos traducidos dinámicamente
        model_names_translated = {
            "Modelo 1.95GHz": f"{t('model')} 1.95GHz",
            "Modelo 2.13GHz": f"{t('model')} 2.13GHz",
            "Modelo 2.65GHz": f"{t('model')} 2.65GHz"
        }
        
        # Mapeo inverso para obtener la clave original
        translated_to_original = {v: k for k, v in model_names_translated.items()}
        
        modelo_seleccionado_display = st.selectbox(
            t("select_model"), 
            list(model_names_translated.values())
        )
        
        # Obtener la clave original del modelo
        modelo_seleccionado = translated_to_original[modelo_seleccionado_display]
        
        # Nombres de carpetas traducidos (sin cambios)
        folder_names_translated = [
            t("layer_structures"),
            t("layer_measurements"),
            t("layer_antenna"),
            t("layer_power")
        ]
        
        # Crear mapeo (sin cambios)
        folder_mapping = {
            folder_names_translated[i]: DATASET_URLS[modelo_seleccionado]["folders"][i]
            for i in range(len(folder_names_translated))
        }
        
        carpeta_nombre = st.selectbox(
            t("select_folder"),
            folder_names_translated
        )
        
        carpeta_seleccionada = folder_mapping[carpeta_nombre]
        
        ruta_base = DATASET_URLS[modelo_seleccionado]["base"]
        ruta_completa = f"{ruta_base}/{carpeta_seleccionada}"
        
        # Botones de descarga
        st.subheader(t("download_datasets"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{t('folder_label')} {carpeta_nombre}**")
            
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
                        {t('download_folder')}
                    </button>
                </a>
                """, unsafe_allow_html=True)
            else:
                st.error(t("dataset_error"))
        
        with col2:
            # CAMBIO: Usar nombre traducido para mostrar
            st.write(f"**{t('complete_dataset')} {modelo_seleccionado_display}**")
            
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
                        {t('download_complete')}
                    </button>
                </a>
                """, unsafe_allow_html=True)
                
                st.warning(t("dataset_warning"))
            else:
                st.error(t("dataset_error"))
        
        st.divider()
        
        # Vista previa de imágenes (sin cambios)
        st.subheader(f"{t('preview_images')}{carpeta_nombre}")
        
        try:
            project = get_project()
            imagenes_preview = get_preview_images(project, ruta_completa, num_images=50)
            
            if imagenes_preview: 
                total_encontradas = len(imagenes_preview)
                st.info(t("showing_images", num=min(50, total_encontradas)))
                
                num_mostrar = min(50, total_encontradas)
                if total_encontradas > 50:
                    imagenes_muestra = random.sample(imagenes_preview, 50)
                else:
                    imagenes_muestra = imagenes_preview
                
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
                
                st.success(t("images_loaded", loaded=imagenes_cargadas, total=num_mostrar))
            else:
                st.warning(t("no_images"))
                
        except Exception as e:
            st.error(t("error_accessing", error=str(e)))

    # ===== SECCIÓN: MODELOS Y EVALUACIÓN =====
    elif vista_seleccionada == "models":
        crear_header_seccion(
            titulo_key="header_models_title",
            subtitulo_key="header_models_subtitle",
            icono="🤖"
        )

        st.subheader(t("model_selection"))

        # CAMBIO: Crear nombres de modelos traducidos
        model_names_translated = {
            "Modelo 1.95GHz": f"{t('model')} 1.95GHz",
            "Modelo 2.13GHz": f"{t('model')} 2.13GHz",
            "Modelo 2.65GHz": f"{t('model')} 2.65GHz"
        }
        
        # Mapeo inverso
        translated_to_original_models = {v: k for k, v in model_names_translated.items()}

        # Crear dos columnas para los selectores
        col1, col2 = st.columns(2) 
    
        with col1:
            st.write(f"**{t('model_to_load')}**")
            modelo_seleccionado_display = st.selectbox(
                t("select_model_load"), 
                list(model_names_translated.values()),
                key="modelo_selector"
            )
            # Obtener clave original
            modelo_seleccionado = translated_to_original_models[modelo_seleccionado_display]

        with col2:
            st.write(f"**{t('dataset_evaluation')}**")
            dataset_seleccionado_display = st.selectbox(
                t("select_dataset_eval"), 
                list(model_names_translated.values()),
                key="dataset_selector"
            )
            # Obtener clave original
            dataset_seleccionado = translated_to_original_models[dataset_seleccionado_display]

        # Botón de descarga del modelo
        st.subheader(t("download_model"))
        st.write(f"**{t('selected_model')} {modelo_seleccionado_display}**")

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
                {t('download_model_button', name=modelo_seleccionado_display)}
            </button>
        </a>
        """, unsafe_allow_html=True)

        st.info(t("model_download_info"))
        st.divider()

        # CAMBIO: Display currently loaded model con traducción
        if st.session_state['modelo_actual'] and st.session_state['nombre_modelo_actual']:
            # Traducir el nombre del modelo cargado
            loaded_model_translated = model_names_translated.get(
                st.session_state['nombre_modelo_actual'], 
                st.session_state['nombre_modelo_actual']
            )
            st.info(f"{t('model_loaded')} **{loaded_model_translated}**")
        else:
            st.warning(t("no_model_loaded"))
        
        # Model button
        if st.button(t("load_model_button")):
            # Clear previous model if exists
            if st.session_state['modelo_actual']:
                st.session_state['modelo_actual'] = None
                import gc
                gc.collect()
                
            # Load new model with progress bar
            #modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado)

            modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado, st.session_state.language)  # ← AGREGAR EL PARÁMETRO


            if modelo:
                show_model_info(modelo)
                st.success(t("model_loaded_success", name=modelo_seleccionado_display))
                st.session_state['modelo_actual'] = modelo
                st.session_state['nombre_modelo_actual'] = modelo_seleccionado
                st.rerun()
        
        # Evaluation section
        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.subheader(t("evaluation_section"))
            st.write(t("evaluation_description"))
        
        with eval_col2:
            is_model_loaded = st.session_state['modelo_actual'] is not None
            evaluar_clicked = st.button(t("evaluate_model_button"), disabled=not is_model_loaded)

        if evaluar_clicked and is_model_loaded:
            dataset_path = DATASET_URLS[dataset_seleccionado]["base"]
            
            # Warning if dataset doesn't match model - CAMBIO: usar nombres traducidos
            if dataset_seleccionado != st.session_state['nombre_modelo_actual']:
                loaded_model_translated = model_names_translated.get(
                    st.session_state['nombre_modelo_actual'],
                    st.session_state['nombre_modelo_actual']
                )
                st.warning(t("evaluation_warning", 
                        model=loaded_model_translated,
                        dataset=dataset_seleccionado_display))
            
            # CAMBIO: Llamar a evaluate_model con parámetro de idioma
            #evaluate_model(st.session_state['modelo_actual'], dataset_path, language=st.session_state.language)
            evaluate_model(st.session_state['modelo_actual'], dataset_path, language=st.session_state.language)

# ===== SECCIÓN: EVALUAR NUEVOS ESCENARIOS =====
    elif vista_seleccionada == "scenarios":
            crear_header_seccion(
                titulo_key="header_models_title",
                subtitulo_key="header_models_subtitle",
                icono="🤖"
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

            def display_image_with_matplotlib(pil_image, title, data_type=None, language='es'):
                """Muestra una imagen con matplotlib con valores reales"""
                from translations import get_text
                
                def t(key, **kwargs):
                    return get_text(key, language, **kwargs)
                
                img_array = np.array(pil_image)
                
                # Configurar vmin, vmax y etiqueta según tipo de datos
                config = {
                    "estructuras": (3, 20, t("height_label")),
                    "trasmisor": (10, 40, t("height_label")),
                    "mediciones": (-100, 40, t("signal_label"))
                }
                
                vmin, vmax, cbar_label = config.get(data_type, (0, 255, t("intensity_label")))
                
                # Crear figura y mostrar imagen
                fig, ax = plt.subplots(figsize=(8, 6))
                norm = Normalize(0, 255)
                im = ax.imshow(img_array, cmap='viridis', extent=[0, 50, 0, 50], norm=norm, origin='upper')
                
                # Configurar ejes y título
                ax.set_xlabel(t("distance_m"))
                ax.set_ylabel(t("distance_m"))
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
            
            def create_common_editor_components(data_type, edit_function, language='es'):
                """Componentes comunes para todos los editores"""
                from translations import get_text
                
                def t(key, **kwargs):
                    return get_text(key, language, **kwargs)
                
                # CSS para personalizar file uploader
                upload_text = t("upload_your_image") if language == 'es' else "Upload your image here"
                browse_text = t("browse_files") if language == 'es' else "Browse files"
                limit_text = t("file_limit") if language == 'es' else "Limit 200MB per file • PNG, JPG, JPEG"
                
                st.markdown(f"""
                <style>
                /* Ocultar completamente TODO el contenido del botón original */
                div[data-testid="stFileUploader"] button {{
                    font-size: 0 !important;
                    position: relative;
                }}
                
                div[data-testid="stFileUploader"] button * {{
                    display: none !important;
                }}
                
                /* Reemplazar con texto en el idioma seleccionado para el botón */
                div[data-testid="stFileUploader"] button::before {{
                    content: "{browse_text}";
                    font-size: 14px;
                    font-weight: 500;
                    color: white;
                    display: block !important;
                }}
                
                /* Ocultar COMPLETAMENTE todo el contenido original del área de arrastre */
                div[data-testid="stFileUploader"] section > div {{
                    display: none !important;
                }}
                
                /* Ocultar cualquier elemento small */
                div[data-testid="stFileUploader"] small {{
                    display: none !important;
                }}
                
                /* Ocultar cualquier SVG o icono original */
                div[data-testid="stFileUploader"] svg {{
                    display: none !important;
                }}
                
                /* Personalizar el área de arrastre */
                div[data-testid="stFileUploader"] section {{
                    position: relative;
                    min-height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }}
                
                /* Crear ÚNICAMENTE nuestro icono de nube personalizado */
                div[data-testid="stFileUploader"] section::before {{
                    content: "☁️";
                    font-size: 32px;
                    margin-bottom: 12px;
                    display: block;
                }}
                
                /* Añadir el texto principal en el idioma seleccionado */
                div[data-testid="stFileUploader"] section::after {{
                    content: "{upload_text}";
                    display: block;
                    text-align: center;
                    color: #666;
                    font-size: 16px;
                    font-weight: 500;
                    margin-bottom: 0;
                }}
                
                /* Crear un elemento adicional para el límite de archivo */
                div[data-testid="stFileUploader"]::after {{
                    content: "{limit_text}";
                    display: block;
                    color: #888;
                    font-size: 12px;
                    text-align: center;
                    margin-top: 8px;
                    padding-top: 4px;
                }}
                
                /* Ocultar el botón adicional que aparece después de cargar archivo */
                div[data-testid="stFileUploader"] div[data-testid="fileUploader"] button:not(:first-child) {{
                    display: none !important;
                }}
                
                /* Ocultar botones adicionales en el área de archivos cargados */
                div[data-testid="stFileUploader"] .uploadedFiles button {{
                    display: none !important;
                }}
                
                /* Ocultar todos los botones que aparecen después del archivo cargado */
                div[data-testid="stFileUploader"] section + * button {{
                    display: none !important;
                }}
                
                /* Asegurar que no se muestren otros elementos de texto */
                div[data-testid="stFileUploader"] section div {{
                    font-size: 0 !important;
                }}
                
                div[data-testid="stFileUploader"] section div * {{
                    font-size: 0 !important;
                }}
                
                /* Mejorar el espaciado general */
                div[data-testid="stFileUploader"] {{
                    margin-bottom: 10px;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                # Traducir el header correctamente
                data_type_translations = {
                    "estructuras": t("layer_structures"),
                    "trasmisor": t("layer_antenna"),
                    "mediciones": t("layer_measurements")
                }
                
                st.subheader(f"{t('load_image_title')} {data_type_translations.get(data_type, data_type)}")
                uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key=f"upload_{data_type}")
                
                if uploaded_file:
                    img, temp_path = process_uploaded_image(uploaded_file, data_type)
                    if img:
                        setattr(st.session_state, f"uploaded_{data_type}", img)
                        setattr(st.session_state, f"uploaded_{data_type}_path", temp_path)

            def create_building_interface():
                # CAMBIO: Traducir el header
                st.subheader(t("editor_structures"))
                
                if 'structures' not in st.session_state:
                    st.session_state.structures = []
                if 'drawing_tool' not in st.session_state:
                    st.session_state.drawing_tool = "linea"
                
                # Crear y mostrar imagen (sin cambios)
                img = Image.new('L', (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw_grid_with_labels(draw)
                
                # Dibujar estructuras existentes (sin cambios)
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
                
                # Mostrar imagen (sin cambios)
                #fig = display_image_with_matplotlib(img, t("layer_structures"), data_type="estructuras")
                fig = display_image_with_matplotlib(img, t("layer_structures"), data_type="estructuras", language=st.session_state.language)

                st.pyplot(fig)
                plt.close(fig)
                
                # Cargar imagen (sin cambios)
                #create_common_editor_components("estructuras", None)
                create_common_editor_components("estructuras", None, language=st.session_state.language)


                # CAMBIO: Traducir controles de dibujo
                st.session_state.drawing_tool = st.radio(
                    t("drawing_tool"),
                    options=["linea", "rectangulo"],
                    format_func=lambda x: t("tool_line") if x == "linea" else t("tool_rectangle")
                )
                
                # CAMBIO: Traducir labels de coordenadas
                col1, col2 = st.columns(2)
                with col1:
                    start_x_m = st.number_input(f"{t('x_initial')} (m)", 0, 50, 25)
                    start_y_m = st.number_input(f"{t('y_initial')} (m)", 0, 50, 25)
                    start_x = int(start_x_m * pixels_per_meter)
                    start_y = int((50 - start_y_m) * pixels_per_meter)
                with col2:
                    end_x_m = st.number_input(f"{t('x_final')} (m)", 0, 50, 35)
                    end_y_m = st.number_input(f"{t('y_final')} (m)", 0, 50, 35)
                    end_x = int(end_x_m * pixels_per_meter)
                    end_y = int((50 - end_y_m) * pixels_per_meter)
                
                # CAMBIO: Traducir altura
                altura = st.slider(f"{t('height')} ({t('meters')})", 3, 20, 10)
                
                # CAMBIO: Traducir botón añadir
                if st.button(t("add_structure"), key="add_structure"):
                    new_structure = {
                        'tipo': st.session_state.drawing_tool,
                        'coords': [(start_x, start_y), (end_x, end_y)],
                        'altura': altura
                    }
                    st.session_state.structures.append(new_structure)
                    st.rerun()
                
                # CAMBIO: Traducir botones de acción
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(t("clear_all"), key="clear_structures"):
                        st.session_state.structures = []
                        st.rerun()
                with col2:
                    if st.button(t("undo"), key="undo_structure"):
                        if st.session_state.structures:
                            st.session_state.structures.pop()
                            st.rerun()
                with col3:
                    if st.button(t("save"), key="save_structures"):
                        save_image_and_data(img, "estructuras", st.session_state.structures)

            def create_point_interface():
                # CAMBIO: Traducir header
                st.subheader(t("editor_antenna"))
                
                if 'reference_point' not in st.session_state:
                    st.session_state.reference_point = None
                
                # Crear y mostrar imagen (sin cambios)
                img = Image.new('L', (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw_grid_with_labels(draw)
                
                # Dibujar punto existente (sin cambios)
                pixels_per_meter = 256/50
                if st.session_state.reference_point:
                    x, y, h = st.session_state.reference_point
                    intensity = calculate_intensity(h, 10, 40)
                    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=intensity)
                
                # Mostrar imagen (CAMBIO: traducir título)
                #fig = display_image_with_matplotlib(img, t("layer_antenna"), data_type="trasmisor")
                fig = display_image_with_matplotlib(img, t("layer_antenna"), data_type="trasmisor", language=st.session_state.language)

                st.pyplot(fig)
                plt.close(fig)
                
                # Cargar imagen (sin cambios)
                #create_common_editor_components("trasmisor", None)
                create_common_editor_components("trasmisor", None, language=st.session_state.language)

                # CAMBIO: Traducir controles de posición
                col1, col2 = st.columns(2)
                with col1:
                    point_x_m = st.number_input(f"{t('position')} X (m)", 0, 50, 25)
                    point_x = int(point_x_m * pixels_per_meter)
                with col2:
                    point_y_m = st.number_input(f"{t('position')} Y (m)", 0, 50, 25)
                    point_y = int((50 - point_y_m) * pixels_per_meter)
                
                # CAMBIO: Traducir altura
                altura = st.slider(f"{t('height')} ({t('meters')})", 10, 40, 25)
                
                # CAMBIO: Traducir botón colocar
                if st.button(t("place_point"), key="add_point"):
                    st.session_state.reference_point = (point_x, point_y, altura)
                    st.rerun()
                
                # CAMBIO: Traducir botones de acción
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t("delete"), key="clear_point"):
                        st.session_state.reference_point = None
                        st.rerun()
                with col2:
                    if st.button(t("save_point"), key="save_point"):
                        save_image_and_data(img, "trasmisor", st.session_state.reference_point)

            def create_pixel_selector():
                # CAMBIO: Traducir header
                st.subheader(t("editor_measurements"))
                
                if 'measurement_points' not in st.session_state:
                    st.session_state.measurement_points = []
                
                # Crear y mostrar imagen (sin cambios)
                img = Image.new('L', (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw_grid_with_labels(draw)
                
                # Dibujar puntos existentes (sin cambios)
                for point in st.session_state.measurement_points:
                    x, y, dbm = point
                    intensity = calculate_intensity(dbm, -100, 40)
                    draw.point((x, y), fill=intensity)

                # Mostrar imagen (CAMBIO: traducir título)
                #fig = display_image_with_matplotlib(img, t("layer_measurements"), data_type="mediciones")
                fig = display_image_with_matplotlib(img, t("layer_measurements"), data_type="mediciones", language=st.session_state.language)

                st.pyplot(fig)
                plt.close(fig)
                
                # Cargar imagen (sin cambios)
                #create_common_editor_components("mediciones", None)
                create_common_editor_components("mediciones", None, language=st.session_state.language)

                # CAMBIO: Traducir controles
                pixels_per_meter = 256/50
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_m = st.number_input("X (m)", 0, 50, 25, key="measurement_x")
                    x = int(x_m * pixels_per_meter)
                with col2:
                    y_m = st.number_input("Y (m)", 0, 50, 25, key="measurement_y")
                    y = int((50 - y_m) * pixels_per_meter)
                with col3:
                    dbm = st.number_input(f"{t('value')} (dBm)", -100, 40, -50)
                
                # CAMBIO: Traducir botón añadir
                if st.button(t("add_measurement"), key="add_measurement"):
                    st.session_state.measurement_points.append((x, y, dbm))
                    st.rerun()
                
                # CAMBIO: Traducir botones de acción
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(t("clear_all"), key="clear_measurements"):
                        st.session_state.measurement_points = []
                        st.rerun()
                with col2:
                    if st.button(t("undo"), key="undo_measurement"):
                        if st.session_state.measurement_points:
                            st.session_state.measurement_points.pop()
                            st.rerun()
                with col3:
                    if st.button(t("save_measurements"), key="save_measurements"):
                        save_image_and_data(img, "mediciones", st.session_state.measurement_points)

            def add_model_evaluation_section():
                """Sección para evaluar escenarios con modelos, calibración adaptativa y resultados detallados"""
                # CAMBIO: Traducir header
                st.header(t("evaluate_with_model"))
                
                # Constantes para normalización/desnormalización (sin cambios)
                STRUCT_MIN, STRUCT_MAX = 0, 20        # metros (altura de estructuras)
                ANTENNA_MIN, ANTENNA_MAX = 10, 40     # metros (altura de antena)
                POWER_MIN = -100                       # dBm (Piso de ruido térmico)
                POWER_MAX = 40                         # dBm (Potencia máxima teórica)
                ACTUAL_POWER_MAX = -10                 # dBm (Máximo real esperado)
                
                # CAMBIO: Traducir selector de modelo
                modelo_seleccionado = st.selectbox(
                    t("select_model_evaluation"),
                    list(MODELOS.keys()),
                    key="modelo_selector"
                )
                
                # Verificar si las imágenes necesarias están cargadas (sin cambios)
                required_images = ['estructuras', 'trasmisor', 'mediciones']
                images_loaded = all(hasattr(st.session_state, f'uploaded_{img}') for img in required_images)
                
                # CAMBIO: Traducir checkbox de calibración
                enable_calibration = st.checkbox(
                    t("enable_calibration"), 
                    value=True,
                    help=t("calibration_help")
                )
                
                # CAMBIO: Traducir botón de evaluación
                if st.button(t("execute_evaluation"), disabled=not images_loaded, key="execute_evaluation_button"):
                    if not images_loaded:
                        # CAMBIO: Traducir warning
                        st.warning(t("load_three_images_warning"))
                        return
                    
                    # CAMBIO: Traducir spinner
                    with st.spinner(t("loading_evaluating")):
                        try:
                            # Cargar modelo base (sin cambios)
                            #modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado)
                            modelo = download_and_load_model_from_gitlab(MODELOS[modelo_seleccionado], modelo_seleccionado, st.session_state.language)  # ← AGREGAR EL PARÁMETRO
                            if modelo:
                                # Preparar imágenes (sin cambios)
                                struct_img = st.session_state.uploaded_estructuras
                                pixel_img = st.session_state.uploaded_mediciones
                                ant_img = st.session_state.uploaded_trasmisor
                                
                                # Verificar si se debe realizar calibración adaptativa
                                if enable_calibration:
                                    # CAMBIO: Traducir info de calibración
                                    st.info(t("performing_calibration"))
                                    
                                    # Realizar calibración adaptativa (sin cambios)
                                    calibration_result = perform_adaptive_calibration(
                                        modelo, struct_img, pixel_img, ant_img
                                    )
                                    
                                    if calibration_result['success']:
                                        # Usar modelo calibrado (sin cambios hasta...)
                                        modelo_final = calibration_result['calibrated_model']
                                        evaluation = calibration_result['evaluation']
                                        calib_params = calibration_result['calibration_params']
                                        input_image = calibration_result['input_image']
                                        
                                        # Obtener predicción del modelo calibrado (sin cambios)
                                        prediction_norm = modelo_final.predict(np.expand_dims(input_image, axis=0))[0]
                                        if prediction_norm.shape[-1] > 1:
                                            prediction_norm = prediction_norm[:, :, 0]
                                        else:
                                            prediction_norm = prediction_norm.squeeze()
                                        
                                        prediction_denorm = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN
                                        pred_np = prediction_denorm
                                        
                                        # CAMBIO: Traducir success message
                                        st.success(t("calibration_completed"))
                                        
                                        # CAMBIO: Traducir subheader de métricas
                                        st.subheader(t("calibration_metrics"))
                                        
                                        # Métricas (sin cambios en los valores, solo labels si es necesario)
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("MAE", f"{evaluation['MAE']:.2f} dB")
                                        with col2:
                                            st.metric("RMSE", f"{evaluation['RMSE']:.2f} dB")
                                        with col3:
                                            st.metric(t("max_error"), f"{evaluation['max_error']:.2f} dB")
                                    else:
                                        # CAMBIO: Traducir error
                                        st.error(f"{t('calibration_error')}: {calibration_result['error']}")
                                        return
                                else:
                                    # Evaluación estándar sin calibración
                                    # CAMBIO: Traducir info
                                    st.info(t("standard_prediction"))

                                    # Preparar entrada IGUAL que en el código de referencia (sin cambios)
                                    struct_array = np.array(st.session_state.uploaded_estructuras) / 255.0
                                    pixels_array = np.array(st.session_state.uploaded_mediciones) / 255.0
                                    antenna_array = np.array(st.session_state.uploaded_trasmisor) / 255.0

                                    # Asegurar dimensiones correctas (sin cambios)
                                    if len(struct_array.shape) == 3:
                                        struct_array = struct_array[:, :, 0] if struct_array.shape[2] == 1 else struct_array.mean(axis=2)
                                    if len(pixels_array.shape) == 3:
                                        pixels_array = pixels_array[:, :, 0] if pixels_array.shape[2] == 1 else pixels_array.mean(axis=2)
                                    if len(antenna_array.shape) == 3:
                                        antenna_array = antenna_array[:, :, 0] if antenna_array.shape[2] == 1 else antenna_array.mean(axis=2)

                                    # Añadir dimensión de canal (sin cambios)
                                    struct_array = np.expand_dims(struct_array, axis=-1)
                                    pixels_array = np.expand_dims(pixels_array, axis=-1)
                                    antenna_array = np.expand_dims(antenna_array, axis=-1)

                                    # Combinar en tensor de entrada (sin cambios)
                                    input_image = np.concatenate([struct_array, pixels_array, antenna_array], axis=-1)
                                    input_batch = np.expand_dims(input_image, axis=0)

                                    # Ejecutar predicción (sin cambios)
                                    prediction_norm = modelo.predict(input_batch)[0]

                                    # Manejar dimensiones de salida (sin cambios)
                                    if len(prediction_norm.shape) > 2:
                                        prediction_norm = prediction_norm[:, :, 0]

                                    # Desnormalizar CORRECTAMENTE (sin cambios)
                                    prediction_denorm = prediction_norm * (POWER_MAX - POWER_MIN) + POWER_MIN

                                    # Convertir a numpy (sin cambios)
                                    pred_np = prediction_denorm.numpy() if hasattr(prediction_denorm, 'numpy') else np.array(prediction_denorm)

                                    # CAMBIO: Traducir success message
                                    st.success(t("prediction_completed"))

                                    # Calcular métricas usando los puntos de medición (sin cambios hasta...)
                                    pixel_array = np.array(st.session_state.uploaded_mediciones)
                                    measurement_coords = np.where(pixel_array > 0)

                                    if len(measurement_coords[0]) > 0:
                                        # Extraer valores reales de las mediciones (sin cambios)
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
                                        
                                        # Convertir a arrays de NumPy (sin cambios)
                                        measured_values = np.array(measured_values)
                                        predicted_values = np.array(predicted_values)
                                        
                                        # Calcular métricas (sin cambios)
                                        errors = predicted_values - measured_values
                                        mae = np.mean(np.abs(errors))
                                        rmse = np.sqrt(np.mean(errors**2))
                                        max_error = np.max(np.abs(errors))
                                        avg_error = np.mean(errors)
                                        
                                        # CAMBIO: Traducir subheader
                                        st.subheader(t("evaluation_metrics"))
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("MAE", f"{mae:.2f} dB")
                                        with col2:
                                            st.metric("RMSE", f"{rmse:.2f} dB")
                                        with col3:
                                            st.metric(t("max_error"), f"{max_error:.2f} dB")
                                    else:
                                        # CAMBIO: Traducir warning
                                        st.warning(t("no_measurement_points"))

                                
                                # Mostrar métricas básicas de la predicción
                                st.subheader(t("prediction_metrics"))
                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric(t("max_power"), f"{np.max(pred_np):.2f} dBm")
                                with metrics_col2:
                                    st.metric(t("min_power"), f"{np.min(pred_np):.2f} dBm")
            
                                # CAMBIO: Traducir título de visualización
                                st.write(f"### {t('input_layers')}")
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                                # Obtener las capas de entrada para visualización (sin cambios)
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

                                # CAMBIO: Traducir títulos de los subplots
                                im1 = ax1.imshow(layer1, extent=[0, 50, 0, 50], cmap='viridis')
                                ax1.set_title(t("layer_structures"), pad=20)
                                plt.colorbar(im1, ax=ax1)

                                im2 = ax2.imshow(layer2, extent=[0, 50, 0, 50], cmap='viridis')
                                ax2.set_title(t("layer_measurements"), pad=20)
                                plt.colorbar(im2, ax=ax2)

                                im3 = ax3.imshow(layer3, extent=[0, 50, 0, 50], cmap='viridis')
                                ax3.set_title(t("layer_antenna"), pad=20)
                                plt.colorbar(im3, ax=ax3)

                                # CAMBIO: Traducir labels de ejes
                                for ax in [ax1, ax2, ax3]:
                                    ax.grid(True, linestyle='--', alpha=0.3)
                                    ax.set_xlabel(t("distance_m"))
                                    ax.set_ylabel(t("distance_m"))

                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # CAMBIO: Traducir subheader de resultado
                                st.subheader(t("prediction_result"))

                                # Calcular rango dinámico real de los datos (sin cambios hasta...)
                                pred_min = np.min(pred_np)
                                pred_max = np.max(pred_np)

                                # Solo usar valores que no sean del fondo (sin cambios)
                                threshold = POWER_MIN + 10
                                mask = pred_np > threshold
                                valid_data = pred_np[mask]

                                if len(valid_data) > 100:
                                    data_min = np.min(valid_data)
                                    data_max = np.max(valid_data)
                                    
                                    buffer_range = (data_max - data_min) * 0.05
                                    vmin_plot = max(data_min - buffer_range, POWER_MIN)
                                    vmax_plot = data_max + buffer_range
                                else:
                                    vmin_plot = pred_min
                                    vmax_plot = pred_max

                                # CAMBIO: Traducir título del gráfico
                                fig, ax = plt.subplots(figsize=(8, 7))
                                im = ax.imshow(pred_np, extent=[0, 50, 0, 50], cmap='viridis',
                                            vmin=vmin_plot, vmax=vmax_plot)

                                ax.set_title(t("predicted_power_map"), pad=20)
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label(t("power_dbm"))
                                ax.grid(True, linestyle='--', alpha=0.3)
                                ax.set_xlabel(t("distance_m"))
                                ax.set_ylabel(t("distance_m"))

                                # CAMBIO: Traducir info del rango
                                st.info(t("visualization_range", vmin=vmin_plot, vmax=vmax_plot))

                                plt.tight_layout()
                                
                                # Guardar la figura para descargarla (sin cambios)
                                fig_result = fig
                                
                                st.pyplot(fig, use_container_width=False)
                                                            
                                # Preparar imagen para descargar (sin cambios hasta...)
                                buffer = io.BytesIO()
                                fig_result.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
                                buffer.seek(0)
                                
                                # Generar nombre de archivo con timestamp (sin cambios)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"mapa_potencia_{timestamp}.png"
                                
                                # CAMBIO: Traducir botón de descarga
                                st.download_button(
                                    label=t("download_prediction_image"),
                                    data=buffer,
                                    file_name=filename,
                                    mime="image/png",
                                    key="download_button"
                                )
                                plt.close(fig)
                                
                                # Histograma de valores predichos
                                st.subheader(t("power_distribution"))
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Crear máscara para ignorar valores de fondo (sin cambios)
                                mask_np = pred_np > (POWER_MIN + 1)
                                prediction_masked = pred_np[mask_np]
                                
                                if len(prediction_masked) > 10:
                                    ax.hist(prediction_masked, bins=50, alpha=0.7, color='blue', label=t("prediction"))
                                    ax.set_xlabel(t("power_dbm"))
                                    ax.set_ylabel(t("frequency"))
                                    ax.set_title(t("predicted_power_distribution"))
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    # CAMBIO: Traducir info
                                    st.info(t("insufficient_data_histogram"))
                                plt.close(fig)

                                # Botón de descarga para la imagen de distribución (sin cambios hasta...)
                                buffer_dist = io.BytesIO()
                                fig.savefig(buffer_dist, format='png', bbox_inches='tight', dpi=300)
                                buffer_dist.seek(0)

                                timestamp_dist = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename_dist = f"distribucion_potencia_{timestamp_dist}.png"

                                # CAMBIO: Traducir botón de descarga
                                st.download_button(
                                    label=t("download_distribution_image"),
                                    data=buffer_dist,
                                    file_name=filename_dist,
                                    mime="image/png",
                                    key="download_dist_button"
                                )

                                # CAMBIO: Traducir subheader de segmentación
                                st.subheader(t("power_zone_segmentation"))
                                try:
                                    # Detectar posición de la antena (sin cambios)
                                    if enable_calibration and calibration_result['success']:
                                        antenna_pos_px = find_antenna_position(input_image[:, :, 2])
                                    else:
                                        antenna_pos_px = find_antenna_position(antenna_array)
                                    
                                    # Crear máscara de segmentación (sin cambios)
                                    pred_mask, pred_r1, pred_r2 = create_segmentation_mask(pred_np, antenna_pos_px)

                                    # Crear visualización EXACTA (sin cambios hasta el título)
                                    fig_seg, ax_seg = plt.subplots(figsize=(8, 7))
                                    ax_seg.imshow(pred_mask)

                                    # Marcar la antena (sin cambios)
                                    ax_seg.plot(antenna_pos_px[0], antenna_pos_px[1], 'w+', markersize=15, markeredgewidth=3)

                                    # Círculos concéntricos (sin cambios)
                                    if pred_r1 > 0:
                                        circle1 = plt.Circle(antenna_pos_px, pred_r1, fill=False, color='white', linewidth=2, linestyle='-')
                                        ax_seg.add_artist(circle1)

                                    if pred_r2 > pred_r1:
                                        circle2 = plt.Circle(antenna_pos_px, pred_r2, fill=False, color='white', linewidth=2, linestyle='--')
                                        ax_seg.add_artist(circle2)

                                    # CAMBIO: Traducir labels de ejes y título
                                    ax_seg.set_xlabel(t("pixels_x"))
                                    ax_seg.set_ylabel(t("pixels_y"))
                                    # Establecer ticks del eje Y para que coincidan con eje X (0, 50, 100, 150, 200, 250)
                                    ax_seg.set_yticks([256, 206, 156, 106, 56, 6])
                                    ax_seg.set_yticklabels(['0', '50', '100', '150', '200', '250'])
                                    ax_seg.grid(True, linestyle='--', alpha=0.3)
                                    ax_seg.set_title(t("power_zone_segmentation"), pad=20)
            
                                    # CAMBIO: Traducir leyenda
                                    from matplotlib.patches import Patch
                                    legend_elements = [
                                        Patch(facecolor='red', label=t("zone_red")),
                                        Patch(facecolor='yellow', label=t("zone_yellow")),
                                        Patch(facecolor='green', label=t("zone_green"))
                                    ]
                                    ax_seg.legend(handles=legend_elements, loc='upper right')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_seg, use_container_width=False)
                                    
                                    # Calcular posición tomando como origen la esquina INFERIOR IZQUIERDA
                                    # En la gráfica: eje X horizontal (columnas), eje Y vertical (filas)
                                    # antenna_pos_px[0] = fila (coordenada Y en sistema original desde arriba)
                                    # antenna_pos_px[1] = columna (coordenada X)
                                    # Para mostrar con origen inferior izquierda:
                                    antenna_x_pixels = antenna_pos_px[0]  # Esto es lo que se ve en el eje X de la gráfica
                                    antenna_y_pixels = 256 - antenna_pos_px[1]  # Invertir para origen inferior

                                    # Convertir a metros
                                    antenna_x_meters = antenna_x_pixels * 50/256
                                    antenna_y_meters = antenna_y_pixels * 50/256

                                    # CAMBIO: Traducir info de segmentación
                                    st.info(t("segmentation_info",
                                            antenna_pos_m=f"({antenna_x_meters:.2f}, {antenna_y_meters:.2f})",
                                            antenna_pos_px=f"({antenna_x_pixels:.1f}, {antenna_y_pixels:.1f})",
                                            red_radius=f"{pred_r1 * 50/256:.2f}",
                                            yellow_radius=f"{pred_r2 * 50/256:.2f}"))

                                    # Botón de descarga (sin cambios hasta...)
                                    buffer_seg = io.BytesIO()
                                    fig_seg.savefig(buffer_seg, format='png', bbox_inches='tight', dpi=300)
                                    buffer_seg.seek(0)
                                    
                                    timestamp_seg = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename_seg = f"segmentacion_potencia_{timestamp_seg}.png"
                                    
                                    # CAMBIO: Traducir botón de descarga
                                    st.download_button(
                                        label=t("download_segmentation_image"),
                                        data=buffer_seg,
                                        file_name=filename_seg,
                                        mime="image/png",
                                        key="download_seg_button"
                                    )
                                    
                                    plt.close(fig_seg)
                                    
                                except Exception as e:
                                    # CAMBIO: Traducir error
                                    st.error(f"{t('segmentation_error')}: {str(e)}")
                                    st.exception(e)
                                                                
                            else:
                                # CAMBIO: Traducir error de carga
                                st.error(t("model_load_error"))
                        except Exception as e:
                            # CAMBIO: Traducir error de evaluación
                            st.error(f"{t('evaluation_error')}: {str(e)}")
                            st.exception(e)
                        
            # REEMPLAZA la función main() completa dentro de scenarios con esto:

            def main():
                # Inicializar el tab activo si no existe
                if 'active_scenario_tab' not in st.session_state:
                    st.session_state.active_scenario_tab = 0
                
                # CSS para ocultar cajas de botones SOLO en scenarios
                st.markdown("""
                <style>
                /* Aplicar solo después de esta clase única */
                div.scenario-tabs-section div[data-testid="stButton"] button {
                    background-color: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                }
                
                div.scenario-tabs-section div[data-testid="stButton"] button:hover {
                    background-color: transparent !important;
                    border: none !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Crear contenedor con clase única
                st.markdown('<div class="scenario-tabs-section">', unsafe_allow_html=True)
                
                # Crear opciones de tabs
                tab_options = [
                    t("tab_structures"),
                    t("tab_antenna"),
                    t("tab_measurements"),
                    t("tab_evaluation")
                ]
                
                # Crear contenedor de tabs horizontal
                cols = st.columns(len(tab_options))
                
                for idx, (col, title) in enumerate(zip(cols, tab_options)):
                    with col:
                        if st.session_state.active_scenario_tab == idx:
                            # Mostrar como tab activo (naranja con línea debajo)
                            st.markdown(f"""
                            <div style="text-align: center; border-bottom: 3px solid #FF6B35; padding: 0.5rem 1rem;">
                                <span style="color: #FF6B35; font-weight: 600; cursor: pointer;">{title}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Mostrar como tab inactivo
                            if st.button(title, key=f"scenario_tab_{idx}", use_container_width=True):
                                st.session_state.active_scenario_tab = idx
                                st.rerun()
                
                # Línea separadora
                st.markdown("""
                <div style="border-bottom: 1px solid rgba(49, 51, 63, 0.2); margin-bottom: 1rem;"></div>
                """, unsafe_allow_html=True)
                
                # Mostrar contenido según tab seleccionado
                if st.session_state.active_scenario_tab == 0:
                    create_building_interface()
                elif st.session_state.active_scenario_tab == 1:
                    create_point_interface()
                elif st.session_state.active_scenario_tab == 2:
                    create_pixel_selector()
                elif st.session_state.active_scenario_tab == 3:
                    add_model_evaluation_section()
                
                # Cerrar contenedor
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Mostrar imágenes guardadas en la barra lateral
                show_saved_images()

            if __name__ == "__main__":
                main()

if __name__ == "__main__":
    import time
    main()