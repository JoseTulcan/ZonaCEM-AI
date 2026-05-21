from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import io, os, base64, math, requests, tempfile
from scipy import stats
from urllib.parse import quote
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

app = FastAPI(title="ZonaCEM AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constantes globales ────────────────────────────────────────────────────
POWER_MIN, POWER_MAX = -100, 40
ACTUAL_POWER_MAX = -10
STRUCT_MAX = 20
ANTENNA_MAX = 40
IMG_SIZE = (256, 256)
AREA_SIZE = (50, 50)

GITLAB_BASE = "https://gitlab.com/tulcanjose0/zonacem-ai/-/raw/main/"

MODELOS = {
    "195": "modelos/Modelo_1.95_GHz.keras",
    "213": "modelos/Modelo_2.13_GHz.keras",
    "265": "modelos/Modelo_2.65_GHz.keras",
}

DATASET_PATHS = {
    "195": "datasets/1.95_GHz_dataset",
    "213": "datasets/2.13_GHz_dataset",
    "265": "datasets/2.65_GHz_dataset",
}

# Cache de modelos en memoria
_model_cache: dict = {}

# ─── Helpers de geometría ────────────────────────────────────────────────────
def pixels_to_meters(px, py, img_size=IMG_SIZE, area_size=AREA_SIZE):
    cx, cy = img_size[0] // 2, img_size[1] // 2
    x = (px - cx) / (img_size[0] / area_size[0])
    y = (cy - py) / (img_size[1] / area_size[1])
    return (x, y)

def meters_to_pixels(coords, img_size=IMG_SIZE, area_size=AREA_SIZE):
    cx, cy = img_size[0] // 2, img_size[1] // 2
    px = int(cx + coords[0] * (img_size[0] / area_size[0]))
    py = int(cy - coords[1] * (img_size[1] / area_size[1]))
    return (px, py)

def calc_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calculate_intensity(value, min_val, max_val):
    return int(((value - min_val) / (max_val - min_val)) * 255)

# ─── Helpers de imagen ───────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def get_gitlab_image(path: str) -> bytes:
    url = GITLAB_BASE + quote(path)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

# ─── Carga de modelos ────────────────────────────────────────────────────────
def load_model(freq_key: str) -> tf.keras.Model:
    if freq_key in _model_cache:
        return _model_cache[freq_key]
    path = MODELOS.get(freq_key)
    if not path:
        raise HTTPException(400, f"Frecuencia inválida: {freq_key}")
    url = GITLAB_BASE + quote(path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as f:
        content = requests.get(url, timeout=120).content
        f.write(content)
        tmp = f.name
    model = tf.keras.models.load_model(tmp)
    os.unlink(tmp)
    _model_cache[freq_key] = model
    return model

# ─── Calibración adaptativa ──────────────────────────────────────────────────
def extract_points_from_array(pixels_array):
    if len(pixels_array.shape) == 3:
        pixels_array = pixels_array[:, :, 0]
    if pixels_array.max() <= 1.0:
        pixels_array = pixels_array * 255.0
    non_zero = pixels_array > 0
    y_coords, x_coords = np.where(non_zero)
    points, powers = {}, {}
    for i, (xp, yp) in enumerate(zip(x_coords, y_coords)):
        xm, ym = pixels_to_meters(xp, yp)
        pv = pixels_array[yp, xp]
        power = (pv / 255.0) * 140.0 - 100.0
        pid = f"P{i+1:02d}"
        points[pid] = (xm, ym)
        powers[pid] = power
    return points, powers

def find_antenna_pos(antenna_arr, threshold=0.5):
    if len(antenna_arr.shape) == 3:
        antenna_arr = antenna_arr[:, :, 0]
    if antenna_arr.max() > 1.0:
        antenna_arr = antenna_arr / 255.0
    yi, xi = np.where(antenna_arr > threshold)
    if len(xi) > 0:
        return (np.mean(xi), np.mean(yi))
    return (antenna_arr.shape[1]//2, antenna_arr.shape[0]//2)

def apply_log_distance(x, y, ant_pos, P0, n):
    d = calc_distance((x, y), ant_pos)
    d = max(d, 0.1)
    return P0 - 10 * n * math.log10(d)

def calibrate_model(model, input_image, points, powers, ant_pos_m):
    pred_norm = model.predict(np.expand_dims(input_image, axis=0))[0]
    if pred_norm.shape[-1] > 1:
        pred_norm = pred_norm[:, :, 0]
    else:
        pred_norm = pred_norm.squeeze()
    pred = pred_norm * (POWER_MAX - POWER_MIN) + POWER_MIN

    distances, real_p, pred_p = [], [], []
    for pid, (x, y) in points.items():
        d = calc_distance((x, y), ant_pos_m)
        distances.append(d)
        real_p.append(powers[pid])
        px_, py_ = meters_to_pixels((x, y))
        if 0 <= px_ < pred.shape[1] and 0 <= py_ < pred.shape[0]:
            pred_p.append(pred[py_, px_])

    log_d = np.log10(distances)
    slope, intercept, r, _, _ = stats.linregress(log_d, real_p)
    n_pl = -slope / 10
    P0_ref = intercept

    power_map = np.zeros(IMG_SIZE)
    for py_ in range(IMG_SIZE[0]):
        for px_ in range(IMG_SIZE[1]):
            pos = pixels_to_meters(px_, py_)
            power_map[py_, px_] = apply_log_distance(pos[0], pos[1], ant_pos_m, P0_ref, n_pl)

    pv_all, lv_all = [], []
    for pid, (x, y) in points.items():
        px_, py_ = meters_to_pixels((x, y))
        if 0 <= px_ < pred.shape[1] and 0 <= py_ < pred.shape[0]:
            pv_all.append(pred[py_, px_])
            lv_all.append(power_map[py_, px_])

    cs, ci, _, _, _ = stats.linregress(pv_all, lv_all)
    t_min = min(powers.values())
    t_max = max(powers.values())
    calib_full = cs * pred + ci
    c_min, c_max = np.min(calib_full), np.max(calib_full)
    scale = (t_max - t_min) / (c_max - c_min) if c_max != c_min else 1
    offset = t_min - scale * c_min

    inputs = tf.keras.Input(shape=model.input_shape[1:])
    base_out = model(inputs)
    if isinstance(base_out, list): base_out = base_out[0]
    denorm = base_out * (POWER_MAX - POWER_MIN) + POWER_MIN
    cal = denorm * cs + ci
    adj = cal * scale + offset
    norm_out = (adj - POWER_MIN) / (POWER_MAX - POWER_MIN)
    calib_model = tf.keras.Model(inputs=inputs, outputs=norm_out)
    calib_model.compile(optimizer='adam', loss='mse')
    return calib_model, r**2

def create_segmentation_mask(power_map, antenna_pos):
    height, width = power_map.shape
    mask = np.zeros((height, width, 3))
    ax, ay = antenna_pos
    y_c, x_c = np.ogrid[:height, :width]
    HIGH_THR, MID_THR = 19.0, 12.0
    hi_mask = power_map > HIGH_THR
    mid_mask = power_map > MID_THR
    dist = np.sqrt((x_c - ax)**2 + (y_c - ay)**2)

    r1 = np.max(np.sqrt((np.where(hi_mask)[1]-ax)**2+(np.where(hi_mask)[0]-ay)**2)) if np.any(hi_mask) else 0
    r2_pts = np.where(mid_mask)
    r2 = np.max(np.sqrt((r2_pts[1]-ax)**2+(r2_pts[0]-ay)**2)) if np.any(mid_mask) else r1
    r2 = max(r2, r1)

    if r1 > 0: mask[dist <= r1] = [1, 0, 0]
    if r2 > r1: mask[(dist > r1) & (dist <= r2)] = [1, 1, 0]
    mask[dist > r2] = [0, 1, 0]
    return mask, r1, r2

# ─── Generación de imágenes desde datos del canvas ──────────────────────────
def build_structure_image(structures: list) -> Image.Image:
    img = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(img)
    for s in structures:
        intensity = calculate_intensity(s['height'], 3, 20)
        if s['type'] == 'line':
            draw.line(s['coords'], fill=intensity, width=2)
        elif s['type'] == 'rect':
            x0, y0 = s['coords'][0]
            x1, y1 = s['coords'][1]
            draw.rectangle([(min(x0,x1), min(y0,y1)), (max(x0,x1), max(y0,y1))],
                           outline=intensity, width=2)
    return img

def build_antenna_image(ax: float, ay: float, height: float) -> Image.Image:
    img = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(img)
    intensity = calculate_intensity(height, 10, 40)
    draw.ellipse([(ax-3, ay-3), (ax+3, ay+3)], fill=intensity)
    return img

def build_pixels_image(measurements: list) -> Image.Image:
    img = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(img)
    for m in measurements:
        x, y, dbm = m['x'], m['y'], m['dbm']
        intensity = calculate_intensity(dbm, -100, 40)
        draw.point((x, y), fill=intensity)
    return img

def arr_from_pil(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert('L'), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(_model_cache.keys())}

@app.get("/api/dataset/preview")
def dataset_preview(freq: str, folder: str, num: int = 30):
    """Devuelve hasta `num` imágenes de una carpeta del dataset."""
    base = DATASET_PATHS.get(freq)
    if not base:
        raise HTTPException(400, "Frecuencia inválida")
    images = []
    for i in range(1, num + 1):
        fname = f"image_{i:03d}.PNG"
        path = f"{base}/{folder}/{fname}"
        try:
            data = get_gitlab_image(path)
            img = Image.open(io.BytesIO(data)).convert('L')
            images.append({
                "name": fname,
                "b64": pil_to_b64(img),
                "index": i
            })
        except Exception:
            continue
    return {"images": images, "total": len(images)}

class LoadModelRequest(BaseModel):
    freq: str  # "195" | "213" | "265"

@app.post("/api/model/load")
def load_model_endpoint(req: LoadModelRequest):
    """Carga un modelo en memoria (o confirma que ya está cargado)."""
    try:
        model = load_model(req.freq)
        return {
            "success": True,
            "freq": req.freq,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

class EvaluateDatasetRequest(BaseModel):
    model_freq: str
    dataset_freq: str
    indices: List[int]

@app.post("/api/model/evaluate-dataset")
def evaluate_dataset(req: EvaluateDatasetRequest):
    """Evalúa el modelo sobre escenarios del dataset de GitLab."""
    model = load_model(req.model_freq)
    base = DATASET_PATHS.get(req.dataset_freq)
    if not base:
        raise HTTPException(400, "Frecuencia de dataset inválida")

    results = []
    for idx in req.indices[:5]:  # Máximo 5 para no agotar timeout
        try:
            fname = f"image_{idx+1:03d}.PNG"
            struct_img = np.array(Image.open(io.BytesIO(get_gitlab_image(f"{base}/structures/{fname}"))).convert('L'), dtype=np.float32)
            pixel_img  = np.array(Image.open(io.BytesIO(get_gitlab_image(f"{base}/selected_pixels/{fname}"))).convert('L'), dtype=np.float32)
            ant_img    = np.array(Image.open(io.BytesIO(get_gitlab_image(f"{base}/antenna_position/{fname}"))).convert('L'), dtype=np.float32)
            power_img  = np.array(Image.open(io.BytesIO(get_gitlab_image(f"{base}/combined_power/{fname}"))).convert('L'), dtype=np.float32)

            sn = struct_img / 255.0
            pn = (pixel_img / 255.0) * (ACTUAL_POWER_MAX - POWER_MIN) / (POWER_MAX - POWER_MIN)
            an = ant_img / 255.0
            power_n = (power_img / 255.0) * (ACTUAL_POWER_MAX - POWER_MIN) / (POWER_MAX - POWER_MIN)

            combined = np.stack([sn, pn, an], axis=-1)
            pred_n = model.predict(np.expand_dims(combined, axis=0))[0].squeeze()
            pred_db = pred_n * (POWER_MAX - POWER_MIN) + POWER_MIN
            power_db = power_n * (POWER_MAX - POWER_MIN) + POWER_MIN

            mask = power_img > 0
            if np.sum(mask) > 0:
                mae = float(np.mean(np.abs(pred_db[mask] - power_db[mask])))
                rmse = float(np.sqrt(np.mean((pred_db[mask] - power_db[mask])**2)))
                ss_res = np.sum((power_db[mask] - pred_db[mask])**2)
                ss_tot = np.sum((power_db[mask] - np.mean(power_db[mask]))**2)
                r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
            else:
                mae, rmse, r2 = 0.0, 0.0, 0.0

            # Generar figura comparativa
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor('#0f0f1a')
            for ax in axes:
                ax.set_facecolor('#0f0f1a')
                ax.tick_params(colors='#aaa')
                ax.xaxis.label.set_color('#aaa')
                ax.yaxis.label.set_color('#aaa')
                ax.title.set_color('#fff')

            im1 = axes[0].imshow(power_db, extent=[0,50,0,50], cmap='plasma')
            axes[0].set_title(f'Real Power (dBm) — #{idx+1}')
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(pred_db, extent=[0,50,0,50], cmap='plasma')
            axes[1].set_title(f'Predicted Power (dBm)')
            plt.colorbar(im2, ax=axes[1])

            plt.tight_layout()
            chart_b64 = fig_to_b64(fig)

            results.append({
                "index": idx + 1,
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 4),
                "chart": chart_b64,
            })
        except Exception as e:
            results.append({"index": idx+1, "error": str(e)})

    return {"results": results}

class ScenarioRequest(BaseModel):
    model_freq: str
    structures: List[dict]        # [{type, coords, height}]
    antenna: dict                 # {x, y, height}  — coords en píxeles
    measurements: List[dict]      # [{x, y, dbm}]   — coords en píxeles
    use_calibration: bool = True

@app.post("/api/scenario/predict")
def predict_scenario(req: ScenarioRequest):
    """Predice el mapa de potencia para un escenario nuevo."""
    model = load_model(req.model_freq)

    # Construir imágenes de entrada
    struct_pil  = build_structure_image(req.structures)
    ant_pil     = build_antenna_image(req.antenna['x'], req.antenna['y'], req.antenna['height'])
    pixels_pil  = build_pixels_image(req.measurements)

    struct_arr  = arr_from_pil(struct_pil)
    ant_arr     = arr_from_pil(ant_pil)
    pixels_arr  = arr_from_pil(pixels_pil)

    input_image = np.concatenate([struct_arr, pixels_arr, ant_arr], axis=-1)

    # Calibración adaptativa
    if req.use_calibration and req.measurements:
        points, powers = extract_points_from_array(pixels_arr)
        ant_pos_m = pixels_to_meters(req.antenna['x'], req.antenna['y'])
        if points:
            try:
                model, _ = calibrate_model(model, input_image, points, powers, ant_pos_m)
            except Exception:
                pass  # Si falla la calibración, usar modelo base

    # Predicción
    pred_norm = model.predict(np.expand_dims(input_image, axis=0))[0].squeeze()
    pred_db   = pred_norm * (POWER_MAX - POWER_MIN) + POWER_MIN

    # Posición antena en píxeles
    ant_pos_px = (req.antenna['x'], req.antenna['y'])

    # Máscara de segmentación
    mask, r1, r2 = create_segmentation_mask(pred_db, ant_pos_px)

    # Figura: mapa de potencia
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    fig1.patch.set_facecolor('#0f0f1a')
    ax1.set_facecolor('#0f0f1a')
    ax1.tick_params(colors='#aaa')
    ax1.xaxis.label.set_color('#aaa')
    ax1.yaxis.label.set_color('#aaa')
    ax1.title.set_color('#fff')
    im = ax1.imshow(pred_db, extent=[0, 50, 0, 50], cmap='plasma',
                    vmin=POWER_MIN, vmax=ACTUAL_POWER_MAX)
    ax1.plot(req.antenna['x'] * 50/256, (256 - req.antenna['y']) * 50/256,
             'w+', markersize=12, markeredgewidth=2)
    plt.colorbar(im, ax=ax1, label='Power (dBm)')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Predicted Power Map')
    ax1.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    power_map_b64 = fig_to_b64(fig1)

    # Figura: segmentación
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.patch.set_facecolor('#0f0f1a')
    ax2.set_facecolor('#0f0f1a')
    ax2.tick_params(colors='#aaa')
    ax2.title.set_color('#fff')
    ax2.imshow(mask)
    ax2.plot(ant_pos_px[0], ant_pos_px[1], 'w+', markersize=15, markeredgewidth=3)
    if r1 > 0:
        ax2.add_artist(plt.Circle(ant_pos_px, r1, fill=False, color='white', lw=2))
    if r2 > r1:
        ax2.add_artist(plt.Circle(ant_pos_px, r2, fill=False, color='white', lw=2, ls='--'))
    legend_elements = [
        Patch(facecolor='red',    label='High Exposure (>19 dBm)'),
        Patch(facecolor='yellow', label='Medium Exposure (12–19 dBm)'),
        Patch(facecolor='green',  label='Low Exposure (<12 dBm)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    ax2.set_title('Power Zone Segmentation')
    ax2.set_yticks([256, 206, 156, 106, 56, 6])
    ax2.set_yticklabels(['0','50','100','150','200','250'])
    ax2.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    seg_b64 = fig_to_b64(fig2)

    ant_x_m = req.antenna['x'] * 50 / 256
    ant_y_m = (256 - req.antenna['y']) * 50 / 256

    return {
        "success": True,
        "power_map": power_map_b64,
        "segmentation": seg_b64,
        "antenna_pos_m": {"x": round(ant_x_m, 2), "y": round(ant_y_m, 2)},
        "red_radius_m": round(r1 * 50 / 256, 2),
        "yellow_radius_m": round(r2 * 50 / 256, 2),
    }
