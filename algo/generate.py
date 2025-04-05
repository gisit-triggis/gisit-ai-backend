import base64
import json
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping, shape, LineString, Point
from shapely.ops import transform as shapely_transform
import requests
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import os
from skimage import morphology
import cv2
from skimage.graph import route_through_array
import pyproj
import time
import traceback
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID", "e04c0b5c-1411-481a-a893-88a379dc0f7b")
CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET", "yy0W4mgn0m46e4NK6inqg8N9aE5fjZlx")
OUTPUT_DIR_ALGO = "output_algo_temp"
os.makedirs(OUTPUT_DIR_ALGO, exist_ok=True)
IMAGE_SIZE = 1024

KM_PER_DEGREE_APPROX = 111.32
FUEL_COST_PER_KM_DIFFICULTY_UNIT = 0.05
CONSTRUCTION_COST_PER_KM_DIFFICULTY_UNIT = 5.0
BASE_FUEL_CONSUMPTION_FACTOR = 1.0
BASE_CONSTRUCTION_COST_FACTOR = 1.0

image_processor = SegformerImageProcessor.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
model = SegformerForSemanticSegmentation.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
ROAD_CLASS_IDS = {1, 3, 4, 5, 6, 7}

def get_access_token():
    url = "https://services.sentinel-hub.com/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting access token: {e}")
        raise Exception(f"Ошибка получения токена: {e}")
    except KeyError:
        print(f"Error parsing token response: {response.text}")
        raise Exception("Ошибка парсинга ответа токена")

def get_satellite_image(geometry, image_type="rgb", date_from="2024-06-01", date_to="2024-08-01"):
    print(f"Requesting {image_type} image...")
    access_token = get_access_token()
    url = "https://services.sentinel-hub.com/api/v1/process"

    if image_type == "rgb":
        evalscript = """
            //VERSION=3
            function setup() { return { input: ["B04", "B03", "B02", "dataMask"], output: { bands: 4 } }; }
            function evaluatePixel(sample) {
              let factor = 2.5;
              return [ sample.B04 * factor, sample.B03 * factor, sample.B02 * factor, sample.dataMask ];
            }"""
        output_format = "image/png"
        accept_header = "image/png"
        data_source_type = "sentinel-2-l2a"
    elif image_type == "ndwi":
         evalscript = """
            //VERSION=3
            function setup() { return { input: ["B03", "B08", "dataMask"], output: { bands: 1, sampleType: "FLOAT32" } }; }
            function evaluatePixel(sample) {
              if (sample.dataMask == 0) { return [-9999]; }
              let denom = sample.B03 + sample.B08;
              if (denom == 0) { return [0];}
              let ndwi = (sample.B03 - sample.B08) / denom;
              return [ndwi];
            }"""
         output_format = "image/tiff"
         accept_header = "image/tiff"
         data_source_type = "sentinel-2-l2a"
    elif image_type == "dem":
        evalscript = """
            //VERSION=3
            function setup() { return { input: ["DEM"], output: { bands: 1, sampleType: "FLOAT32" } }; }
            function evaluatePixel(sample) { return [sample.DEM]; }
            """
        output_format = "image/tiff"
        accept_header = "image/tiff"
        data_source_type = "dem"
    else:
        raise ValueError(f"Неизвестный тип изображения: {image_type}")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": accept_header
    }

    if data_source_type == "sentinel-2-l2a":
        data_source = {
            "type": "sentinel-2-l2a",
            "dataFilter": {
                "timeRange": {
                    "from": f"{date_from}T00:00:00Z",
                    "to": f"{date_to}T23:59:59Z"
                },
                "maxCloudCoverage": 15,
            },
             "processing": {
                 "upsampling": "BICUBIC",
                 "downsampling": "BICUBIC"
             }
        }
    elif data_source_type == "dem":
         data_source = {"type": "dem", "dataFilter": {"demInstance": "COPERNICUS_30"}}
    else:
         # Should not happen based on image_type check, but added for safety
         raise ValueError(f"Неизвестный тип источника данных: {data_source_type}")


    payload = {
        "input": {
            "bounds": { "geometry": geometry },
            "data": [data_source]
        },
        "output": {
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "responses": [{"identifier": "default", "format": {"type": output_format}}]
        },
        "evalscript": evalscript
    }

    img_bytes = None # Initialize img_bytes
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        img_bytes = response.content
        if not img_bytes:
             raise Exception(f"Empty response content received for {image_type}.")

        if output_format == "image/tiff":
             with BytesIO(img_bytes) as memfile:
                 # Ensure rasterio doesn't keep the file open unnecessarily
                 with rasterio.open(memfile) as src:
                    data = src.read(1)
                    profile = src.profile.copy() # Make a copy of the profile
                    return data, profile
        else: # Assumes image/png
             with BytesIO(img_bytes) as memfile:
                 with Image.open(memfile) as img:
                     img.load()
                     return img.copy(), None

    except requests.exceptions.Timeout:
         print(f"Timeout error requesting {image_type} image.")
         raise Exception(f"Таймаут при получении снимка ({image_type})")
    except requests.exceptions.HTTPError as e:
         print(f"HTTP error requesting {image_type} image: {e.response.status_code}")
         print(f"Response body: {e.response.text}") # Log the actual error message from API
         raise Exception(f"Ошибка HTTP ({e.response.status_code}) при получении снимка ({image_type}): {e.response.text}")
    except requests.exceptions.RequestException as e:
         print(f"General request error for {image_type}: {e}")
         raise Exception(f"Ошибка запроса при получении снимка ({image_type}): {e}")
    except Exception as e:
         print(f"Error processing {image_type} image response: {e}")
         traceback.print_exc()
         if img_bytes is not None:
            print(f"Received {len(img_bytes)} bytes.")
            try:
                err_filename = os.path.join(OUTPUT_DIR_ALGO, f"error_{image_type}_response.{output_format.split('/')[-1]}")
                with open(err_filename, "wb") as f_err:
                    f_err.write(img_bytes)
                print(f"Raw response saved to {err_filename}")
            except Exception as save_err:
                print(f"Could not save raw error response: {save_err}")
         else:
             print("No response bytes received.")
         raise Exception(f"Ошибка обработки ответа снимка ({image_type}): {e}")

def detect_water(ndwi_array):
    print("Анализ водных объектов по NDWI...")
    if ndwi_array is None or ndwi_array.size == 0:
        print("ВНИМАНИЕ: Получен пустой NDWI массив.")
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    ndwi_threshold = 0.25
    print(f"Используется порог NDWI: {ndwi_threshold}")

    water_mask = np.where((ndwi_array > ndwi_threshold) & (ndwi_array > -9000), 1, 0).astype(np.uint8) * 255

    print(f"Создана маска воды из NDWI с порогом {ndwi_threshold}.")
    water_pixel_count = np.sum(water_mask > 0)
    if water_pixel_count == 0:
        print(f"ВНИМАНИЕ: Вода не найдена с порогом {ndwi_threshold}.")
    else:
        print(f"Найдено пикселей воды (до морфологии): {water_pixel_count}")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        water_mask_opened = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        water_mask_closed = cv2.morphologyEx(water_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        print(f"Пикселей воды после морфологии (ядро {kernel_size}x{kernel_size}): {np.sum(water_mask_closed > 0)}")
        water_mask = water_mask_closed

    #cv2.imwrite(os.path.join(OUTPUT_DIR_ALGO, "water_mask_from_ndwi.png"), water_mask)
    return water_mask

def detect_slopes(dem_array):
    print("Анализ уклонов по DEM...")
    if dem_array is None or dem_array.size == 0 or np.all(dem_array <= -9000):
        print("ВНИМАНИЕ: Получен пустой или некорректный DEM массив.")
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    nodata_val = -9999
    valid_dem = dem_array[dem_array > nodata_val]
    if valid_dem.size > 0:
        mean_elev = np.mean(valid_dem)
        dem_array[dem_array <= nodata_val] = mean_elev
    else:
        print("ВНИМАНИЕ: DEM массив не содержит валидных данных о высоте.")
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    sobelx = cv2.Sobel(dem_array, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(dem_array, cv2.CV_64F, 0, 1, ksize=3)
    slope_magnitude = np.sqrt(sobelx**2 + sobely**2)

    min_slope, max_slope = np.min(slope_magnitude), np.max(slope_magnitude)
    if max_slope > min_slope:
        slope_normalized = (slope_magnitude - min_slope) / (max_slope - min_slope)
    else:
        slope_normalized = np.zeros_like(slope_magnitude)

    print(f"Карта уклонов создана. Диапазон нормализованных значений: {np.min(slope_normalized):.2f} - {np.max(slope_normalized):.2f}")
    #plt.imsave(os.path.join(OUTPUT_DIR_ALGO, "slope_map_normalized.png"), slope_normalized, cmap='viridis')
    return slope_normalized.astype(np.float32)

def detect_roads(rgb_image):
    print("Детекция дорог с помощью SegFormer-B4...")

    if rgb_image is None:
        print("ВНИМАНИЕ: Нет RGB изображения для детекции дорог.")
        return np.zeros((1024, 1024), dtype=np.uint8)

    try:
        if rgb_image.mode != 'RGB':
            rgb_image = rgb_image.convert('RGB')

        orig_width, orig_height = rgb_image.size

        inputs = image_processor(images=rgb_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            upsampled_logits = F.interpolate(logits, size=(orig_height, orig_width), mode="bilinear", align_corners=False)
            pred_mask = torch.argmax(upsampled_logits, dim=1)[0]

        pred_np = pred_mask.cpu().numpy()
        road_mask = np.isin(pred_np, list(ROAD_CLASS_IDS)).astype(np.uint8) * 255

        print("Дороги успешно детектированы.")
        return road_mask

    except Exception as e:
        print(f"Ошибка при детекции дорог: {e}")
        return np.zeros((1024, 1024), dtype=np.uint8)

def create_weight_map(water_mask, slope_map, road_mask,
                      water_cost_factor=20.0,
                      slope_cost_factor=5.0,
                      road_cost_factor=0.1):
    print("Создание карты весов (стоимости) для маршрутизации...")

    water_mask_norm = water_mask.astype(np.float32) / 255.0
    slope_map_norm = slope_map.astype(np.float32)
    road_mask_norm = road_mask.astype(np.float32) / 255.0

    base_cost = np.ones_like(slope_map_norm, dtype=np.float32)
    slope_cost = slope_map_norm * slope_cost_factor
    water_penalty = water_mask_norm * water_cost_factor

    combined_cost = base_cost + slope_cost + water_penalty
    combined_cost = np.where(road_mask_norm > 0.1,
                             combined_cost * road_cost_factor,
                             combined_cost)

    min_cost_allowed = 0.01
    combined_cost = np.maximum(combined_cost, min_cost_allowed)

    print(f"Карта весов создана. Диапазон значений: {np.min(combined_cost):.2f} - {np.max(combined_cost):.2f}")
    #plt.imsave(os.path.join(OUTPUT_DIR_ALGO, "weight_map_combined.png"), combined_cost, cmap='magma')
    return combined_cost.astype(np.float32)

def generate_routes_algo(cost_map, start_point_px, end_point_px, num_routes=3):
    print(f"Генерация {num_routes} маршрутов от {start_point_px} до {end_point_px}...")
    routes_data = []

    if cost_map is None or cost_map.size == 0:
         print("Ошибка: Карта весов пуста.")
         return []
    if not (0 <= start_point_px[0] < cost_map.shape[0] and 0 <= start_point_px[1] < cost_map.shape[1]):
         print(f"Ошибка: Начальная точка {start_point_px} вне границ карты ({cost_map.shape}).")
         return []
    if not (0 <= end_point_px[0] < cost_map.shape[0] and 0 <= end_point_px[1] < cost_map.shape[1]):
         print(f"Ошибка: Конечная точка {end_point_px} вне границ карты ({cost_map.shape}).")
         return []

    cost_map_float = cost_map.astype(float)

    for i in range(num_routes):
        print(f"Поиск маршрута {i+1}...")
        variation_factor = 0.05 + (i * 0.03)
        variation = np.random.normal(0, variation_factor, cost_map_float.shape)
        varied_cost_map = np.clip(cost_map_float + cost_map_float * variation, 0.01, np.max(cost_map_float) * 1.5)

        try:
            indices, weight = route_through_array(
                varied_cost_map,
                start=start_point_px,
                end=end_point_px,
                fully_connected=True,
                geometric=True
            )

            if indices is None or len(indices) == 0:
                print(f"Не удалось найти маршрут {i+1}.")
                continue

            indices = np.array(indices)
            route = indices.T
            routes_data.append({"path": route, "cost": weight})
            print(f"Маршрут {i+1} найден, стоимость (из route_through_array): {weight:.2f}, точек: {len(indices)}")

        except ValueError as e:
            print(f"Ошибка при поиске маршрута {i+1} (возможно, точки совпадают или недостижимы): {e}")
        except Exception as e:
             print(f"Неожиданная ошибка при поиске маршрута {i+1}: {e}")

    return routes_data


def get_transform_and_bounds(geometry_4326):
    src_crs = "EPSG:4326"
    dst_crs = "EPSG:3857"

    transformer_forward = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    transformer_backward = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True)

    geom_shape = shape(geometry_4326)
    if not geom_shape.is_valid:
        geom_shape = geom_shape.buffer(0)

    geom_mercator = shapely_transform(transformer_forward.transform, geom_shape)

    x_min, y_min, x_max, y_max = geom_mercator.bounds
    width_meters = x_max - x_min
    height_meters = y_max - y_min

    return transformer_backward, x_min, y_min, x_max, y_max, width_meters, height_meters


def coords_to_pixel(lon_lat, x_min_merc, y_min_merc, width_merc, height_merc, img_width, img_height):
    src_crs = "EPSG:4326"
    dst_crs = "EPSG:3857"
    transformer_forward = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    lon, lat = lon_lat
    merc_x, merc_y = transformer_forward.transform(lon, lat)

    x_max_merc = x_min_merc + width_merc
    y_max_merc = y_min_merc + height_merc

    if not (x_min_merc <= merc_x <= x_max_merc and y_min_merc <= merc_y <= y_max_merc):
        print(f"Warning: Point {lon_lat} is outside the geometry bounds calculated for the image. Clamping.")
        merc_x = np.clip(merc_x, x_min_merc, x_max_merc)
        merc_y = np.clip(merc_y, y_min_merc, y_max_merc)

    x_prop = (merc_x - x_min_merc) / width_merc if width_merc > 0 else 0.5
    y_prop = (y_max_merc - merc_y) / height_merc if height_merc > 0 else 0.5

    col = int(round(x_prop * (img_width - 1)))
    row = int(round(y_prop * (img_height - 1)))

    col = np.clip(col, 0, img_width - 1)
    row = np.clip(row, 0, img_height - 1)

    return row, col


def calculate_route_costs(routes_data, transformer_backward, x_min, y_min, width_meters, height_meters, img_width, img_height):
    print("Расчет стоимости для сгенерированных маршрутов...")
    cost_results = []

    if not routes_data:
        return []

    from generated.routegenerator.v1 import routegenerator_pb2

    for i, route_info in enumerate(routes_data):
        route_id = i + 1
        path_pixels = route_info["path"]
        original_path_cost = route_info["cost"]

        if path_pixels.shape[1] < 2:
             print(f"Маршрут {route_id} слишком короткий для расчета расстояния.")
             continue

        coords_mercator = []
        rows, cols = path_pixels[0], path_pixels[1]

        total_distance_meters = 0.0
        for j in range(len(rows)):
             merc_x = x_min + (cols[j] / (img_width -1)) * width_meters
             merc_y = y_min + ((img_height -1 - rows[j]) / (img_height - 1)) * height_meters
             coords_mercator.append((merc_x, merc_y))

             if j > 0:
                  dx = coords_mercator[j][0] - coords_mercator[j-1][0]
                  dy = coords_mercator[j][1] - coords_mercator[j-1][1]
                  segment_distance = np.sqrt(dx**2 + dy**2)
                  total_distance_meters += segment_distance


        total_distance_km = total_distance_meters / 1000.0

        difficulty_factor = original_path_cost / total_distance_meters if total_distance_meters > 0 else 1.0
        difficulty_factor = np.clip(difficulty_factor, 0.5, 10.0)

        estimated_fuel = (
            BASE_FUEL_CONSUMPTION_FACTOR *
            total_distance_km *
            (1 + difficulty_factor * FUEL_COST_PER_KM_DIFFICULTY_UNIT)
        )

        estimated_construction = (
             BASE_CONSTRUCTION_COST_FACTOR *
             total_distance_km *
             (1 + difficulty_factor * CONSTRUCTION_COST_PER_KM_DIFFICULTY_UNIT)
        )

        total_estimated = estimated_fuel + estimated_construction

        cost_results.append(routegenerator_pb2.RouteCost(
            route_id=route_id,
            path_cost=original_path_cost,
            estimated_distance_km=total_distance_km,
            estimated_fuel_cost_units=estimated_fuel,
            estimated_construction_cost_units=estimated_construction,
            total_estimated_cost_units=total_estimated
        ))
        print(f"  Маршрут {route_id}: Дистанция={total_distance_km:.2f} км, Стоимость(путь)={original_path_cost:.2f}, "
              f"Стоимость(топливо)={estimated_fuel:.2f}, Стоимость(стр-во)={estimated_construction:.2f}, Итого={total_estimated:.2f}")

    return cost_results

def apply_winter_effect(rgb_image_pil):
    if rgb_image_pil is None: return None
    try:
        if rgb_image_pil.mode == 'RGBA':
             rgb_array = np.array(rgb_image_pil.convert('RGB'))
        else:
             rgb_array = np.array(rgb_image_pil)

        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        saturation_factor = 0.4
        s_desaturated = np.clip(s * saturation_factor, 0, 255).astype(hsv.dtype)

        brightness_increase = 1.1
        v_brighter = np.clip(v * brightness_increase, 0, 255).astype(hsv.dtype)

        hsv_modified = cv2.merge([h, s_desaturated, v_brighter])
        rgb_desaturated = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB)

        blue_tint_strength = 0.15
        blue_overlay = np.zeros_like(rgb_desaturated, dtype=np.float32)
        blue_overlay[:, :, 2] = 255
        blue_overlay[:, :, 1] = 230

        rgb_float = rgb_desaturated.astype(np.float32)
        rgb_tinted_float = cv2.addWeighted(rgb_float, 1.0 - blue_tint_strength, blue_overlay, blue_tint_strength, 0.0)

        rgb_winter = np.clip(rgb_tinted_float, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb_winter)

    except Exception as e:
        print(f"Ошибка в apply_winter_effect: {e}")
        return rgb_image_pil


def visualize_results(rgb_image_pil, water_mask, slope_map, road_mask, cost_map, routes_data, start_px, end_px):
    print("Визуализация результатов...")

    rgb_winter_pil = apply_winter_effect(rgb_image_pil)
    if rgb_winter_pil:
        display_rgb = np.array(rgb_winter_pil)
    elif rgb_image_pil:
         if rgb_image_pil.mode == 'RGBA':
             display_rgb = np.array(rgb_image_pil.convert('RGB'))
         else:
             display_rgb = np.array(rgb_image_pil)
    else:
        display_rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Анализ и Маршруты (Зимние Условия)', fontsize=16)

    axes[0, 0].imshow(display_rgb)
    axes[0, 0].set_title('RGB Изображение (Зимний вид)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(water_mask, cmap='Blues', vmin=0, vmax=255)
    axes[0, 1].set_title('Детекция Воды/Льда (NDWI)')
    axes[0, 1].axis('off')

    im_slope = axes[0, 2].imshow(slope_map, cmap='viridis')
    axes[0, 2].set_title('Карта Уклонов (Нормализованная)')
    axes[0, 2].axis('off')
    fig.colorbar(im_slope, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(road_mask, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Детекция Дорог (Линии)')
    axes[1, 0].axis('off')

    im_cost = axes[1, 1].imshow(cost_map, cmap='magma')
    axes[1, 1].set_title('Карта Стоимости (Cost Map)')
    axes[1, 1].axis('off')
    fig.colorbar(im_cost, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(display_rgb)
    axes[1, 2].set_title('Сгенерированные Маршруты')
    axes[1, 2].axis('off')

    colors = plt.cm.cool(np.linspace(0, 1, len(routes_data) if routes_data else 1))
    if routes_data:
         for i, route_info in enumerate(routes_data):
             route_pixels = route_info["path"]
             if route_pixels is not None and route_pixels.ndim == 2 and route_pixels.shape[0] == 2:
                 rows, cols = route_pixels
                 axes[1, 2].plot(cols, rows, color=colors[i], linewidth=2.0, label=f'Маршрут {i + 1}')

         if start_px:
             axes[1, 2].plot(start_px[1], start_px[0], 'go', markersize=8, label='Старт')
         if end_px:
             axes[1, 2].plot(end_px[1], end_px[0], 'ro', markersize=8, label='Финиш')
         axes[1, 2].legend(fontsize='small')
    else:
         axes[1, 2].text(0.5, 0.5, 'Маршруты не найдены', horizontalalignment='center', verticalalignment='center', transform=axes[1, 2].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.getvalue()
    buf.close()

    encoded = base64.b64encode(png_bytes).decode('utf-8')

    print("Визуализация создана и готова к отправке.")
    return encoded

def create_routes_geojson(routes_data, transformer_backward, x_min, y_min, width_meters, height_meters, img_width, img_height):
    print("Создание GeoJSON для маршрутов...")
    if not routes_data:
        return json.dumps({"type": "FeatureCollection", "features": []})

    features = []

    for i, route_info in enumerate(routes_data):
        route_pixels = route_info["path"]
        path_cost = route_info["cost"]

        if route_pixels.shape[1] < 2: continue

        coords_lonlat = []
        rows, cols = route_pixels[0], route_pixels[1]

        for j in range(len(rows)):
             merc_x = x_min + (cols[j] / (img_width -1)) * width_meters
             merc_y = y_min + ((img_height -1 - rows[j]) / (img_height - 1)) * height_meters

             lon, lat = transformer_backward.transform(merc_x, merc_y)
             coords_lonlat.append((lon, lat))

        line = LineString(coords_lonlat)
        feature = {
            "type": "Feature",
            "properties": {
                "id": i + 1,
                "name": f"Маршрут {i + 1}",
                "path_cost": path_cost
            },
            "geometry": mapping(line)
        }
        features.append(feature)

    geojson_output = {
        "type": "FeatureCollection",
        "crs": {
             "type": "name",
             "properties": {
                 "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
             }
         },
        "features": features
    }

    print("GeoJSON создан.")
    return json.dumps(geojson_output, ensure_ascii=False)


def generate(geojson_str, num_routes, start_lon_lat=None, end_lon_lat=None):
    start_time_algo = time.time()
    print("\n--- Запуск алгоритма генерации маршрутов ---")
    results = {
        "visualization_png": b'',
        "routes_geojson": json.dumps({"type": "FeatureCollection", "features": []}),
        "route_costs": [],
        "status_message": "Ошибка инициализации",
        "success": False
    }

    try:
        print("1. Парсинг GeoJSON геометрии...")
        try:
            geojson_data = json.loads(geojson_str)
            geometry_dict = None

            if isinstance(geojson_data, dict) and geojson_data.get("type") == "FeatureCollection":
                print("  Обнаружен FeatureCollection, извлекаю первую геометрию...")
                if not geojson_data.get("features"):
                    raise ValueError("FeatureCollection не содержит features")
                if not isinstance(geojson_data["features"], list) or len(geojson_data["features"]) == 0:
                     raise ValueError("Массив features пуст или имеет неверный формат")

                first_feature = geojson_data["features"][0]
                if not isinstance(first_feature, dict) or first_feature.get("type") != "Feature":
                     raise ValueError("Первый элемент в features не является Feature")

                geometry_dict = first_feature.get("geometry")
                if not isinstance(geometry_dict, dict) or "type" not in geometry_dict or "coordinates" not in geometry_dict:
                     raise ValueError("Feature не содержит валидную геометрию")
                print(f"  Извлечен тип геометрии: {geometry_dict.get('type')}")
            elif isinstance(geojson_data, dict) and "type" in geojson_data and "coordinates" in geojson_data:
                 print("  Обнаружен объект геометрии GeoJSON.")
                 geometry_dict = geojson_data
            else:
                 raise ValueError("Входные данные не являются FeatureCollection или валидной геометрией GeoJSON")

            if geometry_dict is None:
                 raise ValueError("Не удалось извлечь или определить геометрию из входных данных")

            target_geometry_dict = geometry_dict
            geom_shape = shape(target_geometry_dict)

            if not geom_shape.is_valid:
                print("Предупреждение: Извлеченная геометрия невалидна, пытаюсь исправить (buffer(0))...")
                geom_shape = geom_shape.buffer(0)
                target_geometry_dict = mapping(geom_shape)

            print(f"  Используемый тип геометрии для обработки: {geom_shape.geom_type}")

        except json.JSONDecodeError as e:
             raise ValueError(f"Ошибка декодирования JSON: {e}")
        except ValueError as e: # Catch specific ValueErrors from checks above
             raise ValueError(f"Ошибка парсинга GeoJSON: {e}")
        except Exception as e: # Catch any other unexpected parsing errors
            # Log or print detailed error for debugging
            traceback.print_exc()
            raise ValueError(f"Неожиданная ошибка при парсинге GeoJSON: {e}")


        print("\n1.5 Получение трансформации и границ...")
        # Ensure subsequent calls use the corrected variable name
        transformer_backward, x_min_merc, y_min_merc, x_max_merc, y_max_merc, width_merc, height_merc = get_transform_and_bounds(target_geometry_dict)


        print("\n2. Загрузка спутниковых снимков...")
        try:
            today = datetime.now()
            date_to_str = (today - timedelta(days=90)).strftime('%Y-%m-%d')
            date_from_str = (today - timedelta(days=180)).strftime('%Y-%m-%d')
            print(f"  Диапазон дат для снимков: {date_from_str} to {date_to_str}")

            # Ensure subsequent calls use the corrected variable name
            rgb_image_pil, _ = get_satellite_image(target_geometry_dict, "rgb", date_from_str, date_to_str)
            ndwi_array, _ = get_satellite_image(target_geometry_dict, "ndwi", date_from_str, date_to_str)
            dem_array, dem_profile = get_satellite_image(target_geometry_dict, "dem")

        except Exception as e:
             traceback.print_exc()
             raise ConnectionError(f"Ошибка загрузки спутниковых снимков: {e}")

        print("\n3. Анализ изображений...")
        water_mask = detect_water(ndwi_array)
        slope_map = detect_slopes(dem_array)
        road_mask = detect_roads(rgb_image_pil)


        print("\n4. Создание карты весов...")
        cost_map = create_weight_map(water_mask, slope_map, road_mask)


        print("\n5. Определение точек старта/финиша...")

        print(f"  DEBUG: Received start_lon_lat = {start_lon_lat} (type: {type(start_lon_lat)})", flush=True)
        print(f"  DEBUG: Received end_lon_lat = {end_lon_lat} (type: {type(end_lon_lat)})", flush=True)

        start_point_px = None
        end_point_px = None

        if start_lon_lat and isinstance(start_lon_lat, list) and len(start_lon_lat) == 2:
            try:
                 start_point_px = coords_to_pixel(start_lon_lat, x_min_merc, y_min_merc, width_merc, height_merc, IMAGE_SIZE, IMAGE_SIZE)
                 print(f"  Стартовая точка из запроса ({start_lon_lat[0]:.4f}, {start_lon_lat[1]:.4f}) -> пиксель {start_point_px}")
            except Exception as coord_err:
                 print(f"  ОШИБКА при конвертации start_lon_lat {start_lon_lat} в пиксели: {coord_err}. Использую дефолт.")
                 start_point_px = None
        else:
             print(f"  Условие для start_lon_lat не выполнено (получено: {start_lon_lat}).")

        if start_point_px is None:
            start_point_px = (int(IMAGE_SIZE * 0.15), int(IMAGE_SIZE * 0.15))
            print(f"  Используется стартовая точка по умолчанию: {start_point_px}")


        if end_lon_lat and isinstance(end_lon_lat, list) and len(end_lon_lat) == 2:
            try:
                 end_point_px = coords_to_pixel(end_lon_lat, x_min_merc, y_min_merc, width_merc, height_merc, IMAGE_SIZE, IMAGE_SIZE)
                 print(f"  Конечная точка из запроса ({end_lon_lat[0]:.4f}, {end_lon_lat[1]:.4f}) -> пиксель {end_point_px}")
            except Exception as coord_err:
                 print(f"  ОШИБКА при конвертации end_lon_lat {end_lon_lat} в пиксели: {coord_err}. Использую дефолт.")
                 end_point_px = None
        else:
             print(f"  Условие для end_lon_lat не выполнено (получено: {end_lon_lat}).")

        if end_point_px is None:
            end_point_px = (int(IMAGE_SIZE * 0.85), int(IMAGE_SIZE * 0.85))
            print(f"  Используется конечная точка по умолчанию: {end_point_px}")


        if start_point_px == end_point_px:
             print("Предупреждение: Старт и финиш совпадают ({start_point_px}), сдвигаю финиш немного...")
             new_end_row = min(end_point_px[0] + 1, IMAGE_SIZE - 1)
             new_end_col = min(end_point_px[1] + 1, IMAGE_SIZE - 1)
             if (new_end_row, new_end_col) == start_point_px:
                  new_end_row = max(end_point_px[0] - 1, 0)
             end_point_px = (new_end_row, new_end_col)
             print(f"  Новая конечная точка: {end_point_px}")

        print("\n6. Генерация маршрутов...")
        num_routes_to_gen = num_routes if num_routes > 0 else 3
        routes_data = generate_routes_algo(cost_map, start_point_px, end_point_px, num_routes_to_gen)

        if not routes_data:
            print("Не удалось сгенерировать ни одного маршрута.")
            results["status_message"] = "Обработка завершена, но маршруты не найдены."
            results["success"] = True
            try:
                results["visualization_png"] = visualize_results(rgb_image_pil, water_mask, slope_map, road_mask, cost_map, [], start_point_px, end_point_px)
            except Exception as viz_err:
                print(f"Ошибка при генерации визуализации без маршрутов: {viz_err}")
            return results


        print("\n7. Расчет стоимости маршрутов...")
        results["route_costs"] = calculate_route_costs(routes_data, transformer_backward, x_min_merc, y_min_merc, width_merc, height_merc, IMAGE_SIZE, IMAGE_SIZE)


        print("\n8. Создание выходных данных...")
        results["visualization_png"] = visualize_results(rgb_image_pil, water_mask, slope_map, road_mask, cost_map, routes_data, start_point_px, end_point_px)
        results["routes_geojson"] = create_routes_geojson(routes_data, transformer_backward, x_min_merc, y_min_merc, width_merc, height_merc, IMAGE_SIZE, IMAGE_SIZE)

        results["status_message"] = f"Успешно сгенерировано {len(routes_data)} маршрутов."
        results["success"] = True

    except ValueError as e:
        print(f"Ошибка входных данных в алгоритме: {e}")
        results["status_message"] = f"Ошибка входных данных: {e}"
        results["success"] = False
    except ConnectionError as e:
         print(f"Ошибка сети/API в алгоритме: {e}")
         results["status_message"] = f"Ошибка сети/API: {e}"
         results["success"] = False
    except Exception as e:
        print(f"Непредвиденная ошибка в алгоритме: {e}")
        traceback.print_exc()
        results["status_message"] = f"Внутренняя ошибка алгоритма: {e}"
        results["success"] = False

    print(f"--- Алгоритм завершен за {time.time() - start_time_algo:.2f} секунд (Успех: {results['success']}) ---")
    return results