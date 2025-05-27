import logging
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import pandas as pd
import rasterio
from shapely import wkb
from src.data_process.v1.data_handlers import RoadNetworkHandler
from src.data_process.v1.data_handlers import WaterSystemHandler
from src.data_process.v1.data_handlers.poi_handler import POIHandler
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Пути к данным
BUILDINGS_PATH = "../../../data/v1/buildings/Buildings_Linan (1).shp"
RASTER_PATH = "../../../data/v1/population/chn_ppp_2020_UNadj_constrained.tif"
OUTPUT_PATH = "../../../data/processed/processed_buildings.parquet"

# Параметры обработки
CHUNK_SIZE = 1000
MAX_WORKERS = 4

# Проецируемая система координат для центроидов (метры)
PROJECTED_CRS = "EPSG:3857"


def transform(buildings_chunk: gpd.GeoDataFrame, raster_path: str) -> pd.DataFrame:
    """
    Трансформирует чанк зданий, извлекая население из растрового слоя.
    """
    # Преобразуем CRS в проецируемую перед вычислением центроидов
    chunk_projected = buildings_chunk.to_crs(PROJECTED_CRS)
    centroids = chunk_projected.geometry.centroid.to_crs("EPSG:4326")

    coords = [(geom.x, geom.y) for geom in centroids]

    with rasterio.open(raster_path) as src:
        pop_values = [x[0] if x[0] != src.nodata else 0 for x in src.sample(coords)]

    result_df = buildings_chunk.copy()
    result_df["population"] = pop_values
    return result_df


def process_chunk(buildings_chunk: gpd.GeoDataFrame, raster_path: str) -> pd.DataFrame:
    try:
        return transform(buildings_chunk, raster_path)
    except Exception as e:
        logger.error(f"Ошибка в чанке: {e}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame в случае ошибки


def process_data_in_chunks(buildings: gpd.GeoDataFrame, chunk_size: int, raster_path: str) -> gpd.GeoDataFrame:
    chunks = [buildings.iloc[i:i + chunk_size] for i in range(0, len(buildings), chunk_size)]
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_chunk, chunk, raster_path) for chunk in chunks]
        for f in tqdm(futures, desc="Обработка чанков"):
            try:
                result = f.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при обработке чанка: {e}")

    df = pd.concat(results, ignore_index=True)

    # Если geometry в WKB, декодируем
    if isinstance(df.iloc[0]["geometry"], (bytes, bytearray)):
        df["geometry"] = df["geometry"].apply(lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g)

    return gpd.GeoDataFrame(df, geometry="geometry", crs=buildings.crs)


def load_properties_data(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Начата обработка зданий по чанкам...")
    buildings = process_data_in_chunks(buildings, CHUNK_SIZE, RASTER_PATH)

    logger.info("Население добавлено к зданиям.")

    # # Инициализировать и загрузить области
    # area_heandler = AreaHandler()
    # area_heandler.load_areas()
    #
    # # Добавить признаки областей к зданиям
    # buildings = area_heandler.add_area_features_to_buildings(buildings)
    #
    # logger.info("Добавлены признаки областей к зданиям.")

    # Инициализировать и загрузить POI
    poi_handler = POIHandler()
    poi_handler.load_all_poi()

    # Добавить признаки POI к зданиям
    buildings = poi_handler.add_poi_features_to_buildings(buildings, buffer_radius=500)
    logger.info("Добавлены признаки POI к зданиям.")

    # Инициализировать и загрузить водную систему
    water_handler = WaterSystemHandler()
    water_handler.load_water_system()

    # Добавить признаки водных объектов к зданиям
    buildings = water_handler.add_water_features_to_buildings(buildings, buffer_radius=500)

    logger.info("Добавлены признаки водных объектов к зданиям.")

    # Инициализировать и загрузить дорожную сеть
    road_handler = RoadNetworkHandler()
    road_handler.load_roads()

    # Добавить признаки дорожной сети к зданиям
    buildings = road_handler.add_road_features_to_buildings(buildings, buffer_radius=500)
    logger.info("Добавлены признаки дорожной сети к зданиям.")

    return buildings


def main():
    logger.info("Загрузка данных зданий...")
    buildings = gpd.read_file(BUILDINGS_PATH)

    # Удаляем колонку LandArea сразу после загрузки
    if 'LandArea' in buildings.columns:
        buildings = buildings.drop(columns=['LandArea'])

    logger.info(f"Загружено объектов: {len(buildings)}")

    processed_buildings = load_properties_data(buildings)

    logger.info("Обработка зданий завершена.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        import pyarrow
        processed_buildings.to_parquet(OUTPUT_PATH)
        logger.info(f"Результат сохранён в {OUTPUT_PATH}")
    except ImportError:
        fallback_path = OUTPUT_PATH.replace(".parquet", ".geojson")
        processed_buildings.to_file(fallback_path, driver="GeoJSON")
        logger.warning(f"pyarrow не найден, данные сохранены в GeoJSON: {fallback_path}")

    # Вывести первые строки атрибутивной таблицы и геометрию
    print("\nПервые строки данных:")
    print(processed_buildings.head())

    # Информация о системе координат
    print("\nCRS:", processed_buildings.crs)


if __name__ == "__main__":
    main()
