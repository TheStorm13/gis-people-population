import os
from typing import Optional

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree


class WaterSystemHandler:
    def __init__(self, water_shapefile: str = "../../data/waterSystem/waterSystem_Linan.shp"):
        """
        Инициализирует анализатор водной системы.

        Parameters:
            water_shapefile (str): Путь к .shp файлу с данными о водной системе.
        """
        self.water_shapefile = water_shapefile
        self.water_polygons: Optional[gpd.GeoDataFrame] = None

    def load_water_system(self) -> gpd.GeoDataFrame:
        """
        Загружает данные о водной системе.

        Returns:
            gpd.GeoDataFrame: Данные о водных объектах.
        """
        if not os.path.exists(self.water_shapefile):
            raise FileNotFoundError(f"Файл не найден: {self.water_shapefile}")

        self.water_polygons = gpd.read_file(self.water_shapefile)
        return self.water_polygons

    def set_water_polygons(self, polygons: gpd.GeoDataFrame):
        """
        Устанавливает внешний GeoDataFrame водной системы.

        Parameters:
            polygons (gpd.GeoDataFrame): Геодатафрейм с водными объектами.
        """
        self.water_polygons = polygons

    def add_water_features_to_buildings(
            self,
            buildings: gpd.GeoDataFrame,
            buffer_radius: float = 500,
            crs_for_analysis: str = "EPSG:3857"
    ) -> gpd.GeoDataFrame:
        """
        Добавляет к зданиям признаки, связанные с водными объектами.

        Parameters:
            buildings (gpd.GeoDataFrame): Здания (полигоны или точки), должны содержать геометрию и CRS.
            buffer_radius (float): Радиус поиска водных объектов вокруг здания в метрах.
            crs_for_analysis (str): CRS для анализа расстояний и площадей.

        Returns:
            gpd.GeoDataFrame: Копия входного DataFrame с колонками:
                - count_water — количество водных объектов в радиусе
                - nearest_water_dist — расстояние до ближайшего водного объекта
        """
        if self.water_polygons is None:
            raise ValueError(
                "Данные о водной системе не загружены. Вызовите load_water_system() или set_water_polygons().")

        if buildings.crs is None:
            raise ValueError("GeoDataFrame зданий должен иметь CRS.")

        original_crs = buildings.crs
        buildings_proj = buildings.to_crs(crs_for_analysis)
        water_proj = self.water_polygons.to_crs(crs_for_analysis)

        # Центры зданий
        building_centers = buildings_proj.geometry.representative_point()

        # Буферы вокруг зданий
        buffers = building_centers.buffer(buffer_radius)
        buffer_gdf = gpd.GeoDataFrame(geometry=buffers, crs=crs_for_analysis)

        # Пространственное соединение для подсчета количества водных объектов
        join = gpd.sjoin(water_proj, buffer_gdf, predicate="intersects")
        count_series = join.groupby("index_right").size()

        # --- Оптимизация расстояния: cKDTree ---
        # Получаем центроиды водных объектов
        water_centroids = water_proj.geometry.centroid
        coords_water = np.array(list(zip(water_centroids.x, water_centroids.y)))

        coords_buildings = np.array(list(zip(building_centers.x, building_centers.y)))

        # Создаем дерево
        tree = cKDTree(coords_water)
        nearest_distances, indices = tree.query(coords_buildings, k=1, distance_upper_bound=buffer_radius + 100)

        # Подготовка результата
        result = buildings.copy()
        result["count_water"] = 0
        result["nearest_water_dist"] = np.inf

        result.loc[count_series.index, "count_water"] = count_series
        result["nearest_water_dist"] = nearest_distances

        return result
