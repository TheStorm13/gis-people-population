import os
from typing import Optional

import geopandas as gpd
import numpy as np


class RoadNetworkHandler:
    def __init__(self, road_shapefile: str = "../../data/roadNetwork/roadNetwork.shp"):
        """
        Инициализирует обработчик дорожной сети.

        Parameters:
            road_shapefile (str): Путь к shapefile с дорожной сетью.
        """
        self.road_shapefile = road_shapefile
        self.roads: Optional[gpd.GeoDataFrame] = None

    def load_roads(self) -> gpd.GeoDataFrame:
        """
        Загружает дорожную сеть из файла.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame с дорогами.
        """
        if not os.path.exists(self.road_shapefile):
            raise FileNotFoundError(f"Файл не найден: {self.road_shapefile}")

        self.roads = gpd.read_file(self.road_shapefile)
        return self.roads

    def add_road_features_to_buildings(
            self,
            buildings: gpd.GeoDataFrame,
            buffer_radius: float = 500,
            crs_for_analysis: str = "EPSG:3857"
    ) -> gpd.GeoDataFrame:
        """
        Добавляет пространственные признаки по дорожной сети к зданиям.

        Признаки:
        - count_roads: количество дорог в радиусе
        - nearest_road_dist: расстояние до ближайшей дороги

        Parameters:
            buildings (gpd.GeoDataFrame): Здания с геометрией.
            buffer_radius (float): Радиус поиска дорог в метрах.
            crs_for_analysis (str): CRS для анализа на основе метров.

        Returns:
            gpd.GeoDataFrame: Копия зданий с новыми колонками.
        """
        if self.roads is None:
            raise ValueError("Дороги не загружены. Вызовите load_roads().")

        if buildings.crs is None or self.roads.crs is None:
            raise ValueError("CRS отсутствует в GeoDataFrame зданий или дорог.")

        original_crs = buildings.crs

        # Перепроецировать данные
        buildings_proj = buildings.to_crs(crs_for_analysis)
        roads_proj = self.roads.to_crs(crs_for_analysis)

        # Проверка наличия колонки FWIDTH
        if "FWIDTH" not in roads_proj.columns:
            raise KeyError("В файле дорог отсутствует колонка 'FWIDTH'")

        # Фильтрация дорог без ширины (нулевые объекты)
        roads_proj = roads_proj[roads_proj["FWIDTH"] > 0]

        # Центроиды зданий
        building_centroids = buildings_proj.geometry.centroid.rename("centroid").reset_index()

        # Пространственное соединение ближайших дорог
        sjoined = gpd.sjoin_nearest(
            building_centroids,
            roads_proj,
            how="left",
            max_distance=buffer_radius,
            distance_col="distance"
        )

        # Подсчёт уникальных дорог в радиусе
        road_counts = (
            sjoined.loc[:, ["index", "index_right"]]
            .drop_duplicates()
            .groupby("index")
            .size()
        )
        buildings_proj["count_roads"] = buildings_proj.index.map(road_counts).fillna(0).astype(int)

        # Расчёт точного расстояния до ближайшей дороги
        nearest_distances = (
            sjoined.groupby("index")["distance"]
            .min()
        )
        buildings_proj["nearest_road_dist"] = buildings_proj.index.map(nearest_distances).fillna(np.inf)

        # Вернуть обратно в исходную проекцию
        return buildings_proj.to_crs(original_crs)
