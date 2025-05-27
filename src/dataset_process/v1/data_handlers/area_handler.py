import os
from typing import Optional

import geopandas as gpd


class AreaHandler:
    def __init__(self, area_shapefile: str = "../../data/area/area_Linan.shp"):
        self.area_shapefile = area_shapefile
        self._areas_gdf: Optional[gpd.GeoDataFrame] = None
        self._crs = None  # Кэшируем CRS для избежания повторных вызовов

    def load_areas(self) -> gpd.GeoDataFrame:
        if self._areas_gdf is not None:
            return self._areas_gdf

        if not os.path.exists(self.area_shapefile):
            raise FileNotFoundError(f"Файл не найден: {self.area_shapefile}")

        self._areas_gdf = gpd.read_file(self.area_shapefile)

        required_columns = {'FNAME', 'Eng_Name', 'OBJECTID', 'geometry'}
        if not required_columns.issubset(self._areas_gdf.columns):
            missing = required_columns - set(self._areas_gdf.columns)
            raise ValueError(f"В файле {self.area_shapefile} отсутствуют необходимые колонки: {missing}")

        # Предварительно переименовываем для sjoin
        self._areas_gdf.rename(columns={
            "FNAME": "area_FNAME",
            "Eng_Name": "area_Eng_Name",
            "OBJECTID": "area_OBJECTID"
        }, inplace=True)

        self._crs = self._areas_gdf.crs
        return self._areas_gdf

    def add_area_features_to_buildings(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        areas_gdf = self.load_areas()

        if buildings.crs is None:
            raise ValueError("GeoDataFrame зданий должен иметь CRS.")

        # Приводим CRS к единому
        if buildings.crs != self._crs:
            buildings = buildings.to_crs(self._crs)

        # Определяем целевую проекцию для точности расчетов (например, UTM Zone автоматически)
        crs_utm = buildings.estimate_utm_crs()  # Поддерживается в geopandas >= 0.12.0

        # Перепроецируем для корректного вычисления centroid
        buildings_projected = buildings.to_crs(crs_utm)

        # Вычисляем центроид в проекционной СК
        buildings["geometry_tmp"] = buildings_projected.geometry.centroid.to_crs(self._crs)

        # Выполняем spatial join
        joined = gpd.sjoin(
            buildings.set_geometry("geometry_tmp"),
            areas_gdf[["area_OBJECTID", "area_FNAME", "area_Eng_Name", "geometry"]],
            how="left",
            predicate="within"
        )

        # Восстанавливаем основную геометрию и очищаем временные данные
        joined = joined.set_geometry("geometry")
        joined.drop(columns=["geometry_tmp", "index_right"], errors="ignore", inplace=True)

        return joined
