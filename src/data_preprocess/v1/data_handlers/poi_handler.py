import glob
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import geopandas as gpd
import pandas as pd


class POIHandler:
    def __init__(self, base_path: str = "../../data/POIs"):
        """
        Инициализирует загрузчик POI из директории.
        """
        self.base_path = base_path
        self.poi_gdfs: Dict[str, gpd.GeoDataFrame] = {}
        self.combined_gdf: Optional[gpd.GeoDataFrame] = None

    def load_all_poi(self, target_crs: Optional[str] = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Загружает все .shp файлы из директории, приводит к target_crs и объединяет в один GeoDataFrame.
        """
        shapefiles = glob.glob(os.path.join(self.base_path, "*.shp"))
        if not shapefiles:
            raise FileNotFoundError(f"В папке {self.base_path} не найдено .shp файлов.")

        def load_and_reproject(file: str):
            try:
                poi_type = os.path.splitext(os.path.basename(file))[0]
                gdf = gpd.read_file(file)
                gdf["poi_type"] = poi_type
                if target_crs:
                    gdf = gdf.to_crs(target_crs)
                return poi_type, gdf
            except Exception as e:
                raise RuntimeError(f"Ошибка при чтении файла {file}: {e}")

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_and_reproject, shapefiles))

        self.poi_gdfs = dict(results)

        self.combined_gdf = gpd.GeoDataFrame(
            pd.concat([gdf for _, gdf in results], ignore_index=True),
            crs=target_crs
        )
        return self.combined_gdf

    def get_poi_by_type(self, poi_type: str) -> Optional[gpd.GeoDataFrame]:
        """
        Возвращает POI заданного типа.
        """
        return self.poi_gdfs.get(poi_type)

    def filter_by_area(self, area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Фильтрует все POI внутри переданного полигона.
        """
        if self.combined_gdf is None:
            raise ValueError("Сначала вызовите load_all_poi().")

        if self.combined_gdf.crs is None or area.crs is None:
            raise ValueError("CRS отсутствует в одном из GeoDataFrame.")

        if self.combined_gdf.crs != area.crs:
            area = area.to_crs(self.combined_gdf.crs)

        result = gpd.sjoin(self.combined_gdf, area[['geometry']], predicate='within', how='inner')
        return result.drop(columns=["index_right"])

    def save_to_file(self, output_path: str = "../../data/processed/all_poi.parquet") -> None:
        """
        Сохраняет объединённый POI GeoDataFrame в Parquet.
        """
        if self.combined_gdf is None:
            raise ValueError("Сначала вызовите load_all_poi().")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.combined_gdf.to_parquet(output_path)

    def add_poi_features_to_buildings(
            self,
            buildings: gpd.GeoDataFrame,
            buffer_radius: float = 1000,
            crs_for_analysis: str = "EPSG:3857"
    ) -> gpd.GeoDataFrame:
        """
        Добавляет к зданиям информацию о количестве и расстоянии до POI по типам.
        """
        if self.combined_gdf is None:
            raise ValueError("Сначала вызовите load_all_poi().")

        if buildings.crs is None or self.combined_gdf.crs is None:
            raise ValueError("Один из GeoDataFrame не содержит CRS.")

        buildings_proj = buildings.to_crs(crs_for_analysis)
        poi_proj = self.combined_gdf.to_crs(crs_for_analysis)

        buildings_proj["centroid"] = buildings_proj.geometry.centroid
        poi_types = poi_proj["poi_type"].unique()
        result = buildings_proj.copy()

        for ptype in poi_types:
            poi_type_gdf = poi_proj[poi_proj["poi_type"] == ptype]
            buffers = buildings_proj["centroid"].buffer(buffer_radius)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffers, crs=crs_for_analysis)

            joined = gpd.sjoin(poi_type_gdf, buffer_gdf, predicate='within', how='inner')
            joined = joined.rename(columns={"index_right": "building_idx"})

            # Кол-во POI рядом
            counts = joined.groupby("building_idx").size()
            result[f"count_{ptype}"] = result.index.map(counts).fillna(0).astype(int)

            # Минимальное расстояние
            distances = (
                joined
                .assign(dist=lambda df: df.geometry.distance(
                    gpd.GeoSeries(
                        df["building_idx"].map(buildings_proj["centroid"]).values,
                        index=df.index,
                        crs=None  # CRS будет установлен ниже
                    ).set_crs(buildings_proj.crs),
                    align=False
                ))
                .groupby("building_idx")["dist"]
                .min()
            )
            result[f"nearest_{ptype}_dist"] = result.index.map(distances)

        result.drop(columns="centroid", inplace=True)
        return result.to_crs(buildings.crs)
