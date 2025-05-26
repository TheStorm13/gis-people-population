import features
import geopandas as gpd
import pandas as pd
import rasterio

BUILDINGS_PATH = "../../data/buildings/Buildings_Linan (1).shp"
POPULATION_PATH = "../../data/population/chn_ppp_2020_UNadj_constrained.tif"

# Перепроектировать здания в WGS84 (для совместимости с WorldPop)
buildings = gpd.read_file(BUILDINGS_PATH).to_crs(epsg=4326)

buildings = buildings[(buildings['Shape_Area'] > 0) & (buildings['Shape_Area'] < 1e6)]

with rasterio.open(POPULATION_PATH) as src:
    pop = src.read(1)
    transform = src.transform


# 1. Создаем растровую маску зданий (1 пиксель = 100 м)
rasterized = features.rasterize(
    [(geom, 1) for geom in buildings.geometry],
    out_shape=pop.shape,
    transform=transform,
    fill=0
)

# 2. Группируем здания по пикселям
pixel_coords = features.sample(buildings.geometry, transform=transform)
buildings['pixel_row'], buildings['pixel_col'] = zip(*pixel_coords)

# 3. Считаем суммарную площадь зданий в каждом пикселе
pixel_area_sum = buildings.groupby(['pixel_row', 'pixel_col'])['Shape_Area'].sum().reset_index()
pixel_area_sum.columns = ['pixel_row', 'pixel_col', 'total_area_in_pixel']

# 4. Объединяем с исходными данными
buildings = pd.merge(buildings, pixel_area_sum, on=['pixel_row', 'pixel_col'])

# 5. Распределяем население пропорционально площади
buildings['pop_assigned'] = (buildings['Shape_Area'] / buildings['total_area_in_pixel']) * pop[buildings['pixel_row'], buildings['pixel_col']]

print(buildings.head())

# 6. Сохраняем результат
buildings.to_file("../../data/dataset/populated_buildings.geojson", driver='GeoJSON')