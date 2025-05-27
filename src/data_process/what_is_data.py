import pandas as pd
import rasterio

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import os
import geopandas as gpd


def process_shapefile(shp_path):
    try:
        # Получаем имя папки и файла
        dir_name = os.path.basename(os.path.dirname(shp_path))
        file_name = os.path.splitext(os.path.basename(shp_path))[0]  # Убираем .shp

        print("-----------------------------------------------")
        print(f"{dir_name} / {file_name}\n")  # Формат: "название_папки / название_файла"
        print(f"Path: {shp_path}\n")

        # Загрузка shapefile
        data = gpd.read_file(shp_path)

        # Вывод списка всех атрибутов (столбцов)
        print("Атрибуты:", list(data.columns))

        # Вывести первые строки атрибутивной таблицы и геометрию
        print("\nПервые строки данных:")
        print(data.head())

        # Информация о системе координат
        print("\nCRS:", data.crs)

        # Показать количество объектов
        print("\nКоличество объектов:", len(data))
        print("\n")

    except Exception as e:
        print(f"Ошибка при обработке файла {shp_path}: {str(e)}\n")


def find_and_process_shp_files(root_dir='../../data'):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.shp'):
                full_path = os.path.join(root, file)
                process_shapefile(full_path)


def find_and_process_tif(root_dir='.'):
    print("-----------------------------------------------")
    print("Population Data\n")
    print("Path: ../../data/population/chn_ppp_2020_UNadj_constrained.tif\n")

    file_path = "../../data/population/chn_ppp_2020_UNadj_constrained.tif"

    with rasterio.open(file_path) as src:
        print("Meta:", src.meta)  # Метаданные растрового файла


if __name__ == "__main__":
    find_and_process_shp_files()
    find_and_process_tif()
