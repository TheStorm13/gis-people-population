import pandas as pd
from geopandas import gpd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Колонки, которые НЕ нужно обрабатывать (оставить как есть)
DO_NOT_TOUCH = ['geometry']  # Пример

# Колонки, которые нужно УДАЛИТЬ из конечных данных
COLUMNS_TO_DROP = ['GlobalID',
                   'count_w_all', 'count_m_all', 'count_w_0_4', 'count_w_5_9', 'count_w_10_14',
                   'count_w_15_19', 'count_w_20_24', 'count_w_25_29', 'count_w_30_34',
                   'count_w_35_39', 'count_w_40_44', 'count_w_45_49', 'count_w_50_54',
                   'count_w_55_59', 'count_w_60_64', 'count_w_65_69', 'count_w_more_70',
                   'count_m_0_4', 'count_m_5_9', 'count_m_10_14', 'count_m_15_19',
                   'count_m_20_24', 'count_m_25_29', 'count_m_30_34', 'count_m_35_39',
                   'count_m_40_44', 'count_m_45_49', 'count_m_50_54', 'count_m_55_59',
                   'count_m_60_64', 'count_m_65_69', 'count_m_more_70']  # Пример


def process_geojson(root_dir='.'):
    print("GEO Data\n")

    file_path = "../../../data/v2/samara_people_model.geojson"

    data = gpd.read_file(file_path)
    print("Атрибуты:", list(data.columns))
    print("\nПервые строки данных:")
    print(data.head())
    print("\nCRS:", data.crs)
    print("\nКоличество объектов:", len(data))
    print("\n")

    # Копируем данные, чтобы не изменять исходные
    df = data.copy()

    # Копируем данные и удаляем ненужные колонки
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # Функция для попытки конвертации в число
    def try_convert_to_numeric(col):
        try:
            return pd.to_numeric(col, errors='raise')
        except:
            return None

    # Проходим по всем колонкам
    for column in df.columns:
        # Пропускаем геометрию
        if column in DO_NOT_TOUCH:
            continue  # Пропускаем колонки из списка "не трогать"

        # Проверяем, является ли колонка строковой или категориальной
        if df[column].dtype == 'object':
            # Пробуем конвертировать в число
            converted_col = try_convert_to_numeric(df[column])

            if converted_col is not None:
                # Если конвертация удалась — заменяем колонку
                df[column] = converted_col
                print(f"Колонка '{column}' преобразована в числовой формат.")
            else:
                # Если не удалось — применяем Label Encoding
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].fillna('MISSING'))
                print(f"\nКолонка '{column}' (Label Encoding):")
                for i, (cls, code) in enumerate(zip(le.classes_, le.transform(le.classes_))):
                    print(f"  {cls} → {code}")
                    if i >= 4:  # Останавливаемся после 5 строк (индексы 0-4)
                        break

    # Результат
    print("\nДанные после обработки:")
    print(df.head())

    # Сохраняем результат
    output_path_json = "../../../data/processed/samara_people_model_processed.geojson"
    output_path_parquet = "../../../data/processed/samara_people_model_processed.parquet"
    df.to_file(output_path_json, driver='GeoJSON')
    df.to_parquet(output_path_parquet, index=False)
    print(f"\nФайл сохранён: {output_path_json}")


if __name__ == "__main__":
    process_geojson()
