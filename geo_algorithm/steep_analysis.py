import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from calc_terrain_status import (
    calc_and_visualize_height,
    calc_and_visualize_shc,
    calc_and_visualize_slope,
    calc_shc_in_steep_area,
)
from rasterio.mask import mask
from shapely.geometry import box
from create_terrain_graph import (
    visualize_slope_area_ratio_histogram,
    visualize_slope_ratio_vs_median_slope,
    visualize_slope_shc_relationship,
    visualize_slope_shc_relationship_with_top_cities,
    visualize_slope_shc_relationship_with_top_cities1,
    visualize_top_cities_on_map,
    visualize_top_steep_cities_comparison,
)

DEM_DIR = "/Users/sakamo/Desktop/GISDATA/DEM_地方別/"
# DEM_地方別/以下に地方名.tifファイルが格納されている
HOUSING_AREA_DIR = "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/"
# 住居系用途地域2019_10万人以上の自治体_全国/A29-19_(都道府県コード)/以下にhousing_(自治体名).shpファイルが格納されている
TARGET_LIST_EXCEL = "/Users/sakamo/Desktop/GISDATA/10man_status.xlsx"
RESULT_XLSX = "result/terrain_statistics.xlsx"

SASEBO_DEM = "/Users/sakamo/Desktop/GISDATA/DEM_地方別/九州.tif"
SASEBO_COMMUNITY = "/Users/sakamo/Desktop/GISDATA/自治協議会/16_182294.SHP" # 自治協議会(町内会、自治会、協議会)
SASEBO_HOUSING_AREA = "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_42/housing_佐世保市.shp"
SASEBO_ROAD_1_4 = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/1項4号道路/1項4号.shp"
SASEBO_ROAD_1_5 = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/1項5号道路/16_188764.SHP"
SASEBO_ROAD_2_IKKATSU = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/2項(一括指定)道路/16_188807.SHP"
SASEBO_ROAD_2_KOBETSU = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/2項(個別指定)道路/16_188797.SHP"
SASEBO_ROAD_3 = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/3項道路/3項.shp"
SASEBO_ROAD_CITY = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/市道(路線)/01_路線.shp"
SASEBO_OBJ = "PATH_TO_PLATEAU_DATA"

files_to_check = [
        DEM_DIR,
        HOUSING_AREA_DIR,
        TARGET_LIST_EXCEL,
        SASEBO_DEM,
        SASEBO_COMMUNITY,
        SASEBO_HOUSING_AREA,
        SASEBO_ROAD_1_4,
        SASEBO_ROAD_1_5,
        SASEBO_ROAD_2_IKKATSU,
        SASEBO_ROAD_2_KOBETSU,
        SASEBO_ROAD_3,
        SASEBO_ROAD_CITY,
    ]


def check_file_exists(file_path: str) -> bool:
    """
    指定されたファイルパスが存在するかチェックする。

    Args:
        file_path (str): チェックするファイルパス

    Returns:
        bool: ファイルが存在する場合はTrue、存在しない場合はFalse
    """
    exists = os.path.exists(file_path)
    if not exists:
        print(f"警告: ファイルが見つかりません: {file_path}")
    return exists

# TODO: 1.データのプロット


# TODO: 2.DEMデータと住民自治組織のデータを重ねて、住民自治組織ごとに傾斜・SHCの算出ができるようにし、全域が斜面市街地の自治組織だけを抽出する。


# TODO: 3.住民自治組織ごとに建物数のカウント、属性情報の取得ができるようにする。


# TODO: 4.住民自治組織ごとに接道不良率、地区の高齢化率、推定高齢者数の算出ができるようにする。


# TODO: 5.住民自治組織ごとに将来の接道不良住宅に住む高齢者数の算出ができるようにする。


if __name__ == "__main__":
    # ファイルの存在確認
    if not all(check_file_exists(file) for file in files_to_check):
        print("一部のファイルが見つかりませんでした。プログラムを終了します。")
        exit(1)
    else:
        print("file path OK")