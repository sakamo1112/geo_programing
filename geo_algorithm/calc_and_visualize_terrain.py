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
    visualize_slope_shc_relationship,
    visualize_slope_shc_relationship_with_top_cities,
    visualize_top_cities_on_map,
    visualize_top_steep_cities_comparison,
)

DEM_DIR = "/Users/sakamo/Desktop/GISDATA/DEM_地方別/"
# DEM_地方別/以下に地方名.tifファイルが格納されている
HOUSING_AREA_DIR = "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/"
# 住居系用途地域2019_10万人以上の自治体_全国/A29-19_(都道府県コード)/以下にhousing_(自治体名).shpファイルが格納されている
TARGET_LIST_EXCEL = "/Users/sakamo/Desktop/GISDATA/10man_status.xlsx"
RESULT_XLSX = "result/terrain_statistics.xlsx"

area_dict = {
    "北海道": ["北海道"],
    "東北": ["青森県", "岩手県", "秋田県", "山形県", "宮城県", "福島県"],
    "関東": ["群馬県", "栃木県", "茨城県", "埼玉県", "千葉県", "東京都", "神奈川県", "長野県", "山梨県"],
    "北陸": [
        "新潟県",
        "富山県",
        "石川県",
        "福井県",
    ],
    "中部": ["岐阜県", "静岡県", "愛知県", "三重県"],
    "近畿": ["滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
    "中国": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
    "四国": ["徳島県", "香川県", "愛媛県", "高知県"],
    "九州": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"],
}
kyogikai_cities = [
    "尾道市",
    "横須賀市",
    "下関市",
    "呉市",
    "佐世保市",
    "小樽市",
    "神戸市",
    "長崎市",
    "熱海市",
    "函館市",
    "別府市",
    "北九州市",
]


def calc_terrain_of_a_city(city_name, housing_area_path, src):
    # 住居系用途地域のSHPファイルを読み込む
    housing_area_gdf = gpd.read_file(housing_area_path)

    try:
        bounds = housing_area_gdf.total_bounds
        bbox = box(*bounds)
        shapes = [bbox]
        bbox_elevation, bbox_transform = mask(
            src, shapes=shapes, crop=True, nodata=np.nan
        )
        bbox_elevation = bbox_elevation[0]
        housing_mask = rasterio.features.rasterize(
            [(geom, 1) for geom in housing_area_gdf.geometry],
            out_shape=bbox_elevation.shape,
            transform=bbox_transform,
            fill=0,
            dtype=np.uint8,
        )
        housing_area_height = calc_and_visualize_height(
            city_name, bbox_elevation, housing_mask, visualize=True
        )
        (
            housing_area_slope,
            housing_area_slope_removed,
            steep_ratio,
        ) = calc_and_visualize_slope(
            city_name, bbox_elevation, housing_mask, visualize=True
        )
        housing_area_shc, housing_area_shc_removed = calc_and_visualize_shc(
            city_name, bbox_elevation, housing_mask, visualize=True
        )
        steep_area_shc = calc_shc_in_steep_area(bbox_elevation, housing_mask)

        steep_mask = (housing_area_slope >= 5) & (~np.isnan(housing_area_slope))
        steep_area_slope = np.where(steep_mask, housing_area_slope, np.nan)

        city_terrain_stats = {
            "都道府県コード": city_name.split("_")[0],
            "都道府県名": city_name.split("_")[1].split("(")[1].split(")")[0],
            "市区町村名": city_name.split("_")[1].split("(")[0],
            "標高_平均値": np.nanmean(housing_area_height),
            "標高_中央値": np.nanmedian(housing_area_height),
            "標高_標準偏差": np.nanstd(housing_area_height),
            "傾斜度_平均値": np.nanmean(housing_area_slope),
            "傾斜度_中央値": np.nanmedian(housing_area_slope),
            "傾斜度_標準偏差": np.nanstd(housing_area_slope),
            "傾斜度(外れ値除去後)_平均値": np.nanmean(housing_area_slope_removed),
            "傾斜度(外れ値除去後)_中央値": np.nanmedian(housing_area_slope_removed),
            "傾斜度(外れ値除去後)_標準偏差": np.nanstd(housing_area_slope_removed),
            "SHC_平均値": np.nanmean(housing_area_shc),
            "SHC_中央値": np.nanmedian(housing_area_shc),
            "SHC_標準偏差": np.nanstd(housing_area_shc),
            "SHC(外れ値除去後)_平均値": np.nanmean(housing_area_shc_removed),
            "SHC(外れ値除去後)_中央値": np.nanmedian(housing_area_shc_removed),
            "SHC(外れ値除去後)_標準偏差": np.nanstd(housing_area_shc_removed),
            "住居系用途地域に占める斜面市街地の割合": steep_ratio,
            "斜面市街地の傾斜度_平均値": np.nanmean(steep_area_slope),
            "斜面市街地の傾斜度_中央値": np.nanmedian(steep_area_slope),
            "斜面市街地の傾斜度_標準偏差": np.nanstd(steep_area_slope),
            "斜面市街地のSHC_平均値": np.nanmean(steep_area_shc),
            "斜面市街地のSHC_中央値": np.nanmedian(steep_area_shc),
            "斜面市街地のSHC_標準偏差": np.nanstd(steep_area_shc),
        }
        return city_terrain_stats
    except ValueError as e:
        print(f"Warning: {city_name}の処理でエラーが発生: {e}")
        return None


def calc_terrain_of_283cities(pref_code_dict):
    """
    283自治体の標高・傾斜度・SHCを計算する関数

    Args:
        pref_code_dict (dict): 都道府県名と都道府県コードの辞書
    Returns:
        df_stats (pandas.DataFrame): 283自治体の標高・傾斜度・SHCの統計情報
    """
    counter = 0
    df_stats = pd.DataFrame()
    for area in area_dict.keys():
        print(f"{area}地方のデータ作成中...")
        # 地方DEMのパスを取得
        dem_list = os.listdir(DEM_DIR)
        for dem_name in dem_list:
            if area in dem_name and dem_name.endswith(".tif"):
                area_dem_path = os.path.join(DEM_DIR, dem_name)
                print(f"DEM Path: {area_dem_path}")
                print("DEM読み込み中...")
                with rasterio.open(area_dem_path) as src:
                    elevation = src.read(1)  # 最初のバンドを取得
                    elevation = np.where(elevation == -9999, np.nan, elevation)
                    print("DEM読み込み完了")

                    # 各県内の住居系用途地域のパスを取得
                    for pref in area_dict[area]:
                        pref_dir_name = f"A29-19_{pref_code_dict[pref]}"
                        housing_area_dir = os.path.join(HOUSING_AREA_DIR, pref_dir_name)
                        housing_area_list = os.listdir(housing_area_dir)
                        for housing_area_name in housing_area_list:
                            # 各自治体の住居系用途地域データ(SHP)に対して傾斜度・SHCを計算
                            if housing_area_name.endswith(".shp"):
                                city_name = (
                                    pref_code_dict[pref]
                                    + "_"
                                    + housing_area_name.split("_")[1].split(".")[0]
                                    + f"({pref})"
                                )
                                housing_area_path = os.path.join(
                                    housing_area_dir, housing_area_name
                                )
                                print(
                                    f"\n-----{city_name}-----\nshapefile Path: {housing_area_path}"
                                )
                                print(f"{city_name}の傾斜度計算中...")
                                city_terrain_stats = calc_terrain_of_a_city(
                                    city_name, housing_area_path, src
                                )
                                df_stats = pd.concat(
                                    [df_stats, pd.DataFrame([city_terrain_stats])],
                                    ignore_index=True,
                                )
                                counter += 1
                    print(df_stats)
    print(f"{counter}件のデータを作成しました。")

    return df_stats


if __name__ == "__main__":
    # 都道府県コードと県名の辞書を作成
    target_list = pd.read_excel(TARGET_LIST_EXCEL)
    # 都道府県名と都道府県コードの辞書を作成
    pref_code_dict = {}
    for _, row in target_list.iterrows():
        pref_name = row["都道府県名"].split("_")[1]  # "(都道府県コード)_都道府県名" から都道府県名を抽出
        pref_code = row["都道府県名"].split("_")[0]  # "(都道府県コード)_都道府県名" から都道府県コードを抽出
        pref_code_dict[pref_name] = pref_code

    if_calc_terrain = False
    if_visualize_terrain = True

    if if_calc_terrain:
        # 283都市の標高・傾斜度・SHCを計算し、エクセルデータを作成
        df_stats = calc_terrain_of_283cities(pref_code_dict)
        df_stats.to_excel(RESULT_XLSX, index=False)

    if if_visualize_terrain:
        # 作成したエクセルデータを読み込み
        df_stats = pd.read_excel(RESULT_XLSX, dtype={"都道府県コード": str})
        top_steep_cities = df_stats.sort_values(
            "住居系用途地域に占める斜面市街地の割合", ascending=False
        ).head(27)
        top_steep_cities = list(top_steep_cities["市区町村名"])

        hazure = False
        only_steep_area = False
        visualize_slope_area_ratio_histogram(df_stats)
        #visualize_top_cities_on_map(df_stats, hazure, if_kanto=True, thr_rank=27)
        visualize_top_steep_cities_comparison(df_stats, top_steep_cities)
        visualize_slope_shc_relationship(
            df_stats,
            kyogikai_cities,
            top_steep_cities,
            only_steep_area,
            "red",
            hazure,
            thr_rank=27,
        )
        visualize_slope_shc_relationship(
            df_stats,
            kyogikai_cities,
            top_steep_cities,
            only_steep_area,
            "orange",
            hazure,
            thr_rank=27,
        )
        visualize_slope_shc_relationship_with_top_cities(
            df_stats, top_steep_cities, only_steep_area, hazure
        )
