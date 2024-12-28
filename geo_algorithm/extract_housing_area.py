import os

import geopandas as gpd
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from scipy.ndimage import convolve, gaussian_filter
from shapely.geometry import shape

DIR_PATH = "/Users/sakamo/Desktop/用途地域2019_全国/"
HOUSING_PATH = "/Users/sakamo/Desktop/住居系用途地域2019_全国/"
HOUSING_10MAN_PATH = "/Users/sakamo/Desktop/住居系用途地域2019_10万人以上の自治体_全国/"
TEN_MAN_PATH = "/Users/sakamo/Desktop/GISDATA/人口10万人以上の自治体_2020国勢調査.xlsx"


def extract_housing_area(dir_path: str):
    """
    47都道府県のディレクトリ内のSHPファイルを全て読み込む関数

    Parameters
    ----------
    dir_path : str
        都道府県フォルダが格納されているディレクトリのパス
        <SHPファイルのカラム>
        A29_002 : 都道府県名
        A29_003 : 市区町村名
        A29_004 : 用途地域コード(住居系用途地域は1,2,3,4,5,6,7,21)
        A29_005 : 用途地域名
        A29_006 : 建蔽率
        A29_007 : 容積率

    Returns
    -------
    gdf : GeoDataFrame
        読み込んだSHPファイルを結合したGeoDataFrame
    """
    gdfs = []

    # 都道府県フォルダを走査
    for pref_dir in os.listdir(dir_path):
        if pref_dir.startswith("A29-19_"):
            # シェープファイルが格納されているディレクトリのパスを作成
            shp_dir = os.path.join(dir_path, pref_dir, "01-01_シェープファイル形式")

            # シェープファイルディレクトリが存在する場合のみ処理
            if os.path.exists(shp_dir):
                # ディレクトリ内のSHPファイルを走査
                for filename in os.listdir(shp_dir):
                    if filename.endswith(".shp"):
                        file_path = os.path.join(shp_dir, filename)
                        df = gpd.read_file(file_path)
                        # WGS84に変換
                        df = df.to_crs("EPSG:4326")
                        # 住居系用途地域のコードを指定
                        housing_codes = [1, 2, 3, 4, 5, 6, 7, 21]
                        # 住居系用途地域のみを抽出
                        housing_df = df[df["A29_004"].isin(housing_codes)]
                        if not housing_df.empty:
                            gdfs.append(housing_df)
                            output_dir = os.path.join(HOUSING_PATH, pref_dir)
                            os.makedirs(output_dir, exist_ok=True)
                            city_name = housing_df["A29_003"].unique()[0]
                            output_path = os.path.join(
                                output_dir, f"housing_{city_name}.shp"
                            )

                            try:
                                housing_df.to_file(output_path, encoding="cp932")
                                print(f"保存成功: {output_path}")
                            except Exception as e:
                                print(f"ファイル保存エラー ({city_name}): {str(e)}")
                            print(f"住居系用途地域の抽出・出力完了: {pref_dir} - {filename}")

    # 全てのGeoDataFrameを結合
    gdf = pd.concat(gdfs, ignore_index=True)

    # 市区町村名（A29_003列）を出力
    print("\n市区町村一覧:")
    print(gdf["A29_003"].unique())

    return gdf


def extract_10man_area():
    """
    10万人以上の自治体のSHPファイルを抽出する関数
    """
    ten_man_df = pd.read_excel(TEN_MAN_PATH)
    # 市区町村名を抽出（都道府県名を除去）
    ten_man_df["都道府県・市区町村名"] = ten_man_df["都道府県・市区町村名"].str.split("_").str[-1]

    return ten_man_df


def extract_housing_10man_area(dir_path: str, ten_man_df: pd.DataFrame):
    """
    10万人以上の自治体の住居系用途地域のSHPファイルを抽出する関数

    Parameters
    ----------
    dir_path : str
        都道府県フォルダが格納されているディレクトリのパス
        <SHPファイルのカラム>
        A29_002 : 都道府県名
        A29_003 : 市区町村名
        A29_004 : 用途地域コード(住居系用途地域は1,2,3,4,5,6,7,21)
        A29_005 : 用途地域名
        A29_006 : 建蔽率
        A29_007 : 容積率

    Returns
    -------
    gdf : GeoDataFrame
        読み込んだSHPファイルを結合したGeoDataFrame
    """
    gdfs = []

    # 都道府県フォルダを走査
    counter = 0
    for pref_dir in os.listdir(dir_path):
        if pref_dir.startswith("A29-19_"):
            # シェープファイルが格納されているディレクトリのパスを作成
            shp_dir = os.path.join(dir_path, pref_dir, "01-01_シェープファイル形式")

            # シェープファイルディレクトリが存在する場合のみ処理
            if os.path.exists(shp_dir):
                # ディレクトリ内のSHPファイルを走査
                for filename in os.listdir(shp_dir):
                    if filename.endswith(".shp"):
                        file_path = os.path.join(shp_dir, filename)
                        df = gpd.read_file(file_path)
                        # WGS84に変換
                        df = df.to_crs("EPSG:4326")
                        # 住居系用途地域のコードを指定
                        housing_codes = [1, 2, 3, 4, 5, 6, 7, 21]
                        # 住居系用途地域のみを抽出
                        housing_df = df[df["A29_004"].isin(housing_codes)]
                        if not housing_df.empty:
                            gdfs.append(housing_df)
                            output_dir = os.path.join(HOUSING_10MAN_PATH, pref_dir)
                            os.makedirs(output_dir, exist_ok=True)
                            city_name = housing_df["A29_003"].unique()[0]
                            output_path = os.path.join(
                                output_dir, f"housing_{city_name}.shp"
                            )
                            # 10万人以上の自治体リストに含まれる場合のみ保存
                            if city_name in ten_man_df["都道府県・市区町村名"].values:
                                try:
                                    housing_df.to_file(output_path, encoding="cp932")
                                    # print(f"保存成功: {output_path}")
                                    ten_man_df.loc[
                                        ten_man_df["都道府県・市区町村名"] == city_name, "status"
                                    ] = "success"

                                    counter += 1
                                except Exception as e:
                                    print(f"ファイル保存エラー ({city_name}): {str(e)}")

    # 全てのGeoDataFrameを結合
    gdf = pd.concat(gdfs, ignore_index=True)
    print(counter)
    print(ten_man_df[ten_man_df["status"] == "fail"])
    # save
    ten_man_df.to_excel("/Users/sakamo/Desktop/10man_status.xlsx", index=False)

    # 市区町村名（A29_003列）を出力
    print(f"10万人以上の自治体の住居系用途地域のSHPファイルの抽出・出力完了: {counter}件")

    return gdf


def extract_kesson_housing_area(dir_path: str):
    """
    住居系用途地域のSHPファイルを抽出する関数
    """
    df = gpd.read_file(dir_path, encoding="cp932")
    # 列名を変更
    df = df[["Pref", "Cityname", "YoutoID", "用途地域", "建ぺい率", "容積率", "geometry"]]
    df = df.rename(
        columns={
            "Pref": "A29_002",
            "Cityname": "A29_003",
            "YoutoID": "A29_004",
            "用途地域": "A29_005",
            "建ぺい率": "A29_006",
            "容積率": "A29_007",
        }
    )
    # 住居系用途地域のコードを指定
    housing_codes = [1, 2, 3, 4, 5, 6, 7, 21]
    # 住居系用途地域のみを抽出
    housing_df = df[df["A29_004"].isin(housing_codes)]
    # 保存
    housing_df.to_file("/Users/sakamo/Desktop/housing_那覇市.shp", encoding="cp932")
    print("保存完了")


def read_housing_10man_file(dir_path: str, ten_man_excel_path: str):
    """
    10万人以上の自治体の住居系用途地域のSHPファイルを読み込む関数
    """
    counter = 0
    ten_man_df = extract_10man_area()
    found_cities = set()  # 見つかった市区町村を記録
    ten_man_cities = set(ten_man_df["都道府県・市区町村名"])  # 10万人以上の自治体のセット
    found_cities = []  # 見つかった全ての市区町村を記録
    print(f"10万人以上の自治体数: {len(ten_man_cities)}")
    # デバッグ用：10万人以上の自治体リストを表示
    print("\n10万人以上の自治体リスト:")
    print(sorted(ten_man_cities))

    # 都道府県フォルダを走査
    for pref_dir in os.listdir(dir_path):
        pref_path = os.path.join(dir_path, pref_dir)
        if os.path.isdir(pref_path):
            # フォルダ内のSHPファイルを走査
            for filename in os.listdir(pref_path):
                if filename.startswith("housing_") and filename.endswith(".shp"):
                    file_path = os.path.join(pref_path, filename)
                    try:
                        df = gpd.read_file(file_path)
                        city_name = df["A29_003"].iloc[0]
                        found_cities.append(city_name)
                        counter += 1
                    except Exception as e:
                        print(f"ファイル読み込みエラー ({filename}): {str(e)}")


if __name__ == "__main__":
    # gdf = extract_housing_area(DIR_PATH)
    # print(gdf.head())
    read_housing_10man_file(HOUSING_10MAN_PATH, TEN_MAN_PATH)
    """ten_man_df = extract_10man_area()
    ten_man_df['status'] = 'fail'
    gdf = extract_housing_10man_area(DIR_PATH, ten_man_df)"""
    """KESSONFILE__PATH = "/Users/sakamo/Desktop/kesson_data/47_那覇市/47201_youto.shp"
    extract_kesson_housing_area(KESSONFILE__PATH)"""
