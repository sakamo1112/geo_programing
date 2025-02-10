import os
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import warp
import matplotlib.pyplot as plt
import japanize_matplotlib
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box
import contextily as ctx  # OpenStreetMap表示用
from matplotlib.patches import Patch  # 凡例用
from matplotlib_scalebar.scalebar import ScaleBar  # スケールバー用
from pathlib import Path
import shapely.ops
from rasterio.plot import show
from matplotlib.lines import Line2D

SASEBO_SLOPE_DEM = "result/slope/slope_佐世保市.tif"

ICHISITEI = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/位置指定道路.shp" # 'R22_003'列が「主路線・道路種別コード」で1,2,3,4,6が国道・県道・高速道路
SHIDO = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/市道(路線)/路線_列改訂版.shp" # 幅員：'平均幅員(m', 延長：'延長(m)'
DRM_ROAD = "/Users/sakamo/Desktop/GISDATA/DRM_data/shape/全道路リンク標高.shp"
SASEBO_COMMUNITY = "/Users/sakamo/Desktop/GISDATA/自治協議会/sasebo_community.shp"
SASEBO_HOUSING_AREA = (
    "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_42/housing_佐世保市.shp"
)

GOGO = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/4m以上の道路バッファ/1項5号.shp"
SHITEI_FOR_BUFFER = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/車両通行可能な道路バッファ/shitei.shp"
SANKO = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/4m以上の道路バッファ/3項.shp"
SHIDO_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/市道_for_use.shp"
KENDO_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/県道_for_use.shp"
KOKUDO_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/国道_for_use.shp"
ICHISITEI_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/位置指定道路_for_use.shp"

ICHISITEI_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/位置指定道路_for_use.shp"
SHIDO_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/道路ポリゴン/市道.shp"
KENDO_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/道路ポリゴン/県道.shp"
KOKUDO_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/道路ポリゴン/国道.shp"


def plot_road_data(ichisitei, shido, national, prefectural, housing_area, community):
    # データをWeb メルカトル投影に変換
    ichisitei = ichisitei.to_crs(epsg=3857)
    shido = shido.to_crs(epsg=3857)
    national = national.to_crs(epsg=3857)
    prefectural = prefectural.to_crs(epsg=3857)
    housing_area = housing_area.to_crs(epsg=3857)

    # 位置指定道路をラインに変換
    ichisitei_line = ichisitei.boundary

    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 住居系用途地域の範囲を取得
    bounds = housing_area.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # 住居系用途地域を薄い黄色で表示（凡例用のダミープロットを追加）
    housing_area.plot(ax=ax, color='yellow', alpha=0.3)
    ax.plot([], [], color='yellow', alpha=0.3, label='住居系用途地域', linewidth=10)
    
    # 各道路の描画
    if not national.empty:
        national.plot(ax=ax, color="orange", label="国道", alpha=0.5, linewidth=1)
    if not prefectural.empty:
        prefectural.plot(ax=ax, color="green", label="県道", alpha=0.5, linewidth=0.8)
    
    # 市道と位置指定道路の描画
    shido.plot(ax=ax, color="blue", label="市道", alpha=1, linewidth=0.5)
    ichisitei_line.plot(ax=ax, color="red", label="指定道路", linewidth=0.5)
    
    # OpenStreetMapを背景に追加
    ctx.add_basemap(ax)
    
    ax.legend()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 画像として保存（DPI=300で高解像度、bbox_inches='tight'で余白を調整）
    plt.savefig('佐世保市道路ネットワーク.png', dpi=300, bbox_inches='tight')
    plt.close()  # メモリ解放のためにfigureを閉じる

def create_road_data():
    """道路データを読み込み、必要な形式に変換して保存する関数"""
    # データの読み込み
    ichisitei = gpd.read_file(ICHISITEI, encoding="latin1")
    shido = gpd.read_file(
        SHIDO, 
        encoding="cp932",
        include_fields=['路線名称', '名称', '平均幅員(m', '延長(m)', 'geometry']
    )
    drm_road = gpd.read_file(DRM_ROAD, encoding="latin1")

    # 座標系の設定と変換
    ichisitei = ichisitei.set_crs(epsg=4326)
    shido = shido.set_crs(epsg=6669)
    drm_road = drm_road.set_crs(epsg=4612)
    shido = shido.to_crs(epsg=4326)
    drm_road = drm_road.to_crs(epsg=4326)
    
    # DRMデータを道路種別ごとに分類
    highway = drm_road[drm_road['R22_003'].isin([1, 2])]
    national = drm_road[drm_road['R22_003'] == 3]
    prefectural = drm_road[drm_road['R22_003'].isin([4, 6])]
    
    # 市道データの保存
    shido = shido.rename(columns={'平均幅員(m': '平均幅員', '延長(m)': '延長'})
    shido.to_file(
        SHIDO_FOR_USE,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )
    
    # 県道データの保存
    prefectural.to_file(
        KENDO_FOR_USE,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )
    
    # 国道データの保存
    national.to_file(
        KOKUDO_FOR_USE,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )
    
    # 位置指定道路データの保存
    ichisitei_selected = ichisitei[['geometry']]
    ichisitei_selected.to_file(
        ICHISITEI_FOR_USE,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )
    
    return ichisitei, shido, highway, national, prefectural

def create_road_polygon(ichisitei, shido, national, prefectural):
    """道路のラインデータを幅員情報に基づいてポリゴンに変換する関数
    
    Args:
        shido (GeoDataFrame): 市道データ
        national (GeoDataFrame): 国道データ
        prefectural (GeoDataFrame): 県道データ
    """
    # 幅員コードから実際の幅員（m）に変換する関数
    def get_width_from_code(code):
        width_dict = {
            1: 13.0,  # 13.0m以上の場合は13.0mとして扱う
            2: 9.25,  # 5.5-13.0mの中間値
            3: 4.25,  # 3.0-5.5mの中間値
            4: 3.0,   # 3.0m未満の場合は3.0mとして扱う
        }
        return width_dict.get(code, 9.25)  # コードが不明な場合は2の幅員を使用
    
    # 市道のポリゴン化（幅員情報を使用）
    shido = shido.to_crs(epsg=3857)  # メートル単位の座標系に変換
    shido['buffer_width'] = shido['平均幅員(m'] / 2
    shido_polygon = shido.apply(
        lambda row: row.geometry.buffer(row.buffer_width, cap_style=2),
        axis=1
    )
    shido_gdf = gpd.GeoDataFrame(geometry=shido_polygon, crs=shido.crs)
    shido_gdf = shido_gdf.to_crs(epsg=4326)  # 保存用に4326に戻す
    
    # 国道のポリゴン化
    national = national.to_crs(epsg=3857)
    national['buffer_width'] = national['R22_005'].apply(get_width_from_code) / 2
    national_polygon = national.apply(
        lambda row: row.geometry.buffer(row.buffer_width, cap_style=2),
        axis=1
    )
    national_gdf = gpd.GeoDataFrame(geometry=national_polygon, crs=national.crs)
    national_gdf = national_gdf.to_crs(epsg=4326)
    
    # 県道のポリゴン化
    prefectural = prefectural.to_crs(epsg=3857)
    prefectural['buffer_width'] = prefectural['R22_005'].apply(get_width_from_code) / 2
    prefectural_polygon = prefectural.apply(
        lambda row: row.geometry.buffer(row.buffer_width, cap_style=2),
        axis=1
    )
    prefectural_gdf = gpd.GeoDataFrame(geometry=prefectural_polygon, crs=prefectural.crs)
    prefectural_gdf = prefectural_gdf.to_crs(epsg=4326)
    
    # 各ポリゴンデータの保存
    shido_gdf.to_file(
        SHIDO_POLYGON,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )
    
    national_gdf.to_file(
        KOKUDO_POLYGON,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )
    
    prefectural_gdf.to_file(
        KENDO_POLYGON,
        driver='ESRI Shapefile',
        encoding='cp932',
        crs='EPSG:4326'
    )

    # 5mバッファを持つポリゴンの生成と保存
    buffer_distance = 10  # メートル単位
    
    # 各道路ポリゴンに5mバッファを追加（平面直角座標系I系で計算）
    shido_buffer = shido_gdf.to_crs(epsg=6669).buffer(buffer_distance).to_crs(epsg=4326)
    national_buffer = national_gdf.to_crs(epsg=6669).buffer(buffer_distance).to_crs(epsg=4326)
    prefectural_buffer = prefectural_gdf.to_crs(epsg=6669).buffer(buffer_distance).to_crs(epsg=4326)
    ichisitei_buffer = ichisitei.to_crs(epsg=6669).buffer(buffer_distance).to_crs(epsg=4326)
    
    # バッファ付きポリゴンの保存
    gpd.GeoDataFrame(geometry=shido_buffer, crs='EPSG:4326').to_file(
        SHIDO_POLYGON.replace('.shp', '_5m_buffer.shp'),
        driver='ESRI Shapefile',
        encoding='cp932'
    )
    
    gpd.GeoDataFrame(geometry=national_buffer, crs='EPSG:4326').to_file(
        KOKUDO_POLYGON.replace('.shp', '_10m_buffer.shp'),
        driver='ESRI Shapefile',
        encoding='cp932'
    )
    
    gpd.GeoDataFrame(geometry=prefectural_buffer, crs='EPSG:4326').to_file(
        KENDO_POLYGON.replace('.shp', '_10m_buffer.shp'),
        driver='ESRI Shapefile',
        encoding='cp932'
    )
    
    gpd.GeoDataFrame(geometry=ichisitei_buffer, crs='EPSG:4326').to_file(
        ICHISITEI_POLYGON.replace('.shp', '_10m_buffer.shp'),
        driver='ESRI Shapefile',
        encoding='cp932'
    )

    # 幅員区分の集計と表示
    print("\n=== 道路幅員区分の集計 ===")
    
    print("\n【国道】")
    national_width_counts = national['R22_005'].value_counts().sort_index()
    for code, count in national_width_counts.items():
        width_range = {
            1: "13.0m以上",
            2: "5.5m以上13.0m未満",
            3: "3.0m以上5.5m未満",
            4: "3.0m未満"
        }.get(code, "不明")
        print(f"幅員区分{code}（{width_range}）: {count}件")
    
    print("\n【県道】")
    prefectural_width_counts = prefectural['R22_005'].value_counts().sort_index()
    for code, count in prefectural_width_counts.items():
        width_range = {
            1: "13.0m以上",
            2: "5.5m以上13.0m未満",
            3: "3.0m以上5.5m未満",
            4: "3.0m未満"
        }.get(code, "不明")
        print(f"幅員区分{code}（{width_range}）: {count}件")
    
    return shido_polygon, national_polygon, prefectural_polygon

def plot_community_and_other_data(community, ichisitei_polygon, shido_polygon, national_polygon, prefectural_polygon):
    """各コミュニティの傾斜度と道路ポリゴンを重ねて表示する関数"""    
    # 出力ディレクトリの作成
    output_dir = "sasebo_com_result/"
    os.makedirs(output_dir, exist_ok=True)
    
    # すべてのデータをWebメルカトル（EPSG:3857）に変換
    community = community.to_crs(epsg=3857)
    ichisitei_polygon = ichisitei_polygon.to_crs(epsg=3857)
    shido_polygon = shido_polygon.to_crs(epsg=3857)
    national_polygon = national_polygon.to_crs(epsg=3857)
    prefectural_polygon = prefectural_polygon.to_crs(epsg=3857)
    
    # 各コミュニティについて処理
    for idx, area in community.iterrows():
        name = area['NAME_x']
        print(f"Processing {name}...")
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # コミュニティの範囲を設定
        bounds = area.geometry.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # 傾斜度データの表示（コミュニティの範囲内のみ）
        with rasterio.open(SASEBO_SLOPE_DEM) as src:
            # DEMデータを読み込み
            data = src.read(1)
            
            # コミュニティの範囲をソース座標系に変換
            area_gdf = gpd.GeoDataFrame(geometry=[area.geometry], crs=community.crs)
            area_gdf = area_gdf.to_crs(src.crs)
            bounds_src = area_gdf.geometry[0].bounds
            
            # ピクセル座標に変換
            window = src.window(*bounds_src)
            window = window.round_lengths()
            window = window.round_offsets()
            
            # データをクリップ
            clipped_data = src.read(1, window=window)
            
            # 傾斜度データの表示（透明度を0.5に調整）
            im = ax.imshow(
                clipped_data,
                extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                cmap='YlOrRd',
                alpha=0.5,  # DEMの透明度を下げる
                vmin=0,
                vmax=45,
                zorder=1
            )
        
        # コミュニティ外を灰色で塗る
        # まず、表示範囲全体を覆う四角形を作成
        display_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        # コミュニティ外の領域を作成（四角形からコミュニティ領域を引く）
        outside_area = display_box.difference(area.geometry)
        # より濃い灰色で塗る
        gpd.GeoSeries([outside_area], crs=community.crs).plot(
            ax=ax,
            color='#404040',  # より濃い灰色に変更
            alpha=0.5,        # 透明度も調整
            zorder=2
        )
        
        # 道路データのクリップと表示（zorderを3に変更して灰色の上に表示）
        try:
            if not national.empty:
                clipped_national = national.clip(area.geometry)
                if not clipped_national.empty:
                    clipped_national.plot(ax=ax, color='orange', alpha=0.7, zorder=3)
        except Exception as e:
            print(f"Warning: Failed to plot national roads - {e}")

        try:
            if not prefectural.empty:
                clipped_prefectural = prefectural.clip(area.geometry)
                if not clipped_prefectural.empty:
                    clipped_prefectural.plot(ax=ax, color='green', alpha=0.7, zorder=3)
        except Exception as e:
            print(f"Warning: Failed to plot prefectural roads - {e}")

        try:
            if not shido_polygon.empty:
                clipped_shido = shido_polygon.clip(area.geometry)
                if not clipped_shido.empty:
                    clipped_shido.plot(ax=ax, color='blue', alpha=0.7, zorder=3)
        except Exception as e:
            print(f"Warning: Failed to plot city roads - {e}")

        try:
            if not ichisitei_polygon.empty:
                clipped_ichisitei = ichisitei_polygon.clip(area.geometry)
                if not clipped_ichisitei.empty:
                    clipped_ichisitei.plot(ax=ax, color='red', alpha=0.7, zorder=3)
        except Exception as e:
            print(f"Warning: Failed to plot ichisitei roads - {e}")
        
        # コミュニティの境界線を表示（最前面に表示）
        gpd.GeoSeries([area.geometry]).boundary.plot(
            ax=ax,
            color='black',
            linewidth=2,
            zorder=5
        )
        
        # カラーバーの追加
        cbar = plt.colorbar(im)
        cbar.set_label('傾斜度 (度)')
        
        # 凡例要素を作成（常に全ての道路種別を表示）
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=4, label='位置指定道路'),
            plt.Line2D([0], [0], color='blue', lw=4, label='市道'),
            plt.Line2D([0], [0], color='green', lw=4, label='県道'),
            plt.Line2D([0], [0], color='orange', lw=4, label='国道')
        ]
        
        # タイトルと凡例の設定
        plt.title(f"{name}の傾斜度と道路網", fontsize=16, pad=20)
        ax.legend(handles=legend_elements, loc='upper right')
        
        # グリッドと軸の設定
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # アスペクト比を設定
        ax.set_aspect('equal')
        
        # 画像の保存
        plt.savefig(
            f"{output_dir}/{name}_slope_road.png",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()

def plot_community(community):
    """地域コミュニティを描画する関数
    
    Args:
        community (GeoDataFrame): コミュニティデータ    
    """
    # Webメルカトル投影に変換
    community = community.to_crs(epsg=3857)
    
    # 図の作成
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # コミュニティの範囲を取得して設定
    bounds = community.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # コミュニティの描画
    community.plot(
        ax=ax,
        color='yellow',
        alpha=0.4,
        edgecolor='black',
        linewidth=1.5
    )
    
    # OpenStreetMapを背景に追加
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik
    )
    
    # 凡例の追加
    legend_elements = [
        Patch(facecolor='yellow', 
              edgecolor='black', 
              alpha=0.7,
              label='町内会・自治会の区域')
    ]
    ax.legend(handles=legend_elements, 
             loc='upper right',
             bbox_to_anchor=(1.0, 1.0),
             fontsize=10)
    
    # スケールバーの追加（5kmを目安に）
    scalebar = ScaleBar(
        1,  # 1 pixel = 1 meter (EPSG:3857はメートル単位)
        location='lower right',  # 右下に配置
        length_fraction=0.3,  # スケールバーの長さ
        fixed_value=5,  # 5kmに固定
        fixed_units='km',  # 単位をkmに
        box_alpha=0.5,  # 背景の透明度
    )
    ax.add_artist(scalebar)
    
    # グリッドと軸の設定
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # タイトルの設定
    plt.title('佐世保市コミュニティマップ', fontsize=14, pad=20)
    
    # 画像として保存
    plt.savefig(
        '佐世保市コミュニティ.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()

def merge_gml_to_shp(input_dir: str, output_shp: str):
    """
    指定されたディレクトリ内の全ての.gmlファイルを読み込み、
    1つのシェープファイルに結合し、描画する関数
    
    Args:
        input_dir (str): .gmlファイルが格納されているディレクトリのパス
        output_shp (str): 出力するシェープファイルのパス
    """
    # .gmlファイルのリストを取得
    gml_files = list(Path(input_dir).glob("*.shp"))
    
    if not gml_files:
        print("指定されたディレクトリに.gmlファイルが見つかりません。")
        return
    
    # 全てのGMLファイルを読み込んでリストに格納
    gdfs = []
    for gml_file in gml_files:
        try:
            print(f"読み込み中: {gml_file.name}")
            gdf = gpd.read_file(gml_file, encoding="cp932", crs="EPSG:4326")
            gdfs.append(gdf)
        except Exception as e:
            print(f"エラー - {gml_file.name}の読み込みに失敗: {str(e)}")
    
    if not gdfs:
        print("読み込み可能なGMLファイルがありませんでした。")
        return
    
    # 全てのGeoDataFrameを結合
    print("ファイルを結合中...")
    merged_gdf = gpd.pd.concat(gdfs, ignore_index=True)
    
    # シェープファイルとして保存
    print(f"シェープファイルとして保存中: {output_shp}")
    merged_gdf.to_file(output_shp, driver='ESRI Shapefile', encoding='utf-8')
    
    print(f"完了！ {len(gdfs)}個のファイルを結合し、{output_shp}に保存しました。")
    
    return merged_gdf

def extract_residential_buildings(building_path, housing_area_path, output_path):
    """住居系用途地域内の住宅関連建物を抽出する関数
    
    Args:
        building_path (str): 建物データのパス
        housing_area_path (str): 住居系用途地域のパス
        output_path (str): 出力ファイルのパス
    """
    # 建物データと住居系用途地域データの読み込み
    print("データを読み込んでいます...")
    buildings = gpd.read_file(building_path, encoding="utf-8")
    housing_area = gpd.read_file(housing_area_path, encoding="cp932")
    
    # 座標系を統一
    buildings = buildings.set_crs(epsg=4326)
    housing_area = housing_area.set_crs(epsg=4326)
    
    # 住宅関連の用途コード
    residential_uses = [
        '住宅', '共同住宅', '店舗等併用住宅', '店舗等併用共同住宅', 
        '作業所併用住宅', '411', '412', '413', '414', '415'
    ]
    
    # 住宅関連建物のみを抽出
    print("住宅関連建物を抽出しています...")
    residential_buildings = buildings[buildings['usage'].isin(residential_uses)]
    
    # 住居系用途地域内の建物のみを抽出
    print("住居系用途地域との空間結合を実行しています...")
    within_housing_area = gpd.sjoin(
        residential_buildings,
        housing_area,
        how="inner",
        predicate="intersects"
    )
    
    # 重複列を削除
    columns_to_drop = [col for col in within_housing_area.columns if col.endswith('_right')]
    within_housing_area = within_housing_area.drop(columns=columns_to_drop)
    
    # 住宅分類ごとの件数を集計
    print("\n=== 住宅分類ごとの件数 ===")
    print("\n【抽出前の全建物】")
    total_counts = buildings['usage'].value_counts()
    for use, count in total_counts.items():
        print(f"{use}: {count}件")
    
    print("\n【住居系用途地域内の住宅建物】")
    residential_counts = within_housing_area['usage'].value_counts()
    for use, count in residential_counts.items():
        print(f"{use}: {count}件")
    
    print(f"\n合計: {len(within_housing_area)}件")
    
    # 結果を保存
    print(f"抽出された建物数: {len(within_housing_area)}")
    print(f"結果を保存しています: {output_path}")
    within_housing_area.to_file(
        output_path,
        driver='ESRI Shapefile',
        encoding='cp932'
    )
    
    return within_housing_area

def create_4m_road_buffer(shido, national, prefectural, gogo_road):
    """4m以上の道路をバッファしてポリゴンを作成する関数。
    バッファ距離は(平均幅員/2 + 10)m

    Args:
        shido (GeoDataFrame): 市道データ
        national (GeoDataFrame): 県道データ
        prefectural (GeoDataFrame): 国道データ
        gogo_road (GeoDataFrame): 1項5号道路データ

    Returns:
        GeoDataFrame: 全ての4m以上道路のバッファを統合したポリゴン
    """
    # 全てのデータを平面直角座標系（I系）に変換
    shido = shido.to_crs(epsg=6669)
    national = national.to_crs(epsg=6669)
    prefectural = prefectural.to_crs(epsg=6669)
    gogo_road = gogo_road.to_crs(epsg=6669)

    buffer_polygons = []

    # 幅員コードから実際の幅員（m）に変換する関数
    def get_width_from_code(code):
        width_dict = {
            1: 13.0,  # 13.0m以上
            2: 9.25,  # 5.5-13.0mの中間値
            3: 4.25,  # 3.0-5.5mの中間値
            4: 3.0,   # 3.0m未満
        }
        return width_dict.get(code, 9.25)  # コードが不明な場合は2の幅員を使用

    # 1項5号道路についてバッファを作成
    gogo_buffer = gogo_road.geometry.buffer(10)
    buffer_polygons.extend(gogo_buffer.tolist())
    
    # 市道の'平均幅員'が4m以上のものを抽出し、バッファを作成
    wide_shido = shido[shido['平均幅員'] >= 4]
    wide_shido['buffer_width'] = wide_shido['平均幅員'].apply(lambda x: x/2 + 10)
    shido_buffer = wide_shido.apply(lambda row: row.geometry.buffer(row.buffer_width), axis=1)
    buffer_polygons.extend(shido_buffer.tolist())

    # 国道についてバッファを作成
    national['width'] = national['R22_005'].apply(get_width_from_code)
    national['buffer_width'] = national['width'].apply(lambda x: x/2 + 10)
    national_buffer = national.apply(lambda row: row.geometry.buffer(row.buffer_width), axis=1)
    buffer_polygons.extend(national_buffer.tolist())

    # 県道についてバッファを作成
    prefectural['width'] = prefectural['R22_005'].apply(get_width_from_code)
    prefectural['buffer_width'] = prefectural['width'].apply(lambda x: x/2 + 10)
    prefectural_buffer = prefectural.apply(lambda row: row.geometry.buffer(row.buffer_width), axis=1)
    buffer_polygons.extend(prefectural_buffer.tolist())

    # 全てのバッファを1つのGeoDataFrameにまとめる
    combined_buffer = gpd.GeoDataFrame(
        geometry=[shapely.ops.unary_union(buffer_polygons)],
        crs='EPSG:6669'
    )

    # WGS84に戻す
    combined_buffer = combined_buffer.to_crs(epsg=4326)

    # 結果を保存
    output_path = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/4m以上の道路バッファ/4m以上道路可変バッファ.shp"
    combined_buffer.to_file(
        output_path,
        driver='ESRI Shapefile',
        encoding='cp932'
    )

    return combined_buffer

def create_3m_road_buffer(shido, national, prefectural, shitei_road):
    """3m以上の道路をバッファしてポリゴンを作成する関数。
    バッファ距離は(平均幅員/2 + 10)m

    Args:
        shido (GeoDataFrame): 市道データ
        national (GeoDataFrame): 県道データ
        prefectural (GeoDataFrame): 国道データ
        shitei_road (GeoDataFrame): 指定道路データ

    Returns:
        GeoDataFrame: 全ての3m以上道路のバッファを統合したポリゴン
    """
    # 全てのデータを平面直角座標系（I系）に変換
    shido = shido.to_crs(epsg=6669)
    national = national.to_crs(epsg=6669)
    prefectural = prefectural.to_crs(epsg=6669)
    shitei_road = shitei_road.to_crs(epsg=6669)

    buffer_polygons = []

    # 幅員コードから実際の幅員（m）に変換する関数
    def get_width_from_code(code):
        width_dict = {
            1: 13.0,  # 13.0m以上
            2: 9.25,  # 5.5-13.0mの中間値
            3: 4.25,  # 3.0-5.5mの中間値
            4: 3.0,   # 3.0m未満
        }
        return width_dict.get(code, 9.25)  # コードが不明な場合は2の幅員を使用

    # 指定道路についてバッファを作成
    shitei_buffer = shitei_road.geometry.buffer(10)
    buffer_polygons.extend(shitei_buffer.tolist())
    
    # 市道の'平均幅員'が3m以上のものを抽出し、バッファを作成
    wide_shido = shido[shido['平均幅員'] >= 3]
    wide_shido['buffer_width'] = wide_shido['平均幅員'].apply(lambda x: x/2 + 10)
    shido_buffer = wide_shido.apply(lambda row: row.geometry.buffer(row.buffer_width), axis=1)
    buffer_polygons.extend(shido_buffer.tolist())

    # 国道についてバッファを作成
    national['width'] = national['R22_005'].apply(get_width_from_code)
    national['buffer_width'] = national['width'].apply(lambda x: x/2 + 10)
    national_buffer = national.apply(lambda row: row.geometry.buffer(row.buffer_width), axis=1)
    buffer_polygons.extend(national_buffer.tolist())

    # 県道についてバッファを作成
    prefectural['width'] = prefectural['R22_005'].apply(get_width_from_code)
    prefectural['buffer_width'] = prefectural['width'].apply(lambda x: x/2 + 10)
    prefectural_buffer = prefectural.apply(lambda row: row.geometry.buffer(row.buffer_width), axis=1)
    buffer_polygons.extend(prefectural_buffer.tolist())

    # 全てのバッファを1つのGeoDataFrameにまとめる
    combined_buffer = gpd.GeoDataFrame(
        geometry=[shapely.ops.unary_union(buffer_polygons)],
        crs='EPSG:6669'
    )

    # WGS84に戻す
    combined_buffer = combined_buffer.to_crs(epsg=4326)

    # 結果を保存
    output_path = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/車両通行可能な道路バッファ/車両通行可能な道路可変バッファ.shp"
    combined_buffer.to_file(
        output_path,
        driver='ESRI Shapefile',
        encoding='cp932'
    )

    return combined_buffer

def calculate_buffer_analysis(residential_buildings_path, buildings_outside_3m_path, buildings_outside_4m_path):
    """各地区における道路バッファ圏外の住宅の数と割合を分析する関数
    
    Args:
        residential_buildings_path (str): 住宅建物データのパス
        buildings_outside_3m_path (str): 3m道路バッファ圏外の建物データのパス
        buildings_outside_4m_path (str): 4m道路バッファ圏外の建物データのパス
    """
    print("データを読み込んでいます...")
    # 建物データの読み込み
    buildings = gpd.read_file(residential_buildings_path, encoding="utf-8")
    buildings_outside_3m = gpd.read_file(buildings_outside_3m_path, encoding="cp932")
    buildings_outside_4m = gpd.read_file(buildings_outside_4m_path, encoding="cp932")
    community = gpd.read_file(SASEBO_COMMUNITY, encoding="cp932")
    
    # 座標系を統一（Web メルカトル図法に変換）
    buildings = buildings.to_crs(epsg=3857)
    buildings_outside_3m = buildings_outside_3m.to_crs(epsg=3857)
    buildings_outside_4m = buildings_outside_4m.to_crs(epsg=3857)
    community = community.to_crs(epsg=3857)  # communityもWeb メルカトルに変換
    
    # 結果を格納するリスト
    results = []
    
    print("\n各地区の分析を開始します...")
    for idx, area in community.iterrows():
        name = area['NAME_x']
        print(f"\n処理中: {name}")
        
        # 地区内の全住宅数を計算
        area_buildings = gpd.sjoin(
            buildings,
            gpd.GeoDataFrame(geometry=[area.geometry], crs=buildings.crs),
            how="inner",
            predicate="within"
        )
        total_buildings = len(area_buildings)
        
        if total_buildings == 0:
            print(f"{name}には住宅建物がありません")
            continue
        
        # 3mバッファ圏外の住宅数を計算
        area_buildings_outside_3m = gpd.sjoin(
            buildings_outside_3m,
            gpd.GeoDataFrame(geometry=[area.geometry], crs=buildings_outside_3m.crs),
            how="inner",
            predicate="within"
        )
        outside_3m_count = len(area_buildings_outside_3m)
        
        # 4mバッファ圏外の住宅数を計算
        area_buildings_outside_4m = gpd.sjoin(
            buildings_outside_4m,
            gpd.GeoDataFrame(geometry=[area.geometry], crs=buildings_outside_4m.crs),
            how="inner",
            predicate="within"
        )
        outside_4m_count = len(area_buildings_outside_4m)
        
        # 結果を格納
        results.append({
            'コミュニティ名': name,
            '総住宅数': total_buildings,
            '3mバッファ圏外住宅数': outside_3m_count,
            '3mバッファ圏外住宅割合(%)': (outside_3m_count / total_buildings * 100),
            '4mバッファ圏外住宅数': outside_4m_count,
            '4mバッファ圏外住宅割合(%)': (outside_4m_count / total_buildings * 100)
        })
        
        print(f"  総住宅数: {total_buildings}")
        print(f"  3mバッファ圏外: {outside_3m_count}件 ({outside_3m_count/total_buildings*100:.1f}%)")
        print(f"  4mバッファ圏外: {outside_4m_count}件 ({outside_4m_count/total_buildings*100:.1f}%)")
    
    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)
    
    # 結果を降順でソート（4mバッファ圏外住宅割合で）
    results_df = results_df.sort_values('4mバッファ圏外住宅割合(%)', ascending=False)
    
    # Excelファイルとして保存
    output_path = "コミュニティ別バッファ圏外住宅分析.xlsx"
    print(f"\nExcelファイルを作成中: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # データフレームをExcelに書き込み
        results_df.to_excel(writer, sheet_name='分析結果', index=False)
        
        # ワークシートを取得
        worksheet = writer.sheets['分析結果']
        
        # 列幅の自動調整
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        # パーセンテージ列の書式設定
        from openpyxl.styles import numbers
        for row in worksheet.iter_rows(min_row=2):  # ヘッダーを除く
            for cell in [row[3], row[5]]:  # D列とF列（パーセンテージ列）
                cell.number_format = '0.0"%"'
    
    print(f"分析が完了しました。結果を{output_path}に保存しました。")
    
    return results_df

def visualize_buffer_analysis_results(community, results_df):
    """バッファ分析結果を地図上に可視化する関数
    
    Args:
        community (GeoDataFrame): コミュニティデータ
        results_df (DataFrame): バッファ分析結果のデータフレーム
    """
    # データの読み込みと座標系の統一
    community = community.merge(
        results_df,
        left_on='NAME_x',
        right_on='コミュニティ名',
        how='left'
    ).copy()  # コピーを作成して元のデータを保護
    
    print(f"処理するコミュニティの総数: {len(community)}")  # デバッグ用出力を追加
    
    # 住宅データと道路ポリゴンの読み込み
    buildings = gpd.read_file(residential_buildings, encoding="cp932")
    buildings_outside_3m = gpd.read_file(buildings_outside_three_m_road_buffer, encoding="cp932")
    buildings_outside_4m = gpd.read_file(buildings_outside_four_m_road_buffer, encoding="cp932")
    
    # 道路ポリゴンの読み込み
    shido_polygon = gpd.read_file(SHIDO_POLYGON, encoding="cp932")
    kokudo_polygon = gpd.read_file(KOKUDO_POLYGON, encoding="cp932")
    kendo_polygon = gpd.read_file(KENDO_POLYGON, encoding="cp932")
    ichisitei_polygon = gpd.read_file(ICHISITEI_POLYGON, encoding="cp932")
    
    # 座標系をWeb メルカトル（EPSG:6669）に統一
    community = community.to_crs(epsg=6669)
    buildings = buildings.to_crs(epsg=6669)
    buildings_outside_3m = buildings_outside_3m.to_crs(epsg=6669)
    buildings_outside_4m = buildings_outside_4m.to_crs(epsg=6669)
    shido_polygon = shido_polygon.to_crs(epsg=6669)
    kokudo_polygon = kokudo_polygon.to_crs(epsg=6669)
    kendo_polygon = kendo_polygon.to_crs(epsg=6669)
    ichisitei_polygon = ichisitei_polygon.to_crs(epsg=6669)

    # 傾斜度データの読み込み----よう修正
    with rasterio.open(dem_file) as src:
        slope_data = src.read(1)
        dem_bounds = src.bounds
        
        # マスクの作成（10度以上の部分のみTrue）
        slope_mask = slope_data >= 10
        masked_slope = np.where(slope_mask, slope_data, np.nan)
    
    # 処理済みのコミュニティを追跡
    processed_communities = set()
    
    # 各コミュニティについて処理
    for idx, row in community.iterrows():
        name = row['コミュニティ名']
        
        # 既に処理済みのコミュニティはスキップ
        if name in processed_communities:
            print(f"スキップ: {name} (既に処理済み)")
            continue
            
        print(f'処理中 ({len(processed_communities) + 1}/{len(community)}): {name}')
        processed_communities.add(name)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # コミュニティの境界を基に描画範囲を設定
        bounds = row.geometry.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # 表示範囲全体を覆う四角形を作成
        display_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        # コミュニティ外の領域を作成（四角形からコミュニティ領域を引く）
        outside_area = display_box.difference(row.geometry)
        # コミュニティ外を濃い灰色で塗る（最背面）
        gpd.GeoSeries([outside_area], crs=community.crs).plot(
            ax=ax,
            color='#404040',
            alpha=0.7,
            zorder=1
        )
        
        # 道路ポリゴンの描画（コミュニティでクリップ）
        for road_polygon, label in [
            (ichisitei_polygon, '位置指定道路'),
            (shido_polygon, '市道'),
            (kendo_polygon, '県道'),
            (kokudo_polygon, '国道')
        ]:
            try:
                clipped_road = road_polygon.clip(row.geometry)
                if not clipped_road.empty:
                    clipped_road.plot(
                        ax=ax,
                        color='black',
                        alpha=0.3,
                        zorder=2
                    )
            except Exception as e:
                print(f"Warning: Failed to plot {label} - {e}")
        
        # 全住宅を描画
        buildings_clipped = buildings.copy()
        buildings_clipped.geometry = buildings_clipped.geometry.buffer(0)
        buildings_clipped = buildings_clipped.clip(row.geometry)
        buildings_clipped.plot(
            ax=ax,
            color='gray',
            alpha=0.3,
            markersize=5,
            zorder=3
        )
        
        # バッファ圏外の建物を描画
        buildings_outside_4m_clipped = buildings_outside_4m.copy()
        buildings_outside_4m_clipped.geometry = buildings_outside_4m_clipped.geometry.buffer(0)
        buildings_outside_4m_clipped = buildings_outside_4m_clipped.clip(row.geometry)
        buildings_outside_4m_clipped.plot(
            ax=ax,
            color='red',
            alpha=0.7,
            zorder=4
        )
        
        buildings_outside_3m_clipped = buildings_outside_3m.copy()
        buildings_outside_3m_clipped.geometry = buildings_outside_3m_clipped.geometry.buffer(0)
        buildings_outside_3m_clipped = buildings_outside_3m_clipped.clip(row.geometry)
        buildings_outside_3m_clipped.plot(
            ax=ax,
            color='blue',
            alpha=0.7,
            zorder=5
        )
        
        # 凡例要素を更新
        legend_elements = [
            Patch(facecolor='gray', alpha=0.3, label='住宅'),
            Patch(facecolor='blue', alpha=0.7, label='車両通行可能な道路に面していない住宅'),
            Patch(facecolor='red', alpha=0.7, label='幅員4mの道路に面していない住宅'),
            Patch(facecolor='black', alpha=0.3, label='道路'),
            Patch(facecolor='none', edgecolor='black', label='コミュニティ境界')
        ]
        
        # 凡例を追加
        legend = ax.legend(handles=legend_elements, loc='upper right')
        legend.set_zorder(10)  # 凡例のzorderを後から設定
        
        # コミュニティの境界線を描画
        gpd.GeoSeries([row.geometry]).boundary.plot(
            ax=ax,
            color='black',
            linewidth=2,
            zorder=6
        )
        
        # コミュニティの境界を基に描画範囲を設定
        print('描画範囲を設定中...')
        bounds = row.geometry.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # 背景地図の追加（ズームレベルを19に制限）
        print('背景地図を追加中...')
        ctx.add_basemap(ax, 
                        crs="EPSG:6669", 
                        source=ctx.providers.OpenStreetMap.Mapnik, 
                        alpha=0.5,
                        zoom=19)
        
        ax.set_aspect('equal')
        
        # スケールバーを追加
        print('スケールバーを追加中...')
        scale_bar = ScaleBar(
            dx=1,  # 1 meter per unit
            units="m",
            location="lower right",
            scale_loc="bottom",
            label_loc="bottom",
            height_fraction=0.02,
            length_fraction=0.3,
            box_alpha=0,
        )
        ax.add_artist(scale_bar)
        
        # テキストボックスを追加（フォーマット指定子を修正）
        print('テキストボックスを追加中...')
        textstr = f'コミュニティ名: {row["コミュニティ名"]}\n'
        textstr += f'総住宅数: {int(row["総住宅数"]) if pd.notna(row["総住宅数"]) else 0}件\n'
        
        # 3mバッファの情報
        buildings_3m = int(row["3mバッファ圏外住宅数"]) if pd.notna(row["3mバッファ圏外住宅数"]) else 0
        percent_3m = row["3mバッファ圏外住宅割合(%)"] if pd.notna(row["3mバッファ圏外住宅割合(%)"]) else 0.0
        textstr += f'車両通行可能な道路に面していない住宅: {buildings_3m}件 ({percent_3m:.1f}%)\n'
        
        # 4mバッファの情報
        buildings_4m = int(row["4mバッファ圏外住宅数"]) if pd.notna(row["4mバッファ圏外住宅数"]) else 0
        percent_4m = row["4mバッファ圏外住宅割合(%)"] if pd.notna(row["4mバッファ圏外住宅割合(%)"]) else 0.0
        textstr += f'幅員4mの道路に面していない住宅: {buildings_4m}件 ({percent_4m:.1f}%)'
        
        print('テキストボックスを追加中...')
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        text_box = ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
            zorder=1000
        )
        
        # タイトルの設定
        plt.title(f'{row["コミュニティ名"]}の道路アクセス性分析', fontsize=14, pad=20)
        
        # グリッドと軸の設定
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 画像として保存
        print('画像を保存中...')
        plt.savefig(
            f"community_analysis_maps/{row['コミュニティ名']}_道路アクセス性分析.png",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()
    
    print("可視化が完了しました。")

if __name__ == "__main__":
    # ichisitei, shido, highway, national, prefectural = create_road_data()
    ichisitei = gpd.read_file(ICHISITEI_FOR_USE, encoding="cp932")
    shido = gpd.read_file(SHIDO_FOR_USE, encoding="cp932")
    national = gpd.read_file(KOKUDO_FOR_USE, encoding="cp932")
    prefectural = gpd.read_file(KENDO_FOR_USE, encoding="cp932")
    housing_area = gpd.read_file(SASEBO_HOUSING_AREA, encoding="cp932")
    community = gpd.read_file(SASEBO_COMMUNITY, encoding="cp932")
    gogo_road = gpd.read_file(GOGO, encoding="cp932")

    ichisitei_polygon = gpd.read_file(ICHISITEI_POLYGON, encoding="cp932")
    shido_polygon = gpd.read_file(SHIDO_POLYGON, encoding="cp932")
    national_polygon = gpd.read_file(KOKUDO_POLYGON, encoding="cp932")
    prefectural_polygon = gpd.read_file(KENDO_POLYGON, encoding="cp932")

    shitei_road = gpd.read_file(SHITEI_FOR_BUFFER, encoding="cp932")

    ichisitei = ichisitei.set_crs(epsg=4326)
    shido = shido.set_crs(epsg=4326)
    national = national.set_crs(epsg=4326)
    prefectural = prefectural.set_crs(epsg=4326)
    housing_area = housing_area.set_crs(epsg=4326)
    community = community.set_crs(epsg=4326)

    residential_buildings = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/udx/bldg/residential_buildings.shp"
    four_m_road_buffer = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/4m以上の道路バッファ/4m以上道路可変バッファ.shp"
    three_m_road_buffer = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/車両通行可能な道路バッファ/車両通行可能な道路可変バッファ.shp"

    # 4m以上の道路：指定道路は1項5号道路、市道は'平均幅員'列が4以上のもの、県道・国道
    # 車両通行可能な道路：指定道路は1項5号・2項・3項、市道は'平均幅員'列が3以上のもの、県道・国道

    # 4m以上の道路のバッファを作成
    # four_m_road_buffer = create_4m_road_buffer(shido, national, prefectural, gogo_road)
    # 車両通行可能な道路のバッファを作成
    # three_m_road_buffer = create_3m_road_buffer(shido, national, prefectural, shitei_road)


    # shido_polygon, national_polygon, prefectural_polygon = create_road_polygon(ichisitei, shido, national, prefectural)
    # plot_road_data(ichisitei, shido, national, prefectural, housing_area, community)

    # plot_community(community)
    # plot_community_and_other_data(community, ichisitei_polygon, shido_polygon, national_polygon, prefectural_polygon)
    
    # merged_data = merge_gml_to_shp(input_directory, output_shapefile)

    """# 住居系用途地域内の住宅建物を抽出
    output_residential = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/udx/bldg/residential_buildings.shp"
    residential_buildings = extract_residential_buildings(
        output_shapefile,
        SASEBO_HOUSING_AREA,
        output_residential
    )"""

    """# バッファ圏外の住宅を抽出
    output_path = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/buildings_outside_3m_buffer.shp"
    residential_buildings = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/buildings_outside_4m_buffer.shp"
    buildings_outside = extract_buildings_outside_buffer(
        residential_buildings,
        three_m_road_buffer,
        output_path
    )"""

    """residential_buildings = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/udx/bldg/residential_buildings.shp"
    buildings_outside_three_m_road_buffer = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/buildings_outside_3m_buffer.shp"
    buildings_outside_four_m_road_buffer = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/buildings_outside_4m_buffer.shp"
    # 各地区について、バッファ圏外の住宅の数と割合を算出
    results_df = calculate_buffer_analysis(
        residential_buildings,
        buildings_outside_three_m_road_buffer,
        buildings_outside_four_m_road_buffer
    )
    visualize_buffer_analysis_results(community, results_df)"""
    plot_road_data(ichisitei, shido, national, prefectural, housing_area, community)