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
SASEBO_COMMUNITY = "/Users/sakamo/Desktop/GISDATA/自治協議会/sasebo_steep_community.shp"
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
    print(f"コミュニティ数: {len(community)}")
    community = community.drop_duplicates(subset=['NAME_x'], keep='first')
    print(f"コミュニティ数(重複削除後): {len(community)}")
    
    # 座標系を統一（Web メルカトル図法に変換）
    buildings = buildings.to_crs(epsg=3857)
    buildings_outside_3m = buildings_outside_3m.to_crs(epsg=3857)
    buildings_outside_4m = buildings_outside_4m.to_crs(epsg=3857)
    community = community.to_crs(epsg=3857)  # communityもWeb メルカトルに変換
    
    # 結果を格納するリスト
    results = []
    
    print("\n各地区の分析を開始します...")
    com_counter = 0
    ienasi_counter = 0
    for idx, area in community.iterrows():
        name = area['NAME_x']
    
        # 処理済みのコミュニティを追跡
        processed_communities = set()
        # 既に処理済みのコミュニティはスキップ
        if name in processed_communities:
            print(f"スキップ: {name} (既に処理済み)")
            continue
        com_counter += 1
        print(f'処理中 : {name}({com_counter}件目)')
        processed_communities.add(name)

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
            ienasi_counter += 1
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
    print(f"イエナシコミュニティ数: {ienasi_counter}")
    return results_df

def visualize_buffer_analysis_results(community_gdf, results_df):
    """バッファ分析結果を地図上に可視化する関数
    
    Args:
        community (GeoDataFrame): コミュニティデータ
        results_df (DataFrame): バッファ分析結果のデータフレーム
    """
    # データの読み込みと座標系の統一
    community_gdf = community_gdf.merge(
        results_df,
        left_on='NAME_x',
        right_on='コミュニティ名',
        how='left'
    ).copy()  # コピーを作成して元のデータを保護
    
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
    community_gdf = community_gdf.to_crs(epsg=6669)
    buildings = buildings.to_crs(epsg=6669)
    buildings_outside_3m = buildings_outside_3m.to_crs(epsg=6669)
    buildings_outside_4m = buildings_outside_4m.to_crs(epsg=6669)
    shido_polygon = shido_polygon.to_crs(epsg=6669)
    kokudo_polygon = kokudo_polygon.to_crs(epsg=6669)
    kendo_polygon = kendo_polygon.to_crs(epsg=6669)
    ichisitei_polygon = ichisitei_polygon.to_crs(epsg=6669)

    # 傾斜度データの読み込み----よう修正
    with rasterio.open(SASEBO_SLOPE_DEM) as src:    
        # 処理済みのコミュニティを追跡
        processed_communities = set()
        
        # 各コミュニティについて処理
        com_counter = 0
        for idx, community in community_gdf.iterrows():
            name = community['NAME_x']
            
            # 既に処理済みのコミュニティはスキップ
            if name in processed_communities:
                print(f"スキップ: {name} (既に処理済み)")
                continue
            com_counter += 1
            print(f'処理中 : {name}({com_counter}件目)')
            processed_communities.add(name)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # コミュニティの境界を基に描画範囲を設定
            com_bounds = community.geometry.bounds
            center_x = (com_bounds[0] + com_bounds[2]) / 2
            center_y = (com_bounds[1] + com_bounds[3]) / 2
            com_width = com_bounds[2] - com_bounds[0]
            com_height = com_bounds[3] - com_bounds[1]
            width_margin = com_width * 1.2  # 横方向に20%マージン
            height_margin = com_height * 1.05  # 縦方向に5%マージン
            display_bounds = (
                center_x - width_margin,  # xmin
                center_y - height_margin,  # ymin
                center_x + width_margin,  # xmax
                center_y + height_margin   # ymax
            )
            ax.set_xlim(center_x - width_margin/2, center_x + width_margin/2)
            ax.set_ylim(center_y - height_margin/2, center_y + height_margin/2)
            
            # コミュニティ外の領域を作成（表示範囲からコミュニティ領域を引く）
            outside_area = box(*display_bounds).difference(community.geometry)
            gpd.GeoDataFrame(
                geometry=[outside_area],
                crs=community_gdf.crs
            ).plot(
                ax=ax,
                color='gray',
                alpha=0.5,
                zorder=1  # 背景地図の下に表示
            )

            # 背景地図の追加（ズームレベルを19に制限）
            print('背景地図を追加中...')
            ctx.add_basemap(ax, 
                            crs="EPSG:6669", 
                            source=ctx.providers.OpenStreetMap.Mapnik, 
                            alpha=0.7,
                            zoom=19)

            """# コミュニティ外を灰色で塗る
            xmin, ymin, xmax, ymax = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
            boundary = gpd.GeoDataFrame({
                'geometry': [box(xmin, ymin, xmax, ymax)]
            })"""
            
            """# コミュニティ外の領域を作成
            outside_area = boundary.overlay(community, how='difference')
            outside_area.plot(ax=ax, color='gray', alpha=0.5)"""

            # 傾斜度データの描画
            window = src.window(*display_bounds)
            window = window.round_lengths()
            window = window.round_offsets()
            # 切り出したデータの表示範囲を計算
            window_bounds = rasterio.windows.bounds(window, src.transform)
            slope_data = src.read(1, window=window)
            im = ax.imshow(
                slope_data,
                extent=[window_bounds[0], window_bounds[2], window_bounds[1], window_bounds[3]],
                cmap='YlOrRd',
                vmin=0,
                vmax=45,
                alpha=0.3
            )
            # カラーバーの追加
            cbar = fig.colorbar(im, orientation='horizontal', 
                            label='傾斜度 (度)', shrink=0.8, pad=0.02)

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
            
            # 道路ポリゴンの描画（コミュニティでクリップ）
            for road_polygon, label in [
                (ichisitei_polygon, '位置指定道路'),
                (shido_polygon, '市道'),
                (kendo_polygon, '県道'),
                (kokudo_polygon, '国道')
            ]:
                try:
                    clipped_road = road_polygon.clip(community.geometry)
                    if not clipped_road.empty:
                        clipped_road.plot(
                            ax=ax,
                            color='black',
                            alpha=0.3,
                        )
                except Exception as e:
                    print(f"Warning: Failed to plot {label} - {e}")
            
            # 全住宅を描画
            buildings_clipped = buildings.copy()
            buildings_clipped.geometry = buildings_clipped.geometry.buffer(0)
            buildings_clipped = buildings_clipped.clip(community.geometry)
            buildings_clipped.plot(
                ax=ax,
                color='gray',
                alpha=0.3,
                markersize=5,
            )
            
            # バッファ圏外の建物を描画
            buildings_outside_4m_clipped = buildings_outside_4m.copy()
            buildings_outside_4m_clipped.geometry = buildings_outside_4m_clipped.geometry.buffer(0)
            buildings_outside_4m_clipped = buildings_outside_4m_clipped.clip(community.geometry)
            buildings_outside_4m_clipped.plot(
                ax=ax,
                color='blue',
                alpha=0.7,
            )
            
            buildings_outside_3m_clipped = buildings_outside_3m.copy()
            buildings_outside_3m_clipped.geometry = buildings_outside_3m_clipped.geometry.buffer(0)
            buildings_outside_3m_clipped = buildings_outside_3m_clipped.clip(community.geometry)
            buildings_outside_3m_clipped.plot(
                ax=ax,
                color='red',
                alpha=0.7,
            )
            
            # コミュニティの境界線を描画
            gpd.GeoSeries([community.geometry]).boundary.plot(
                ax=ax,
                color='black',
                linewidth=2,
            )

            # 凡例要素を更新
            legend_elements = [
                Patch(facecolor='gray', alpha=0.3, label='住宅'),
                Patch(facecolor='red', alpha=0.7, label='車両通行可能な道路に面していない住宅'),
                Patch(facecolor='blue', alpha=0.7, label='幅員4mの道路に面していない住宅'),
                Patch(facecolor='black', alpha=0.3, label='道路'),
                Patch(facecolor='none', edgecolor='black', label='コミュニティ境界')
            ]
            
            # 凡例を追加
            legend = ax.legend(handles=legend_elements, loc='upper right')
            
            # テキストボックスを追加（フォーマット指定子を修正）
            print('テキストボックスを追加中...')
            textstr = f'コミュニティ名: {name}\n'
            textstr += f'総住宅数: {int(community["総住宅数"]) if pd.notna(community["総住宅数"]) else 0}件\n'
            
            # 3mバッファの情報
            buildings_3m = int(community["3mバッファ圏外住宅数"]) if pd.notna(community["3mバッファ圏外住宅数"]) else 0
            percent_3m = community["3mバッファ圏外住宅割合(%)"] if pd.notna(community["3mバッファ圏外住宅割合(%)"]) else 0.0
            textstr += f'車両通行可能な道路に面していない住宅: {buildings_3m}件 ({percent_3m:.1f}%)\n'
            
            # 4mバッファの情報
            buildings_4m = int(community["4mバッファ圏外住宅数"]) if pd.notna(community["4mバッファ圏外住宅数"]) else 0
            percent_4m = community["4mバッファ圏外住宅割合(%)"] if pd.notna(community["4mバッファ圏外住宅割合(%)"]) else 0.0
            textstr += f'幅員4mの道路に面していない住宅: {buildings_4m}件 ({percent_4m:.1f}%)'
            
            print('テキストボックスを追加中...')
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            text_box = ax.text(
                0.02, 0.98, textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=props,
            )
            
            # タイトルの設定
            plt.title(f'{name}の道路アクセス性分析', fontsize=14, pad=20)
            
            # グリッドと軸の設定
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 画像として保存
            print('画像を保存中...')
            plt.savefig(
                f"community_analysis_maps/{name}_道路アクセス性分析.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1
            )
            plt.close()
        print(f"可視化が完了しました。{com_counter}件")

if __name__ == "__main__":
    # スリープを防ぐためのcaffeinateコマンドを実行
    import subprocess
    caffeinate_process = subprocess.Popen(['caffeinate', '-i'])

    try:
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

        residential_buildings = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/udx/bldg/residential_buildings.shp"
        buildings_outside_three_m_road_buffer = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/buildings_outside_3m_buffer.shp"
        buildings_outside_four_m_road_buffer = "/Users/sakamo/Desktop/GISDATA/sasebo_citygml/buildings_outside_4m_buffer.shp"
        # 各地区について、バッファ圏外の住宅の数と割合を算出
        results_df = calculate_buffer_analysis(
            residential_buildings,
            buildings_outside_three_m_road_buffer,
            buildings_outside_four_m_road_buffer
        )
        #results_df = pd.read_excel("コミュニティ別バッファ圏外住宅分析.xlsx", sheet_name="分析結果")
        #visualize_buffer_analysis_results(community, results_df)
    finally:
        # スクリプト終了時にcaffeinateプロセスを終了
        caffeinate_process.terminate()
