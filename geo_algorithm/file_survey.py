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


FILE_DIR_PATH = "community_analysis_maps/survey/"


if __name__ == "__main__":
    results_df = pd.read_excel("コミュニティ別バッファ圏外住宅分析.xlsx", sheet_name="分析結果")
    
    # 分析対象の列とファイル名のマッピング
    column_file_mapping = {
        '3mバッファ圏外住宅数': '車両通行可能な道路に面していない住宅の数',
        '3mバッファ圏外住宅割合(%)': '車両通行可能な道路に面していない住宅の割合',
        '4mバッファ圏外住宅数': '幅員4m以上の道路に面していない住宅の数',
        '4mバッファ圏外住宅割合(%)': '幅員4m以上の道路に面していない住宅の割合'
    }
    
    # 上位コミュニティの画像を保存するディレクトリを作成
    for column, file_name in column_file_mapping.items():
        target_dir = os.path.join(FILE_DIR_PATH, file_name.replace(' ', '_'))
        os.makedirs(target_dir, exist_ok=True)
        
        # 割合を評価する場合のみ、総住宅数10件以上でフィルタリング
        if '割合' in column:
            analysis_df = results_df[results_df['総住宅数'] >= 10].copy()
        else:
            analysis_df = results_df.copy()
            
        # 上位20位のコミュニティを抽出
        top20 = analysis_df.nlargest(20, column)[['コミュニティ名', column]]
        
        # 各コミュニティの画像をコピー
        for community in top20['コミュニティ名']:
            source_path = f'community_analysis_maps/{community}_道路アクセス性分析.png'
            if os.path.exists(source_path):
                dest_path = os.path.join(target_dir, f'{community}.png')
                subprocess.run(['cp', source_path, dest_path])
            else:
                print(f"警告: {community}の画像が見つかりません")

    # コミュニティ区域のデータ読み込み
    communities = gpd.read_file(SASEBO_COMMUNITY)
    housing_area = gpd.read_file(SASEBO_HOUSING_AREA)

    # 結合を実行
    # 重複を確認
    print("\nDuplicate rows:", communities.duplicated().sum())

    # 必要に応じて重複を除去
    communities = communities.drop_duplicates()
    communities = communities.merge(
        results_df,
        left_on='NAME_x',
        right_on='コミュニティ名',
        how='left'
    )
    
    # カラム名を英語に変換
    column_mapping = {
        'コミュニティ名': 'com_name',
        '総住宅数': 'num_houses',
        '3mバッファ圏外住宅数': 'n_car_num',
        '3mバッファ圏外住宅割合(%)': 'n_car_rat',
        '4mバッファ圏外住宅数': 'n4r_num',
        '4mバッファ圏外住宅割合(%)': 'n4r_rat',
    }
    communities = communities.rename(columns=column_mapping)
    
    # 必要な列のみを選択
    keep_columns = ['elev_mean', 'elev_med', 'elev_std', 'slope_mean', 'slope_med',
                    'slope_std', 'shc_mean', 'shc_med', 'shc_std', 'steep_rati', 'geometry'] + list(column_mapping.values())
    communities = communities[keep_columns]
    
    # Shapefileとして保存
    communities.to_file(f'{FILE_DIR_PATH}community_areas.shp')
    
    # 各指標について上位20位のコミュニティを抽出し、地図化
    for column, file_name in column_file_mapping.items():
        # 割合を評価する場合のみ、総住宅数10件以上でフィルタリング
        if '割合' in column:
            analysis_df = results_df[results_df['総住宅数'] >= 10].copy()
        else:
            analysis_df = results_df.copy()
            
        # 上位20位のコミュニティを抽出
        top20 = analysis_df.nlargest(20, column)[['コミュニティ名', column]]
        
        fig, ax = plt.subplots(figsize=(15, 10))
        housing_area.plot(ax=ax, color='yellow', alpha=0.3, label='住居系用途地域')
        communities.plot(ax=ax, color='white', edgecolor='gray', alpha=0.1)
        communities[communities['com_name'].isin(top20['コミュニティ名'])].plot(
            ax=ax, color='red', alpha=0.5, label='上位20位のコミュニティ'
        )
        
        ctx.add_basemap(ax, crs=communities.crs.to_string())
        ax.add_artist(ScaleBar(1, location='lower right'))
        plt.title(f'{file_name}の上位20位コミュニティ')
        # plt.legend()
        
        os.makedirs(FILE_DIR_PATH, exist_ok=True)
        plt.savefig(f'{FILE_DIR_PATH}{file_name}_top20.png', dpi=300, bbox_inches='tight')
        plt.close()


    # 散布図の作成
    fig, ax = plt.subplots(figsize=(12, 8))
    filtered_df = results_df[results_df['総住宅数'] >= 10].copy()
    scatter = ax.scatter(
        filtered_df['4mバッファ圏外住宅割合(%)'],
        filtered_df['3mバッファ圏外住宅割合(%)'],
        alpha=0.7,
        c='blue'
    )
    
    # 両方の指標が高いコミュニティ（上位10位）にラベルを付ける
    filtered_df['3割合_rank'] = filtered_df['3mバッファ圏外住宅割合(%)'].rank(ascending=False)
    filtered_df['4割合_rank'] = filtered_df['4mバッファ圏外住宅割合(%)'].rank(ascending=False)
    filtered_df['総合スコア'] = (filtered_df['3割合_rank'] + filtered_df['4割合_rank']) / 2
    # 総合スコアで上位20位を抽出
    top20 = filtered_df.nsmallest(20, '総合スコア')
    
    # ラベルを付ける
    for idx, row in top20.iterrows():
        ax.annotate(
            row['コミュニティ名'],
            (row['4mバッファ圏外住宅割合(%)'], row['3mバッファ圏外住宅割合(%)']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8
        )

    # グラフの設定
    ax.set_xlabel('幅員4m以上の道路に面していない住宅の割合(%)')
    ax.set_ylabel('車両通行可能な道路に面していない住宅の割合(%)')
    ax.set_title('コミュニティ別 道路に面していない住宅の分布')
    
    # グリッド線を追加
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存
    plt.savefig(f'{FILE_DIR_PATH}車両通行可能な道路に面していない住宅の分布.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
