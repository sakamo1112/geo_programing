import os

import contextily as ctx
import geopandas as gpd
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.transform import Affine, from_bounds, from_origin
from scipy.ndimage import convolve
from shapely.geometry import box

SASEBO_SLOPE_DEM = "result/slope/slope_佐世保市.tif"
NAGASAKI_SLOPE_DEM = "result/slope/slope_長崎市.tif"
YOKOSUKA_SLOPE_DEM = "result/slope/slope_横須賀市.tif"
SASEBO_HOUSING_AREA = (
    "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_42/housing_佐世保市.shp"
)
NAGASAKI_HOUSING_AREA = (
    "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_42/housing_長崎市.shp"
)
YOKOSUKA_HOUSING_AREA = (
    "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_14/housing_横須賀市.shp"
)


crs_dict = {
    "佐世保市": "EPSG:6669",
    "長崎市": "EPSG:6669",
    "横須賀市": "EPSG:6677",
}


if __name__ == "__main__":
    # 3つの市のDEMファイルパスとシェープファイルパスを設定
    city_data = [
        (SASEBO_SLOPE_DEM, SASEBO_HOUSING_AREA, "佐世保市"),
        (NAGASAKI_SLOPE_DEM, NAGASAKI_HOUSING_AREA, "長崎市"),
        (YOKOSUKA_SLOPE_DEM, YOKOSUKA_HOUSING_AREA, "横須賀市"),
    ]

    # 各市のデータを個別に描画
    """for dem_file, housing_file, title in city_data:
        # 1つの図を作成
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 傾斜度データの読み込み
        with rasterio.open(dem_file) as src:
            slope_data = src.read(1)
            dem_bounds = src.bounds
            
            # マスクの作成（10度以上の部分のみTrue）
            slope_mask = slope_data >= 10
            masked_slope = np.where(slope_mask, slope_data, np.nan)
            
        # 住居系用途地域の読み込み
        housing = gpd.read_file(housing_file)
        housing = housing.to_crs(crs_dict[title])
        
        # DEMの中心点と範囲を計算
        center_x = (dem_bounds.left + dem_bounds.right) / 2
        center_y = (dem_bounds.bottom + dem_bounds.top) / 2
        dem_width = dem_bounds.right - dem_bounds.left
        dem_height = dem_bounds.top - dem_bounds.bottom
        
        # DEMの範囲に基づいて表示範囲を設定（横方向に20%マージン）
        width_margin = dem_width * 1.2  # 横方向に20%マージン
        height_margin = dem_height * 1.05  # 縦方向に5%マージン
        ax.set_xlim(center_x - width_margin/2, center_x + width_margin/2)
        ax.set_ylim(center_y - height_margin/2, center_y + height_margin/2)
        
        # OpenStreetMapの追加（zoomレベルを指定）
        ctx.add_basemap(ax, crs=crs_dict[title], source=ctx.providers.OpenStreetMap.Mapnik, 
                       zoom=13)
        
        # 住居系用途地域外を灰色で塗る
        xmin, ymin, xmax, ymax = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
        boundary = gpd.GeoDataFrame({
            'geometry': [box(xmin, ymin, xmax, ymax)]
        }, crs=housing.crs)
        
        # 住居系用途地域外の領域を作成
        outside_housing = boundary.overlay(housing, how='difference')
        outside_housing.plot(ax=ax, color='gray', alpha=0.5)
        
        # 傾斜度データの描画（10度以上のみ）
        im = ax.imshow(
            masked_slope,
            extent=[dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top],
            cmap='YlOrRd',
            vmin=10,
            vmax=45,
            alpha=0.7
        )
        
        # 軸の目盛りを削除
        ax.set_xticks([])
        ax.set_yticks([])
        
        # スケールバーを5kmに固定（単位表示を変更）
        scale_bar = ScaleBar(
            dx=1,  # 1 meter per unit
            units="m",
            fixed_value=5000,  # 5000m = 5km
            location="lower right",
            scale_loc="bottom",
            label_loc="bottom",
            height_fraction=0.02,
            length_fraction=0.3,
            box_alpha=0,
            scale_formatter=lambda value, unit: "5 km"  # 常に"5 km"と表示
        )
        ax.add_artist(scale_bar)
        ax.set_title(title)
        
        # カラーバーの追加
        cbar = fig.colorbar(im, orientation='horizontal', 
                          label='傾斜度 (度)', shrink=0.8, pad=0.02)
        
        # 画像の保存（ファイル名に都市名を含める）
        plt.savefig(f'result/slope_{title}.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()"""

    # 長崎市と佐世保市を縦に並べたレイアウトを作成
    nagasaki_sasebo_data = [
        (SASEBO_SLOPE_DEM, SASEBO_HOUSING_AREA, "佐世保市"),
        (NAGASAKI_SLOPE_DEM, NAGASAKI_HOUSING_AREA, "長崎市"),
    ]

    # フォントサイズの基本設定
    plt.rcParams["font.size"] = 14  # 基本のフォントサイズ

    # 2つの図を縦に並べて作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))

    # サブプロット間の間隔の設定
    plt.subplots_adjust(right=0.82, hspace=0.01, bottom=0.05, top=0.95)

    for ax, (dem_file, housing_file, title) in zip(axes, nagasaki_sasebo_data):
        # 傾斜度データの読み込み
        with rasterio.open(dem_file) as src:
            slope_data = src.read(1)
            dem_bounds = src.bounds

            # マスクの作成（10度以上の部分のみTrue）
            slope_mask = slope_data >= 10
            masked_slope = np.where(slope_mask, slope_data, np.nan)

        # 住居系用途地域の読み込み
        housing = gpd.read_file(housing_file)
        housing = housing.to_crs(crs_dict[title])

        # DEMの中心点と範囲を計算
        center_x = (dem_bounds.left + dem_bounds.right) / 2
        center_y = (dem_bounds.bottom + dem_bounds.top) / 2
        dem_width = dem_bounds.right - dem_bounds.left
        dem_height = dem_bounds.top - dem_bounds.bottom

        # DEMの範囲に基づいて表示範囲を設定（横方向に20%マージン）
        width_margin = dem_width * 1.2  # 横方向に20%マージン
        height_margin = dem_height * 1.05  # 縦方向に5%マージン
        ax.set_xlim(center_x - width_margin / 2, center_x + width_margin / 2)
        ax.set_ylim(center_y - height_margin / 2, center_y + height_margin / 2)

        # OpenStreetMapの追加（zoomレベルを指定）
        ctx.add_basemap(
            ax, crs=crs_dict[title], source=ctx.providers.OpenStreetMap.Mapnik, zoom=13
        )

        # 住居系用途地域外を灰色で塗る
        xmin, ymin, xmax, ymax = (
            ax.get_xlim()[0],
            ax.get_ylim()[0],
            ax.get_xlim()[1],
            ax.get_ylim()[1],
        )
        boundary = gpd.GeoDataFrame(
            {"geometry": [box(xmin, ymin, xmax, ymax)]}, crs=housing.crs
        )

        # 住居系用途地域外の領域を作成
        outside_housing = boundary.overlay(housing, how="difference")
        outside_housing.plot(ax=ax, color="gray", alpha=0.5)

        # 傾斜度データの描画（10度以上のみ）
        im = ax.imshow(
            masked_slope,
            extent=[
                dem_bounds.left,
                dem_bounds.right,
                dem_bounds.bottom,
                dem_bounds.top,
            ],
            cmap="YlOrRd",
            vmin=10,
            vmax=45,
            alpha=0.7,
        )

        # 軸の目盛りを削除
        ax.set_xticks([])
        ax.set_yticks([])

        # スケールバーを5kmに固定
        scale_bar = ScaleBar(
            dx=1,
            units="m",
            fixed_value=5000,
            location="lower right",
            scale_loc="bottom",
            label_loc="bottom",
            height_fraction=0.02,
            length_fraction=0.3,
            box_alpha=0,
            scale_formatter=lambda value, unit: "5 km",
        )
        ax.add_artist(scale_bar)
        ax.set_title(title, pad=20, y=-0.1, fontsize=16)

        # 軸ラベルのフォントサイズを設定
        ax.tick_params(axis="both", labelsize=12)

    # カラーバーの追加（フォントサイズを大きく）
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("傾斜度 (度)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # カラーバーの目盛りのフォントサイズ

    # tight_layoutを削除（subplots_adjustで制御するため）
    # plt.tight_layout()

    # 縦に並べた画像の保存（右側のマージンを増やしてカラーバーが収まるようにする）
    plt.savefig(
        "result/slope_nagasaki_sasebo.png", dpi=300, bbox_inches="tight", pad_inches=0.2
    )
    plt.close()
