import os
import subprocess

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import pandas as pd
import rasterio
import fiona
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
RESULT_XLSX_SASEBO = "result/sasebo_terrain_statistics.xlsx"

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
RESULT_SHP_SASEBO = "/Users/sakamo/Desktop/GISDATA/自治協議会/sasebo_community.shp"

YOKOSUKA_DEM = "/Users/sakamo/Desktop/GISDATA/DEM_地方別/関東.tif"
YOKOSUKA_COMMUNITY = "/Users/sakamo/Desktop/GISDATA/横須賀市_地域コミュニティ/D_20231219_043504_504D273F.shp"
RESULT_SHP_YOKOSUKA = "/Users/sakamo/Desktop/GISDATA/横須賀市_地域コミュニティ/yokosuka_community.shp"
YOKOSUKA_HOUSING_AREA = "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_14/housing_横須賀市.shp"
RESULT_XLSX_YOKOSUKA = "result/yokosuka_terrain_statistics.xlsx"

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
        YOKOSUKA_DEM,
        YOKOSUKA_COMMUNITY,
        YOKOSUKA_HOUSING_AREA,
    ]


def calc_and_visualize_height(
    com_id: str,
    com_name: str,
    bbox_elevation: np.ndarray,
    com_mask: np.ndarray,
    housing_mask: np.ndarray,
    visualize: bool,
) -> np.ndarray:
    """
    標高の計算と可視化を行い、コミュニティの標高データを返す。

    Args:
        com_name (str): コミュニティ名(例: 16_182294)
        bbox_elevation (numpy.ndarray): コミュニティを囲むBBoxの標高データ
        com_mask (numpy.ndarray): コミュニティのマスク
    Returns:
        com_area_height (numpy.ndarray): コミュニティの標高データ
    """
    com_area_height = np.where(com_mask == 1, bbox_elevation, np.nan)
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        # 標高データの表示
        im = ax.imshow(bbox_elevation, cmap="terrain", aspect="equal", vmin=0, vmax=500)
        
        # コミュニティ外を灰色で表示
        ax.imshow(
            np.where(com_mask == 0, 0.7, np.nan),
            cmap="gray_r",
            alpha=0.8,
            aspect="equal",
        )
        
        # 住居系用途地域外のハッチング描画を修正
        non_housing_area = np.where((com_mask == 1) & (housing_mask == 0), 1, np.nan)
        
        # マスクの位置に合わせてハッチングを描画
        for i in range(non_housing_area.shape[0]):
            for j in range(non_housing_area.shape[1]):
                if not np.isnan(non_housing_area[i, j]):
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5),  # 位置を調整
                        1, 1,                 # 幅と高さ
                        fill=False,
                        hatch='///',
                        alpha=0.3
                    ))

        cbar = plt.colorbar(im)
        cbar.set_label("標高 [m]")
        plt.title(f"{com_name}の標高分布\n(斜線部: 住居系用途地域外)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"result/sasebo_com_elevation/elevation_{com_id}.png", dpi=300)
        plt.close()

    return com_area_height


def calc_and_visualize_slope(
    com_id: str,
    com_name: str,
    bbox_elevation: np.ndarray,
    com_mask: np.ndarray,
    housing_mask: np.ndarray,
    visualize: bool,
) -> np.ndarray:
    """
    傾斜度の計算と可視化を行い、コミュニティの傾斜度データを返す。

    Args:
        com_name (str): コミュニティ名(例: 16_182294)
        bbox_elevation (numpy.ndarray): コミュニティを囲むBBoxの標高データ
        com_mask (numpy.ndarray): コミュニティのマスク
        housing_mask (numpy.ndarray): 住居系用途地域のマスク
    """
    # 傾斜度の計算
    slope = calc_slope(bbox_elevation)
    
    # コミュニティ内かつ住居系用途地域内のデータのみを抽出
    com_area_slope = np.where((com_mask == 1) & (housing_mask == 1), slope, np.nan)
    
    # 有効なデータがあるか確認
    valid_data = com_area_slope[~np.isnan(com_area_slope)]
    if len(valid_data) == 0:
        print(f"警告: {com_name}の傾斜度データが存在しません")
        return com_area_slope, np.nan

    steep_ratio = sum(valid_data >= 5) / len(valid_data) * 100

    if visualize:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3.5, 1]}
        )
        plt.subplots_adjust(left=0, right=0.85)

        # 左側：傾斜度の分布図
        im = ax1.imshow(slope, cmap="autumn_r", aspect="equal", vmin=0, vmax=45)
        
        # コミュニティ外を灰色で表示
        ax1.imshow(
            np.where(com_mask == 0, 0.7, np.nan),
            cmap="gray_r",
            alpha=0.8,
            aspect="equal",
        )
        
        # 住居系用途地域外（コミュニティ内）に網掛けを追加
        non_housing_area = np.where((com_mask == 1) & (housing_mask == 0), 1, np.nan)
        for i in range(non_housing_area.shape[0]):
            for j in range(non_housing_area.shape[1]):
                if not np.isnan(non_housing_area[i, j]):
                    ax1.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=False,
                        hatch='///',
                        alpha=0.3
                    ))
        # カラーバーの描画（axを明示的に指定）
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label("傾斜度 (度)")
        
        ax1.set_title(f"{com_name}の傾斜度分布\n(斜線部: 住居系用途地域外)")
        ax1.axis("off")

        # 右側：積み上げ棒グラフ
        bins = [0, 5, 10, 15, 20, 25, np.inf]
        labels = ["0-5度", "5-10度", "10-15度", "15-20度", "20-25度", "25度以上"]
        hist, _ = np.histogram(valid_data, bins=bins)
        percentages = hist / len(valid_data) * 100
        steep_ratio = sum(percentages[1:])  # 5度以上の割合の合計

        colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "red"]
        bottom = 0
        for i, (percentage, color) in enumerate(zip(percentages, colors)):
            ax2.bar(0, percentage, bottom=bottom, color=color, label=labels[i])
            if percentage >= 3:
                ax2.text(
                    0,
                    bottom + percentage / 2,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                )
            bottom += percentage

        ax2.set_ylabel("割合 (%)")
        ax2.set_title("傾斜度の区分別割合\n(コミュニティ内かつ住居系用途地域内)")
        ax2.set_xticks([])
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # 統計情報の表示
        stats_text = (
            "傾斜度の統計情報\n(コミュニティ内かつ\n住居系用途地域内)\n"
            f"最小値: {np.nanmin(valid_data):.1f}[度]\n"
            f"最大値: {np.nanmax(valid_data):.1f}[度]\n"
            f"平均値: {np.nanmean(valid_data):.1f}[度]\n"
            f"中央値: {np.nanmedian(valid_data):.1f}[度]"
        )
        ax2.text(
            1.05,
            0.6,
            stats_text,
            transform=ax2.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="none", edgecolor="lightgray", pad=4),
        )

        plt.tight_layout()
        plt.savefig(f"result/sasebo_com_slope/slope_{com_id}.png", dpi=300)
        plt.close()

    return com_area_slope, steep_ratio


def calc_and_visualize_shc(
    com_id: str,
    com_name: str,
    bbox_elevation: np.ndarray,
    com_mask: np.ndarray,
    housing_mask: np.ndarray,
    visualize: bool,
) -> np.ndarray:
    """
    SHCの計算と可視化を行い、コミュニティのSHCデータを返す。

    Args:
        com_name (str): コミュニティ名(例: 16_182294)
        bbox_elevation (numpy.ndarray): コミュニティを囲むBBoxの標高データ
        com_mask (numpy.ndarray): コミュニティのマスク
        housing_mask (numpy.ndarray): 住居系用途地域のマスク
    """
    window_size = 10
    _, plan_curv = calc_profile_and_plan_curvature(bbox_elevation)
    bbox_shc = calc_shc(plan_curv, window_size)
    
    # コミュニティ内かつ住居系用途地域内のデータのみを抽出
    com_area_shc = np.where((com_mask == 1) & (housing_mask == 1), bbox_shc, np.nan)
    
    # 有効なデータがあるか確認
    valid_data = com_area_shc[~np.isnan(com_area_shc)]
    if len(valid_data) == 0:
        print(f"警告: {com_name}のSHCデータが存在しません")
        return com_area_shc

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = 0.15

        im = ax.imshow(bbox_shc, cmap="viridis", aspect="equal", vmin=0, vmax=vmax)
        
        # コミュニティ外を灰色で表示
        ax.imshow(
            np.where(com_mask == 0, 0.7, np.nan),
            cmap="gray_r",
            alpha=0.8,
            aspect="equal",
        )
        
        # 住居系用途地域外（コミュニティ内）に網掛けを追加
        non_housing_area = np.where((com_mask == 1) & (housing_mask == 0), 1, np.nan)
        for i in range(non_housing_area.shape[0]):
            for j in range(non_housing_area.shape[1]):
                if not np.isnan(non_housing_area[i, j]):
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=False,
                        hatch='///',
                        alpha=0.3
                    ))

        # カラーバーの描画（axを明示的に指定）
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("SHC")
        
        ax.set_title(f"{com_name}の平面曲率標準偏差(SHC)\n"
                    f"(移動窓: 半径{window_size*10}mの円)\n"
                    "(斜線部: 住居系用途地域外)")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"result/sasebo_com_shc/shc_{com_id}.png", dpi=300)
        plt.close()

    return com_area_shc


def calc_slope(elevation):
    """
    Prewittフィルタを用いた傾斜度計算
    より自然な地形の勾配を検出

    Args:
        elevation (numpy.ndarray): 標高データ
    Returns:
        slope (numpy.ndarray): 傾斜度データ
    """
    pixel_size_x = 10
    pixel_size_y = 10

    # Prewittフィルタの定義
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / (6.0 * pixel_size_x)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / (6.0 * pixel_size_y)
    dx = convolve(elevation, kernel_x, mode="mirror")
    dy = convolve(elevation, kernel_y, mode="mirror")

    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    return slope


def calc_profile_and_plan_curvature(elevation):
    """
    DEMから縦断勾配と平面曲率を計算する関数

    Args:
        elevation (numpy.ndarray): 標高データ
        window_size (int): 評価する窓サイズ

    Returns:
        tuple: (profile_curvature, plan_curvature)
    """
    pixel_size = 10

    kernel_x = np.array([[-1, 0, 1]]) / (2 * pixel_size)
    kernel_y = kernel_x.T
    dx = convolve(elevation, kernel_x, mode="mirror")
    dy = convolve(elevation, kernel_y, mode="mirror")

    kernel_xx = np.array([[1, -2, 1]]) / (pixel_size**2)
    kernel_yy = kernel_xx.T
    kernel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / (4 * pixel_size**2)

    dxx = convolve(elevation, kernel_xx, mode="mirror")
    dyy = convolve(elevation, kernel_yy, mode="mirror")
    dxy = convolve(elevation, kernel_xy, mode="mirror")

    # 勾配の大きさ
    p = dx**2 + dy**2
    # ゼロ除算を防ぐためのマスク作成
    mask = p > 1e-10
    q = np.sqrt(1 + p[mask])

    # 初期値を0で初期化（平坦な領域は曲率0）
    profile_curvature = np.zeros_like(dx, dtype=np.float64)
    plan_curvature = np.zeros_like(dx, dtype=np.float64)

    # 縦断勾配（profile curvature）
    profile_curvature[mask] = (
        dxx[mask] * dx[mask] ** 2
        + 2 * dxy[mask] * dx[mask] * dy[mask]
        + dyy[mask] * dy[mask] ** 2
    ) / (p[mask] * q**3)

    # 平面曲率（plan curvature）
    plan_curvature[mask] = (
        dxx[mask] * dy[mask] ** 2
        - 2 * dxy[mask] * dx[mask] * dy[mask]
        + dyy[mask] * dx[mask] ** 2
    ) / (p[mask] ** 1.5)

    return profile_curvature, plan_curvature


def calc_shc(plan_curvature, window_size):
    """
    平面曲率の標準偏差（SHC）を計算する関数（高速化版）
    """
    # 円形のカーネルを作成
    y, x = np.ogrid[-window_size : window_size + 1, -window_size : window_size + 1]
    circular_mask = x * x + y * y <= window_size * window_size
    kernel = circular_mask.astype(float)
    kernel /= kernel.sum()  # 正規化

    # 有効なデータのマスクを作成
    valid_mask = ~np.isnan(plan_curvature)

    # 有効なデータ数を計算（一度だけ）
    n_valid = convolve(valid_mask.astype(float), kernel, mode="reflect")

    # 最小有効データ数の閾値（カーネル合計の50%）
    min_valid = 0.5 * np.sum(kernel)

    # データの合計と二乗の合計を同時に計算
    masked_data = np.where(valid_mask, plan_curvature, 0)
    sum_data = convolve(masked_data, kernel, mode="reflect")
    sum_sq_data = convolve(masked_data * masked_data, kernel, mode="reflect")

    # 十分なデータがある位置でのみ統計量を計算
    valid_positions = n_valid >= min_valid

    # 平均と分散を一度に計算
    mean = np.where(valid_positions, sum_data / n_valid, np.nan)
    variance = np.where(
        valid_positions, (sum_sq_data / n_valid) - (mean * mean), np.nan
    )

    # 数値誤差による負の分散を補正して標準偏差を計算
    shc = np.sqrt(np.maximum(variance, 0))

    return shc


def calc_shc_in_steep_area(bbox_elevation, housing_mask):
    """
    傾斜度5度以上の範囲でのSHCを計算・可視化する関数

    Args:
        bbox_elevation (numpy.ndarray): 標高データ
        housing_mask (numpy.ndarray): 住居系用途地域のマスク

    Returns:
        steep_area_shc (numpy.ndarray): 傾斜5度以上の範囲のSHCデータ
    """
    slope = calc_slope(bbox_elevation)
    steep_mask = (slope >= 5) & (housing_mask == 1)

    window_size = 10
    _, plan_curv = calc_profile_and_plan_curvature(bbox_elevation)
    bbox_shc = calc_shc(plan_curv, window_size)
    steep_area_shc = np.where(steep_mask, bbox_shc, np.nan)

    return steep_area_shc


def visualize_pixel_histogram(data, com_name, data_type):
    """
    ピクセルレベルでの傾斜度またはSHCのヒストグラムを作成する関数

    Args:
        data (numpy.ndarray): 傾斜度またはSHCのデータ配列
        com_name (str): コミュニティ名
        data_type (str): データの種類（'slope'または'shc'）
    """
    # nanを除外したデータを取得
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        print(f"Warning: {com_name}の{data_type}データが存在しません")
        return

    # 通常データの平均値を計算
    original_mean = np.mean(valid_data)
    original_median = np.median(valid_data)

    plt.figure(figsize=(10, 6))

    # データの前処理とビンの設定
    if data_type == "slope":
        # 25度以上のデータを25度に置き換え
        plot_data = valid_data
        bins = np.arange(0, 27, 1)  # 0-25度まで1度間隔
        xlabel = "傾斜度 [度]"
        title_type = "傾斜度"
        last_bin_label = "25度以上"
    else:  # shc
        # 0.20以上のデータを0.20に置き換え
        plot_data = valid_data
        bins = np.arange(0, 0.22, 0.01)  # 0-0.20まで0.01間隔
        xlabel = "SHC"
        title_type = "SHC"
        last_bin_label = "0.20以上"

    # ヒストグラムをプロット
    if data_type == "slope":
        n, bins, patches = plt.hist(
            plot_data, bins=bins, edgecolor="black", color="#940b25", alpha=0.65
        )
    else:
        n, bins, patches = plt.hist(
            plot_data, bins=bins, edgecolor="black", color="#5d940b", alpha=0.65
        )

    # 最後のビンの色を変更して目立たせる
    if len(n) > 0:  # データが存在する場合
        patches[-1].set_facecolor("red")  # 最後のビンを赤色に
        patches[-1].set_alpha(0.6)  # 透明度を調整

        last_bin_center = (bins[-2] + bins[-1]) / 2
        plt.annotate(
            last_bin_label,
            xy=(last_bin_center, n[-1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ymax = plt.ylim()[1]
    plt.vlines(
        x=min(original_mean, bins[-1]),
        ymin=0,
        ymax=ymax,
        colors="red",
        linestyles="dashed",
        label=f"平均値: {original_mean:.2f}",
    )
    plt.vlines(
        x=min(original_median, bins[-1]),
        ymin=0,
        ymax=ymax,
        colors="green",
        linestyles="dashed",
        label=f"中央値: {original_median:.2f}",
    )
    plt.legend(loc="lower left")

    # 基本統計情報を計算（表示するデータに応じて）
    target_data = valid_data
    stats_text = (
        f"面積: {len(target_data)/100:.1f}[ha]\n"
        f"     ({len(target_data):,}[pixel])\n"
        f"平均値: {np.mean(target_data):.3f}\n"
        f"中央値: {np.median(target_data):.3f}\n"
        f"標準偏差: {np.std(target_data):.3f}\n"
    )

    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.xlabel(xlabel)
    plt.ylabel("度数")
    plt.title(f"{com_name}の{title_type}分布")
    plt.grid(True, alpha=0.3)

    # x軸の範囲を設定
    if data_type == "slope":
        plt.xlim(0, 25)
    else:
        plt.xlim(0, 0.20)

    plt.ylim(bottom=0)
    plt.legend()

    plt.savefig(
        f"result/sasebo_com_slope_hist/slope_{com_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


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
def plot_data():
    pass


# TODO: 2.DEMデータと住民自治組織のデータを重ねて、住民自治組織ごとに傾斜・SHCの算出ができるようにし、全域が斜面市街地の自治組織だけを抽出する。
def calc_slope_and_shc(com_id, name, com_area, src, sasebo_housing_area):
    try:
        print(f"コミュニティID: {com_id}, コミュニティ名: {name}")
        print('maskの作成')
        bounds = com_area.total_bounds
        bbox = box(*bounds)
        shapes = [bbox]
        bbox_elevation, bbox_transform = mask(
            src, shapes=shapes, crop=True, nodata=np.nan
        )
        bbox_elevation = bbox_elevation[0]
        com_mask = rasterio.features.rasterize(
            [(geom, 1) for geom in com_area.geometry],
            out_shape=bbox_elevation.shape,
            transform=bbox_transform,
            fill=0,
            dtype=np.uint8,
        )
        # 住居系用途地域のマスク作成
        housing_mask = rasterio.features.rasterize(
            [(geom, 1) for geom in sasebo_housing_area.geometry],
            out_shape=bbox_elevation.shape,
            transform=bbox_transform,
            fill=0,
            dtype=np.uint8,
        )
        print('標高')
        com_height = calc_and_visualize_height(
            com_id, name, bbox_elevation, com_mask, housing_mask, visualize=True
        )
        print('傾斜')
        (
            com_slope,
            steep_ratio,
        ) = calc_and_visualize_slope(
            com_id, name, bbox_elevation, com_mask, housing_mask, visualize=True
        )
        print('shc')
        com_shc = calc_and_visualize_shc(
            com_id, name, bbox_elevation, com_mask, housing_mask, visualize=True
        )

        com_terrain_stats = {
            "ID": com_id,
            "NAME": name,
            "標高_平均値": np.nan if np.all(np.isnan(com_height)) else np.nanmean(com_height),
            "標高_中央値": np.nan if np.all(np.isnan(com_height)) else np.nanmedian(com_height),
            "標高_標準偏差": np.nan if np.all(np.isnan(com_height)) else np.nanstd(com_height),
            "傾斜度_平均値": np.nan if np.all(np.isnan(com_slope)) else np.nanmean(com_slope),
            "傾斜度_中央値": np.nan if np.all(np.isnan(com_slope)) else np.nanmedian(com_slope),
            "傾斜度_標準偏差": np.nan if np.all(np.isnan(com_slope)) else np.nanstd(com_slope),
            "SHC_平均値": np.nan if np.all(np.isnan(com_shc)) else np.nanmean(com_shc),
            "SHC_中央値": np.nan if np.all(np.isnan(com_shc)) else np.nanmedian(com_shc),
            "SHC_標準偏差": np.nan if np.all(np.isnan(com_shc)) else np.nanstd(com_shc),
            "住居系用途地域に占める斜面市街地の割合": steep_ratio,
        }
        return com_terrain_stats
    except ValueError as e:
        print(f"警告: {com_id}の処理でエラーが発生: {e}")
        return None

# TODO: 3.住民自治組織ごとに建物数のカウント、属性情報の取得ができるようにする。


# TODO: 4.住民自治組織ごとに接道不良率、地区の高齢化率、推定高齢者数の算出ができるようにする。


# TODO: 5.住民自治組織ごとに将来の接道不良住宅に住む高齢者数の算出ができるようにする。


def create_slope_map(sasebo_community, sasebo_housing_area, stats_df):
    """
    傾斜度の中央値に基づいたマップを作成する

    Args:
        sasebo_community (GeoDataFrame): コミュニティのジオデータフレーム
        sasebo_housing_area (GeoDataFrame): 住居系用途地域のジオデータフレーム
        stats_df (DataFrame): 各コミュニティの地形統計情報のデータフレーム
    """    
    # コミュニティデータと統計情報をマージ
    merged_data = sasebo_community.merge(stats_df, left_on='KANRIID', right_on='ID')
    
    # コミュニティを住居系用途地域で切り抜く
    housing_union = sasebo_housing_area.unary_union
    merged_data['geometry'] = merged_data.geometry.intersection(housing_union)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 住居系用途地域を灰色で表示
    housing_area = sasebo_housing_area.plot(
        ax=ax,
        color='lightgrey',
        alpha=0.5,
        label='住居系用途地域'
    )
    
    # カスタムカラーマップの作成
    colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "red"]
    bounds = [0, 5, 10, 15, 20, 25, 30]  # np.infを具体的な値に変更
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, len(colors))
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # NaNを含むデータを黒で表示するため、まず黒で塗りつぶす
    merged_data[merged_data['傾斜度_中央値'].isna()].plot(
        ax=ax,
        color='black',
        alpha=0.7
    )
    
    # 有効なデータを色分けして表示
    merged_data[merged_data['傾斜度_中央値'].notna()].plot(
        column='傾斜度_中央値',
        ax=ax,
        cmap=cmap,
        norm=norm,
        legend=True,
        legend_kwds={
            'label': '傾斜度の中央値 [度]',
            'orientation': 'vertical',
            'shrink': 0.8,
            'boundaries': bounds,
            'ticks': bounds[:-1],
            'format': '%g'
        },
        alpha=0.7
    )
    
    # コミュニティの境界線を黒で表示
    merged_data.boundary.plot(
        ax=ax,
        color='black',
        linewidth=0.5,
        alpha=0.5
    )
    
    ax.set_title('コミュニティごとの傾斜度中央値\n(住居系用途地域内)', pad=20, fontsize=14)
    ax.axis('off')
    
    # 凡例を追加（handles引数を明示的に指定）
    handles = [housing_area]
    labels = ['住居系用途地域']
    ax.legend(handles=handles, labels=labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('result/slope_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_shc_map(sasebo_community, sasebo_housing_area, stats_df):
    """
    SHCの平均値に基づいたマップを作成する

    Args:
        sasebo_community (GeoDataFrame): コミュニティのジオデータフレーム
        sasebo_housing_area (GeoDataFrame): 住居系用途地域のジオデータフレーム
        terrain_stats_list (list): 各コミュニティの地形統計情報のリスト
    """
    # コミュニティデータと統計情報をマージ
    merged_data = sasebo_community.merge(stats_df, left_on='KANRIID', right_on='ID')
    
    # コミュニティを住居系用途地域で切り抜く
    housing_union = sasebo_housing_area.unary_union
    merged_data['geometry'] = merged_data.geometry.intersection(housing_union)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 住居系用途地域を灰色で表示
    housing_area = sasebo_housing_area.plot(
        ax=ax,
        color='lightgrey',
        alpha=0.5,
        label='住居系用途地域'
    )
    
    # SHCの平均値でコミュニティを色分け
    merged_data.plot(
        column='SHC_平均値',
        ax=ax,
        legend=True,
        legend_kwds={
            'label': 'SHCの平均値',
            'orientation': 'vertical',
            'shrink': 0.8
        },
        cmap='viridis',
        alpha=0.7,
        missing_kwds={'color': 'white'}
    )
    
    # コミュニティの境界線を黒で表示
    merged_data.boundary.plot(
        ax=ax,
        color='black',
        linewidth=0.5,
        alpha=0.5
    )
    
    ax.set_title('コミュニティごとのSHC平均値\n(住居系用途地域内)', pad=20, fontsize=14)
    ax.axis('off')
    
    # 凡例を追加（handles引数を明示的に指定）
    handles = [housing_area]
    labels = ['住居系用途地域']
    ax.legend(handles=handles, labels=labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('result/shc_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_com_shp(community, housing_area, terrain_stats_df, output_path):
    """
    地形統計情報を含むシェープファイルを作成する関数

    Args:
        community (GeoDataFrame): コミュニティのジオデータフレーム
        housing_area (GeoDataFrame): 住居系用途地域のジオデータフレーム
        terrain_stats_df (DataFrame): 地形統計情報のデータフレーム
        output_path (str): 出力するシェープファイルのパス
    """
    # 地形統計情報をコミュニティのシェープファイルに結合
    merged_community = community.merge(
        terrain_stats_df,
        left_on='KANRIID',
        right_on='ID',
        how='left'
    )
    
    # 不要な列を削除
    if 'ID' in merged_community.columns:
        merged_community = merged_community.drop('ID', axis=1)
    
    # コミュニティを住居系用途地域で切り抜く
    housing_union = housing_area.unary_union
    merged_community['geometry'] = merged_community.geometry.intersection(housing_union)
    
    # 空の（完全に切り取られた）ジオメトリを持つ行を削除
    merged_community = merged_community[~merged_community.geometry.is_empty]
    
    # カラム名を短く、英語に変更（シェープファイルの制限に対応）
    column_mapping = {
        '標高_平均値': 'elev_mean',
        '標高_中央値': 'elev_med',
        '標高_標準偏差': 'elev_std',
        '傾斜度_平均値': 'slope_mean',
        '傾斜度_中央値': 'slope_med',
        '傾斜度_標準偏差': 'slope_std',
        'SHC_平均値': 'shc_mean',
        'SHC_中央値': 'shc_med',
        'SHC_標準偏差': 'shc_std',
        '住居系用途地域に占める斜面市街地の割合': 'steep_ratio'
    }
    merged_community = merged_community.rename(columns=column_mapping)
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # シェープファイルとして出力
    merged_community.to_file(output_path, encoding='cp932', driver='ESRI Shapefile')
    print(f"地形統計情報を含むシェープファイルを出力しました: {output_path}")
    
    return merged_community

if __name__ == "__main__":
    # caffeinate プロセスを開始
    caffeinate_process = subprocess.Popen(['caffeinate', '-i'])
    
    try:
        # ファイルの存在確認
        if not all(check_file_exists(file) for file in files_to_check):
            print("一部のファイルが見つかりませんでした。プログラムを終了します。")
            exit(1)
        else:
            print("file path OK")
            
        # データの読み込み
        sasebo_community = gpd.read_file(SASEBO_COMMUNITY, encoding='cp932')
        yokosuka_community = gpd.read_file(YOKOSUKA_COMMUNITY, encoding='cp932')
        sasebo_housing_area = gpd.read_file(SASEBO_HOUSING_AREA, encoding='cp932')
        yokosuka_housing_area = gpd.read_file(YOKOSUKA_HOUSING_AREA, encoding='cp932')
        road_1_4 = gpd.read_file(SASEBO_ROAD_1_4, encoding='cp932')
        road_1_5 = gpd.read_file(SASEBO_ROAD_1_5, encoding='cp932')
        road_2_ikkatsu = gpd.read_file(SASEBO_ROAD_2_IKKATSU, encoding='cp932')
        road_2_kobetsu = gpd.read_file(SASEBO_ROAD_2_KOBETSU, encoding='cp932')
        road_3 = gpd.read_file(SASEBO_ROAD_3, encoding='cp932')
        road_city = gpd.read_file(SASEBO_ROAD_CITY, encoding='cp932', ignore_fields=['認定年月日'])
        # CRSの設定
        sasebo_community = sasebo_community.set_crs(epsg=6669).to_crs(epsg=4326)
        sasebo_housing_area = sasebo_housing_area.set_crs(epsg=4326)
        road_1_4 = road_1_4.set_crs(epsg=6669).to_crs(epsg=4326)
        road_1_5 = road_1_5.set_crs(epsg=6669).to_crs(epsg=4326)
        road_2_ikkatsu = road_2_ikkatsu.set_crs(epsg=6669).to_crs(epsg=4326)
        road_2_kobetsu = road_2_kobetsu.set_crs(epsg=6669).to_crs(epsg=4326)
        road_3 = road_3.set_crs(epsg=6669).to_crs(epsg=4326)
        road_city = road_city.set_crs(epsg=6669).to_crs(epsg=4326)

        # 地形統計情報を格納するリスト
        terrain_stats_df = pd.DataFrame()

        community = sasebo_community
        housing_area = sasebo_housing_area
        DEM_PATH = SASEBO_DEM
        RESULT_XLSX_PATH = RESULT_XLSX_SASEBO
        RESULT_SHP_PATH = RESULT_SHP_SASEBO

        # 住居系用途地域と重なるコミュニティを抽出
        housing_area_union = housing_area.geometry.buffer(0).unary_union
        community['geometry'] = community.geometry.buffer(0)
        community = community[community.geometry.intersects(housing_area_union)]
        print(f"住居系用途地域と重なるコミュニティ数: {len(community)}")

        with rasterio.open(DEM_PATH) as src:
            print("DEM読み込み完了")
            # 各コミュニティの傾斜度・SHCを計算
            for com_id, name in zip(community['KANRIID'], community['NAME']):
                com_area = community[community['KANRIID'] == com_id].copy()
                terrain_stats = calc_slope_and_shc(com_id, name, com_area, src, housing_area)
                if terrain_stats is not None:
                    terrain_stats_df = pd.concat(
                                        [terrain_stats_df, pd.DataFrame([terrain_stats])],
                                        ignore_index=True,
                                    )
                    print(f"{len(terrain_stats_df)}/{len(community)}の処理が完了。")
                    print(terrain_stats)
        terrain_stats_df.to_excel(RESULT_XLSX_PATH, index=False)
        create_slope_map(community, housing_area, terrain_stats_df)
        create_shc_map(community, housing_area, terrain_stats_df)
        merged_community = create_com_shp(community, housing_area, terrain_stats_df, RESULT_SHP_PATH)

    finally:
        caffeinate_process.terminate()
