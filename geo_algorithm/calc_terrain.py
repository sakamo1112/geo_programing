import rasterio
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import geopandas as gpd
from rasterio.mask import mask
import pandas as pd
from shapely.geometry import shape
# DEMデータのパス
ONOMICHI_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/尾道市_DEM/尾道市_DEM.tiff"
YOKOSUKA_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/横須賀市_DEM/横須賀市_DEM.tif"
SIMONOSEKI_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/下関市_DEM/下関市_DEM.tif"
KURE_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/呉市_DEM/呉市_DEM.tiff"
SASEBO_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/佐世保市_DEM/佐世保市_DEM.tiff"
OTARU_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/小樽市_DEM/小樽市_DEM.tiff"
KOBE_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/神戸市_DEM/神戸市_DEM.tif"
NAGASAKI_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/長崎市_DEM/長崎市_DEM.tiff"
ATAMI_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/熱海市_DEM/熱海市_DEM.tiff"
HAKODATE_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/函館市_DEM/函館市_DEM.tiff"
BEPPU_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/別府市_DEM/別府市_DEM.tiff"
KITAKYUSHU_DEM = "/Users/sakamo/Desktop/GISDATA/DEM/北九州市_DEM/北九州市_DEM.tif"

ONOMICHI_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/尾道市_住居系用途地域/尾道市_住居系用途地域.shp"
YOKOSUKA_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/横須賀市_住居系用途地域/横須賀市_住居系用途地域.shp"
SIMONOSEKI_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/下関市_住居系用途地域/下関市_住居系用途地域.shp"
KAMAKURA_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/鎌倉市_住居系用途地域/鎌倉市_住居系用途地域.shp"
KURE_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/呉市_住居系用途地域/呉市_住居系用途地域.shp"
SASEBO_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/佐世保市_住居系用途地域/佐世保市_住居系用途地域.shp"
OTARU_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/小樽市_住居系用途地域/小樽市_住居系用途地域.shp"
KOBE_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/神戸市_住居系用途地域/神戸市_住居系用途地域.shp"
IKOMA_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/生駒市_住居系用途地域/生駒市_住居系用途地域.shp"
TAZIMI_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/多治見市_住居系用途地域/多治見市_住居系用途地域.shp"
NAGASAKI_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/長崎市_住居系用途地域/長崎市_住居系用途地域.shp"
ATAMI_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/熱海市_住居系用途地域/熱海市_住居系用途地域.shp"
HAKODATE_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/函館市_住居系用途地域/函館市_住居系用途地域.shp"
BEPPU_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/別府市_住居系用途地域/別府市_住居系用途地域.shp"
KITAKYUSHU_H = "/Users/sakamo/Desktop/GISDATA/住居系用途地域リスト_wgs/北九州市_住居系用途地域/北九州市_住居系用途地域.shp"


def visualize_height(elevation, city_name, housing_area):
    """
    DEMデータを読み込んで可視化する関数
    
    Parameters
    ----------
    dem_path : str
        DEMデータ(GeoTIFF)のパス
    """ 
    # 標高値の統計量を表示
    print(f"標高の統計情報:")
    print(f"最小値: {np.nanmin(elevation):.1f}m")
    print(f"最大値: {np.nanmax(elevation):.1f}m") 
    print(f"平均値: {np.nanmean(elevation):.1f}m")
    print(f"中央値: {np.nanmedian(elevation):.1f}m")

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # マスク画像を作成（住居系用途地域外を灰色にする）
    with rasterio.open(dem_paths[city_name]) as src:
        # 住居系用途地域でDEMをマスク
        if src.crs != housing_area.crs:
            housing_area = housing_area.to_crs(src.crs)
        mask = rasterio.features.rasterize(
            housing_area.geometry,
            out_shape=elevation.shape,
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
    
    # カラーマップの設定
    vmin = 0
    vmax = 300
    
    # 標高データの表示
    im = ax.imshow(elevation, 
                   cmap='terrain',
                   aspect='equal',
                   vmin=vmin,
                   vmax=vmax)
    
    # マスク部分を灰色で表示（住居系用途地域外を灰色に）
    ax.imshow(np.where(mask == 0, 0.7, np.nan),  # mask==0の部分(住居系用途地域外)を灰色(0.7)に、それ以外を透明に
              cmap='gray_r',
              alpha=0.8,
              aspect='equal')
    
    # カラーバーの追加
    cbar = plt.colorbar(im)
    cbar.set_label('標高 (m)')
    
    plt.title(f'{city_name}の数値標高モデル(DEM)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"terrain_result/height_{city_name}.png", dpi=300)

def visualize_all_dems():
    """
    全ての自治体のDEMを可視化し、グリッド状に並べて表示する関数
    """
    # 全てのDEMパスを取得
    dem_paths = [ONOMICHI_DEM, YOKOSUKA_DEM, SIMONOSEKI_DEM, KURE_DEM,
                 SASEBO_DEM, OTARU_DEM, KOBE_DEM, NAGASAKI_DEM,
                 ATAMI_DEM, HAKODATE_DEM, BEPPU_DEM, KITAKYUSHU_DEM]
    
    # グリッドのサイズを設定（4行3列）
    rows, cols = 4, 3
    
    # 全体の図を作成
    fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
    fig.suptitle('各自治体の数値標高モデル(DEM)', fontsize=16, y=0.95)
    
    for i, dem_path in enumerate(dem_paths):
        ax = axes[i // cols, i % cols]
        
        # DEMデータの読み込みと処理
        with rasterio.open(dem_path) as src:
            elevation = src.read(1)
            nodata = src.nodata
        elevation = np.where(elevation == nodata, np.nan, elevation)
        
        # 可視化
        im = ax.imshow(elevation, cmap='terrain', aspect='equal')
        # ディレクトリ名から自治体名を抽出（例："/path/to/尾道市_DEM" → "尾道市"）
        city_name = os.path.basename(os.path.dirname(dem_path)).replace('_DEM', '')
        ax.set_title(city_name)
        ax.axis('off')
    
    # カラーバーを追加
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('標高 (m)')
    
    plt.tight_layout()
    plt.show()

def calc_slope(elevation):
    """
    指定されたDEMパスから傾斜度を計算する関数
    
    Parameters
    ----------
    dem_path : str
        DEMデータ(GeoTIFF)のパス
        
    Returns
    -------
    slope : numpy.ndarray
        傾斜度（度単位）の配列
    """
    pixel_size_x = 5
    pixel_size_y = 5
    
    # 3x3の移動窓で傾斜度を計算
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) / (8.0 * pixel_size_x)
    
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) / (8.0 * pixel_size_y)
    
    # 畳み込みで勾配を計算
    dx = convolve(elevation, kernel_x, mode='mirror')
    dy = convolve(elevation, kernel_y, mode='mirror')
    
    # 傾斜度を度単位で計算
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    return slope

def visualize_slope(elevation, city_name: str, housing_area, store_data: bool = False):
    """
    傾斜度を計算して可視化する関数
    
    Parameters
    ----------
    elevation : numpy.ndarray
        標高データ
    city_name : str
        都市名
    housing_area : GeoDataFrame
        住居系用途地域のジオメトリ
    store_data : bool
        データを返すかどうか
    """
    # 傾斜度の計算
    slope = calc_slope(elevation)

    # マスク画像を作成（住居系用途地域外を灰色にする）
    with rasterio.open(dem_paths[city_name]) as src:
        # 住居系用途地域でDEMをマスク
        if src.crs != housing_area.crs:
            housing_area = housing_area.to_crs(src.crs)
        mask = rasterio.features.rasterize(
            housing_area.geometry,
            out_shape=elevation.shape,
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

    # 住居系用途地域内のデータのみを使用して統計を計算
    masked_slope = np.where(mask == 1, slope, np.nan)

    # 基本統計情報の表示
    print(f"傾斜度の統計情報:")
    print(f"最小値: {np.nanmin(masked_slope):.1f}度")
    print(f"最大値: {np.nanmax(masked_slope):.1f}度")
    print(f"平均値: {np.nanmean(masked_slope):.1f}度")
    print(f"中央値: {np.nanmedian(masked_slope):.1f}度")
    
    # パーセンタイル値の表示
    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        print(f"{p}パーセンタイル: {np.nanpercentile(masked_slope, p):.1f}度")

    # 傾斜度の区分ごとの割合を計算（住居系用途地域内のみ）
    bins = [0, 5, 10, 15, 20, 25, np.inf]
    labels = ['0-5度', '5-10度', '10-15度', '15-20度', '20-25度', '25度以上']
    hist, _ = np.histogram(masked_slope[~np.isnan(masked_slope)], bins=bins)
    percentages = hist / len(masked_slope[~np.isnan(masked_slope)]) * 100

    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [3.5, 1]})
    plt.subplots_adjust(left=0, right=0.85)
    
    # 傾斜度の空間分布
    im = ax1.imshow(slope, cmap='Reds', aspect='equal', vmin=0, vmax=45)
    
    # マスク部分を灰色で表示（住居系用途地域外を灰色に）
    ax1.imshow(np.where(mask == 0, 0.7, np.nan),
              cmap='gray_r', 
              alpha=0.8,
              aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('傾斜度 (度)')
    ax1.set_title(f'{city_name}の傾斜度分布')
    ax1.axis('off')
    
    # 以下は変更なし（積み上げ棒グラフと統計情報の表示）
    colors = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', 'red']
    bottom = 0
    for i, (percentage, color) in enumerate(zip(percentages, colors)):
        bar = ax2.bar(0, percentage, bottom=bottom, color=color, label=labels[i])
        if percentage >= 3:
            ax2.text(0, bottom + percentage/2, f'{percentage:.1f}%',
                    ha='center', va='center')
        bottom += percentage
    
    ax2.set_ylabel('割合 (%)')
    ax2.set_title('傾斜度の区分別割合(住居系用途地域内)')
    ax2.set_xticks([])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    stats_text = ("傾斜度の統計情報\n(住居系用途地域内)\n"
                 f"最小値: {np.nanmin(masked_slope):.1f}[度]\n"
                 f"最大値: {np.nanmax(masked_slope):.1f}[度]\n"
                 f"平均値: {np.nanmean(masked_slope):.1f}[度]\n"
                 f"中央値: {np.nanmedian(masked_slope):.1f}[度]")
    ax2.text(1.05, 0.6, stats_text, transform=ax2.transAxes,
             ha='left', va='top', bbox=dict(facecolor='none', edgecolor='lightgray',
                                          pad=4))
    
    plt.savefig(f"terrain_result/slope_distribution_{city_name}.png", dpi=300)

    if store_data:
        return {label: percentage for label, percentage in zip(labels, percentages)}
    return None

def calc_curvature(elevation, window_size):
    """
    指定されたDEMパスから曲率を計算する関数

    Args:
        dem_path : str
            DEMデータのパス
        window_size : int
            評価する窓サイズ（奇数を指定）。窓のサイズは5*window_size[m]四方となる。
    """
    pixel_size = 5
    sigma = window_size / 6.0  # 窓サイズに応じたsigma値
    smoothed = gaussian_filter(elevation, sigma=sigma)
    
    # 2次微分カーネルを作成
    kernel_size = window_size
    half_size = kernel_size // 2
    kernel_x = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        dist = i - half_size
        if dist == 0:
            kernel_x[half_size, i] = -2
        else:
            kernel_x[half_size, i] = 1 / (dist**2)
    kernel_y = kernel_x.T
    
    # カーネルの正規化（合計が0になるように）
    kernel_x[half_size, half_size] = -np.sum(kernel_x[half_size, :]) + kernel_x[half_size, half_size]
    kernel_y[half_size, half_size] = -np.sum(kernel_y[:, half_size]) + kernel_y[half_size, half_size]
    # スケール調整
    kernel_x = kernel_x / (pixel_size**2 * window_size)
    kernel_y = kernel_y / (pixel_size**2 * window_size)
    
    d2x = convolve(smoothed, kernel_x, mode='mirror')
    d2y = convolve(smoothed, kernel_y, mode='mirror')
    curvature = d2x + d2y
    
    return curvature

def visualize_curvature(elevation, city_name: str, housing_area, window_size: int, store_data: bool = False):
    # 曲率の計算
    curvature = calc_curvature(elevation, window_size=window_size)

    # マスク画像を作成（住居系用途地域外を灰色にする）
    with rasterio.open(dem_paths[city_name]) as src:
        # 住居系用途地域でDEMをマスク
        if src.crs != housing_area.crs:
            housing_area = housing_area.to_crs(src.crs)
        mask = rasterio.features.rasterize(
            housing_area.geometry,
            out_shape=elevation.shape,
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

    # 住居系用途地域内のデータのみを使用して統計を計算
    masked_curvature = np.where(mask == 1, curvature, np.nan)

    # 基本統計情報の表示（マスクされたデータを使用）
    print(f"曲率の統計情報\n(住居系用途地域内):")
    print(f"最小値: {np.nanmin(masked_curvature):.3f}")
    print(f"最大値: {np.nanmax(masked_curvature):.3f}")
    print(f"平均値: {np.nanmean(masked_curvature):.3f}")
    print(f"中央値: {np.nanmedian(masked_curvature):.3f}")
    
    # パーセンタイル値の表示（マスクされたデータを使用）
    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        print(f"{p}パーセンタイル: {np.nanpercentile(masked_curvature, p):.3f}")

    # 曲率の区分ごとの割合を計算（マスクされたデータを使用）
    bins = [-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf]
    labels = ['-(-0.02)', '(-0.02)-(-0.01)', '(-0.01)-0.01', '0.01-0.02', '0.02-']
    hist, _ = np.histogram(masked_curvature[~np.isnan(masked_curvature)], bins=bins)
    percentages = hist / len(masked_curvature[~np.isnan(masked_curvature)]) * 100

    print("\n曲率の区分ごとの割合(住居系用途地域内):")
    for label, percentage in zip(labels, percentages):
        print(f"{label}: {percentage:.1f}%")

    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(left=0, right=0.85)
    
    # 曲率の空間分布
    vmax = np.nanpercentile(np.abs(masked_curvature), 95)  # マスクされたデータを使用
    im = ax1.imshow(curvature, 
                    cmap='RdYlBu_r',
                    aspect='equal',
                    vmin=-vmax,
                    vmax=vmax)
    
    # マスク部分を灰色で表示（住居系用途地域外を灰色に）
    ax1.imshow(np.where(mask == 0, 0.7, np.nan),
              cmap='gray_r', 
              alpha=0.8,
              aspect='equal')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('曲率')
    ax1.set_title(f'{city_name}の曲率分布(窓サイズ: {window_size*5}m×{window_size*5}m)')
    ax1.axis('off')

    # カラーマップから色を生成
    import matplotlib.cm as cm
    cmap = cm.RdYlBu_r
    n_categories = len(labels)
    colors = [cmap(i/(n_categories-1)) for i in range(n_categories)]
    
    # 積み上げ棒グラフ
    bottom = 0
    for i, (percentage, color) in enumerate(zip(percentages, colors)):
        ax2.bar(0, percentage, bottom=bottom, color=color, label=labels[i])
        if percentage >= 3:  # 3%以上の場合のみ表示
            ax2.text(0, bottom + percentage/2, f'{percentage:.1f}%',
                    ha='center', va='center')
        bottom += percentage
    
    ax2.set_ylabel('割合 (%)')
    ax2.set_title('曲率の区分別割合(住居系用途地域内)')
    ax2.set_xticks([])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 統計情報をax2の凡例の下に配置
    stats_text = ("曲率の統計情報\n"
                 f"最小値: {np.nanmin(curvature):.3f}\n"
                 f"最大値: {np.nanmax(curvature):.3f}\n"
                 f"平均値: {np.nanmean(curvature):.3f}\n"
                 f"中央値: {np.nanmedian(curvature):.3f}")
    ax2.text(1.05, 0.6, stats_text, transform=ax2.transAxes,
             ha='left', va='top', bbox=dict(facecolor='none', edgecolor='lightgray',
                                          pad=4))
    
    plt.savefig(f"terrain_result/curvature_distribution_{city_name}.png", dpi=300)

    if store_data:
        return {label: percentage for label, percentage in zip(labels, percentages)}
    return None

def calc_profile_and_plan_curvature(elevation, window_size):
    """
    DEMから縦断勾配と平面曲率を計算する関数
    
    Parameters
    ----------
    elevation : numpy.ndarray
        標高データ
    window_size : int
        評価する窓サイズ
        
    Returns
    -------
    tuple
        (profile_curvature, plan_curvature)
    """
    pixel_size = 5
    sigma = window_size / 6.0
    smoothed = gaussian_filter(elevation, sigma=sigma)
    
    # 1次微分と2次微分の計算
    kernel_x = np.array([[-1, 0, 1]]) / (2 * pixel_size)
    kernel_y = kernel_x.T
    
    # 1次微分
    dx = convolve(smoothed, kernel_x, mode='mirror')
    dy = convolve(smoothed, kernel_y, mode='mirror')
    
    # 2次微分
    kernel_xx = np.array([[1, -2, 1]]) / (pixel_size**2)
    kernel_yy = kernel_xx.T
    kernel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / (4 * pixel_size**2)
    
    dxx = convolve(smoothed, kernel_xx, mode='mirror')
    dyy = convolve(smoothed, kernel_yy, mode='mirror')
    dxy = convolve(smoothed, kernel_xy, mode='mirror')
    
    # 勾配の大きさ
    p = dx**2 + dy**2
    q = np.sqrt(1 + p)
    
    # 縦断勾配（profile curvature）
    profile_curvature = ((dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) /
                        (p * q**3)) if np.any(p != 0) else np.zeros_like(dx)
    
    # 平面曲率（plan curvature）
    plan_curvature = ((dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) /
                     (p**1.5)) if np.any(p != 0) else np.zeros_like(dx)
    
    return profile_curvature, plan_curvature

def visualize_profile_and_plan_curvature(elevation, city_name: str, housing_area, window_size: int):
    """
    縦断勾配と平面曲率を可視化する関数
    """
    profile_curv, plan_curv = calc_profile_and_plan_curvature(elevation, window_size)
    
    # マスク画像の作成
    with rasterio.open(dem_paths[city_name]) as src:
        if src.crs != housing_area.crs:
            housing_area = housing_area.to_crs(src.crs)
        mask = rasterio.features.rasterize(
            housing_area.geometry,
            out_shape=elevation.shape,
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
    
    # 2つの図を横に並べて表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # 95パーセンタイルをvmaxとして設定
    vmax_profile = np.nanpercentile(np.abs(profile_curv), 95)
    vmax_plan = np.nanpercentile(np.abs(plan_curv), 95)
    
    # 縦断勾配の表示
    im1 = ax1.imshow(profile_curv, 
                     cmap='RdYlBu_r',
                     aspect='equal',
                     vmin=-vmax_profile,
                     vmax=vmax_profile)
    ax1.imshow(np.where(mask == 0, 0.7, np.nan),
               cmap='gray_r',
               alpha=0.8,
               aspect='equal')
    plt.colorbar(im1, ax=ax1, label='縦断勾配')
    ax1.set_title(f'{city_name}の縦断勾配\n(窓サイズ: {window_size*5}m×{window_size*5}m)')
    ax1.axis('off')
    
    # 平面曲率の表示
    im2 = ax2.imshow(plan_curv,
                     cmap='RdYlBu_r',
                     aspect='equal',
                     vmin=-vmax_plan,
                     vmax=vmax_plan)
    ax2.imshow(np.where(mask == 0, 0.7, np.nan),
               cmap='gray_r',
               alpha=0.8,
               aspect='equal')
    plt.colorbar(im2, ax=ax2, label='平面曲率')
    ax2.set_title(f'{city_name}の平面曲率\n(窓サイズ: {window_size*5}m×{window_size*5}m)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"terrain_result/profile_plan_curvature_{city_name}.png", dpi=300, bbox_inches='tight')

def compare_curvature_distributions(curvature_data, window_size):
    """
    全自治体の曲率分布を積み上げ棒グラフで比較

    Parameters
    ----------
    curvature_data : dict
        都市名をキーとし、曲率の区分別割合を値とする辞書
    """
    labels = ['-(-0.02)', '(-0.02)-(-0.01)', '(-0.01)-0.01', '0.01-0.02', '0.02-']
    cities = list(curvature_data.keys())
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
    
    # 極端な曲率（絶対値が0.01以上）の割合で都市を並び替え
    extreme_percentages = {}
    for city in cities:
        extreme_percent = curvature_data[city]['-(-0.02)'] + curvature_data[city]['0.02-'] + curvature_data[city]['(-0.02)-(-0.01)'] + curvature_data[city]['0.01-0.02']
        extreme_percentages[city] = extreme_percent
    
    cities = sorted(cities, key=lambda x: extreme_percentages[x], reverse=True)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(cities))
    bottom = np.zeros(len(cities))
    
    # カラーマップから色を生成
    import matplotlib.cm as cm
    cmap = cm.RdYlBu_r
    n_categories = len(labels)
    colors = [cmap(i/(n_categories-1)) for i in range(n_categories)]

    for label, color in zip(labels, colors):
        percentages = [curvature_data[city][label] for city in cities]
        ax.bar(x, percentages, bottom=bottom, label=label, color=color)
        
        for i, percentage in enumerate(percentages):
            if percentage >= 3:
                ax.text(x[i], bottom[i] + percentage/2, f'{percentage:.1f}%',
                       ha='center', va='center')
        bottom += np.array(percentages)
    
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    ax.set_xlabel('自治体名')
    ax.set_ylabel('割合 [%]')
    ax.set_title(f'各自治体の住居系用途地域における曲率(窓サイズ: {window_size*5}m×{window_size*5}m)', pad=20)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplots_adjust(right=0.85)
    plt.savefig(f"terrain_result/curvature_distribution_comparison.png", dpi=300)

def extract_clipped_dem(dem_path, clip_area):
    with rasterio.open(dem_path) as src:
        # 住居系用途地域データでDEMをクリップ
        if src.crs != clip_area.crs:
            clip_area = clip_area.to_crs(src.crs)
        out_image, out_transform = mask(src, clip_area.geometry, crop=True)
        clipped_dem = out_image[0]  # 最初のバンドを取得
        clipped_dem = np.where(clipped_dem == -9999, np.nan, clipped_dem)
    return clipped_dem

def compare_slope_distributions(slopes_data):
    """
    全自治体の傾斜度分布を積み上げ棒グラフで比較

    Parameters
    ----------
    slopes_data : dict
        都市名をキーとし、傾斜度の区分別割合を値とする辞書
    """
    labels = ['0-5度', '5-10度', '10-15度', '15-20度', '20-25度', '25度以上']
    cities = list(slopes_data.keys())
    colors = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', 'red']
    
    # 5度以上の割合を計算して、その割合で都市を並び替え
    steep_percentages = {}
    for city in cities:
        steep_percent = sum(slopes_data[city][label] for label in labels[1:])  # '0-5度'以外の合計
        steep_percentages[city] = steep_percent
    
    # 5度以上の割合で降順にソート
    cities = sorted(cities, key=lambda x: steep_percentages[x], reverse=True)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(cities))
    bottom = np.zeros(len(cities))
    
    bars = []
    for label, color in zip(labels, colors):
        percentages = [slopes_data[city][label] for city in cities]
        bars.append(ax.bar(x, percentages, bottom=bottom, label=label, color=color))
        
        # 各セグメントの中央に割合を表示（3%以上の場合のみ）
        for i, percentage in enumerate(percentages):
            if percentage >= 3:
                y_pos = bottom[i] + percentage/2  # bottomを配列としてアクセス
                ax.text(x[i], y_pos, f'{percentage:.1f}%',
                       ha='center', va='center')
        bottom += np.array(percentages)  # 明示的にNumPy配列として加算
    
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    ax.set_xlabel('自治体名')
    ax.set_ylabel('割合 [%]')
    ax.set_title('各自治体の住居系用途地域における傾斜度', pad=20)
    
    # 凡例は1回だけ設定
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # マージンを手動で調整
    plt.subplots_adjust(right=0.85)  # 凡例用のスペースを確保
    plt.savefig(f"terrain_result/slope_distribution_comparison.png", dpi=300)

def compare_slope_boxplots(elevation_data, housing_areas):
    """
    全自治体の傾斜度分布を箱ひげ図で比較し、統計値を返す関数
    
    Parameters
    ----------
    elevation_data : dict
        都市名をキーとし、標高データを値とする辞書
    housing_areas : dict
        都市名をキーとし、住居系用途地域のGeoDataFrameを値とする辞書
        
    Returns
    -------
    dict
        都市名をキーとし、平均値、中央値、標準偏差を値とする辞書
    """
    # データを格納するリスト
    slope_data = []
    city_labels = []
    stats_dict = {}
    
    # 各都市の傾斜度データを計算
    for city, elevation in elevation_data.items():
        # 傾斜度の計算
        slope = calc_slope(elevation)
        
        # マスク画像の作成
        with rasterio.open(dem_paths[city]) as src:
            if src.crs != housing_areas[city].crs:
                housing_area = housing_areas[city].to_crs(src.crs)
            else:
                housing_area = housing_areas[city]
            mask = rasterio.features.rasterize(
                housing_area.geometry,
                out_shape=elevation.shape,
                transform=src.transform,
                fill=0,
                default_value=1,
                dtype=np.uint8
            )
        
        # 住居系用途地域内のデータのみを抽出
        masked_slope = slope[mask == 1]
        masked_slope = masked_slope[~np.isnan(masked_slope)]
        
        # 統計値の計算と保存
        stats_dict[city] = {
            'mean': np.mean(masked_slope),
            'median': np.median(masked_slope),
            'std': np.std(masked_slope)
        }
        
        # データの追加
        slope_data.append(masked_slope)
        city_labels.append(city)
    
    # 中央値で都市を並び替え
    city_order = []
    medians = []
    for i, city in enumerate(city_labels):
        median = stats_dict[city]['median']
        medians.append((city, median))
    
    # 中央値で降順にソート
    medians.sort(key=lambda x: x[1], reverse=True)
    city_order = [x[0] for x in medians]
    
    # データを並び替え
    slope_data = [slope_data[city_labels.index(city)] for city in city_order]
    
    # 箱ひげ図の作成
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 箱ひげ図のプロット
    bp = ax.boxplot(slope_data, 
                    labels=city_order,
                    whis=[10, 90],  # 10-90パーセンタイルを表示
                    showfliers=False)  # 外れ値は表示しない
    
    # グラフの装飾
    ax.set_xticklabels(city_order, rotation=45)
    ax.set_ylabel('傾斜度 [度]')
    ax.set_title('各自治体の住居系用途地域における傾斜度分布')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 統計情報をテキストで表示
    stats_text = "箱ひげ図の説明:\n"
    stats_text += "箱の上端: 75パーセンタイル\n"
    stats_text += "箱の中線: 中央値\n"
    stats_text += "箱の下端: 25パーセンタイル\n"
    stats_text += "ヒゲの上端: 90パーセンタイル\n"
    stats_text += "ヒゲの下端: 10パーセンタイル"
    
    ax.text(1.02, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("terrain_result/slope_boxplot_comparison.png", dpi=300, bbox_inches='tight')
    
    return stats_dict

def calc_shc(plan_curvature, window_size):
    """
    平面曲率の標準偏差（SHC）を計算する関数（最適化版）
    """
    from scipy.ndimage import uniform_filter
    import numpy.ma as ma
    
    # nanを含むデータをマスクする
    masked_curvature = ma.masked_invalid(plan_curvature)
    
    # 平均値の計算
    mean = uniform_filter(masked_curvature.filled(0), size=window_size, mode='mirror')
    mean_mask = uniform_filter(~masked_curvature.mask * 1.0, size=window_size, mode='mirror')
    mean = np.where(mean_mask > 0, mean / mean_mask, np.nan)
    
    # 二乗の平均を計算
    mean_sq = uniform_filter((masked_curvature.filled(0))**2, size=window_size, mode='mirror')
    mean_sq = np.where(mean_mask > 0, mean_sq / mean_mask, np.nan)
    
    # 標準偏差の計算
    variance = mean_sq - mean**2
    # 数値誤差で負の値になることがあるため、クリップする
    variance = np.clip(variance, 0, None)
    shc = np.sqrt(variance)
    
    return shc

def visualize_shc(elevation, city_name: str, housing_area, window_size: int):
    """
    平面曲率の標準偏差（SHC）を可視化する関数
    
    Parameters
    ----------
    elevation : numpy.ndarray
        標高データ
    city_name : str
        都市名
    housing_area : GeoDataFrame
        住居系用途地域のジオメトリ
    window_size : int
        移動窓のサイズ
    """
    # 平面曲率を計算
    print(f'{city_name}の平面曲率を計算中...')
    _, plan_curv = calc_profile_and_plan_curvature(elevation, window_size)
    
    # SHCを計算
    print(f'{city_name}の平面曲率標準偏差(SHC)を計算中...')
    shc = calc_shc(plan_curv, window_size)
    
    # マスク画像の作成
    print(f'{city_name}のマスク画像を作成中...')
    with rasterio.open(dem_paths[city_name]) as src:
        if src.crs != housing_area.crs:
            housing_area = housing_area.to_crs(src.crs)
        mask = rasterio.features.rasterize(
            housing_area.geometry,
            out_shape=elevation.shape,
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
    
    # 住居系用途地域内のデータのみを使用して統計を計算
    print(f'{city_name}のSHCをマスク中...')
    masked_shc = np.where(mask == 1, shc, np.nan)
    
    # 基本統計情報の表示
    print(f"\nSHCの統計情報（住居系用途地域内）:")
    print(f"最小値: {np.nanmin(masked_shc):.6f}")
    print(f"最大値: {np.nanmax(masked_shc):.6f}")
    print(f"平均値: {np.nanmean(masked_shc):.6f}")
    print(f"中央値: {np.nanmedian(masked_shc):.6f}")
    
    # パーセンタイル値の表示
    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        print(f"{p}パーセンタイル: {np.nanpercentile(masked_shc, p):.6f}")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # vmaxを95パーセンタイルに設定
    vmax = np.nanpercentile(masked_shc, 95)
    
    # SHCの空間分布
    im = ax.imshow(shc,
                   cmap='viridis',
                   aspect='equal',
                   vmin=0,
                   vmax=vmax)
    
    # マスク部分を灰色で表示
    ax.imshow(np.where(mask == 0, 0.7, np.nan),
              cmap='gray_r',
              alpha=0.8,
              aspect='equal')
    
    # カラーバーの追加
    cbar = plt.colorbar(im)
    cbar.set_label('SHC')
    
    ax.set_title(f'{city_name}の平面曲率標準偏差(SHC)\n(窓サイズ: {window_size*5}m×{window_size*5}m)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"terrain_result/shc_distribution_{city_name}.png", dpi=300)
    
    return masked_shc

def compare_shc_boxplots(elevation_data, housing_areas, window_size):
    """
    全自治体のSHC分布を箱ひげ図で比較し、統計値を返す関数
    
    Parameters
    ----------
    elevation_data : dict
        都市名をキーとし、標高データを値とする辞書
    housing_areas : dict
        都市名をキーとし、住居系用途地域のGeoDataFrameを値とする辞書
    window_size : int
        移動窓のサイズ
        
    Returns
    -------
    dict
        都市名をキーとし、平均値、中央値、標準偏差を値とする辞書
    """
    shc_data = []
    city_labels = []
    stats_dict = {}
    
    for city, elevation in elevation_data.items():
        print(f"{city}のSHCを計算中...")
        masked_shc = visualize_shc(elevation, city, housing_areas[city], window_size)
        
        # 有効なデータのみを抽出
        valid_shc = masked_shc[~np.isnan(masked_shc)]
        
        # 統計値の計算と保存
        stats_dict[city] = {
            'mean': np.mean(valid_shc),
            'median': np.median(valid_shc),
            'std': np.std(valid_shc),
        }
        
        shc_data.append(valid_shc)
        city_labels.append(city)
    
    # 中央値で都市を並び替え
    medians = [(city, np.median(data)) for city, data in zip(city_labels, shc_data)]
    medians.sort(key=lambda x: x[1], reverse=True)
    city_order = [x[0] for x in medians]
    
    # データを並び替え
    shc_data = [shc_data[city_labels.index(city)] for city in city_order]
    
    # 箱ひげ図の作成
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bp = ax.boxplot(shc_data,
                    labels=city_order,
                    whis=[10, 90],
                    showfliers=False)
    
    ax.set_xticklabels(city_order, rotation=45)
    ax.set_ylabel('SHC')
    ax.set_title('各自治体の住居系用途地域における平面曲率標準偏差(SHC)分布')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 統計情報の表示
    stats_text = "箱ひげ図の説明:\n"
    stats_text += "箱の上端: 75パーセンタイル\n"
    stats_text += "箱の中線: 中央値\n"
    stats_text += "箱の下端: 25パーセンタイル\n"
    stats_text += "ヒゲの上端: 90パーセンタイル\n"
    stats_text += "ヒゲの下端: 10パーセンタイル"
    
    ax.text(1.02, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("terrain_result/shc_boxplot_comparison.png", dpi=300, bbox_inches='tight')
    
    return stats_dict

def create_terrain_statistics(cities, stats_dict_slope, stats_dict_shc):
    """
    傾斜度とSHCの統計値をデータフレームにまとめる関数
    
    Parameters
    ----------
    cities : list
        都市名のリスト
    stats_dict_slope : dict
        傾斜度の統計値を含む辞書
    stats_dict_shc : dict
        SHCの統計値を含む辞書
        
    Returns
    -------
    pandas.DataFrame
        統計値をまとめたデータフレーム
    """
    # データフレームを作成
    df_stats = pd.DataFrame(index=cities)
    
    # 傾斜度の統計値を追加
    df_stats['傾斜度_平均値'] = [stats_dict_slope[city]['mean'] for city in cities]
    df_stats['傾斜度_中央値'] = [stats_dict_slope[city]['median'] for city in cities]
    df_stats['傾斜度_標準偏差'] = [stats_dict_slope[city]['std'] for city in cities]
    
    # SHCの統計値を追加
    df_stats['SHC_平均値'] = [stats_dict_shc[city]['mean'] for city in cities]
    df_stats['SHC_中央値'] = [stats_dict_shc[city]['median'] for city in cities]
    df_stats['SHC_標準偏差'] = [stats_dict_shc[city]['std'] for city in cities]
    
    # 小数点以下3桁に丸める
    df_stats = df_stats.round(3)
    
    # Excelファイルとして出力
    df_stats.to_excel('terrain_result/terrain_statistics.xlsx', 
                      sheet_name='地形統計値')
    
    return df_stats

def visualize_slope_shc_relationship(df_stats, cities):
    """
    傾斜度とSHCの関係を散布図で可視化する関数
    
    Parameters
    ----------
    df_stats : pandas.DataFrame
        統計値をまとめたデータフレーム
    cities : list
        都市名のリスト
    """
    plt.figure(figsize=(10, 8))
    
    # 各都市のデータをプロット
    plt.scatter(df_stats['SHC_中央値'], df_stats['傾斜度_中央値'])
    
    # 都市名のラベルを付ける
    for i, city in enumerate(cities):
        plt.annotate(city, 
                    (df_stats['SHC_中央値'][i], df_stats['傾斜度_中央値'][i]),
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    # グラフの装飾
    plt.xlabel('デコボコ度合い(SHCの中央値)')
    plt.ylabel('急峻さ(傾斜度の中央値) [度]')
    plt.title('各自治体の傾斜度とSHCの関係')
    plt.grid(True, alpha=0.3)
    
    # グラフを保存
    plt.savefig('terrain_result/slope_shc_scatter.png', 
                dpi=300, 
                bbox_inches='tight')


if __name__ == "__main__":
    cities = ["尾道市", "横須賀市", "下関市", "呉市",
              "佐世保市", "小樽市", "神戸市", "長崎市",
              "熱海市", "函館市", "別府市", "北九州市"]
    dem_paths = {"尾道市": ONOMICHI_DEM, "横須賀市": YOKOSUKA_DEM, "下関市": SIMONOSEKI_DEM, "呉市": KURE_DEM,
                 "佐世保市": SASEBO_DEM, "小樽市": OTARU_DEM, "神戸市": KOBE_DEM, "長崎市": NAGASAKI_DEM,
                 "熱海市": ATAMI_DEM, "函館市": HAKODATE_DEM, "別府市": BEPPU_DEM, "北九州市": KITAKYUSHU_DEM}
    onomichi_h = gpd.read_file(ONOMICHI_H, encoding="latin1").to_crs(epsg=6671)
    yokosuka_h = gpd.read_file(YOKOSUKA_H, encoding="latin1").to_crs(epsg=6677)
    simonoseki_h = gpd.read_file(SIMONOSEKI_H, encoding="latin1").to_crs(epsg=6671)
    kamakura_h = gpd.read_file(KAMAKURA_H, encoding="latin1").to_crs(epsg=6677)
    kure_h = gpd.read_file(KURE_H, encoding="latin1").to_crs(epsg=6671)
    sasebo_h = gpd.read_file(SASEBO_H, encoding="latin1").to_crs(epsg=6669)
    otaru_h = gpd.read_file(OTARU_H, encoding="latin1").to_crs(epsg=6679)
    kobe_h = gpd.read_file(KOBE_H, encoding="latin1").to_crs(epsg=6673)
    ikoma_h = gpd.read_file(IKOMA_H, encoding="latin1").to_crs(epsg=6674)
    tazimi_h = gpd.read_file(TAZIMI_H, encoding="latin1").to_crs(epsg=6675)
    nagasaki_h = gpd.read_file(NAGASAKI_H, encoding="latin1").to_crs(epsg=6669)
    atami_h = gpd.read_file(ATAMI_H, encoding="latin1").to_crs(epsg=6676)
    hakodate_h = gpd.read_file(HAKODATE_H, encoding="latin1").to_crs(epsg=6679)
    beppu_h = gpd.read_file(BEPPU_H, encoding="latin1").to_crs(epsg=6670)
    kitakyushu_h = gpd.read_file(KITAKYUSHU_H, encoding="latin1").to_crs(epsg=6670)
    housing_areas = {"尾道市": onomichi_h, "横須賀市": yokosuka_h, "下関市": simonoseki_h, "呉市": kure_h,
                     "佐世保市": sasebo_h, "小樽市": otaru_h, "神戸市": kobe_h, "長崎市": nagasaki_h,
                     "熱海市": atami_h, "函館市": hakodate_h, "別府市": beppu_h, "北九州市": kitakyushu_h}

    # 傾斜度データを格納する辞書
    elevation_data = {}
    all_slope_data = {}
    all_curvature_data = {}
    curvature_window_size = 9
    for city in cities:
        print(f'{city}--------------------------------')
        with rasterio.open(dem_paths[city]) as src:
            if src.crs != housing_areas[city].crs:
                housing_areas[city] = housing_areas[city].to_crs(src.crs)
            elevation = src.read(1)  # 最初のバンドを取得
            elevation = np.where(elevation == -9999, np.nan, elevation)
            elevation_data[city] = elevation
        slope_data = visualize_slope(elevation, city, housing_areas[city], store_data=True)
        """visualize_height(elevation, city, housing_areas[city])
        slope_data = visualize_slope(elevation, city, housing_areas[city], store_data=True)
        all_slope_data[city] = slope_data
        print(f'{city}の縦断勾配と平面曲率を計算中...')
        visualize_profile_and_plan_curvature(elevation, city, housing_areas[city], window_size=curvature_window_size)"""
    
    # 全自治体の傾斜度分布を比較
    # compare_slope_distributions(all_slope_data)
    stats_dict_slope = compare_slope_boxplots(elevation_data, housing_areas)

    shc_window_size = 21  # 45m四方の移動窓
    stats_dict_shc = compare_shc_boxplots(elevation_data, housing_areas, shc_window_size)
    # 傾斜度とSHCの統計値をdfにまとめる
    df_stats = create_terrain_statistics(cities, stats_dict_slope, stats_dict_shc)
    visualize_slope_shc_relationship(df_stats, cities)
    

