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
from shapely.geometry import shape, box
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    "北陸": ["新潟県", "富山県", "石川県", "福井県",],
    "中部": ["岐阜県", "静岡県", "愛知県", "三重県"],
    "近畿": ["滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
    "中国": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
    "四国": ["徳島県", "香川県", "愛媛県", "高知県"],
    "九州": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]
}
kyogikai_cities = ["尾道市", "横須賀市", "下関市", "呉市",
                   "佐世保市", "小樽市", "神戸市", "長崎市",
                   "熱海市", "函館市", "別府市", "北九州市"]


def calc_and_visualize_height(city_name, bbox_elevation, housing_mask, visualize=True):
    """
    標高の(必要ないが、他の処理と入出力を合わせるため)計算と可視化を行い、
    住居系用途地域の標高データを返す。

    Args:
        city_name (str): 自治体名(例: 11_戸田市(埼玉県))
        bbox_elevation (numpy.ndarray): 住居系用途地域を囲むBBoxの標高データ
        housing_mask (numpy.ndarray): 住居系用途地域のマスク
    Returns:
        housing_area_height (numpy.ndarray): 住居系用途地域の標高データ
    """
    housing_area_height = np.where(housing_mask == 1, bbox_elevation, np.nan)
    # 標高の可視化
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(bbox_elevation, 
                    cmap='terrain',
                    aspect='equal',
                    vmin=0,
                    vmax=300)
        # マスク部分を灰色で表示（住居系用途地域外を灰色に）
        ax.imshow(np.where(housing_mask == 0, 0.7, np.nan),  # mask==0の部分(住居系用途地域外)を灰色(0.7)に、それ以外を透明に
                cmap='gray_r',
                alpha=0.8,
                aspect='equal')

        cbar = plt.colorbar(im)
        cbar.set_label('標高 (m)')
        plt.title(f'{city_name}の標高分布')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"result/elevation/elevation_{city_name}.png", dpi=300)
        plt.close()

    return housing_area_height


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
    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]]) / (6.0 * pixel_size_x)
    
    kernel_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]]) / (6.0 * pixel_size_y)
    
    # 勾配計算
    dx = convolve(elevation, kernel_x, mode='mirror')
    dy = convolve(elevation, kernel_y, mode='mirror')
    
    # 傾斜度を度単位で計算
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    return slope


def calc_and_visualize_slope(city_name, bbox_elevation, housing_mask, visualize=True):
    """
    傾斜度の計算と可視化を行い、住居系用途地域の傾斜度データを返す。

    Args:
        city_name (str): 自治体名(例: 11_戸田市(埼玉県))
        src (rasterio.DatasetReader): DEMデータソース
        bbox_elevation (numpy.ndarray): 住居系用途地域を囲むBBoxの標高データ
        housing_mask (numpy.ndarray): 住居系用途地域のマスク

    Returns:
        housing_area_slope (numpy.ndarray): 住居系用途地域の傾斜度データ
    """
    # 傾斜度の計算
    slope = calc_slope(bbox_elevation)
    
    # 住居系用途地域内のデータのみを抽出
    housing_area_slope = np.where(housing_mask == 1, slope, np.nan)

    # 傾斜度の可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [3.5, 1]})
    plt.subplots_adjust(left=0, right=0.85)
    
    # 傾斜度の空間分布
    if visualize:
        im = ax1.imshow(slope, 
                        cmap='autumn_r', 
                        aspect='equal', 
                        vmin=0, 
                        vmax=45)
    
        # マスク部分を灰色で表示（住居系用途地域外を灰色に）
        ax1.imshow(np.where(housing_mask == 0, 0.7, np.nan),
                cmap='gray_r', 
                alpha=0.8,
                aspect='equal')
    
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('傾斜度 (度)')
        ax1.set_title(f'{city_name}の傾斜度分布')
        ax1.axis('off')

    # 傾斜度の区分ごとの割合を計算
    bins = [0, 5, 10, 15, 20, 25, np.inf]
    labels = ['0-5度', '5-10度', '10-15度', '15-20度', '20-25度', '25度以上']
    hist, _ = np.histogram(housing_area_slope[~np.isnan(housing_area_slope)], bins=bins)
    percentages = hist / len(housing_area_slope[~np.isnan(housing_area_slope)]) * 100
    steep_ratio = sum(percentages[1:])  # 5度以上の割合の合計

    # 積み上げ棒グラフの作成
    if visualize:
        colors = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', 'red']
        bottom = 0
        for i, (percentage, color) in enumerate(zip(percentages, colors)):
            ax2.bar(0, percentage, bottom=bottom, color=color, label=labels[i])
            if percentage >= 3:
                ax2.text(0, bottom + percentage/2, f'{percentage:.1f}%',
                        ha='center', va='center')
            bottom += percentage
        
        ax2.set_ylabel('割合 (%)')
        ax2.set_title('傾斜度の区分別割合\n(住居系用途地域内)')
        ax2.set_xticks([])
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 統計情報の表示
    if visualize:
        stats_text = ("傾斜度の統計情報\n(住居系用途地域内)\n"
                    f"最小値: {np.nanmin(housing_area_slope):.1f}[度]\n"
                    f"最大値: {np.nanmax(housing_area_slope):.1f}[度]\n"
                    f"平均値: {np.nanmean(housing_area_slope):.1f}[度]\n"
                    f"中央値: {np.nanmedian(housing_area_slope):.1f}[度]")
        ax2.text(1.05, 0.6, stats_text, transform=ax2.transAxes,
                ha='left', va='top', bbox=dict(facecolor='none', edgecolor='lightgray', pad=4))
    
        plt.savefig(f"result/slope/slope_{city_name}.png", dpi=300)
        plt.close()

    return housing_area_slope, steep_ratio


def visualize_top_steep_cities_comparison(top_steep_cities):
    """
    与えられたDataFrameの各都市について、傾斜度区分割合を比較するグラフを作成する。
    
    Args:
        df_stats (pandas.DataFrame): 統計データを含むDataFrame。以下のカラムが必要:
            - 市区町村名
            - 都道府県名
            - 都道府県コード
    """
    slope_data = {}
    steep_ratio_15deg = {}  # 15度以上の割合を保存する辞書
    
    # 各都市について傾斜度区分の計算を行う
    for _, row in top_steep_cities.iterrows():
        city_name = row['市区町村名']
        pref_name = row['都道府県名']
        pref_code = row['都道府県コード']
        
        # DEMファイルの確認
        area_name = None
        for area, prefs in area_dict.items():
            if pref_name in prefs:
                area_name = area
                break
        
        dem_path = None
        for dem_name in os.listdir(DEM_DIR):
            if area_name in dem_name and dem_name.endswith('.tif'):
                dem_path = os.path.join(DEM_DIR, dem_name)
                break
        
        # 住居系用途地域のシェープファイルを読み込む
        housing_area_path = os.path.join(
            HOUSING_AREA_DIR,
            f"A29-19_{pref_code}",
            f"housing_{city_name}.shp"
        )
        
        if dem_path and os.path.exists(housing_area_path):
            with rasterio.open(dem_path) as src:
                housing_area_gdf = gpd.read_file(housing_area_path)
                bounds = housing_area_gdf.total_bounds
                bbox = box(*bounds)
                shapes = [bbox]
                bbox_elevation, bbox_transform = mask(
                    src,
                    shapes=shapes,
                    crop=True,
                    nodata=np.nan
                )
                bbox_elevation = bbox_elevation[0]
                housing_mask = rasterio.features.rasterize(
                    [(geom, 1) for geom in housing_area_gdf.geometry],
                    out_shape=bbox_elevation.shape,
                    transform=bbox_transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                # 傾斜度の計算
                slope = calc_slope(bbox_elevation)
                housing_area_slope = np.where(housing_mask == 1, slope, np.nan)
                
                # 傾斜度の区分ごとの割合を計算
                bins = [0, 5, 10, 15, 20, 25, np.inf]
                hist, _ = np.histogram(
                    housing_area_slope[~np.isnan(housing_area_slope)],
                    bins=bins
                )
                percentages = hist / len(housing_area_slope[~np.isnan(housing_area_slope)]) * 100
                slope_data[f"{city_name}"] = percentages
                # 15度以上の割合を計算（15度以上の3区分の合計）
                steep_ratio_15deg[city_name] = sum(percentages[3:])
    
    # 2つのバージョンのグラフを作成
    for sort_by_steep in [False, True]:
        fig, ax = plt.subplots(figsize=(18, 8))
        
        labels = ['0-5度', '5-10度', '10-15度', '15-20度', '20-25度', '25度以上']
        colors = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', 'red']
        
        # 15度以上の割合で並べ替え（オプション）
        if sort_by_steep:
            sorted_cities = dict(sorted(steep_ratio_15deg.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
            sorted_slope_data = {k: slope_data[k] for k in sorted_cities.keys()}
        else:
            sorted_slope_data = slope_data
        
        x = np.arange(len(sorted_slope_data))
        bottom = np.zeros(len(sorted_slope_data))
        
        # 積み上げ棒グラフの作成
        for i, (label, color) in enumerate(zip(labels, colors)):
            values = [data[i] for data in sorted_slope_data.values()]
            ax.bar(x, values, bottom=bottom, label=label, color=color)
            
            # パーセンテージの表示（5%以上の場合のみ）
            for j, v in enumerate(values):
                if v >= 5:
                    ax.text(j, bottom[j] + v/2, f'{v:.1f}%',
                           ha='center', va='center')
            bottom += values
        
        # グラフの装飾
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_slope_data.keys(), rotation=45, ha='right')
        ax.set_ylabel('割合 (%)')
        title = '各都市の傾斜度区分割合の比較'
        if sort_by_steep:
            title += '\n（15度以上の割合で降順に並べ替え）'
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = 'result/steep_ratio_comparison'
        if sort_by_steep:
            filename += '_sorted_by_15deg'
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()



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
    pixel_size = 10
    
    # 1次微分と2次微分の計算
    kernel_x = np.array([[-1, 0, 1]]) / (2 * pixel_size)
    kernel_y = kernel_x.T
    
    # 1次微分
    dx = convolve(elevation, kernel_x, mode='mirror')
    dy = convolve(elevation, kernel_y, mode='mirror')
    
    # 2次微分
    kernel_xx = np.array([[1, -2, 1]]) / (pixel_size**2)
    kernel_yy = kernel_xx.T
    kernel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / (4 * pixel_size**2)
    
    dxx = convolve(elevation, kernel_xx, mode='mirror')
    dyy = convolve(elevation, kernel_yy, mode='mirror')
    dxy = convolve(elevation, kernel_xy, mode='mirror')
    
    # 勾配の大きさ
    p = dx**2 + dy**2
    # ゼロ除算を防ぐためのマスク作成
    mask = p > 1e-10  # 十分小さい閾値
    q = np.sqrt(1 + p[mask])

    # 初期値を0で初期化（平坦な領域は曲率0）
    profile_curvature = np.zeros_like(dx, dtype=np.float64)
    plan_curvature = np.zeros_like(dx, dtype=np.float64)
    
    # 縦断勾配（profile curvature）
    profile_curvature[mask] = ((dxx[mask] * dx[mask]**2 + 
                               2 * dxy[mask] * dx[mask] * dy[mask] + 
                               dyy[mask] * dy[mask]**2) /
                              (p[mask] * q**3))
    
    # 平面曲率（plan curvature）
    plan_curvature[mask] = ((dxx[mask] * dy[mask]**2 - 
                            2 * dxy[mask] * dx[mask] * dy[mask] + 
                            dyy[mask] * dx[mask]**2) /
                           (p[mask]**1.5))
    
    return profile_curvature, plan_curvature


def calc_shc(plan_curvature, window_size):
    """
    平面曲率の標準偏差（SHC）を計算する関数（高速化版）
    """
    import numpy as np
    from scipy.ndimage import uniform_filter
    
    # 円形のカーネルを作成
    y, x = np.ogrid[-window_size:window_size+1, -window_size:window_size+1]
    circular_mask = x*x + y*y <= window_size*window_size
    kernel = circular_mask.astype(float)
    kernel /= kernel.sum()  # 正規化
    
    # uniform_filterを使用して移動窓の計算を高速化
    mean = uniform_filter(np.nan_to_num(plan_curvature, 0), 
                         size=2*window_size+1,
                         mode='mirror')
    mean_sq = uniform_filter(np.nan_to_num(plan_curvature, 0)**2,
                           size=2*window_size+1,
                           mode='mirror')
    
    # 分散と標準偏差の計算
    variance = mean_sq - mean**2
    variance = np.clip(variance, 0, None)  # 数値誤差対策
    
    # オリジナルのnanを維持
    shc = np.where(np.isnan(plan_curvature), np.nan, np.sqrt(variance))
    
    return shc


def calc_and_visualize_shc(city_name, bbox_elevation, housing_mask, visualize=True):
    """
    SHCの計算と可視化を行い、住居系用途地域のSHCデータを返す。

    Args:
        city_name (str): 自治体名(例: 11_戸田市(埼玉県))
        bbox_elevation (numpy.ndarray): 住居系用途地域を囲むBBoxの標高データ
        housing_mask (numpy.ndarray): 住居系用途地域のマスク

    Returns:
        housing_area_shc (numpy.ndarray): 住居系用途地域のSHCデータ
    """
    window_size = 100 # 移動窓サイズ
    
    # 平面曲率を計算
    print(f'{city_name}の平面曲率を計算中...')
    _, plan_curv = calc_profile_and_plan_curvature(bbox_elevation, window_size)
    
    # SHCを計算
    print(f'{city_name}の平面曲率標準偏差(SHC)を計算中...')
    bbox_shc = calc_shc(plan_curv, window_size)
    
    # 住居系用途地域内のデータのみを使用して統計を計算
    print(f'{city_name}のSHCをマスク中...')
    housing_area_shc = np.where(housing_mask == 1, bbox_shc, np.nan)
    
    # 可視化
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        # vmaxを95パーセンタイルに設定
        # vmax = np.nanpercentile(housing_area_shc, 95)
        vmax = 0.08

        # SHCの空間分布
        im = ax.imshow(bbox_shc,
                       cmap='viridis',
                       aspect='equal',
                       vmin=0,
                       vmax=vmax)
        
        # マスク部分を灰色で表示
        ax.imshow(np.where(housing_mask == 0, 0.7, np.nan),
                  cmap='gray_r',
                  alpha=0.8,
                  aspect='equal')
        
        # カラーバーの追加
        cbar = plt.colorbar(im)
        cbar.set_label('SHC')
        
        ax.set_title(f'{city_name}の平面曲率標準偏差(SHC)\n(窓サイズ: {window_size*10}m×{window_size*10}m)')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"result/shc/shc_{city_name}.png", dpi=300)
        plt.close()
    
    return housing_area_shc


def calc_shc_in_steep_area(city_name, bbox_elevation, housing_mask):
    """
    傾斜度5度以上の範囲でのSHCを計算・可視化する関数
    
    Args:
        city_name (str): 自治体名
        bbox_elevation (numpy.ndarray): 標高データ
        housing_mask (numpy.ndarray): 住居系用途地域のマスク
    
    Returns:
        steep_area_shc (numpy.ndarray): 傾斜5度以上の範囲のSHCデータ
    """
    # 傾斜度の計算
    slope = calc_slope(bbox_elevation)
    
    # 傾斜5度以上かつ住居系用途地域内の領域のマスク作成
    steep_mask = (slope >= 5) & (housing_mask == 1)
    
    # 平面曲率とSHCの計算
    window_size = 10
    _, plan_curv = calc_profile_and_plan_curvature(bbox_elevation, window_size)
    bbox_shc = calc_shc(plan_curv, window_size)
    
    # 傾斜5度以上の範囲のSHCを抽出
    steep_area_shc = np.where(steep_mask, bbox_shc, np.nan)
    
    return steep_area_shc


def calc_terrain_of_a_city(city_name, housing_area_path, elevation, src):
    # 住居系用途地域のSHPファイルを読み込む
    housing_area_gdf = gpd.read_file(housing_area_path)

    try:
        bounds = housing_area_gdf.total_bounds
        bbox = box(*bounds)
        shapes = [bbox]
        bbox_elevation, bbox_transform = mask(
            src,
            shapes=shapes,
            crop=True,
            nodata=np.nan
        )
        bbox_elevation = bbox_elevation[0]
        housing_mask = rasterio.features.rasterize(
            [(geom, 1) for geom in housing_area_gdf.geometry],
            out_shape=bbox_elevation.shape,
            transform=bbox_transform,
            fill=0,
            dtype=np.uint8
        )
        housing_area_height = calc_and_visualize_height(city_name, bbox_elevation, housing_mask, visualize=False)
        housing_area_slope, steep_ratio = calc_and_visualize_slope(city_name, bbox_elevation, housing_mask, visualize=False)
        housing_area_shc = calc_and_visualize_shc(city_name, bbox_elevation, housing_mask, visualize=False)
        steep_area_shc = calc_shc_in_steep_area(city_name, bbox_elevation, housing_mask)

        steep_mask = (housing_area_slope >= 5) & (~np.isnan(housing_area_slope))
        steep_area_slope = np.where(steep_mask, housing_area_slope, np.nan)


        city_terrain_stats = {
            '都道府県コード': city_name.split('_')[0],
            '都道府県名': city_name.split('_')[1].split('(')[1].split(')')[0],
            '市区町村名': city_name.split('_')[1].split('(')[0],
            '標高_平均値': np.nanmean(housing_area_height),
            '標高_中央値': np.nanmedian(housing_area_height),
            '標高_標準偏差': np.nanstd(housing_area_height),
            '傾斜度_平均値': np.nanmean(housing_area_slope),
            '傾斜度_中央値': np.nanmedian(housing_area_slope),
            '傾斜度_標準偏差': np.nanstd(housing_area_slope),
            'SHC_平均値': np.nanmean(housing_area_shc),
            'SHC_中央値': np.nanmedian(housing_area_shc),
            'SHC_標準偏差': np.nanstd(housing_area_shc),
            '住居系用途地域に占める斜面市街地の割合': steep_ratio,
            '斜面市街地の傾斜度_平均値': np.nanmean(steep_area_slope),
            '斜面市街地の傾斜度_中央値': np.nanmedian(steep_area_slope),
            '斜面市街地の傾斜度_標準偏差': np.nanstd(steep_area_slope),
            '斜面市街地のSHC_平均値': np.nanmean(steep_area_shc),
            '斜面市街地のSHC_中央値': np.nanmedian(steep_area_shc),
            '斜面市街地のSHC_標準偏差': np.nanstd(steep_area_shc),
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
        print(f'{area}地方のデータ作成中...')
        # 地方DEMのパスを取得
        dem_list = os.listdir(DEM_DIR)
        for dem_name in dem_list:
            if area in dem_name and dem_name.endswith('.tif'):
                area_dem_path = os.path.join(DEM_DIR, dem_name)
                print(f'DEM Path: {area_dem_path}')
                print("DEM読み込み中...")
                with rasterio.open(area_dem_path) as src:
                    elevation = src.read(1)  # 最初のバンドを取得
                    elevation = np.where(elevation == -9999, np.nan, elevation)
                    print("DEM読み込み完了")

                    # 各県内の住居系用途地域のパスを取得
                    for pref in area_dict[area]:
                        pref_dir_name = f'A29-19_{pref_code_dict[pref]}'
                        housing_area_dir = os.path.join(HOUSING_AREA_DIR, pref_dir_name)
                        housing_area_list = os.listdir(housing_area_dir)
                        for housing_area_name in housing_area_list:
                            # 各自治体の住居系用途地域データ(SHP)に対して傾斜度・SHCを計算
                            if housing_area_name.endswith('.shp'):
                                city_name = pref_code_dict[pref] + '_' + housing_area_name.split('_')[1].split('.')[0] + f'({pref})'
                                housing_area_path = os.path.join(housing_area_dir, housing_area_name)
                                print(f'\n-----{city_name}-----\nshapefile Path: {housing_area_path}')
                                print(f'{city_name}の傾斜度計算中...')
                                city_terrain_stats = calc_terrain_of_a_city(city_name, housing_area_path, elevation, src)
                                df_stats = pd.concat([df_stats, pd.DataFrame([city_terrain_stats])], ignore_index=True)
                                counter += 1
                    print(df_stats)
    print(f'{counter}件のデータを作成しました。')

    return df_stats


def visualize_slope_shc_relationship(df_stats, kyogikai_cities, top_steep_cities, only_steep_area,mode, mean_or_med, thr_rank: int = 30):
    """
    傾斜度とSHCの関係を散布図で可視化する関数
    
    Parameters
    ----------
    df_stats : pandas.DataFrame
        統計値をまとめたデータフレーム
    kyogikai_cities : list
        協議会都市名のリスト
    """
    plt.figure(figsize=(10, 8))

    if only_steep_area:
        df_stats = df_stats[df_stats['住居系用途地域に占める斜面市街地の割合'] >= 5]
        makura = '斜面市街地の'
    else:
        makura = ''
    
    # 傾斜度とSHCの上位thr_rank自治体を取得
    top_slope = df_stats.nlargest(thr_rank, f'{makura}傾斜度_平均値')['市区町村名'].tolist()
    top_shc = df_stats.nlargest(thr_rank, f'{makura}SHC_平均値')['市区町村名'].tolist()
    
    # 凡例用のフラグを追加
    legend_added = {
        'both': False,
        'slope': False,
        'shc': False,
        'other': False,
        'top_steep': False,
        'kyogikai': False
    }

    # まず、その他の都市をプロット
    for i, row in df_stats.iterrows():
        city = row['市区町村名']
        x, y = row[f'{makura}SHC_{mean_or_med}'], row[f'{makura}傾斜度_{mean_or_med}']

        if city in top_slope and city in top_shc:
            # 両方のTop20に入っている都市は紫の星でプロット
            plt.scatter(x, y, color='purple', marker='*', s=100, zorder=3,
                    label=f'傾斜度・SHC両方で上位{thr_rank}位以内' if not legend_added['both'] else '', alpha=0.5)
            legend_added['both'] = True
            """if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(city, (x, y), xytext=(5, 5),
                            textcoords='offset points', color='purple', zorder=3, fontsize=6)"""
        elif city in top_slope:
            plt.scatter(x, y, color='green', marker='s', s=50, zorder=2,
                    label=f'傾斜度上位{thr_rank}位以内' if not legend_added['slope'] else '', alpha=0.5)
            legend_added['slope'] = True
            """if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(city, (x, y), xytext=(5, 5),
                            textcoords='offset points', color='green', zorder=2, fontsize=6)"""
        elif city in top_shc:
            plt.scatter(x, y, color='blue', marker='^', s=50, zorder=2,
                    label=f'SHC上位{thr_rank}位以内' if not legend_added['shc'] else '', alpha=0.5)
            legend_added['shc'] = True
            """if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(city, (x, y), xytext=(5, 5),
                            textcoords='offset points', color='blue', zorder=2, fontsize=6)"""
        else:
            # それ以外の都市は灰色でプロット
            plt.scatter(x, y, color='gray', alpha=0.5, s=30, zorder=1,
                    label='その他の都市' if not legend_added['other'] else '')
            legend_added['other'] = True

    # 次に、協議会都市または主要な斜面都市をプロット（最前面に表示）
    for i, row in df_stats.iterrows():
        city = row['市区町村名']
        x, y = row[f'{makura}SHC_{mean_or_med}'], row[f'{makura}傾斜度_{mean_or_med}']

        if mode == 'red' and city in kyogikai_cities:
            plt.scatter(x, y, color='red', marker='o', s=20, zorder=10,
                    label='全国斜面都市連絡協議会に加盟している自治体' if not legend_added['kyogikai'] else '')
            legend_added['kyogikai'] = True
            plt.annotate(city, (x, y), xytext=(5, 5), 
                        textcoords='offset points',
                        color='black', fontweight='bold', zorder=10, fontsize=8)
        elif mode == 'orange' and city in top_steep_cities:
            plt.scatter(x, y, color='orange', marker='o', s=20, zorder=10,
                    label='主要な斜面都市' if not legend_added['top_steep'] else '')
            legend_added['top_steep'] = True
            plt.annotate(city, (x, y), xytext=(5, 5), 
                        textcoords='offset points',
                        color='black', fontweight='bold', zorder=10, fontsize=8)
    
    # グラフの装飾
    plt.xlabel(f'起伏の大きさ(SHCの{mean_or_med})')
    plt.ylabel(f'急峻さ(傾斜度の{mean_or_med}) [度]')
    if only_steep_area:
        plt.title(f'各自治体の斜面市街地における傾斜度とSHCの関係')
    else:
        plt.title(f'各自治体における傾斜度とSHCの関係')
    plt.grid(True, alpha=0.3)
    
    # 凡例を表示（重複を除去）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='lower left')
    
    # グラフを保存
    if mode == 'red':
        plt.savefig(f'result/slope_shc_scatter_kyogikai_{mean_or_med}.png', 
                    dpi=300, bbox_inches='tight')
    elif mode == 'orange':
        plt.savefig(f'result/slope_shc_scatter_top_steep_{mean_or_med}.png', 
                    dpi=300, bbox_inches='tight')
    plt.close()


def visualize_slope_shc_relationship_with_top_cities(df_stats, top_steep_cities, only_steep_area, mean_or_med):
    """
    傾斜度とSHCの関係を散布図で可視化する関数
    
    Parameters
    ----------
    df_stats : pandas.DataFrame
        統計値をまとめたデータフレーム
    top_steep_cities : list
        主要な斜面都市名のリスト
    """
    plt.figure(figsize=(10, 8))

    if only_steep_area:
        df_stats = df_stats[df_stats['住居系用途地域に占める斜面市街地の割合'] >= 5]
        makura = '斜面市街地の'
    else:
        makura = ''

    # まず、その他の都市をプロット
    for i, row in df_stats.iterrows():
        city = row['市区町村名']
        x, y = row[f'{makura}SHC_{mean_or_med}'], row[f'{makura}傾斜度_{mean_or_med}']

    # 次に、主要な斜面都市をプロット（最前面に表示）
    for i, row in df_stats.iterrows():
        city = row['市区町村名']
        x, y = row[f'{makura}SHC_{mean_or_med}'], row[f'{makura}傾斜度_{mean_or_med}']

        if city in top_steep_cities:
            plt.scatter(x, y, color='orange', marker='o', s=30, zorder=10)
            plt.annotate(city, (x, y), xytext=(5, 5), 
                        textcoords='offset points',fontweight='bold', zorder=10, fontsize=10)
    
    # グラフの装飾
    plt.xlabel(f'起伏の大きさ(SHCの{mean_or_med})')
    plt.ylabel(f'急峻さ(傾斜度の{mean_or_med}) [度]')
    if only_steep_area:
        plt.title(f'主要な斜面都市の斜面市街地における傾斜度とSHCの関係')
    else:
        plt.title(f'主要な斜面都市における傾斜度とSHCの関係')
    plt.grid(True, alpha=0.3)
    
    # 凡例を表示（重複を除去）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='lower left')
    
    # グラフを保存
    plt.savefig(f'result/slope_shc_scatter_top_steep_with_top_cities_{mean_or_med}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def visualize_top_cities_on_map(df_stats, mean_or_med, thr_rank: int = 30):
    # 傾斜度とSHCの上位都市を抽出
    slope_top = df_stats.nlargest(thr_rank, f'傾斜度_{mean_or_med}')['市区町村名'].tolist()
    shc_top = df_stats.nlargest(thr_rank, f'SHC_{mean_or_med}')['市区町村名'].tolist()

    # 両方のリストに含まれる都市を抽出
    both_top = list(set(slope_top) & set(shc_top))

    # 地図の作成
    import contextily as ctx
    fig, ax = plt.subplots(figsize=(15, 10))

    # 日本の大まかな範囲を設定
    japan_bounds = [
        14200000,  # 左端（西）
        16400000,  # 右端（東）
        3500000,   # 下端（南）
        5800000    # 上端（北）
    ]
    kanto_bounds = [
        15480000,  # 左端（西）
        15650000,  # 右端（東）
        4165000,   # 下端（南）
        4308000    # 上端（北）
    ]
    # 地図の表示範囲を設定
    ax.set_xlim(japan_bounds[0], japan_bounds[1])
    ax.set_ylim(japan_bounds[2], japan_bounds[3])
    #ax.set_xlim(kanto_bounds[0], kanto_bounds[1])
    #ax.set_ylim(kanto_bounds[2], kanto_bounds[3])
    
    # 各都市の住居系用途地域を描画
    for _, row in df_stats.iterrows():
        city_name = row['市区町村名']
        pref_name = row['都道府県名']
        pref_code = pref_code_dict[pref_name]
        
        # 住居系用途地域のシェープファイルを読み込む
        shp_path = os.path.join(HOUSING_AREA_DIR, 
                               f"A29-19_{pref_code}", 
                               f"housing_{city_name}.shp")
        if os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path)
            # CRSをWeb Mercatorに変換
            gdf = gdf.to_crs(epsg=3857)
            
            # 都市の中心座標を取得
            center_point = gdf.geometry.unary_union.centroid
            
            # マーカーの設定 関東版は500, 400, 300
            if city_name in both_top:
                marker = '*'  # 星
                color = 'purple'
                size = 50
            elif city_name in slope_top:
                marker = 's'  # 三角
                color = 'green' 
                size = 40
            elif city_name in shc_top:
                marker = '^'  # 四角
                color = 'blue'
                size = 40
            else:
                continue
            
            # マーカーを描画
            ax.scatter(center_point.x, center_point.y, 
                      c=color, marker=marker, s=size, 
                      edgecolor='black', linewidth=1)


    # OSMを背景地図として追加
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)  # alphaパラメータを追加して透明度を設定

    # 縦横比を固定
    ax.set_aspect('equal')

    # 凡例を追加
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='purple',
               markersize=15, label=f'傾斜度・SHC共に上位{thr_rank}位以内'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
               markersize=10, label=f'傾斜度上位{thr_rank}位以内'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=10, label=f'SHC上位{thr_rank}位以内')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.title(f'住居系用途地域の傾斜度・SHC上位{thr_rank}都市')
    ax.set_axis_off()
    plt.savefig(f'result/top{thr_rank}_cities_map_japan_{mean_or_med}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def visualize_slope_area_ratio_histogram(df_stats):
    print('斜面市街地の割合の度数分布表を作成中...')
    
    # 2%間隔でビンを作成
    bins = np.arange(0, df_stats['住居系用途地域に占める斜面市街地の割合'].max() + 2, 2)
    
    # 度数分布表の作成
    hist, bin_edges = np.histogram(df_stats['住居系用途地域に占める斜面市街地の割合'], bins=bins)
    
    # 度数分布表の表示
    print('\n斜面市街地の割合の度数分布表:')
    print('範囲(%)    度数')
    print('-' * 20)
    for i in range(len(hist)):
        print(f'{bin_edges[i]:>5.1f}-{bin_edges[i+1]:>5.1f}: {hist[i]:>4d}')
    
    # ヒストグラムの作成
    plt.figure(figsize=(10, 6))
    plt.hist(df_stats['住居系用途地域に占める斜面市街地の割合'], 
             bins=bins, 
             edgecolor='black',
             color='#0b5394',
             alpha=0.8
             )
    plt.xlabel('住居系用途地域に占める斜面市街地の割合 (%)')
    plt.ylabel('自治体数')
    plt.title('斜面市街地の割合の分布')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)  # x軸の左端を0に設定
    plt.savefig('result/slope_area_ratio_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_slope_correlation(df_stats):
    """
    住居系用途地域の斜面市街地割合と傾斜度の相関関係を分析する関数
    
    Parameters:
    -----------
    df_stats : pandas.DataFrame
        分析対象のデータフレーム
    
    Returns:
    --------
    correlation : float
        相関係数
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    
    # 相関係数の計算
    correlation = df_stats['住居系用途地域に占める斜面市街地の割合'].corr(df_stats['傾斜度_平均値'])
    
    # 散布図の作成
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_stats, 
                    x='住居系用途地域に占める斜面市街地の割合', 
                    y='傾斜度_平均値')
    
    # タイトルと軸ラベルの設定
    plt.title('住居系用途地域の斜面市街地割合と傾斜度の相関関係')
    plt.xlabel('斜面市街地割合 [％]')
    plt.ylabel('平均傾斜度 [度]')
    
    # 相関係数を図に追加
    plt.text(0.05, 0.95, f'相関係数: {correlation:.3f}', 
             transform=plt.gca().transAxes)
    
    # グラフを保存
    plt.savefig('result/slope_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation


def analyze_terrain_indicators(df_stats):
    """
    地形指標間の相関分析と主成分分析を行う関数
    
    Args:
        df_stats (pandas.DataFrame): 地形統計データ
    """
    # 斜面市街地割合で降順にソートし、上位28位までを抽出
    df_stats = df_stats.sort_values('住居系用途地域に占める斜面市街地の割合', 
                                        ascending=False).head(28)
    print("\n=== 分析対象都市（斜面市街地割合上位28位） ===")
    for i, (_, row) in enumerate(df_stats.iterrows(), 1):
        print(f"{i}位: {row['市区町村名']} ({row['住居系用途地域に占める斜面市街地の割合']:.1f}%)")

    # 分析対象の変数を選択
    target_cols = [
        '標高_平均値',
        '傾斜度_平均値',
        'SHC_平均値',
        '住居系用途地域に占める斜面市街地の割合',
        '斜面市街地の傾斜度_平均値',
        '斜面市街地のSHC_平均値'
    ]
    
    # 相関分析
    correlation_matrix = df_stats[target_cols].corr()
    
    # 相関行列のヒートマップを作成
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,  # 相関係数を表示
                cmap='coolwarm',  # カラーマップ
                center=0,  # 0を中心に
                fmt='.2f',  # 小数点2桁まで表示
                square=True)  # マスを正方形に
    plt.title('地形指標間の相関行列')
    plt.tight_layout()
    plt.savefig('result/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 主成分分析
    # データの標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_stats[target_cols])
    
    # PCAの実行
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # 寄与率を計算
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # スクリープロットの作成
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
            cumulative_variance_ratio, 
            'bo-')
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio, 
            'ro-')
    plt.xlabel('主成分の数')
    plt.ylabel('寄与率')
    plt.title('スクリープロット')
    plt.grid(True)
    plt.legend(['累積寄与率', '寄与率'])
    plt.savefig('result/scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 第1・第2主成分の固有ベクトルを可視化
    pc_cols = ['PC{}'.format(i+1) for i in range(len(target_cols))]
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pc_cols,
        index=target_cols
    )
    
    plt.figure(figsize=(10, 10))
    plt.scatter(loadings['PC1'], loadings['PC2'])
    for i, variable in enumerate(target_cols):
        plt.annotate(variable, (loadings['PC1'][i], loadings['PC2'][i]))
    plt.xlabel('第1主成分')
    plt.ylabel('第2主成分')
    plt.title('主成分負荷量')
    plt.grid(True)
    # 軸の範囲を-1から1に設定
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # 原点に線を追加
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.savefig('result/pca_loadings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 主成分得点の散布図
    plt.figure(figsize=(10, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    # 都市名をプロット
    for i, city in enumerate(df_stats['市区町村名']):
        plt.annotate(city, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    plt.xlabel('第1主成分')
    plt.ylabel('第2主成分')
    plt.title('主成分得点散布図')
    plt.grid(True)
    plt.savefig('result/pca_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果をDataFrameにまとめて返す
    pca_summary = pd.DataFrame({
        '寄与率': explained_variance_ratio,
        '累積寄与率': cumulative_variance_ratio
    }, index=pc_cols)
    
    return correlation_matrix, pca_summary, loadings


if __name__ == "__main__":
    # 都道府県コードと県名の辞書を作成
    target_list = pd.read_excel(TARGET_LIST_EXCEL)
    # 都道府県名と都道府県コードの辞書を作成
    pref_code_dict = {}
    for _, row in target_list.iterrows():
        pref_name = row['都道府県名'].split('_')[1]  # "(都道府県コード)_都道府県名" から都道府県名を抽出
        pref_code = row['都道府県名'].split('_')[0]  # "(都道府県コード)_都道府県名" から都道府県コードを抽出
        pref_code_dict[pref_name] = pref_code
    
    """# 283都市の標高・傾斜度・SHCを計算し、エクセルデータを作成
    df_stats = calc_terrain_of_283cities(pref_code_dict)
    df_stats.to_excel(RESULT_XLSX, index=False)"""

    # 作成したエクセルデータを読み込み
    df_stats = pd.read_excel(RESULT_XLSX, dtype={'都道府県コード': str})
    top_steep_cities = df_stats.sort_values('住居系用途地域に占める斜面市街地の割合', ascending=False).head(27)

    mean_or_med = '平均値'
    # correlation_matrix, pca_summary, loadings = analyze_terrain_indicators(df_stats)
    # correlation = analyze_slope_correlation(df_stats)
    # visualize_slope_area_ratio_histogram(df_stats)
    visualize_top_cities_on_map(df_stats, mean_or_med, thr_rank=28)
    visualize_top_steep_cities_comparison(top_steep_cities)
    top_steep_cities = list(top_steep_cities['市区町村名'])
    visualize_slope_shc_relationship(df_stats, kyogikai_cities, top_steep_cities, False, 'red', mean_or_med, thr_rank=27)
    visualize_slope_shc_relationship(df_stats, kyogikai_cities, top_steep_cities, False, 'orange', mean_or_med, thr_rank=27)
    visualize_slope_shc_relationship_with_top_cities(df_stats, top_steep_cities, False, mean_or_med)
