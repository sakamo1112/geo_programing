import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def calc_and_visualize_height(
    city_name: str,
    bbox_elevation: np.ndarray,
    housing_mask: np.ndarray,
    visualize: bool,
) -> np.ndarray:
    """
    標高の計算と可視化を行い、住居系用途地域の標高データを返す。

    Args:
        city_name (str): 自治体名(例: 11_戸田市(埼玉県))
        bbox_elevation (numpy.ndarray): 住居系用途地域を囲むBBoxの標高データ
        housing_mask (numpy.ndarray): 住居系用途地域のマスク
    Returns:
        housing_area_height (numpy.ndarray): 住居系用途地域の標高データ
    """
    housing_area_height = np.where(housing_mask == 1, bbox_elevation, np.nan)
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(bbox_elevation, cmap="terrain", aspect="equal", vmin=0, vmax=300)
        ax.imshow(
            np.where(housing_mask == 0, 0.7, np.nan),
            cmap="gray_r",
            alpha=0.8,
            aspect="equal",
        )

        cbar = plt.colorbar(im)
        cbar.set_label("標高 [m]")
        plt.title(f"{city_name}の標高分布")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"result/elevation/elevation_{city_name}.png", dpi=300)
        plt.close()

    return housing_area_height


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
    housing_area_slope_removed = remove_local_outliers(housing_area_slope, "傾斜度")

    # 傾斜度の可視化
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3.5, 1]}
    )
    plt.subplots_adjust(left=0, right=0.85)

    # 傾斜度の空間分布
    if visualize:
        visualize_pixel_histogram(housing_area_slope, city_name, "slope", hazure=False)
        im = ax1.imshow(slope, cmap="autumn_r", aspect="equal", vmin=0, vmax=45)

        # マスク部分を灰色で表示（住居系用途地域外を灰色に）
        ax1.imshow(
            np.where(housing_mask == 0, 0.7, np.nan),
            cmap="gray_r",
            alpha=0.8,
            aspect="equal",
        )

        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("傾斜度 (度)")
        ax1.set_title(f"{city_name}の傾斜度分布")
        ax1.axis("off")

    # 傾斜度の区分ごとの割合を計算
    bins = [0, 5, 10, 15, 20, 25, np.inf]
    labels = ["0-5度", "5-10度", "10-15度", "15-20度", "20-25度", "25度以上"]
    hist, _ = np.histogram(housing_area_slope[~np.isnan(housing_area_slope)], bins=bins)
    percentages = hist / len(housing_area_slope[~np.isnan(housing_area_slope)]) * 100
    steep_ratio = sum(percentages[1:])  # 5度以上の割合の合計

    # 積み上げ棒グラフの作成
    if visualize:
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
        ax2.set_title("傾斜度の区分別割合\n(住居系用途地域内)")
        ax2.set_xticks([])
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 統計情報の表示
    if visualize:
        stats_text = (
            "傾斜度の統計情報\n(住居系用途地域内)\n"
            f"最小値: {np.nanmin(housing_area_slope):.1f}[度]\n"
            f"最大値: {np.nanmax(housing_area_slope):.1f}[度]\n"
            f"平均値: {np.nanmean(housing_area_slope):.1f}[度]\n"
            f"中央値: {np.nanmedian(housing_area_slope):.1f}[度]"
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

        plt.savefig(f"result/slope/slope_{city_name}.png", dpi=300)
        plt.close()

    return housing_area_slope, housing_area_slope_removed, steep_ratio


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
    window_size = 10  # 移動窓サイズ

    # 平面曲率を計算
    print(f"{city_name}の平面曲率を計算中...")
    _, plan_curv = calc_profile_and_plan_curvature(bbox_elevation)

    # SHCを計算
    print(f"{city_name}の平面曲率標準偏差(SHC)を計算中...")
    bbox_shc = calc_shc(plan_curv, window_size)

    # 住居系用途地域内のデータのみを使用して統計を計算
    print(f"{city_name}のSHCをマスク中...")
    housing_area_shc = np.where(housing_mask == 1, bbox_shc, np.nan)
    housing_area_shc_removed = remove_local_outliers(housing_area_shc, "SHC")

    if visualize:
        visualize_pixel_histogram(housing_area_shc, city_name, "shc", hazure=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        # vmaxを95パーセンタイルに設定
        # vmax = np.nanpercentile(housing_area_shc, 95)
        vmax = 0.15

        # SHCの空間分布
        im = ax.imshow(bbox_shc, cmap="viridis", aspect="equal", vmin=0, vmax=vmax)
        ax.imshow(
            np.where(housing_mask == 0, 0.7, np.nan),
            cmap="gray_r",
            alpha=0.8,
            aspect="equal",
        )

        cbar = plt.colorbar(im)
        cbar.set_label("SHC")

        ax.set_title(f"{city_name}の平面曲率標準偏差(SHC)\n(移動窓: 半径{window_size*10}mの円)")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"result/shc/shc_{city_name}.png", dpi=300)
        plt.close()

    return housing_area_shc, housing_area_shc_removed


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


def remove_local_outliers(data, column_name):
    """
    各都市内の局所的な外れ値を除去する関数

    Args:
        data (numpy.ndarray): 分析対象のデータ配列
        column_name (str): データの種類（ログ出力用）

    Returns:
        numpy.ndarray: 外れ値を除去したデータ配列
    """
    valid_data = data[~np.isnan(data)]
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_no_outliers = np.where(
        (data < lower_bound) | (data > upper_bound), np.nan, data
    )

    outlier_ratio = (
        np.sum((data < lower_bound) | (data > upper_bound))
        / np.sum(~np.isnan(data))
        * 100
    )

    print(f"{column_name}の外れ値除去: {outlier_ratio:.1f}%のピクセルを除外")

    return data_no_outliers


def visualize_pixel_histogram(data, city_name, data_type, hazure=False):
    """
    ピクセルレベルでの傾斜度またはSHCのヒストグラムを作成する関数

    Args:
        data (numpy.ndarray): 傾斜度またはSHCのデータ配列
        city_name (str): 自治体名
        data_type (str): データの種類（'slope'または'shc'）
        hazure (bool): 外れ値除去後のデータを使用するかどうか
    """
    # nanを除外したデータを取得
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        print(f"Warning: {city_name}の{data_type}データが存在しません")
        return

    # 通常データの平均値を計算
    original_mean = np.mean(valid_data)
    original_median = np.median(valid_data)

    # 外れ値除去後のデータと平均値を計算
    removed_data = remove_local_outliers(data, data_type)
    removed_valid_data = removed_data[~np.isnan(removed_data)]

    plt.figure(figsize=(10, 6))

    # データの前処理とビンの設定
    if data_type == "slope":
        # 25度以上のデータを25度に置き換え
        plot_data = valid_data if not hazure else removed_valid_data
        bins = np.arange(0, 27, 1)  # 0-25度まで1度間隔
        xlabel = "傾斜度 [度]"
        title_type = "傾斜度"
        last_bin_label = "25度以上"
    else:  # shc
        # 0.20以上のデータを0.20に置き換え
        plot_data = valid_data if not hazure else removed_valid_data
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
    target_data = removed_valid_data if hazure else valid_data
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
    hazure_str = "（外れ値除去後）" if hazure else ""
    plt.title(f"{city_name}の{title_type}分布{hazure_str}")
    plt.grid(True, alpha=0.3)

    # x軸の範囲を設定
    if data_type == "slope":
        plt.xlim(0, 25)
    else:
        plt.xlim(0, 0.20)

    plt.ylim(bottom=0)
    plt.legend()

    data_dir = "slope" if data_type == "slope" else "shc"
    hazure_suffix = "_hazure" if hazure else ""
    plt.savefig(
        f"result/{data_dir}_hist/{data_type}_hist_{city_name}{hazure_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
