import os

import contextily as ctx
import geopandas as gpd
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from calc_terrain_status import calc_slope
from matplotlib.lines import Line2D
from rasterio.mask import mask
from shapely.geometry import box

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


target_list = pd.read_excel(TARGET_LIST_EXCEL)
pref_code_dict = {}
for _, row in target_list.iterrows():
    pref_name = row["都道府県名"].split("_")[1]  # "(都道府県コード)_都道府県名" から都道府県名を抽出
    pref_code = row["都道府県名"].split("_")[0]  # "(都道府県コード)_都道府県名" から都道府県コードを抽出
    pref_code_dict[pref_name] = pref_code


def visualize_top_steep_cities_comparison(df_stats, top_steep_cities):
    """
    与えられたDataFrameの各都市について、傾斜度区分割合を比較するグラフを作成する。

    Args:
        df_stats (pandas.DataFrame): 統計データを含むDataFrame。以下のカラムが必要:
            - 市区町村名
            - 都道府県名
            - 都道府県コード
        top_steep_cities (list): 上位の斜面都市名のリスト
    """
    slope_data = {}
    steep_ratio_15deg = {}

    # 各都市について傾斜度区分の計算を行う
    for city_name in top_steep_cities:
        city_row = df_stats[df_stats["市区町村名"] == city_name].iloc[0]
        pref_name = city_row["都道府県名"]
        pref_code = city_row["都道府県コード"]

        # DEMファイルの確認
        area_name = None
        for area, prefs in area_dict.items():
            if pref_name in prefs:
                area_name = area
                break

        dem_path = None
        for dem_name in os.listdir(DEM_DIR):
            if area_name in dem_name and dem_name.endswith(".tif"):
                dem_path = os.path.join(DEM_DIR, dem_name)
                break

        # 住居系用途地域のシェープファイルを読み込む
        housing_area_path = os.path.join(
            HOUSING_AREA_DIR, f"A29-19_{pref_code}", f"housing_{city_name}.shp"
        )

        if dem_path and os.path.exists(housing_area_path):
            with rasterio.open(dem_path) as src:
                housing_area_gdf = gpd.read_file(housing_area_path)
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

                # 傾斜度の計算
                slope = calc_slope(bbox_elevation)
                housing_area_slope = np.where(housing_mask == 1, slope, np.nan)

                # 傾斜度の区分ごとの割合を計算
                bins = [0, 5, 10, 15, 20, 25, np.inf]
                hist, _ = np.histogram(
                    housing_area_slope[~np.isnan(housing_area_slope)], bins=bins
                )
                percentages = (
                    hist / len(housing_area_slope[~np.isnan(housing_area_slope)]) * 100
                )
                slope_data[f"{city_name}"] = percentages
                # 15度以上の割合を計算（15度以上の3区分の合計）
                steep_ratio_15deg[city_name] = sum(percentages[3:])

    # 2つのバージョンのグラフを作成
    for sort_by_steep in [False, True]:
        fig, ax = plt.subplots(figsize=(18, 8))

        labels = ["0-5度", "5-10度", "10-15度", "15-20度", "20-25度", "25度以上"]
        colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "red"]

        # 15度以上の割合で並べ替え（オプション）
        if sort_by_steep:
            sorted_cities = dict(
                sorted(steep_ratio_15deg.items(), key=lambda x: x[1], reverse=True)
            )
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
                    ax.text(j, bottom[j] + v / 2, f"{v:.1f}%", ha="center", va="center")
            bottom += values

        # グラフの装飾
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_slope_data.keys(), rotation=45, ha="right")
        ax.set_ylabel("割合 (%)")
        title = "各都市の傾斜度区分割合の比較"
        if sort_by_steep:
            title += "\n（15度以上の割合で降順に並べ替え）"
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        filename = "result/steep_ratio_comparison"
        if sort_by_steep:
            filename += "_sorted_by_15deg"
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()


def visualize_slope_shc_relationship(
    df_stats,
    kyogikai_cities,
    top_steep_cities,
    only_steep_area,
    mode,
    hazure,
    thr_rank,
):
    """
    傾斜度とSHCの関係を散布図で可視化する関数

    Args:
        df_stats (pandas.DataFrame): 統計値をまとめたデータフレーム
        kyogikai_cities (list): 協議会都市名のリスト
        top_steep_cities (list): 上位の斜面都市名のリスト
        only_steep_area (bool): 斜面市街地のみを表示するかどうか
        mode (str): 色分けのモード（"red" または "orange"）
        hazure (bool): 外れ値除去を行うかどうか
        thr_rank (int): 上位何位までを表示するか
    """
    plt.figure(figsize=(10, 8))

    if only_steep_area:
        df_stats = df_stats[df_stats["住居系用途地域に占める斜面市街地の割合"] >= 5]
        print(f"斜面市街地の割合が5%以上の都市: {len(df_stats)}件")
        makura = "斜面市街地の"
    else:
        makura = ""

    if hazure:
        hazure = "(外れ値除去後)"
    else:
        hazure = ""

    top_slope = df_stats.nlargest(thr_rank, f"{makura}傾斜度{hazure}_中央値")[
        "市区町村名"
    ].tolist()
    top_shc = df_stats.nlargest(thr_rank, f"{makura}SHC{hazure}_平均値")["市区町村名"].tolist()

    # 凡例用のフラグを追加
    legend_added = {
        "both": False,
        "slope": False,
        "shc": False,
        "other": False,
        "top_steep": False,
        "kyogikai": False,
    }

    # まず、その他の都市をプロット
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

        if city in top_slope and city in top_shc:
            # 両方のTop20に入っている都市は紫の星でプロット
            plt.scatter(
                x,
                y,
                color="purple",
                marker="*",
                s=100,
                zorder=3,
                label=f"傾斜度・SHC両方で上位{thr_rank}位以内" if not legend_added["both"] else "",
                alpha=0.5,
            )
            legend_added["both"] = True
            if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(
                    city,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="purple",
                    zorder=3,
                    fontsize=6,
                )
        elif city in top_slope:
            plt.scatter(
                x,
                y,
                color="green",
                marker="s",
                s=50,
                zorder=2,
                label=f"傾斜度上位{thr_rank}位以内" if not legend_added["slope"] else "",
                alpha=0.5,
            )
            legend_added["slope"] = True
            if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(
                    city,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="green",
                    zorder=2,
                    fontsize=6,
                )
        elif city in top_shc:
            plt.scatter(
                x,
                y,
                color="blue",
                marker="^",
                s=50,
                zorder=2,
                label=f"SHC上位{thr_rank}位以内" if not legend_added["shc"] else "",
                alpha=0.5,
            )
            legend_added["shc"] = True
            if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(
                    city,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="blue",
                    zorder=2,
                    fontsize=6,
                )
        else:
            # それ以外の都市は灰色でプロット
            plt.scatter(
                x,
                y,
                color="gray",
                alpha=0.5,
                s=30,
                zorder=1,
                label="その他の都市" if not legend_added["other"] else "",
            )
            legend_added["other"] = True

    # 次に、協議会都市または主要な斜面都市をプロット（最前面に表示）
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

        if mode == "red" and city in kyogikai_cities:
            plt.scatter(
                x,
                y,
                color="red",
                marker="o",
                s=20,
                zorder=10,
                label="全国斜面都市連絡協議会に加盟している自治体" if not legend_added["kyogikai"] else "",
            )
            legend_added["kyogikai"] = True
            plt.annotate(
                city,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color="black",
                fontweight="bold",
                zorder=10,
                fontsize=8,
            )
        elif mode == "orange" and city in top_steep_cities:
            plt.scatter(
                x,
                y,
                color="orange",
                marker="o",
                s=20,
                zorder=10,
                label="主要な斜面都市" if not legend_added["top_steep"] else "",
            )
            legend_added["top_steep"] = True
            plt.annotate(
                city,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color="black",
                fontweight="bold",
                zorder=10,
                fontsize=8,
            )

    plt.xlabel(f"起伏の大きさ(SHC{hazure}の平均値)")
    plt.ylabel(f"急峻さ(傾斜度{hazure}の中央値) [度]")
    if only_steep_area:
        plt.title(f"各自治体の斜面市街地における傾斜度とSHCの関係")
    else:
        plt.title(f"各自治体における傾斜度とSHCの関係")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower left")

    hazure_suffix = "_hazure" if hazure else ""
    if mode == "red":
        plt.savefig(
            f"result/slope_shc_scatter_kyogikai{hazure_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
    elif mode == "orange":
        plt.savefig(
            f"result/slope_shc_scatter_top_steep{hazure_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def visualize_slope_shc_relationship_with_top_cities(
    df_stats, top_steep_cities, only_steep_area, hazure
):
    """
    傾斜度とSHCの関係を散布図で可視化する関数

    Args:
        df_stats (pandas.DataFrame): 統計値をまとめたデータフレーム
        top_steep_cities (list): 主要な斜面都市名のリスト
        only_steep_area (bool): 斜面市街地のみを表示するかどうか
        hazure (bool): 外れ値除去を行うかどうか
    """
    plt.figure(figsize=(10, 8))

    if only_steep_area:
        df_stats = df_stats[df_stats["住居系用途地域に占める斜面市街地の割合"] >= 5]
        makura = "斜面市街地の"
    else:
        makura = ""

    if hazure:
        hazure = "(外れ値除去後)"
    else:
        hazure = ""

    # まず、その他の都市をプロット
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

    # 次に、主要な斜面都市をプロット（最前面に表示）
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

        if city in top_steep_cities:
            plt.scatter(x, y, color="orange", marker="o", s=30, zorder=10)
            plt.annotate(
                city,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
                zorder=10,
                fontsize=10,
            )

    plt.xlabel(f"起伏の大きさ(SHC{hazure}の平均値)")
    plt.ylabel(f"急峻さ(傾斜度{hazure}の中央値) [度]")
    if only_steep_area:
        plt.title(f"主要な斜面都市の斜面市街地における傾斜度とSHCの関係")
    else:
        plt.title(f"主要な斜面都市における傾斜度とSHCの関係")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower left")

    hazure_suffix = "_hazure" if hazure else ""
    plt.savefig(
        f"result/slope_shc_scatter_top_steep_with_top_cities{hazure_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualize_top_cities_on_map(df_stats, hazure, if_kanto, thr_rank):
    """
    傾斜度とSHCの上位都市を地図上に可視化する関数

    Args:
        df_stats (pandas.DataFrame): 統計値をまとめたデータフレーム
        hazure (bool): 外れ値除去を行うかどうか
        if_kanto (bool): 関東版の地図を表示するかどうか
        thr_rank (int): 上位何位までを表示するか
    """
    if hazure:
        hazure = "(外れ値除去後)"
    else:
        hazure = ""

    slope_top = df_stats.nlargest(thr_rank, f"傾斜度{hazure}_中央値")["市区町村名"].tolist()
    shc_top = df_stats.nlargest(thr_rank, f"SHC{hazure}_平均値")["市区町村名"].tolist()
    both_top = list(set(slope_top) & set(shc_top))

    fig, ax = plt.subplots(figsize=(15, 10))

    japan_bounds = [
        14200000,  # 左端（西）
        16400000,  # 右端（東）
        3500000,  # 下端（南）
        5800000,  # 上端（北）
    ]
    kanto_bounds = [
        15480000,  # 左端（西）
        15650000,  # 右端（東）
        4165000,  # 下端（南）
        4308000,  # 上端（北）
    ]

    if if_kanto:
        ax.set_xlim(kanto_bounds[0], kanto_bounds[1])
        ax.set_ylim(kanto_bounds[2], kanto_bounds[3])
    else:
        ax.set_xlim(japan_bounds[0], japan_bounds[1])
        ax.set_ylim(japan_bounds[2], japan_bounds[3])

    # 各都市の住居系用途地域を描画
    for _, row in df_stats.iterrows():
        city_name = row["市区町村名"]
        pref_name = row["都道府県名"]
        pref_code = pref_code_dict[pref_name]

        shp_path = os.path.join(
            HOUSING_AREA_DIR, f"A29-19_{pref_code}", f"housing_{city_name}.shp"
        )
        if os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path)
            gdf = gdf.to_crs(epsg=3857)

            center_point = gdf.geometry.unary_union.centroid

            # マーカーの設定 関東版は10倍
            baffer = 1
            if if_kanto:
                baffer *= 10

            if city_name in both_top:
                marker = "*"  # 星
                color = "purple"
                size = 50 * baffer
            elif city_name in slope_top:
                marker = "s"  # 三角
                color = "green"
                size = 40 * baffer
            elif city_name in shc_top:
                marker = "^"  # 四角
                color = "blue"
                size = 40 * baffer
            else:
                continue

            # マーカーを描画
            ax.scatter(
                center_point.x,
                center_point.y,
                c=color,
                marker=marker,
                s=size,
                edgecolor="black",
                linewidth=1,
            )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)

    ax.set_aspect("equal")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="purple",
            markersize=15,
            label=f"傾斜度・SHC共に上位{thr_rank}位以内",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label=f"傾斜度上位{thr_rank}位以内",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label=f"SHC上位{thr_rank}位以内",
        ),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.title(f"住居系用途地域の傾斜度・SHC上位{thr_rank}都市")
    ax.set_axis_off()
    hazure_suffix = "_hazure" if hazure else ""
    if if_kanto:
        plt.savefig(
            f"result/top{thr_rank}_cities_map_kanto{hazure_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"result/top{thr_rank}_cities_map_japan{hazure_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def visualize_slope_area_ratio_histogram(df_stats):
    """
    斜面市街地の割合のヒストグラムを作成する関数

    Args:
        df_stats (pandas.DataFrame): 統計値をまとめたデータフレーム
    """
    print("斜面市街地の割合のヒストグラムを作成中...")

    # 2%間隔でビンを作成
    bins = np.arange(0, df_stats["住居系用途地域に占める斜面市街地の割合"].max() + 2, 2)

    # ヒストグラムの作成
    plt.figure(figsize=(10, 6))
    plt.hist(
        df_stats["住居系用途地域に占める斜面市街地の割合"],
        bins=bins,
        edgecolor="black",
        color="#0b5394",
        alpha=0.8,
    )
    plt.xlabel("住居系用途地域に占める斜面市街地の割合 (%)")
    plt.ylabel("自治体数")
    plt.title("斜面市街地の割合の分布")
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)  # x軸の左端を0に設定
    plt.savefig("result/slope_area_ratio_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_slope_shc_relationship(
    df_stats,
    kyogikai_cities,
    top_steep_cities,
    only_steep_area,
    mode,
    hazure,
    thr_rank,
):
    """
    傾斜度とSHCの関係を散布図で可視化する関数

    Args:
        df_stats (pandas.DataFrame): 統計値をまとめたデータフレーム
        kyogikai_cities (list): 協議会都市名のリスト
        top_steep_cities (list): 主要な斜面都市名のリスト
        only_steep_area (bool): 斜面市街地のみを表示するかどうか
        mode (str): モード ("red" or "orange")
        hazure (bool): 外れ値除去を行うかどうか
        thr_rank (int): 上位何位までを表示するか
    """
    plt.figure(figsize=(10, 8))

    if only_steep_area:
        df_stats = df_stats[df_stats["住居系用途地域に占める斜面市街地の割合"] >= 5]
        print(f"斜面市街地の割合が5%以上の都市: {len(df_stats)}件")
        makura = "斜面市街地の"
    else:
        makura = ""

    if hazure:
        hazure = "(外れ値除去後)"
    else:
        hazure = ""

    # 傾斜度とSHCの上位thr_rank自治体を取得
    top_slope = df_stats.nlargest(thr_rank, f"{makura}傾斜度{hazure}_中央値")[
        "市区町村名"
    ].tolist()
    top_shc = df_stats.nlargest(thr_rank, f"{makura}SHC{hazure}_平均値")["市区町村名"].tolist()

    # 凡例用のフラグを追加
    legend_added = {
        "both": False,
        "slope": False,
        "shc": False,
        "other": False,
        "top_steep": False,
        "kyogikai": False,
    }

    # まず、その他の都市をプロット
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

        if city in top_slope and city in top_shc:
            plt.scatter(
                x,
                y,
                color="purple",
                marker="*",
                s=100,
                zorder=3,
                label=f"傾斜度・SHC両方で上位{thr_rank}位以内" if not legend_added["both"] else "",
                alpha=0.5,
            )
            legend_added["both"] = True
            if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(
                    city,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="purple",
                    zorder=3,
                    fontsize=6,
                )
        elif city in top_slope:
            plt.scatter(
                x,
                y,
                color="green",
                marker="s",
                s=50,
                zorder=2,
                label=f"傾斜度上位{thr_rank}位以内" if not legend_added["slope"] else "",
                alpha=0.5,
            )
            legend_added["slope"] = True
            if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(
                    city,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="green",
                    zorder=2,
                    fontsize=6,
                )
        elif city in top_shc:
            plt.scatter(
                x,
                y,
                color="blue",
                marker="^",
                s=50,
                zorder=2,
                label=f"SHC上位{thr_rank}位以内" if not legend_added["shc"] else "",
                alpha=0.5,
            )
            legend_added["shc"] = True
            if city not in kyogikai_cities and city not in top_steep_cities:
                plt.annotate(
                    city,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="blue",
                    zorder=2,
                    fontsize=6,
                )
        else:
            plt.scatter(
                x,
                y,
                color="gray",
                alpha=0.5,
                s=30,
                zorder=1,
                label="その他の都市" if not legend_added["other"] else "",
            )
            legend_added["other"] = True

    # 次に、協議会都市または主要な斜面都市をプロット（最前面に表示）
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

        if mode == "red" and city in kyogikai_cities:
            plt.scatter(
                x,
                y,
                color="red",
                marker="o",
                s=20,
                zorder=10,
                label="全国斜面都市連絡協議会に加盟している自治体" if not legend_added["kyogikai"] else "",
            )
            legend_added["kyogikai"] = True
            plt.annotate(
                city,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color="black",
                fontweight="bold",
                zorder=10,
                fontsize=8,
            )
        elif mode == "orange" and city in top_steep_cities:
            plt.scatter(
                x,
                y,
                color="orange",
                marker="o",
                s=20,
                zorder=10,
                label="主要な斜面都市" if not legend_added["top_steep"] else "",
            )
            legend_added["top_steep"] = True
            plt.annotate(
                city,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color="black",
                fontweight="bold",
                zorder=10,
                fontsize=8,
            )

    plt.xlabel(f"起伏の大きさ(SHC{hazure}の平均値)")
    plt.ylabel(f"急峻さ(傾斜度{hazure}の中央値) [度]")
    if only_steep_area:
        plt.title(f"各自治体の斜面市街地における傾斜度とSHCの関係")
    else:
        plt.title(f"各自治体における傾斜度とSHCの関係")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower left")

    hazure_suffix = "_hazure" if hazure else ""
    steep_suffix = "_os" if only_steep_area else ""
    if mode == "red":
        plt.savefig(
            f"result/slope_shc_scatter_kyogikai{hazure_suffix}{steep_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
    elif mode == "orange":
        plt.savefig(
            f"result/slope_shc_scatter_top_steep{hazure_suffix}{steep_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def visualize_slope_shc_relationship_with_top_cities(
    df_stats, top_steep_cities, only_steep_area, hazure
):
    """
    傾斜度とSHCの関係を散布図で可視化する関数

    Args:
        df_stats (pandas.DataFrame): 統計値をまとめたデータフレーム
        top_steep_cities (list): 主要な斜面都市名のリスト
        only_steep_area (bool): 斜面市街地のみを表示するかどうか
        hazure (bool): 外れ値除去を行うかどうか
    """
    plt.figure(figsize=(10, 8))

    if only_steep_area:
        df_stats = df_stats[df_stats["住居系用途地域に占める斜面市街地の割合"] >= 5]
        makura = "斜面市街地の"
    else:
        makura = ""

    if hazure:
        hazure = "(外れ値除去後)"
    else:
        hazure = ""

    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

    # 次に、主要な斜面都市をプロット（最前面に表示）
    for i, row in df_stats.iterrows():
        city = row["市区町村名"]
        x, y = (
            row[f"{makura}SHC{hazure}_平均値"],
            row[f"{makura}傾斜度{hazure}_中央値"],
        )

        if city in top_steep_cities:
            plt.scatter(x, y, color="orange", marker="o", s=30, zorder=10)
            plt.annotate(
                city,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
                zorder=10,
                fontsize=10,
            )

    plt.xlabel(f"起伏の大きさ(SHC{hazure}の平均値)")
    plt.ylabel(f"急峻さ(傾斜度{hazure}の中央値) [度]")
    if only_steep_area:
        plt.title(f"主要な斜面都市の斜面市街地における傾斜度とSHCの関係")
    else:
        plt.title(f"主要な斜面都市における傾斜度とSHCの関係")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower left")

    hazure_suffix = "_hazure" if hazure else ""
    steep_suffix = "_os" if only_steep_area else ""
    plt.savefig(
        f"result/slope_shc_scatter_top_steep_with_top_cities{hazure_suffix}{steep_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
