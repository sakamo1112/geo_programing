import argparse
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mojimoji
import numpy as np
import pandas as pd  # type: ignore

area_list = [
    "全市",
    "追浜",
    "本庁",
    "田浦",
    "大津",
    "北下浦",
    "久里浜",
    "大楠",
    "衣笠",
    "逸見",
    "浦賀",
    "長井",
    "武山",
]
year_list = [
    "1995",
    "2000",
    "2005",
    "2010",
    "2015",
    "2020",
    "2025",
    "2030",
    "2035",
    "2040",
]


def create_population_pyramid(
    year: str, xlsx_path: str, area_name: str, age_column_name: str
):
    # データの読み込み
    data = pd.read_excel(open(xlsx_path, "rb"), sheet_name=area_name)
    man_column_name = year + "M"
    woman_column_name = year + "W"
    df = data[[age_column_name, man_column_name, woman_column_name]]

    # データの整形
    x = df[age_column_name][0:18]
    x = [s.replace("～", "-") for s in x]
    x = [s.replace(" ", "") for s in x]
    x = [mojimoji.zen_to_han(s) for s in x]
    x = [re.sub(r"[^a-zA-Z0-9=" + "-" + "]", "", s) for s in x]

    man_max_value = max(df[man_column_name][0:18])
    woman_max_value = max(df[woman_column_name][0:18])
    max_value = max(man_max_value, woman_max_value)
    margin = max_value * 0.1
    range = max_value + margin  # rangeを定数にすれば値の範囲を固定できる

    # プロット
    fig = plt.figure(figsize=(7, 8))  # windowのサイズ
    plt.rcParams["font.size"] = "12"
    ax1 = fig.add_subplot(111)
    for i, row in df[0:18].iterrows():
        plt.barh(
            row[age_column_name],
            [row[man_column_name], -row[woman_column_name]],
            color=["#008AB8", "#CC6699"],
            height=0.8,
            align="center",
            edgecolor="none",
        )

    # 軸の設定
    plt.title("Population of {} in {}".format(area_name, year), fontname="IPAexGothic")
    plt.ylim(-0.6, 17.6)
    plt.xlim(-range, range)
    plt.yticks(np.arange(0, 18), x, fontsize=12)
    plt.xticks(
        np.arange(-range, range + 1, range / 4),
        [
            "{}".format(int(abs(x) / 1)) if x != 0 else "0"
            for x in np.arange(-range, range + 1, range / 4)
        ],
    )

    # 軸のラベル設定
    plt.ylabel("Age")
    plt.xlabel("Number of people")
    a = mpatches.Patch(color="#008AB8", label="male")
    b = mpatches.Patch(color="#CC6699", label="female")
    plt.legend(handles=[a, b])
    plt.savefig("result/{}-{}.png".format(area_name, year))
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a point cloud")
    parser.add_argument("--year", type=str)
    parser.add_argument(
        "--xlsx_path",
        type=str,
        default="data/横須賀_人口推測.xlsx",
    )
    parser.add_argument("--area_name", type=str)
    parser.add_argument("--age_column_name", type=str, default="Age")
    args = parser.parse_args()

    if args.area_name and args.year:
        print(f"{args.area_name}について{args.year}年の人口ピラミッドを作成")
        create_population_pyramid(
            args.year, args.xlsx_path, args.area_name, args.age_column_name
        )

    elif not args.area_name and args.year:
        print(f"13地域について{args.year}年の人口ピラミッドを作成")
        for area_name in area_list:
            create_population_pyramid(
                args.year, args.xlsx_path, area_name, args.age_column_name
            )

    elif args.area_name and not args.year:
        print(f"{args.area_name}について1995-2040年の人口ピラミッドを作成")
        for year in year_list:
            create_population_pyramid(
                year, args.xlsx_path, args.area_name, args.age_column_name
            )

    else:
        print("13地域について1995-2040年の人口ピラミッドを作成")
        for year in year_list:
            for area_name in area_list:
                create_population_pyramid(
                    year, args.xlsx_path, area_name, args.age_column_name
                )
