import random

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt

"""
都道府県名
（N03_001）	当該区域を含む都道府県名称	文字列型（CharacterString）

支庁・振興局名
（N03_002）	当該都道府県が「北海道」の場合、該当する支庁・振興局の名称	文字列型（CharacterString）

郡・政令都市名
（N03_003）	当該行政区の郡又は政令市の名称	文字列型（CharacterString）

市区町村名
（N03_004）	当該行政区の市区町村の名称	文字列型（CharacterString）

行政区域コード
（N03_007）	都道府県コードと市区町村コードからなる、行政区を特定するためのコード	コードリスト「行政区域コード」
"""
# 世界地図を表示
world = gpd.datasets.get_path("naturalearth_lowres")  # 世界地図
cities = gpd.datasets.get_path("naturalearth_cities")  # 都市
df_world = gpd.read_file(world)
df_cities = gpd.read_file(cities)
ax = df_world.plot()
df_world.plot(ax=ax, color="white", edgecolor="black")
df_cities.plot(ax=ax, marker="o", color="red", markersize=5)
plt.show()


# 東京都(本土データ)を行政区域ごとに色分けして表示
shape_file = r"shp_files/tokyo_gyoseikai/N03-23_13_230101.shp"
data = gpd.read_file(shape_file, encoding="shift-jis")
print(data["geometry"].head())
ax_tokyo = data[0:151].plot()

for i in range(151):
    color_RGB = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
    data_new = data[i : i + 1]
    data_new.plot(ax=ax_tokyo, color=color_RGB, edgecolor="black")
plt.show()
