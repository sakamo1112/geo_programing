import random

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt

continent_color_dict = {
    "Oceania": (1.000, 0.294, 0.000),
    "Africa": (0.000, 0.353, 1.000),
    "North America": (0.012, 0.686, 0.478),
    "Europe": (0.302, 0.769, 1.000),
    "Asia": (0.965, 0.667, 0.000),
    "South America": (1.000, 0.945, 0.000),
    "Antarctica": (0.255, 0.078, 0.271),
    "Australia": (0.5000, 0.5000, 0.5000),
    "Others": (0, 0, 0),
}

world = gpd.datasets.get_path("naturalearth_lowres")  # 世界地図
df_world = gpd.read_file(world)

# 大陸ごとに色分け表示
ax_continent = df_world.plot()
continents = df_world.dissolve(by="continent")  # 大陸ごとにまとめる
for i in range(len(continents)):
    continent = continents[i : i + 1]
    continent_name = continent.index.values[0]
    if continent_name in continent_color_dict:
        continent.plot(
            ax=ax_continent,
            color=continent_color_dict[continent_name],
            edgecolor="black",
        )
    else:
        continent.plot(
            ax=ax_continent, color=continent_color_dict["Others"], edgecolor="black"
        )
plt.show()

# 国ごとに色分け表示
ax_country = df_world.plot()
for i in range(len(df_world)):
    A_Country = df_world[i : i + 1]
    RGB = (random.uniform(0.3, 1), random.uniform(0.3, 1), random.uniform(0.3, 1))
    A_Country.plot(ax=ax_country, color=RGB, edgecolor="black")
plt.show()
