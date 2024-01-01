import random

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

color_dict = {
    "1": (1.000, 0.294, 0.000),
    "2": (0.000, 0.353, 1.000),
    "3": (0.012, 0.686, 0.478),
    "4": (0.302, 0.769, 1.000),
    "5": (0.965, 0.667, 0.000),
    "6": (1.000, 0.945, 0.000),
    "7": (0.255, 0.078, 0.271),
}


def gen_cmap_name(cols):
    nmax = float(len(cols) - 1)
    color_list = []
    for n, c in enumerate(cols):
        color_list.append((n / nmax, c))

    return mpl.colors.LinearSegmentedColormap.from_list("cmap", color_list)


cmap = gen_cmap_name(["r", "yellow", "g", "b", "magenta"])


# 洪水浸水想定区域のshapefileを読み込む
shp_flood = r"shp_files/洪水浸水想定区域/10_計画規模/A31-10-22_10_5339.shp"
shp_tokyo = r"shp_files/tokyo150/tokyo_gyoseikai150.shp"
data_flood = gpd.read_file(shp_flood, encoding="shift-jis")
data_tokyo = gpd.read_file(shp_tokyo, encoding="shift-jis")

ax_tokyo = data_tokyo.plot()
for i in range(151):
    color_RGB = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
    data_new = data_tokyo[i : i + 1]
    data_new.plot(ax=ax_tokyo, color=color_RGB, edgecolor="black")

for i in range(len(data_flood)):
    data_flood_new = data_flood[i : i + 1]
    rank = data_flood_new["A31_101"].values[0]
    data_flood_new.plot(ax=ax_tokyo, color=color_dict[str(rank)])
plt.show()
