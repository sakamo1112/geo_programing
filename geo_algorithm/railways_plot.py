import random

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# 東京都(本土データ)を行政区域ごとに色分けして表示
tokyo_shp = r"shp_files/tokyo150/tokyo_gyoseikai150.shp"
saitama_shp = r"shp_files/saitama_gyoseikai/N03-23_11_230101.shp"
chiba_shp = r"shp_files/chiba_gyoseikai/N03-23_12_230101.shp"
kanagawa_shp = r"shp_files/kanagawa_gyoseikai/N03-23_14_230101.shp"
data_tokyo = gpd.read_file(tokyo_shp, encoding="shift-jis")
data_saitama = gpd.read_file(saitama_shp, encoding="shift-jis")
data_chiba = gpd.read_file(chiba_shp, encoding="shift-jis")
data_kanagawa = gpd.read_file(kanagawa_shp, encoding="shift-jis")
tokyoken_list = [data_tokyo, data_saitama, data_chiba, data_kanagawa]

merged_tokyoken = gpd.GeoDataFrame(pd.concat(tokyoken_list, ignore_index=True))

ax_tokyoken = merged_tokyoken.plot()
for i in range(len(merged_tokyoken)):
    data_sanken_new = merged_tokyoken[i : i + 1]
    color_RGB = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
    data_sanken_new.plot(ax=ax_tokyoken, color=color_RGB, edgecolor="black")
plt.show()
