import os
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import warp
import matplotlib.pyplot as plt
import japanize_matplotlib
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box, Point
import contextily as ctx  # OpenStreetMap表示用
from matplotlib.patches import Patch  # 凡例用
from matplotlib_scalebar.scalebar import ScaleBar  # スケールバー用
from pathlib import Path
import shapely.ops
from rasterio.plot import show
from matplotlib.lines import Line2D
import open3d as o3d

SASEBO_SLOPE_DEM = "result/slope/slope_佐世保市.tif"

ICHISITEI = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/位置指定道路.shp" # 'R22_003'列が「主路線・道路種別コード」で1,2,3,4,6が国道・県道・高速道路
SHIDO = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/市道(路線)/路線_列改訂版.shp" # 幅員：'平均幅員(m', 延長：'延長(m)'
DRM_ROAD = "/Users/sakamo/Desktop/GISDATA/DRM_data/shape/全道路リンク標高.shp"
SASEBO_COMMUNITY = "/Users/sakamo/Desktop/GISDATA/自治協議会/sasebo_steep_community.shp"
SASEBO_HOUSING_AREA = (
    "/Users/sakamo/Desktop/GISDATA/住居系用途地域2019_10万人以上の自治体_全国/A29-19_42/housing_佐世保市.shp"
)

GOGO = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/4m以上の道路バッファ/1項5号.shp"
SHITEI_FOR_BUFFER = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/車両通行可能な道路バッファ/shitei.shp"
SANKO = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/4m以上の道路バッファ/3項.shp"
SHIDO_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/市道_for_use.shp"
KENDO_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/県道_for_use.shp"
KOKUDO_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/国道_for_use.shp"
ICHISITEI_FOR_USE = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/位置指定道路_for_use.shp"

ICHISITEI_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/位置指定道路_for_use.shp"
SHIDO_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/道路ポリゴン/市道.shp"
KENDO_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/道路ポリゴン/県道.shp"
KOKUDO_POLYGON = "/Users/sakamo/Desktop/GISDATA/佐世保市_道路/道路ポリゴン/国道.shp"

# LAS: epsg:6669
# 道路: epsg:6669


TERRAIN_LAS = "/Users/sakamo/Desktop/01JE4523.pcd"


if __name__ == '__main__':
    # 道路のデータを読み込む
    road_data = gpd.read_file(SHIDO) # EPSG:6669
    terrain = o3d.io.read_point_cloud(TERRAIN_LAS)
    o3d.visualization.draw_geometries([terrain])