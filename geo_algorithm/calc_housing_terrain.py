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

HOUSING_HOKKAIDO = "/Users/sakamo/Desktop/住居系用途地域2019/北海道/"
HOUSING_TOHOKU = "/Users/sakamo/Desktop/住居系用途地域2019/東北/"
HOUSING_KANTO = "/Users/sakamo/Desktop/住居系用途地域2019/関東1/"
HOUSING_KANTO = "/Users/sakamo/Desktop/住居系用途地域2019/関東2/"
HOUSING_KANTO = "/Users/sakamo/Desktop/住居系用途地域2019/関東3/"
HOUSING_HOKURIKU = "/Users/sakamo/Desktop/住居系用途地域2019/北陸/"
HOUSING_CHUBU = "/Users/sakamo/Desktop/住居系用途地域2019/中部/"
HOUSING_KANSAI = "/Users/sakamo/Desktop/住居系用途地域2019/近畿/"
HOUSING_CHUGOKU = "/Users/sakamo/Desktop/住居系用途地域2019/中国/"
HOUSING_SHIKOKU = "/Users/sakamo/Desktop/住居系用途地域2019/四国/"
HOUSING_KYUSHU = "/Users/sakamo/Desktop/住居系用途地域2019/九州/"

DEM_HOKKAIDO = "/Users/sakamo/Desktop/GISDATA/DEM/北海道_DEM/北海道_DEM.tiff"
DEM_TOHOKU = "/Users/sakamo/Desktop/GISDATA/DEM/東北_DEM/東北_DEM.tiff"
DEM_KANTO = "/Users/sakamo/Desktop/GISDATA/DEM/関東_DEM/関東1_DEM.tiff"
DEM_KANTO = "/Users/sakamo/Desktop/GISDATA/DEM/関東_DEM/関東2_DEM.tiff"
DEM_KANTO = "/Users/sakamo/Desktop/GISDATA/DEM/関東_DEM/関東3_DEM.tiff"
DEM_HOKURIKU = "/Users/sakamo/Desktop/GISDATA/DEM/北陸_DEM/北陸_DEM.tiff"
DEM_CHUBU = "/Users/sakamo/Desktop/GISDATA/DEM/中部_DEM/中部_DEM.tiff"
DEM_KINKI = "/Users/sakamo/Desktop/GISDATA/DEM/近畿_DEM/近畿_DEM.tiff"
DEM_CHUGOKU = "/Users/sakamo/Desktop/GISDATA/DEM/中国_DEM/中国_DEM.tiff"
DEM_SHIKOKU = "/Users/sakamo/Desktop/GISDATA/DEM/四国_DEM/四国_DEM.tiff"
DEM_KYUSHU = "/Users/sakamo/Desktop/GISDATA/DEM/九州_DEM/九州_DEM.tiff"

# 地方ごとに、住居系用途地域内における傾斜度5度以上の面積割合を計算する。

if __name__ == "__main__":
    pass