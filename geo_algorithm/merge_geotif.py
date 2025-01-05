import glob

import numpy as np
import rasterio
from rasterio.merge import merge


def merge_in_batches(file_list, batch_size=10):
    mosaic = None
    out_trans = None

    # マージオプションを追加
    merge_options = {
        "method": "first",  # オーバーラップする場合の処理方法
        "bounds": None,  # 出力範囲の指定
        "res": None,  # 解像度の指定
        "nodata": -9999,  # NoDataの値を指定
    }

    for i in range(0, len(file_list), batch_size):
        batch = file_list[i : i + batch_size]
        src_files = [rasterio.open(fp) for fp in batch]

        temp_mosaic, temp_trans = merge(src_files, **merge_options)

        if mosaic is None:
            mosaic = temp_mosaic
            out_trans = temp_trans
        else:
            # 大きい方のサイズに合わせてパディング
            max_height = max(mosaic.shape[1], temp_mosaic.shape[1])
            max_width = max(mosaic.shape[2], temp_mosaic.shape[2])

            # 既存のモザイクをパディング
            padded_mosaic = np.full(
                (1, max_height, max_width), -9999, dtype=mosaic.dtype
            )
            padded_mosaic[:, : mosaic.shape[1], : mosaic.shape[2]] = mosaic

            # 新しいモザイクをパディング
            padded_temp = np.full(
                (1, max_height, max_width), -9999, dtype=temp_mosaic.dtype
            )
            padded_temp[:, : temp_mosaic.shape[1], : temp_mosaic.shape[2]] = temp_mosaic

            mosaic = np.maximum(padded_mosaic, padded_temp)

        # ファイルを閉じる
        for src in src_files:
            src.close()

    return mosaic, out_trans


# ファイルリストを取得
# file_list = glob.glob("/Users/sakamo/Desktop/GISDATA/DEM_全国/北海道*.tif")
file_list = [
    "/Users/sakamo/Desktop/GISDATA/DEM_backup/九州1.tif",
    "/Users/sakamo/Desktop/GISDATA/DEM_backup/九州2.tif",
    "/Users/sakamo/Desktop/GISDATA/DEM_backup/九州3.tif",
    "/Users/sakamo/Desktop/GISDATA/別府.tif",
]
# 入力ファイルの座標系が全て同じか確認
for fp in file_list:
    with rasterio.open(fp) as src:
        print(f"File: {fp}")
        print(f"CRS: {src.crs}")

# バッチ処理でマージ
mosaic, out_trans = merge_in_batches(file_list, batch_size=5)

# メタデータを更新
out_meta = rasterio.open(file_list[0]).meta.copy()
out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "nodata": -9999,
    }
)

# 結果を保存
with rasterio.open("/Users/sakamo/Desktop/output.tif", "w", **out_meta) as dest:
    dest.write(mosaic)
