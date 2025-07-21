import rasterio

# 打开 TIFF 文件
with rasterio.open('/root/autodl-tmp/new/111.tif') as src:
    # 获取波段数量
    band_count = src.count
    print(f"图像包含 {band_count} 个波段")