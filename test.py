import rasterio

def compare_geoinfo(tif1_path, tif2_path):
    with rasterio.open(tif1_path) as src1, rasterio.open(tif2_path) as src2:
        # 1. 比较坐标参考系统 (CRS)
        crs1 = src1.crs
        crs2 = src2.crs
        crs_equal = crs1 == crs2
        print(f"CRS 相同: {crs_equal}")
        if not crs_equal:
            print(f"  - 文件1 CRS: {crs1}")
            print(f"  - 文件2 CRS: {crs2}")

        # 2. 比较仿射变换矩阵（决定了图像的几何变换和坐标）
        transform1 = src1.transform
        transform2 = src2.transform
        transform_equal = transform1 == transform2
        print(f"仿射变换相同: {transform_equal}")
        if not transform_equal:
            print(f"  - 文件1 变换: {transform1}")
            print(f"  - 文件2 变换: {transform2}")

        # 3. 比较图像边界范围（Bounding Box）
        bounds1 = src1.bounds
        bounds2 = src2.bounds
        bounds_equal = bounds1 == bounds2
        print(f"边界范围相同: {bounds_equal}")
        if not bounds_equal:
            print(f"  - 文件1 边界: {bounds1}")
            print(f"  - 文件2 边界: {bounds2}")

        # 总体判断
        if crs_equal and transform_equal and bounds_equal:
            print("✅ 所有地理坐标信息相同。")
        else:
            print("❌ 地理坐标信息不一致。")

# 示例调用
tif1 = '/root/autodl-tmp/new/111.tif'
tif2 = '/root/autodl-tmp/new/results/completed_predict/sensor_detect/111.tif'
compare_geoinfo(tif1, tif2)