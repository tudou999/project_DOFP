import rasterio
from rasterio.windows import Window

def crop_tif_to_5000x5000(input_path, output_path):
    with rasterio.open(input_path) as src:
        # 获取图像基本信息
        width = src.width
        height = src.height
        count = src.count
        dtype = src.dtypes[0]
        crs = src.crs
        transform = src.transform
        metadata = src.meta.copy()

        # 检查是否足够大
        if width < 5000 or height < 5000:
            raise ValueError(f"输入图像尺寸 {width}x{height} 小于 5000x5000，无法裁剪")

        # 定义裁剪窗口：左上角 5000x5000
        window = Window(col_off=0, row_off=height - 5000, width=5000, height=5000)

        # 读取裁剪区域的数据
        data = src.read(window=window)

        # 更新元数据
        metadata.update({
            'width': 5000,
            'height': 5000,
            'transform': src.window_transform(window)  # 更新变换矩阵
        })

        # 写入裁剪后的图像
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(data)

        print(f"✅ 裁剪完成，结果保存至: {output_path}")

# 示例调用
input_tif = "./images/111.tif"
output_tif = "./image/222.tif"
crop_tif_to_5000x5000(input_tif, output_tif)