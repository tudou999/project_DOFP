import os
import sys
import cv2
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
import rasterio

def to_absolute(value, dim):
    return round(value * dim) if value <= 1.0 else round(value)

def draw_predictions_on_image(
    image_path, results_file_path, class_labels, class_names, completed_output_path
):
    # 确保类别标签和类别名称数量一致
    assert len(class_labels) == len(
        class_names
    ), "类别标签数量应与类别名称数量一致。"

    # 定义类别对应的颜色
    colors = [
        (255, 0, 0),  # head: 红色
    ]

    # 定义类别标签到类别名称的映射
    label_map = dict(zip(class_labels, class_names))

    # 定义每个类别对应的颜色
    color_map = dict(zip(class_labels, colors))

    # OpenCV读取是BGR，tif一般是RGB，如果需要可转换
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 可选：如需保持RGB顺序

    # 读取存放变换结果的文本文件
    with open(results_file_path, 'r') as file:
        lines = file.readlines()

    # 遍历每行结果
    for line in lines:
        line = line.strip().split(' ')
        # 支持有经纬度的格式
        if len(line) >= 8:
            class_label, x, y, w, h, conf, lon, lat = map(float, line[:8])
            lonlat_str = f"{lon:.4f},{lat:.4f}"
        else:
            class_label, x, y, w, h, conf = map(float, line[:6])
            lonlat_str = ""

        # 计算边界框的坐标
        image_height, image_width, _ = image.shape
        abs_x = to_absolute(x, image_width)
        abs_y = to_absolute(y, image_height)
        abs_w = to_absolute(w, image_width)
        abs_h = to_absolute(h, image_height)
        x_min = abs_x - abs_w // 2
        y_min = abs_y - abs_h // 2
        x_max = abs_x + abs_w // 2
        y_max = abs_y + abs_h // 2

        # 获取类别名称和颜色
        class_name = label_map.get(int(class_label), "Unknown")
        color = color_map[int(class_label)]

        # 绘制边界框和类别标签
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            image,
            f"{class_name}: {conf:.2f}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        # 绘制经纬度
        if lonlat_str:
            # 解析lon,lat字符串为float
            try:
                lon_val, lat_val = map(float, lonlat_str.split(','))
                ns = 'N' if lat_val >= 0 else 'S'
                ew = 'E' if lon_val >= 0 else 'W'
                lat_fmt = f"{ns}:{abs(lat_val):.4f}"
                lon_fmt = f"{ew}:{abs(lon_val):.4f}"
                nswe_str = f"{lat_fmt} {lon_fmt}"
            except Exception:
                nswe_str = lonlat_str
            cv2.putText(
                image,
                nswe_str,
                (x_min, y_min - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # 蓝色
                2,
            )

    # 保存绘制结果
    filename = os.path.basename(image_path)
    if not os.path.exists(completed_output_path):
        os.makedirs(completed_output_path)

    output_image_path = os.path.join(completed_output_path, filename)
    if os.path.exists(output_image_path):
        import logging
        os.remove(output_image_path)
        logging.warning(f"图片 {filename} 的预测可视化结果已存在，原内容将被覆盖！")

    with rasterio.open(image_path) as src:
        profile = src.profile

    # (H, W, 3) -> (3, H, W)
    image_to_save = image.transpose(2, 0, 1)

    profile.update({
        "count": 3,
        "dtype": image_to_save.dtype
    })

    with rasterio.open(output_image_path, 'w', **profile) as dst:
        dst.write(image_to_save)

    print(f"图片 {filename} 的预测可视化结果已保存至: {output_image_path}")