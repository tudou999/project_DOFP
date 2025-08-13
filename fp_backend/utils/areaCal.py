import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj
import numpy as np

def calculate_mask_area(tif_path, mask_pixel_points):
    """
    计算TIFF图像中光伏板掩码的真实世界面积。

    参数:
    tif_path (str): 输入TIFF图像的路径。
    mask_pixel_points (list of tuples): 掩码的多边形顶点，格式为 [(col1, row1), (col2, row2), ...]。
                                        这里的col和row是图像的像素坐标。

    返回:
    float: 掩码在现实世界中的面积（平方米）。如果无法计算，返回None。
    """
    try:
        with rasterio.open(tif_path) as src:
            # 1. 获取TIFF图像的CRS和地理变换信息
            image_crs = src.crs
            image_transform = src.transform
            print(f"图像CRS: {image_crs}")
            print(f"图像地理变换: {image_transform}")

            # 2. 将掩码的像素坐标转换为世界坐标
            # rasterio.transform.xy 可以将像素行列号转换为世界坐标 (x, y)
            # x 通常是经度或东向坐标，y 通常是纬度或北向坐标
            world_coords = []
            for col, row in mask_pixel_points:
                x_world, y_world = rasterio.transform.xy(image_transform, row, col)  # 注意：xy函数通常接受 (row, col)
                world_coords.append((x_world, y_world))
            print(f"转换后的世界坐标（部分）: {world_coords[:5]}")

            # 3. 创建Shapely多边形对象
            # Shapely Polygon 接受 (x, y) 坐标列表
            polygon_geographic = Polygon(world_coords)
            print(f"Shapely多边形创建成功。")

            # 4. (如果需要) 重投影到等面积CRS并计算面积
            # 检查图像CRS是否是地理坐标系统 (如WGS84, EPSG:4326)
            if image_crs.is_geographic:
                print("图像CRS是地理坐标系统，需要重投影以准确计算面积。")
                # 确定一个合适的投影坐标系统
                # 对于全球范围，可以使用ESRI:54009 (World_Mollweide) 等面积投影，
                # 但对于小区域，通常使用UTM（通用横轴墨卡托）投影更精确，因为它以米为单位。
                # UTM区域的选择取决于多边形的中心点经度。

                # 获取多边形的质心，用于确定UTM区域
                centroid_lon = polygon_geographic.centroid.x
                utm_zone = int((centroid_lon + 180) / 6) + 1

                # 判断南北半球，确定UTM EPSG代码
                # 如果纬度为负（南半球），则在EPSG代码后面加600，例如327XX
                # 如果纬度为正（北半球），则为326XX
                centroid_lat = polygon_geographic.centroid.y
                if centroid_lat >= 0:
                    utm_epsg = f"EPSG:326{utm_zone}"  # 北半球
                else:
                    utm_epsg = f"EPSG:327{utm_zone}"  # 南半球

                print(f"根据质心 ({centroid_lon:.2f}, {centroid_lat:.2f}) 确定UTM区域: {utm_epsg}")

                # 定义投影转换器
                # from_crs(source_crs, target_crs, always_xy=True) 确保输入输出都是 (x, y) 顺序
                project = pyproj.Transformer.from_crs(
                    image_crs,
                    pyproj.CRS(utm_epsg),
                    always_xy=True
                ).transform

                # 应用重投影
                polygon_projected = transform(project, polygon_geographic)
                print("多边形已重投影到UTM坐标系。")

                # 计算面积
                area_sq_m = polygon_projected.area
                print(f"计算面积成功。")
                return area_sq_m

            elif image_crs.is_projected and image_crs.linear_units == 'metre':
                print("图像CRS已经是投影坐标系统且单位是米，直接计算面积。")
                # 如果图像本身就是投影坐标系且单位是米，可以直接计算面积
                area_sq_m = polygon_geographic.area
                return area_sq_m
            else:
                print(f"图像CRS ({image_crs}) 既不是地理坐标系统也不是以米为单位的投影坐标系统，无法直接计算面积。")
                return None

    except rasterio.errors.RasterioIOError as e:
        print(f"错误：无法打开或读取TIFF文件。请检查文件路径和权限。错误信息: {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None


def read_segmentation_txt(txt_path):
    """
    读取分割轮廓点txt文件
    
    参数:
    txt_path (str): 分割轮廓点txt文件路径
    
    返回:
    list: 包含所有轮廓的列表，每个轮廓是一个点坐标列表 [(x1, y1), (x2, y2), ...]
    """
    contours = []
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 7:  # 至少需要class_id + 3个点(6个坐标)
                    continue
                    
                # 第一个是class_id，后面都是坐标
                class_id = int(parts[0])
                coordinates = parts[1:]
                
                # 将归一化坐标转换为像素坐标
                points = []
                for i in range(0, len(coordinates), 2):
                    if i + 1 < len(coordinates):
                        x_norm = float(coordinates[i])
                        y_norm = float(coordinates[i + 1])
                        points.append((x_norm, y_norm))
                
                if len(points) >= 3:  # 至少需要3个点
                    contours.append(points)
        
        print(f"成功读取 {len(contours)} 个轮廓")
        return contours
        
    except Exception as e:
        print(f"读取txt文件失败: {e}")
        return []


def calculate_area_from_txt(tif_path, txt_path):
    """
    从分割轮廓点txt文件计算光伏板面积
    
    参数:
    tif_path (str): 原始TIFF图像路径
    txt_path (str): 分割轮廓点txt文件路径
    
    返回:
    dict: 包含每个轮廓的面积信息
    """
    # 读取txt文件中的轮廓点
    contours = read_segmentation_txt(txt_path)
    
    if not contours:
        print("未找到有效的轮廓数据")
        return {}
    
    # 获取图像尺寸
    try:
        with rasterio.open(tif_path) as src:
            img_width = src.width
            img_height = src.height
            print(f"图像尺寸: {img_width} x {img_height}")
    except Exception as e:
        print(f"无法读取图像尺寸: {e}")
        return {}
    
    results = {}
    
    for i, contour in enumerate(contours):
        print(f"\n处理轮廓 {i+1}:")
        
        # 将归一化坐标转换为像素坐标
        pixel_points = []
        for x_norm, y_norm in contour:
            x_pixel = int(x_norm * img_width)
            y_pixel = int(y_norm * img_height)
            pixel_points.append((x_pixel, y_pixel))
        
        print(f"轮廓 {i+1} 有 {len(pixel_points)} 个点")
        print(f"像素坐标范围: X({min(p[0] for p in pixel_points)}-{max(p[0] for p in pixel_points)}), "
              f"Y({min(p[1] for p in pixel_points)}-{max(p[1] for p in pixel_points)})")
        
        # 计算面积
        area = calculate_mask_area(tif_path, pixel_points)
        
        if area is not None:
            results[f"contour_{i+1}"] = {
                "area_sq_m": area,
                "area_hectares": area / 10000,
                "area_sq_km": area / 1000000,
                "point_count": len(pixel_points)
            }
            print(f"轮廓 {i+1} 面积: {area:.2f} 平方米 ({area/10000:.4f} 公顷)")
        else:
            print(f"轮廓 {i+1} 面积计算失败")
    
    return results


def batch_calculate_areas(tif_dir, txt_dir, output_file=None):
    import os
    import json

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"在目录 {txt_dir} 中未找到txt文件")
        return

    all_results = {}
    total_area_sq_m = 0  # 总面积（平方米）
    total_area_hectares = 0  # 总面积（公顷）
    total_area_sq_km = 0  # 总面积（平方公里）

    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        base_name = os.path.splitext(txt_file)[0]
        tif_path = os.path.join(tif_dir, base_name + '.tif')
        if not os.path.exists(tif_path):
            for ext in ['.tiff', '.TIF', '.TIFF']:
                alt_tif_path = os.path.join(tif_dir, base_name + ext)
                if os.path.exists(alt_tif_path):
                    tif_path = alt_tif_path
                    break
        if not os.path.exists(tif_path):
            print(f"未找到与 {txt_file} 对应的TIFF文件")
            continue

        results = calculate_area_from_txt(tif_path, txt_path)
        if results:
            # 计算当前文件的总面积
            file_total = sum(r["area_sq_m"] for r in results.values())

            # 累加到总面积
            total_area_sq_m += file_total
            total_area_hectares += file_total / 10000
            total_area_sq_km += file_total / 1000000

            # 将当前文件结果添加到总结果中
            all_results[txt_file] = {
                "file_total_area_sq_m": file_total,
                "file_total_area_hectares": file_total / 10000,
                "file_total_area_sq_km": file_total / 1000000,
                "contours": results
            }
            print(f"文件 {txt_file} 处理完成，总面积: {file_total:.2f} 平方米")

    # 创建最终结果字典
    result_data = {
        "total_area_sq_m": total_area_sq_m,
        "total_area_hectares": total_area_hectares,
        "total_area_sq_km": total_area_sq_km,
        "files": all_results
    }

    # 输出结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    # 打印总面积
    print("\n总面积统计:")
    print(f"总计: {total_area_sq_m:.2f} 平方米")
    print(f"总计: {total_area_hectares:.4f} 公顷")
    print(f"总计: {total_area_sq_km:.6f} 平方公里")

    return result_data


# --- 使用示例 ---
if __name__ == "__main__":
    # 假设您的TIFF文件路径
    # 请替换为您的实际TIFF文件路径
    my_tif_file = "../runs/seg/exp20250803_2354/FINAL/3.tif"

    # 假设这是您示例分割得到的光伏板掩码的像素点
    # 这些点应该是构成多边形的顺序点，例如从左上角顺时针或逆时针
    # 这里的 (col, row) 对应图像的像素列和像素行
    # 这是一个简单的矩形示例，实际中可能更复杂
    example_mask_points = [
        (100, 50),  # 像素列，像素行
        (200, 50),
        (200, 150),
        (100, 150),
        (100, 50)  # 闭合多边形，第一个点和最后一个点相同
    ]

    # 为了使示例可运行，我们创建一个虚拟的GeoTIFF文件
    # 在实际使用中，您应该使用自己的真实GeoTIFF文件
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    import os

    # 创建一个虚拟的GeoTIFF文件
    dummy_tif_path = "dummy_georeferenced_image.tif"
    if not os.path.exists(dummy_tif_path):
        print(f"创建虚拟GeoTIFF文件: {dummy_tif_path}")
        # 假设图像分辨率为1米/像素，起始点在经度116.3，纬度39.9（北京附近）
        # 这是一个WGS84（EPSG:4326）地理坐标系的图像
        transform_matrix = from_origin(116.3, 39.9, 0.00001, 0.00001)  # 经度每像素0.00001度，纬度每像素0.00001度

        # 创建一个1000x1000像素的图像
        width, height = 1000, 1000
        # 图像数据，这里用全0填充
        data = np.zeros((1, height, width), dtype=np.uint8)

        new_dataset_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,  # 波段数
            'dtype': data.dtype,
            'crs': CRS.from_epsg(4326),  # WGS84
            'transform': transform_matrix,
            'nodata': 0  # 示例：将0设为无数据值
        }

        with rasterio.open(dummy_tif_path, 'w', **new_dataset_profile) as dst:
            dst.write(data)
        print("虚拟GeoTIFF文件创建完成。")

    my_tif_file = dummy_tif_path  # 将文件路径指向虚拟文件

    # 调用函数计算面积
    area = calculate_mask_area(my_tif_file, example_mask_points)

    if area is not None:
        print(f"\n光伏板掩码的现实世界面积约为: {area:.2f} 平方米")
        print(f"约合 {area / 10000:.2f} 公顷")
        print(f"约合 {area / 1000000:.2f} 平方公里")
    else:
        print("\n面积计算失败。")

    # 清理虚拟文件（可选）
    # os.remove(dummy_tif_path)
    # print(f"已删除虚拟GeoTIFF文件: {dummy_tif_path}")

    # 示例1: 处理单个txt文件
    print("示例1: 处理单个分割轮廓点txt文件")
    print("-" * 40)
    
    # 请替换为您的实际文件路径
    example_tif = "runs/seg/exp20250803_2354/FINAL/3.tif"
    example_txt = "runs/seg/exp20250803_2354/FINAL/3_mask.tif"
    
    if os.path.exists(example_tif) and os.path.exists(example_txt):
        results = calculate_area_from_txt(example_tif, example_txt)
        if results:
            total_area = sum(r["area_sq_m"] for r in results.values())
            print(f"\n总光伏板面积: {total_area:.2f} 平方米")
    
    # 示例2: 批量处理
    print("\n\n示例2: 批量处理多个txt文件")
    print("-" * 40)
    
    # 请替换为您的实际目录路径
    tif_directory = "path/to/your/tif/images"
    txt_directory = "path/to/your/txt/results"
    output_file = "area_calculation_results.json"
    
    if os.path.exists(tif_directory) and os.path.exists(txt_directory):
        batch_calculate_areas(tif_directory, txt_directory, output_file)
    else:
        print("请设置正确的目录路径")