import os
import sys
import numpy as np
import cv2
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

from .nms import apply_nms
import rasterio

def convert_coordinates(
    txt_label_path, output_file_dir, iou_threshold, confidence_threshold, area_weight, slice_sep, orgimg_dir
):
    # txt_file_path: 存放 YOLOv8 小图检测结果的 TXT 文件的上级路径
    # output_file_path: 变换后的结果存放的 TXT 文件路径
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
        print(f"已创建文件夹 {output_file_dir}")
    output_lines = dict()  # 存储转换后的结果

    # orgimg_w = 0
    # orgimg_h = 0
    # 遍历文件夹中的每个 TXT 文件
    # 格式为：类别、框的x中心点、框的y中心点、宽、高、置信度
    # 名字格式：y、x、切片大小、原图大小
    for root, dirs, files in os.walk(txt_label_path):
        for index, filename in enumerate(files):
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)

                # 解析文件名中的信息
                slice_info = filename.split(".")[0].split(slice_sep)
                y0 = int(slice_info[-6])
                x0 = int(slice_info[-5])
                sliceHeight = int(slice_info[-4])
                sliceWidth = int(slice_info[-3])
                orgimg_w = int(slice_info[-2])
                orgimg_h = int(slice_info[-1])

                exclude_imgname_char = slice_sep + str(y0) + slice_sep + str(x0) + slice_sep + str(sliceHeight) + \
                slice_sep + str(sliceWidth) + slice_sep + str(orgimg_w) + slice_sep + str(orgimg_h)
                exclude_imgname_index = filename.split(".")[0].index(exclude_imgname_char)
                imgname = filename.split(".")[0][:exclude_imgname_index]

                # 读取小图检测结果的坐标信息
                with open(filepath, "r") as f:
                    lines = f.readlines()

                # 将边界框坐标转换到原图的坐标空间，并将结果存储到列表中
                converted_lines = []
                for line in lines:
                    class_label, x, y, w, h, conf = line.strip().split(" ")
                    x = float(x) * sliceWidth
                    y = float(y) * sliceHeight
                    w = float(w) * sliceWidth
                    h = float(h) * sliceHeight

                    x_in_original = float(x) + x0
                    y_in_original = float(y) + y0
                    w_in_original = float(w)
                    h_in_original = float(h)

                    # 计算中心点经纬度
                    tif_path = os.path.join(orgimg_dir, imgname + ".tif")
                    if os.path.exists(tif_path):
                        with rasterio.open(tif_path) as src:
                            lon, lat = src.transform * (x_in_original, y_in_original)
                    else:
                        lon, lat = 0.0, 0.0  # 若找不到tif，填0

                    converted_line = [
                        int(class_label),
                        x_in_original,
                        y_in_original,
                        w_in_original,
                        h_in_original,
                        float(conf),
                        float(lon),
                        float(lat)
                    ]
                    converted_lines.append(converted_line)

                # 将转换后的结果添加到输出列表中
                if imgname not in output_lines.keys():
                    output_lines[imgname] = converted_lines
                else:
                    output_lines[imgname].extend(converted_lines)
                    
    # print(f'orgimg_w-------{orgimg_w}')
    # print(f'orgimg_h-------{orgimg_h}')

    outputs_file_path_list = []
    for key, value in output_lines.items():
        nms_output_lines = apply_nms(
            value,
            iou_threshold,
            confidence_threshold,
            area_weight,
        )

        # 将转换后的结果写入输出文件
        output_file_path = os.path.join(output_file_dir, f"{key}.txt")
        if os.path.exists(output_file_path):
            # import shutil
            import logging
            os.remove(output_file_path)
            logging.warning(f"图片 {key} 的预测txt结果已存在，原内容将被覆盖！")

        with open(output_file_path, "w") as f:
            for line in nms_output_lines:
                # line: [class, x, y, w, h, conf, lon, lat]
                if isinstance(line, (list, tuple)) and len(line) >= 8:
                    f.write(f"{line[0]} {line[1]:.2f} {line[2]:.2f} {line[3]:.2f} {line[4]:.2f} {line[5]:.2f} {line[6]:.4f} {line[7]:.4f}\n")
                else:
                    f.write(str(line))
        print(f"图片 {key} 的预测txt结果已保存至: {output_file_path}")
        outputs_file_path_list.append(output_file_path)

    return outputs_file_path_list

def convert_coordinates_seg(
        mask_files,
        output_dir,
        iou_threshold,
        confidence_threshold,
        area_weight,
        slice_sep,
        orgimg_dir
):
    """
    处理分割任务的掩码转换：
    1. 将切片掩码拼接回完整大图
    2. 应用掩码级NMS处理重叠区域
    3. 优化重叠区域的处理逻辑
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 存储每张大图的完整掩码
    full_masks = {}

    # 预先扫描原始图像文件，构建原图名映射
    org_images = {}
    for img_file in os.listdir(orgimg_dir):
        if img_file.endswith(('.tif', '.tiff', '.jpg', '.png')):
            base_name = os.path.splitext(img_file)[0]
            # 移除可能的数字后缀
            clean_name = re.sub(r'_\d+$', '', base_name)
            org_images[clean_name] = base_name
            print(f"注册原图: {clean_name} -> {img_file}")

    print(f"[INFO] 开始处理 {len(mask_files)} 个掩码文件...")

    for idx, mask_path in enumerate(mask_files):
        print(f"[INFO] 处理掩码 {idx + 1}/{len(mask_files)}: {os.path.basename(mask_path)}")

        # 获取文件名（不含扩展名）
        filename = os.path.basename(mask_path)
        filename_without_ext = os.path.splitext(filename)[0]

        # 清理文件名：移除可能的 "_mask" 后缀
        clean_filename = filename_without_ext.replace("_mask", "")

        # 统一解析逻辑
        parts = clean_filename.split(slice_sep)
        if len(parts) < 7:
            # 尝试直接使用原始文件名解析
            parts = filename_without_ext.split(slice_sep)
            if len(parts) < 7:
                print(f"[ERROR] 无效文件名格式: {filename} - 需要至少7部分")
                continue

        try:
            # 最后6个元素总是位置/尺寸信息
            y0 = int(parts[-6])
            x0 = int(parts[-5])
            slice_h = int(parts[-4])
            slice_w = int(parts[-3])
            org_w = int(parts[-2])
            org_h = int(parts[-1])

            # 原图名 = 除最后6部分外的所有部分
            raw_imgname = slice_sep.join(parts[:-6])

            # 清理原图名：移除可能的数字后缀
            imgname = re.sub(r'_\d+$', '', raw_imgname)

            # 尝试匹配注册的原图名
            matched = False
            for prefix in org_images:
                if imgname.startswith(prefix):
                    imgname = prefix
                    matched = True
                    break

            if not matched and imgname in org_images:
                matched = True

            print(f"解析成功: {filename} -> "
                  f"原图:{imgname}, 位置:({y0},{x0}), "
                  f"切片尺寸:{slice_w}x{slice_h}, 原图尺寸:{org_w}x{org_h}")

        except (ValueError, IndexError) as e:
            print(f"文件名解析错误: {filename} - {str(e)}")
            continue

        # 加载模型输出的掩码
        try:
            # 使用OpenCV加载
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("加载失败")

            # 二值化处理
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            print(f"成功加载掩码: {filename}, 尺寸: {mask.shape}")
        except Exception as e:
            print(f"加载掩码失败: {mask_path} - {e}")
            continue

        # 初始化或获取全尺寸掩码
        if imgname not in full_masks:
            # 使用解析出的原图尺寸或实际原图尺寸
            actual_org_w, actual_org_h = org_w, org_h

            # 检查实际原图尺寸
            org_img_path = os.path.join(orgimg_dir, f"{org_images.get(imgname, imgname)}.tif")
            if os.path.exists(org_img_path):
                with rasterio.open(org_img_path) as src:
                    actual_org_h, actual_org_w = src.height, src.width
                    if abs(actual_org_w - org_w) > 100 or abs(actual_org_h - org_h) > 100:
                        print(f"尺寸不匹配: 解析尺寸({org_w}x{org_h}) vs 实际尺寸({actual_org_w}x{actual_org_h})")

            full_masks[imgname] = {
                "mask": np.zeros((actual_org_h, actual_org_w), dtype=np.uint8),
                "dims": (actual_org_w, actual_org_h),
                "count": 0,
                "overlap_count": np.zeros((actual_org_h, actual_org_w), dtype=np.uint8)  # 记录重叠次数
            }

        full_data = full_masks[imgname]
        full_mask = full_data["mask"]
        overlap_count = full_data["overlap_count"]
        actual_org_w, actual_org_h = full_data["dims"]

        # 增强边界检查
        y0 = max(0, min(y0, actual_org_h - 1))
        x0 = max(0, min(x0, actual_org_w - 1))

        y_end = min(y0 + mask.shape[0], actual_org_h)
        x_end = min(x0 + mask.shape[1], actual_org_w)
        actual_h = y_end - y0
        actual_w = x_end - x0

        # 检查有效性
        if actual_h <= 0 or actual_w <= 0:
            print(f"跳过无效位置: {filename} - 位置({y0},{x0}) "
                  f"切片尺寸({mask.shape[0]}x{mask.shape[1]}) "
                  f"原图尺寸({actual_org_w}x{actual_org_h})")
            continue

        # 优化的重叠区域处理
        try:
            roi = full_mask[y0:y_end, x0:x_end]
            mask_roi = mask[:actual_h, :actual_w]
            overlap_roi = overlap_count[y0:y_end, x0:x_end]

            # 更新重叠计数
            overlap_roi[mask_roi > 0] += 1

            # 使用加权平均合并重叠区域
            # 对于重叠区域，使用平均值来平滑结果
            combined = np.where(
                overlap_roi > 1,
                (roi + mask_roi) // 2,  # 重叠区域取平均
                np.maximum(roi, mask_roi)  # 非重叠区域取最大值
            )

            full_mask[y0:y_end, x0:x_end] = combined
            overlap_count[y0:y_end, x0:x_end] = overlap_roi
            full_data["count"] += 1

            print(f"成功合并掩码: {filename}")
        except Exception as e:
            print(f"合并掩码失败: {filename} - {e}")
            continue

    # 应用NMS并保存结果
    output_files = []
    for imgname, data in full_masks.items():
        if data["count"] == 0:
            print(f"[WARNING] 未找到 {imgname} 的掩码")
            continue

        full_mask = data["mask"]
        org_w, org_h = data["dims"]
        print(f"处理图像: {imgname}, 合并了 {data['count']} 个切片")

        # 后处理：移除小的噪声区域
        try:
            # 形态学操作去除小噪点
            kernel = np.ones((3, 3), np.uint8)
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

            # 应用NMS
            final_mask = apply_mask_nms(
                full_mask,
                iou_threshold,
                confidence_threshold,
                area_weight
            )
            print(f"NMS处理完成: {imgname}")
        except Exception as e:
            print(f"[ERROR] NMS处理失败: {str(e)}")
            final_mask = full_mask

        # 保存最终掩码
        output_path = os.path.join(output_dir, f"{imgname}_mask.tif")
        try:
            # 使用rasterio保存地理信息
            org_img_path = os.path.join(orgimg_dir, f"{org_images.get(imgname, imgname)}.tif")
            if os.path.exists(org_img_path):
                with rasterio.open(org_img_path) as src:
                    profile = src.profile
                    profile.update(
                        dtype=rasterio.uint8,
                        count=1,
                        nodata=0
                    )
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(final_mask.astype(np.uint8), 1)
            else:
                # 如果没有原图，保存为普通TIFF
                cv2.imwrite(output_path, final_mask)

            print(f"已保存 {imgname} 的拼接掩码至: {output_path}")
            output_files.append(output_path)
        except Exception as e:
            print(f"保存掩码失败: {output_path} - {e}")

    return output_files

def mask_to_txt(mask_path, output_txt_path, class_id=0):
    """
    将掩码文件转换为txt格式（分割轮廓点）
    
    Args:
        mask_path: 掩码文件路径 (.tif, .png等)
        output_txt_path: 输出的txt文件路径
        class_id: 类别ID，默认为0
    """
    try:
        # 读取掩码文件
        if mask_path.endswith('.tif') or mask_path.endswith('.tiff'):
            # 使用rasterio读取地理信息文件
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # 读取第一个波段
        else:
            # 使用OpenCV读取其他格式
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"[ERROR] 无法读取掩码文件: {mask_path}")
            return False
            
        # 二值化处理
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        
        # 获取图像尺寸用于归一化
        img_height, img_width = mask.shape
        
        # 写入txt文件
        with open(output_txt_path, 'w') as f:
            for contour in contours:
                # 简化轮廓点（减少点的数量）
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # 提取轮廓点坐标
                points = []
                for point in approx_contour:
                    x, y = point[0]
                    # 归一化坐标到[0,1]范围
                    x_norm = x / img_width
                    y_norm = y / img_height
                    points.extend([x_norm, y_norm])
                
                # 写入分割格式：class_id x1 y1 x2 y2 x3 y3 ...
                if len(points) >= 6:  # 至少需要3个点（6个坐标值）
                    line = f"{class_id} " + " ".join([f"{p:.6f}" for p in points])
                    f.write(line + "\n")
        
        print(f"[SUCCESS] 掩码已转换为分割txt格式: {output_txt_path}")
        print(f"[INFO] 找到 {len(contours)} 个轮廓")
        return True
        
    except Exception as e:
        print(f"[ERROR] 掩码转txt失败: {str(e)}")
        return False


def convert_masks_to_txt(mask_files, output_dir, class_id=0):
    """
    批量将掩码文件转换为txt格式
    
    Args:
        mask_files: 掩码文件路径列表
        output_dir: 输出目录
        class_id: 类别ID，默认为0
    
    Returns:
        list: 成功转换的txt文件路径列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    converted_files = []
    
    for mask_path in mask_files:
        # 生成对应的txt文件名
        mask_name = os.path.basename(mask_path)
        txt_name = os.path.splitext(mask_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)
        
        # 转换掩码为txt
        if mask_to_txt(mask_path, txt_path, class_id):
            converted_files.append(txt_path)
    
    print(f"[SUCCESS] 成功转换 {len(converted_files)} 个掩码文件为txt格式")
    return converted_files