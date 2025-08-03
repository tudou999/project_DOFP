import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

from skimage.measure import label, regionprops

def calculate_area(box):
    """
    计算边界框的面积
    box的格式：[xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    return area

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（Intersection over Union）
    box1和box2的格式：[xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算交集的坐标
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        # 两个边界框没有交集
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def nms_per_class(
    boxes, scores, classes, iou_threshold, confidence_threshold, area_weight
):
    """
    使用NMS对不同类别的边界框进行后处理
    boxes: 边界框列表，每个边界框的格式为 [xmin, ymin, xmax, ymax]
    scores: 每个边界框的置信度得分列表
    classes: 每个边界框的类别列表
    threshold: 重叠度阈值，高于该阈值的边界框将被抑制
    """
    # 过滤置信度低于阈值的边界框
    filtered_indices = np.where(np.array(scores) >= confidence_threshold)[0]
    boxes = [boxes[i] for i in filtered_indices]
    scores = [scores[i] for i in filtered_indices]
    classes = [classes[i] for i in filtered_indices]

    # 将边界框、置信度、类别转换为NumPy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    areas = np.array([calculate_area(box) for box in boxes])

    # 初始化空列表来存储保留的边界框索引
    keep_indices = []

    # 获取所有唯一的类别标签
    unique_classes = np.unique(classes)

    for cls in unique_classes:
        # 获取属于当前类别的边界框索引
        cls_indices = np.where(classes == cls)[0]

        # 根据当前类别的置信度得分和面积对边界框进行排序
        sorted_indices = np.lexsort(
            (scores[cls_indices], areas[cls_indices] * area_weight)
        )[::-1]
        # sorted_indices = np.argsort(areas[cls_indices])[::-1]
        cls_indices = cls_indices[sorted_indices]
        while len(cls_indices) > 0:
            # 选择当前得分最高的边界框
            current_index = cls_indices[0]
            current_box = boxes[current_index]
            keep_indices.append(filtered_indices[current_index])

            # 计算当前边界框与其他边界框的IoU
            other_indices = cls_indices[1:]
            ious = np.array(
                [calculate_iou(current_box, boxes[i]) for i in other_indices]
            )

            # 找到重叠度低于阈值的边界框索引
            low_iou_indices = np.where(ious < iou_threshold)[0]

            # 更新剩余边界框索引
            cls_indices = cls_indices[1:][low_iou_indices]

    return keep_indices


def apply_nms(
        outputs, iou_threshold, confidence_threshold, area_weight
    ):
        # 将边界框列表转换为NumPy数组
        outputs = np.array(outputs)

        boxes = []
        scores = []
        class_ids = []
        extra_params = []  # 用于保留第七、第八个参数
        for out in outputs:
            x = out[1]
            y = out[2]
            w = out[3]
            h = out[4]
            score = out[5]
            class_id = int(out[0])
            # 保留第七、第八个参数（假设为out[6], out[7]，如有更多可扩展）
            extra = out[6:8] if out.shape[0] > 7 else []

            # 计算边界
            left = float(x - w / 2)
            top = float(y - h / 2)
            right = float(x + w / 2)
            bottom = float(y + h / 2)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(score)
            boxes.append([left, top, right, bottom])
            extra_params.append(extra)

        # 应用NMS
        indices = nms_per_class(
            boxes=boxes,
            scores=scores,
            classes=class_ids,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            area_weight=area_weight,
        )

        # 选择通过NMS过滤后的边界框
        nms_out_lines = []
        for i in indices:
            # Get the box, score, class ID, and extra params corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            extra = extra_params[i]
            x = float(box[0] + (box[2] - box[0]) / 2)
            y = float(box[1] + (box[3] - box[1]) / 2)
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            # 拼接额外参数
            extra_str = " ".join(str(e) for e in extra) if len(extra) > 0 else ""
            nms_out_line = f"{class_id} {x} {y} {w} {h} {score}"
            if extra_str:
                nms_out_line += f" {extra_str}"
            nms_out_line += "\n"
            nms_out_lines.append(nms_out_line)
        return nms_out_lines


def apply_mask_nms(
        mask,
        iou_threshold=0.5,
        confidence_threshold=0.25,
        area_weight=1.0
):
    """
    对掩码应用NMS处理：
    1. 标记连通区域
    2. 根据置信度和面积过滤
    3. 基于掩码IoU进行非极大值抑制
    4. 优化小区域处理
    """
    # 预处理：确保掩码是二值的
    binary_mask = (mask > 0).astype(np.uint8)

    # 标记连通区域
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)

    # 提取区域属性
    boxes = []
    masks = []
    scores = []
    areas = []

    print(f"[INFO] 找到 {len(regions)} 个连通区域")

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        bbox = [minc, minr, maxc, maxr]  # xmin, ymin, xmax, ymax
        region_mask = (labeled_mask == region.label)

        # 计算区域面积
        area = region.area

        # 使用区域面积作为置信度代理，但考虑面积权重
        conf = (area / mask.size) * area_weight

        # 过滤太小的区域
        min_area = mask.size * 0.0001  # 最小面积阈值
        if area < min_area:
            print(f"[INFO] 过滤小区域: 面积={area}, 阈值={min_area}")
            continue

        if conf < confidence_threshold:
            print(f"[INFO] 过滤低置信度区域: conf={conf:.4f}, 阈值={confidence_threshold}")
            continue

        boxes.append(bbox)
        masks.append(region_mask)
        scores.append(conf)
        areas.append(area)
        print(f"[INFO] 保留区域: 面积={area}, 置信度={conf:.4f}")

    if not boxes:
        print("[INFO] 没有符合条件的区域，返回空掩码")
        return np.zeros_like(mask, dtype=np.uint8)

    print(f"[INFO] 开始NMS处理 {len(boxes)} 个区域...")

    # 按置信度排序
    sorted_indices = np.argsort(scores)[::-1]

    # NMS处理
    keep = []
    while sorted_indices.size > 0:
        i = sorted_indices[0]
        keep.append(i)

        if sorted_indices.size == 1:
            break

        # 计算当前掩码与其他掩码的IoU
        current_mask = masks[i]
        ious = []
        for j in sorted_indices[1:]:
            other_mask = masks[j]
            intersection = np.logical_and(current_mask, other_mask).sum()
            union = np.logical_or(current_mask, other_mask).sum()
            iou = intersection / (union + 1e-7)
            ious.append(iou)

        # 保留IoU低于阈值的索引
        ious = np.array(ious)
        low_iou_indices = np.where(ious <= iou_threshold)[0]
        sorted_indices = sorted_indices[1:][low_iou_indices]

    print(f"[INFO] NMS后保留 {len(keep)} 个区域")

    # 创建最终掩码
    final_mask = np.zeros_like(mask, dtype=np.uint8)
    for idx in keep:
        final_mask[masks[idx]] = 255  # 使用255而不是1，便于可视化

    return final_mask