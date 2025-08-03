import os
import sys
import argparse
import datetime
import numpy as np
import cv2

# 设置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

# 滑动窗口公用
from utils import slice_image
# 预测用函数
from utils import convert_coordinates, draw_predictions_on_image
# 分割用函数
from utils import convert_coordinates_seg, draw_segs_on_image

from ultralytics import YOLO

def get_exp_dir(base_dir, task):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = f"exp{now}"
    return os.path.join(base_dir, task, exp_dir)

def predict(
    images_dir=os.path.join(PROJECT_ROOT, "images"),
    im_ext=".tif",
    sliceHeight=1024,
    sliceWidth=1024,
    overlap=0.6,
    slice_sep='_',
    overwrite=False,
    out_ext='.png',
    model="yoloFan.pt",
    conf=0.25,
    iou=0.7,
    half=False,
    device=None,
    show=False,
    save=True,
    save_txt=True,
    save_conf=True,
    save_crop=False,
    hide_labels=False,
    hide_conf=False,
    max_det=300,
    vid_stride=False,
    line_width=None,
    visualize=False,
    augment=False,
    agnostic_nms=False,
    retina_masks=False,
    classes=None,
    boxes=True,
    iou_threshold=0.01,
    confidence_threshold=0.6,
    area_weight=5,
    class_labels=[0],
    class_names=["Fan",],
    task='det',
):
    if task == 'det':
        model = "yoloFan.pt"
        class_names=["Fan"]
        images_dir = os.path.join(images_dir, 'det_images')
    elif task == 'seg':
        model = "yoloPanel.pt"
        class_names=["Panel"]
        images_dir = os.path.join(images_dir, 'seg_images')

    # 统一时间戳目录
    exp_dir = get_exp_dir(os.path.join(PROJECT_ROOT, 'runs'), task)
    outdir_slice_ims = os.path.join(exp_dir, 'window')
    completed_output_path = os.path.join(exp_dir, 'FINAL')

    im_list = [z for z in os.listdir(images_dir) if z.lower().endswith(im_ext.lower())]

    if not os.path.exists(outdir_slice_ims):
        os.makedirs(outdir_slice_ims)
    else:
        import shutil
        shutil.rmtree(outdir_slice_ims)
        os.makedirs(outdir_slice_ims)
        print(f"{outdir_slice_ims} 已存在，原有内容将被覆盖！")

    for i, im_name in enumerate(im_list):
        im_path = os.path.join(images_dir, im_name)
        print("=========================== ", im_name, "--", i + 1, "/", len(im_list),
              " =========================== ")
        slice_image(
            im_path,
            outdir_slice_ims,
            sliceHeight=sliceHeight,
            sliceWidth=sliceWidth,
            overlap=overlap,
            slice_sep=slice_sep,
            overwrite=overwrite,
            out_ext=out_ext,
        )

    # 检测任务
    if task == 'det':
        det_results = os.path.join("runs/det", exp_dir)
        predict_shell = (
            'yolo predict model={} source={} project={} conf={} iou={} half={} device={} show={} '
            'save={} save_txt={} save_conf={} save_crop={} hide_labels={} hide_conf={} '
            'max_det={} vid_stride={} line_width={} visualize={} augment={} agnostic_nms={} '
            'retina_masks={} classes={} boxes={} project={}'.format(
                model,
                outdir_slice_ims,
                f'results/yolov8_detect',
                conf,
                iou,
                half,
                device,
                show,
                save,
                save_txt,
                save_conf,
                save_crop,
                hide_labels,
                hide_conf,
                max_det,
                vid_stride,
                line_width,
                visualize,
                augment,
                agnostic_nms,
                retina_masks,
                classes,
                boxes,
                det_results
            )
        )
        print('\n')
        print(f"预测命令: {predict_shell}")
        print('\n')

        os.system(predict_shell)

        txt_label_path = os.path.join(det_results, 'predict', 'labels')

        txt_regress_path_list = convert_coordinates(
            txt_label_path=txt_label_path,
            output_file_dir=completed_output_path,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            area_weight=area_weight,
            slice_sep=slice_sep,
            orgimg_dir=images_dir,
        )
        for txt_regress_path in txt_regress_path_list:
            image_name = os.path.basename(txt_regress_path).split('.')[0]
            image_path = os.path.join(images_dir, image_name + im_ext)
            draw_predictions_on_image(
                image_path=image_path,
                results_file_path=txt_regress_path,
                class_labels=class_labels,
                class_names=class_names,
                completed_output_path=completed_output_path,
            )
        print("===== 风机检测任务完成 =====")


    # 分割任务 outdir_slice_ims
    elif task == 'seg':
        yolo_model = YOLO(model)
        seg_results = os.path.join("runs/seg", exp_dir)
        results = yolo_model.predict(
            source=outdir_slice_ims,
            project=seg_results,
            name="predict",
            task='seg',  # 明确指定分割任务
            save=True,
            save_txt=True,
            save_conf=True,
            conf=conf,
            iou=iou,
            retina_masks=True,  # 确保生成高质量掩码
            device=device if device else None,
            exist_ok=True,  # 允许覆盖现有结果
            verbose=True  # 输出详细日志
        )

        # 获取实际保存目录
        print(f"[SUCCESS] 预测完成，结果保存在: {seg_results}")

        # 打印目录结构
        print("\n[DEBUG] 预测目录内容:")
        for item in os.listdir(seg_results):
            print(f"|- {item}")

        # === 分割掩码处理 ===
        print("\n" + "=" * 50)
        print("开始处理分割掩码")
        print("=" * 50)

        # 创建掩码提取目录
        mask_extract_dir = os.path.join(seg_results, "extracted_masks")
        os.makedirs(mask_extract_dir, exist_ok=True)

        # 直接从YOLO结果中提取掩码
        extracted_mask_files = []
        print(f"[INFO] 从 {len(results)} 个预测结果中提取掩码...")

        for i, r in enumerate(results):
            if r.masks is not None and len(r.masks) > 0:
                # 获取原始切片文件名
                source_path = r.path
                if source_path:
                    slice_name = os.path.basename(source_path)
                    slice_base = os.path.splitext(slice_name)[0]

                    # 合并该切片的所有掩码
                    combined_mask = None
                    for j, mask in enumerate(r.masks):
                        mask_data = mask.data[0].cpu().numpy()

                        if combined_mask is None:
                            combined_mask = mask_data
                        else:
                            combined_mask = np.logical_or(combined_mask, mask_data)

                    if combined_mask is not None:
                        # 保存合并后的掩码
                        mask_filename = f"{slice_base}_mask.png"
                        mask_path = os.path.join(mask_extract_dir, mask_filename)

                        # 转换为uint8格式并保存
                        mask_uint8 = (combined_mask * 255).astype(np.uint8)
                        cv2.imwrite(mask_path, mask_uint8)
                        extracted_mask_files.append(mask_path)
                        print(f"提取掩码: {mask_filename}")
                else:
                    print(f"[WARNING] 结果 {i} 缺少源文件路径信息")
            else:
                print(f"[INFO] 结果 {i} 没有检测到掩码")

        print(f"[SUCCESS] 成功提取 {len(extracted_mask_files)} 个掩码文件")

        # 查找其他可能的掩码文件
        existing_mask_files = []
        for root, _, files in os.walk(seg_results):
            for file in files:
                file_lower = file.lower()
                if ('mask' in file_lower or 'seg' in file_lower) and file_lower.endswith(('.png', '.tif')):
                    existing_mask_files.append(os.path.join(root, file))

        # 合并所有掩码文件
        all_mask_files = extracted_mask_files + existing_mask_files

        if not all_mask_files:
            raise FileNotFoundError(
                f"未找到任何掩码文件！\n"
                f"目录内容: {os.listdir(seg_results)}\n"
                f"请检查YOLO预测设置"
            )

        print(f"[SUCCESS] 总共找到 {len(all_mask_files)} 个掩码文件")
        print(f"第一个掩码文件: {all_mask_files[0]}")

        # === 调用掩码转换函数 ===
        try:
            mask_outputs = convert_coordinates_seg(
                mask_files=all_mask_files,
                output_dir=completed_output_path,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
                area_weight=area_weight,
                slice_sep=slice_sep,
                orgimg_dir=images_dir,
            )
            print(f"[SUCCESS] 掩码转换完成，输出到: {completed_output_path}")

        except Exception as e:
            print(f"[ERROR] 掩码转换失败: {str(e)}")
            raise

        # === 可视化分割结果 ===
        print("\n" + "=" * 50)
        print("开始生成可视化结果")
        print("=" * 50)

        for mask_path in mask_outputs:
            img_name = os.path.basename(mask_path).replace("_mask.tif", "")
            image_path = os.path.join(images_dir, img_name + im_ext)

            if os.path.exists(image_path):
                output_img_path = os.path.join(
                    completed_output_path, f"{img_name}_vis{im_ext}"
                )

                # 创建输出目录
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

                try:
                    draw_segs_on_image(
                        image_path=image_path,
                        mask_path=mask_path,
                        output_path=output_img_path,
                        color=(0, 255, 0),  # 光伏板显示为绿色
                        alpha=0.3
                    )
                    print(f"生成可视化: {img_name}_vis{im_ext}")
                except Exception as e:
                    print(f"[ERROR] 可视化失败 {img_name}: {str(e)}")
            else:
                print(f"[WARNING] 未找到原图: {image_path}")

        print("===== 光伏板分割任务完成 =====")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=os.path.join(PROJECT_ROOT, 'images'))
    parser.add_argument("--im_ext", type=str, default=".tif")
    parser.add_argument("--sliceHeight", type=int, default=640)
    parser.add_argument("--sliceWidth", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--slice_sep", type=str, default="_")
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--out_ext", type=str, default=".png")
    parser.add_argument("--model", type=str, default="yoloPanel.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", type=bool, default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save_txt", type=bool, default=True)
    parser.add_argument("--save_conf", type=bool, default=True)
    parser.add_argument("--save_crop", type=bool, default=False)
    parser.add_argument("--hide_labels", type=bool, default=False)
    parser.add_argument("--hide_conf", type=bool, default=False)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--vid_stride", type=bool, default=False)
    parser.add_argument("--line_width", type=float, default=None)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--augment", type=bool, default=False)
    parser.add_argument("--agnostic_nms", type=bool, default=False)
    parser.add_argument("--retina_masks", type=bool, default=False)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--boxes", type=bool, default=True)
    parser.add_argument("--iou_threshold", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--area_weight", type=float, default=5)
    parser.add_argument("--class_labels", type=int, nargs="+", default=[0])
    parser.add_argument("--class_names", type=str, nargs="+", default=["Fan"])
    parser.add_argument("--task", type=str, default='seg', choices=['det', 'seg'])
    args = parser.parse_args()
    predict(**vars(args))