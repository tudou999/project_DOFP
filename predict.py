import os
import sys
import argparse
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from utils import slice_image, convert_coordinates, draw_predictions_on_image

def get_exp_dir(base_dir):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = f"exp{now}"
    return os.path.join(base_dir, exp_dir)

def predict(
    images_dir=os.path.join(PROJECT_ROOT, "images"),
    project_name="Fan",
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
):
    # 统一时间戳目录
    exp_dir = get_exp_dir(os.path.join(PROJECT_ROOT, 'runs'))
    outdir_slice_ims = os.path.join(exp_dir, 'window')
    output_file_dir = os.path.join(exp_dir, 'txt')
    completed_output_path = os.path.join(exp_dir, 'predict')

    im_list = [z for z in os.listdir(images_dir) if z.lower().endswith(im_ext.lower())]

    if not os.path.exists(os.path.join(outdir_slice_ims, project_name)):
        os.makedirs(os.path.join(outdir_slice_ims, project_name))
    else:
        import shutil
        shutil.rmtree(os.path.join(outdir_slice_ims, project_name))
        os.makedirs(os.path.join(outdir_slice_ims, project_name))
        print(f"{os.path.join(outdir_slice_ims, project_name)} 已存在，原有内容将被覆盖！")

    for i, im_name in enumerate(im_list):
        im_path = os.path.join(images_dir, im_name)
        print("=========================== ", im_name, "--", i + 1, "/", len(im_list), " =========================== ")
        slice_image(
            im_path,
            project_name,
            outdir_slice_ims,
            sliceHeight=sliceHeight,
            sliceWidth=sliceWidth,
            overlap=overlap,
            slice_sep=slice_sep,
            overwrite=overwrite,
            out_ext=out_ext,
        )

    yolov8_predict_results_path = os.path.join(PROJECT_ROOT, 'results', 'yolov8_detect', project_name)

    if os.path.exists(yolov8_predict_results_path):
        import shutil
        import logging
        shutil.rmtree(yolov8_predict_results_path)
        logging.warning(f"检测结果路径: {yolov8_predict_results_path} 已存在，原有内容将被覆盖！")

    predict_shell = (
        'yolo predict model={} source={} project={} name={} conf={} iou={} half={} device={} show={} '
        'save={} save_txt={} save_conf={} save_crop={} hide_labels={} hide_conf={} '
        'max_det={} vid_stride={} line_width={} visualize={} augment={} agnostic_nms={} '
        'retina_masks={} classes={} boxes={}'.format(
            model,
            os.path.join(outdir_slice_ims, project_name),
            f'results/yolov8_detect',
            project_name,
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
        )
    )
    print('\n')
    print(f"预测命令: {predict_shell}")
    print('\n')

    os.system(predict_shell)

    txt_label_path = os.path.join(yolov8_predict_results_path, 'labels')

    txt_regress_path_list = convert_coordinates(
        txt_label_path=txt_label_path,
        output_file_dir=os.path.join(output_file_dir, project_name),
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
            completed_output_path=os.path.join(completed_output_path, project_name),
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=os.path.join(PROJECT_ROOT, 'images'))
    parser.add_argument("--project_name", type=str, default="FanPanel_detect")
    parser.add_argument("--im_ext", type=str, default=".tif")
    parser.add_argument("--sliceHeight", type=int, default=1088)
    parser.add_argument("--sliceWidth", type=int, default=1088)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--slice_sep", type=str, default="_")
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--out_ext", type=str, default=".png")
    parser.add_argument("--model", type=str, default="yoloFan.pt")
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
    args = parser.parse_args()
    predict(**vars(args))