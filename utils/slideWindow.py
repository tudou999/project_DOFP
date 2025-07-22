import os
import cv2
import time

def slice_image(
    image_path,
    preject_name,
    out_dir_all_images,
    sliceHeight=416,
    sliceWidth=416,
    overlap=0.1,
    slice_sep="_",
    overwrite=False,
    out_ext=".png",
):
    if len(out_ext) == 0:
        im_ext = "." + image_path.split(".")[-1]
    else:
        im_ext = out_ext

    t0 = time.time()
    image = cv2.imread(image_path)
    print("图片尺寸：", image.shape)

    image_name = os.path.basename(image_path).split('.')[0]
    win_h, win_w = image.shape[:2]

    dx = int((1.0 - overlap) * sliceWidth)
    dy = int((1.0 - overlap) * sliceHeight)

    out_dir_image = os.path.join(out_dir_all_images, preject_name)

    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            n_ims += 1

            if (n_ims % 100) == 0:
                print(f"已切分 {n_ims} 张小图")

            if y0 + sliceHeight > image.shape[0]:
                y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image.shape[1]:
                x = image.shape[1] - sliceWidth
            else:
                x = x0

            window_c = image[y : y + sliceHeight, x : x + sliceWidth]
            outpath = os.path.join(
                out_dir_image,
                image_name
                + slice_sep
                + str(y)
                + "_"
                + str(x)
                + "_"
                + str(sliceHeight)
                + "_"
                + str(sliceWidth)
                + "_"
                + str(win_w)
                + "_"
                + str(win_h)
                + im_ext,
            )
            if not os.path.exists(outpath):
                cv2.imwrite(outpath, window_c)
            elif overwrite:
                cv2.imwrite(outpath, window_c)
            else:
                print(f"文件 {outpath} 已存在，跳过该切片")

    print("切分得到小图数量:", n_ims, "小图高度:", sliceHeight, "小图宽度:", sliceWidth)
    print("切分", image_path, "耗时", time.time() - t0, "秒")
    print(
        f"{os.path.basename(image_path)} 的切片结果已保存至: {out_dir_image}"
    )
    return out_dir_image
