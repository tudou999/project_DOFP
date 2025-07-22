import os
import datetime

def get_yolo_style_dir(base_dir, task='detect'):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = f"exp{now}"
    return os.path.join(base_dir, task, exp_dir)