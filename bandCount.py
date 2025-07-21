import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
images_dir = os.path.join(PROJECT_ROOT, "images"),

print(f"Images directory: {images_dir}")