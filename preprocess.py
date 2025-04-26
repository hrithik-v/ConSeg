"""
Preprocess the dataset: rename files and organize them into images and masks directories.
"""
import os
import shutil

base_dir = "./dataset"
subdirs = ["train", "valid", "test"]

for subdir in subdirs:
    source_dir = os.path.join(base_dir, subdir)
    images_dir = os.path.join(source_dir, "images")
    masks_dir = os.path.join(source_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            new_filename = filename.replace('.rf.', '_')
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(source_dir, new_filename)
            os.rename(src_file, dst_file)
            # Place masks and images in their respective folders
            if "mask" in new_filename.lower() or "_mask" in new_filename.lower():
                shutil.move(dst_file, os.path.join(masks_dir, new_filename))
            else:
                shutil.move(dst_file, os.path.join(images_dir, new_filename))
    print(f"Files have been segregated and renamed in the '{subdir}' directory.")
