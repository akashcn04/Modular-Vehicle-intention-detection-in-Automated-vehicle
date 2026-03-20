import os
import random
import shutil

image_dir = "dataset/Bstld/images/train"
label_dir = "dataset/Bstld/labels/train"

val_img_dir = "dataset/Bstld/images/val"
val_lbl_dir = "dataset/Bstld/labels/val"

split_ratio = 0.2  # 20% validation

images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

random.shuffle(images)

val_count = int(len(images) * split_ratio)

val_images = images[:val_count]

for img in val_images:
    label = img.replace(".png", ".txt")

    shutil.move(
        os.path.join(image_dir, img),
        os.path.join(val_img_dir, img)
    )

    shutil.move(
        os.path.join(label_dir, label),
        os.path.join(val_lbl_dir, label)
    )

print(f"Moved {val_count} images to validation set")
