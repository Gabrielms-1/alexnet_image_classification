import os
import cv2
import albumentations as A
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor

def generate_augmented_dataset():
    transform = A.Compose([
        A.Resize(width=224, height=224),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomBrightnessContrast(p=0.8, brightness_limit=0.2, contrast_limit=0.2),
        A.Rotate(limit=180, p=0.5, border_mode=cv2.BORDER_REPLICATE),
        A.RandomShadow(p=0.8, num_shadows_limit=(1, 4), shadow_intensity_range=(0.2, 0.5)),
        A.ElasticTransform(p=0.6, alpha=0.3, sigma=70),
        A.GridDistortion(p=0.8, distort_limit=0.3),
        A.OpticalDistortion(distort_limit=0.2, p=0.7),
        A.Sharpen(p=0.8, alpha=(0.2, 0.5), lightness=(0.2, 0.6)),
        A.Emboss(p=0.8, alpha=(0.1, 0.3), strength=(0.1, 0.3)),
        A.ImageCompression(quality_range=(70, 90), p=0.8),
    ])

    input_dir = "data/processed"
    output_dir = "data/augmented"

    os.makedirs(output_dir, exist_ok=True)

    jpg_images = glob.glob(os.path.join(input_dir, "**/*.jpg"), recursive=True)
    png_images = glob.glob(os.path.join(input_dir, "**/*.png"), recursive=True)
    jpeg_images = glob.glob(os.path.join(input_dir, "**/*.jpeg"), recursive=True)

    images = jpg_images + png_images + jpeg_images

    for image_path in images:
        image = cv2.imread(image_path)
        output_path = image_path.replace("/processed", "/augmented")
        basedir = os.path.dirname(output_path)
        os.makedirs(basedir, exist_ok=True)
        # for i in range(1): 
        augmented = transform(image=image)
        aug_image = augmented["image"]

        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(basedir, f"{img_name}_aug.jpg")

        cv2.imwrite(output_path, aug_image)

def copy_image(image_path, output_dir):
    output_path = image_path.replace("/augmented", "/full")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(image_path, output_path)

def merge_datasets():
    augmented_dir = "data/augmented"
    processed_dir = "data/processed"
    output_dir = "data/full"

    os.makedirs(output_dir, exist_ok=True)

    jpg_images = glob.glob(os.path.join(augmented_dir, "**/*.jpg"), recursive=True)
    png_images = glob.glob(os.path.join(augmented_dir, "**/*.png"), recursive=True)
    jpeg_images = glob.glob(os.path.join(augmented_dir, "**/*.jpeg"), recursive=True)

    processed_jpg_images = glob.glob(os.path.join(processed_dir, "**/*.jpg"), recursive=True)
    processed_png_images = glob.glob(os.path.join(processed_dir, "**/*.png"), recursive=True)
    processed_jpeg_images = glob.glob(os.path.join(processed_dir, "**/*.jpeg"), recursive=True)

    images = jpg_images + png_images + jpeg_images + processed_jpg_images + processed_png_images + processed_jpeg_images

    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(copy_image, images, [output_dir]*len(images))

if __name__ == "__main__":
    generate_augmented_dataset()
    #merge_datasets()
