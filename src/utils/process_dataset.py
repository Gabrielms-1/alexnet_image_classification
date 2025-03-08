import os
import random
import shutil

def move_images(images: list[str], source_path: str, target_path: str):
    os.makedirs(target_path, exist_ok=True)
    for image in images:
        shutil.copy2(os.path.join(source_path, image), os.path.join(target_path, image))

def process_dataset(dataset_path: str, output_path: str, split_ratio: list[float]):
    """
    Process the dataset and save it to the output path.
    """
    
    classes = os.listdir(dataset_path)
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        class_images_len = len(os.listdir(class_path))
        train_images_len = int(class_images_len * split_ratio[0])
        valid_images_len = int(class_images_len * split_ratio[1])
        test_images_len = class_images_len - train_images_len - valid_images_len
        
        train_images = random.sample(os.listdir(class_path), train_images_len)
        valid_images = random.sample(os.listdir(class_path), valid_images_len)
        test_images = random.sample(os.listdir(class_path), test_images_len)
        
        move_images(images=train_images, source_path=class_path, target_path=os.path.join(output_path, "train", class_name))
        move_images(images=valid_images, source_path=class_path, target_path=os.path.join(output_path, "valid", class_name))
        move_images(images=test_images, source_path=class_path, target_path=os.path.join(output_path, "test", class_name))



if __name__ == "__main__":
    dataset_path = "data/raw"
    output_path = "data/processed"
    split_ratio = [0.7, 0.15, 0.15]
    process_dataset(dataset_path=dataset_path, output_path=output_path, split_ratio=split_ratio)