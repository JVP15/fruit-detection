import os
import zipfile
import tarfile
import shutil

import numpy as np
import pandas as pd
import wget
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, ConcatDataset, random_split

DATASET_DIR = 'dataset'

FRUIT_NAME = ['apple', 'papaya', 'pineapple']

def get_dataset(name, url, output_dir, replace=False):
    """
    Downloads the dataset from the and extracts it to the output directory.
    If the compressed dataset file already exists, it will not be downloaded again unless replace=True
    """

    filename = url.split('/')[-1]
    filename = os.path.join(output_dir, filename)

    if replace or not os.path.exists(filename):
        print(f'Downloading {name} dataset...')
        wget.download(url, out=output_dir)

    print(f'Extracting {name} dataset...')

    if  filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif  filename.endswith('.tar.gz'):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        raise ValueError(f'Cannot extract dataset file. Unknown file type: {filename}')

    print(f'{name} dataset downloaded.')

def save_dataset(dataset, image_folder, label_folder):
    for i, (img_path, boxes) in tqdm(enumerate(dataset), total=len(dataset), desc=f'Saving dataset to {image_folder} and {label_folder}'):
        img_extension = img_path.split('.')[-1]
        img_dest = os.path.join(image_folder, f'image_{i}.{img_extension}')
        label_dest = os.path.join(label_folder, f'image_{i}.txt')

        # moving the file instead of copying it to save disk space
        shutil.move(img_path, img_dest)
        boxes.to_csv(label_dest, sep=' ', header=False, index=False)

class MultifruitDataset(Dataset):

    download_url = 'https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/acfr-multifruit-2016.zip'

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir

        self.multifruit_dir = os.path.join(self.dataset_dir, 'acfr-fruit-dataset')
        self.zip_file = os.path.join(self.multifruit_dir, 'acfr-multifruit-2016.zip')


        if not os.path.exists(self.multifruit_dir):
            get_dataset(name='Multifruit', url=self.download_url, output_dir=self.dataset_dir)

        # sort the filenames so that the images are in the same order as the annotations
        self.imgs = list(sorted(os.listdir(os.path.join(self.multifruit_dir, 'apples', 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(self.multifruit_dir, 'apples', 'annotations'))))
        self.h = 202
        self.w = 308

    def annotation_to_bounding_box(self, filename):
        """Converts a circle annotation to a bounding box annotation.
        """
        circles = pd.read_csv(filename)
        bounding_boxes = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])
        for index, row in circles.iterrows():
            bounding_boxes.loc[index] = [0, row['c-x'], row['c-y'], 2 * row['radius'], 2 * row['radius']]

        # normalize x, y, w, h to 0-1
        bounding_boxes['x'] = bounding_boxes['x'] / self.w
        bounding_boxes['y'] = bounding_boxes['y'] / self.h
        bounding_boxes['w'] = bounding_boxes['w'] / self.w
        bounding_boxes['h'] = bounding_boxes['h'] / self.h

        return bounding_boxes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        annotation_name = self.annotations[idx]
        img_path = os.path.join(self.multifruit_dir, 'apples', 'images', img_name)
        annotation_path = os.path.join(self.multifruit_dir, 'apples', 'annotations', annotation_name)

        return img_path, self.annotation_to_bounding_box(annotation_path)

    def clean(self, remove_zip=False):
        shutil.rmtree(self.multifruit_dir)
        if remove_zip:
            os.remove(self.zip_file)

        print('Multifruit Dataset cleaned.')

class MinneappleDataset(Dataset):
    """
    Dataset class that represents the Minneapple dataset.
    Modified from:https://github.com/nicolaihaeni/MinneApple/blob/master/data/apple_dataset.py
    """
    download_url = 'https://conservancy.umn.edu/bitstream/handle/11299/206575/detection.tar.gz'

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir

        self.detection_dir = os.path.join(self.dataset_dir, 'detection')
        # only the training folder has masks, so we have to ignore the test folder
        self.minneapple_dir = os.path.join(self.detection_dir, 'train')
        self.zip_file = os.path.join(self.dataset_dir, 'detection.tar.gz')

        if not os.path.exists(self.minneapple_dir):
            get_dataset(name='Minneapple', url=self.download_url, output_dir=self.dataset_dir)

        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.minneapple_dir, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(self.minneapple_dir, 'masks'))))
        self.h = 1280
        self.w = 720

    def mask_to_bounding_box(self, mask_path):
        # Each color of mask corresponds to a different instance with 0 being the background
        mask = Image.open(
            mask_path)

        # Convert the PIL image to np array
        mask = np.array(mask)
        obj_ids = np.unique(mask)

        # Remove background id
        obj_ids = obj_ids[1:]

        # Split the color-encoded masks into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bbox coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []

        for ii in range(num_objs):
            pos = np.where(masks[ii])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue

            # Yolo-v5 expects the values to be normalized from 0 to 1
            xmin = np.clip(xmin, a_min=0, a_max=self.w) / self.w
            xmax = np.clip(xmax, a_min=0, a_max=self.w) / self.w
            ymin = np.clip(ymin, a_min=0, a_max=self.h) / self.h
            ymax = np.clip(ymax, a_min=0, a_max=self.h) / self.h

            # it also expects the x and y values to represent the center of the bounding box
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            boxes.append({'class': 0, 'x': x, 'y': y, 'w': xmax - xmin, 'h': ymax - ymin})

        bounding_boxes = pd.DataFrame(boxes)

        return bounding_boxes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        mask_name = self.masks[idx]
        img_path = os.path.join(self.minneapple_dir, 'images', img_name)
        mask_path = os.path.join(self.minneapple_dir, 'masks', mask_name)

        return img_path, self.mask_to_bounding_box(mask_path)

    def clean(self, remove_zip=False):
        shutil.rmtree(self.detection_dir)
        if remove_zip:
            os.remove(self.zip_file)

        print('Minneapple Dataset cleaned.')


if __name__ == '__main__':
    image_train_dir = os.path.join(DATASET_DIR, 'images', 'train')
    label_train_dir = os.path.join(DATASET_DIR, 'labels', 'train')
    image_test_dir = os.path.join(DATASET_DIR, 'images', 'test')
    label_test_dir = os.path.join(DATASET_DIR, 'labels', 'test')

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(image_test_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)

    datasets = [MultifruitDataset(), MinneappleDataset()]

    yolov5_dataset = ConcatDataset(datasets)

    test_percent = .2
    train_size = int(len(yolov5_dataset) * (1 - test_percent))
    test_size = len(yolov5_dataset) - train_size

    train_dataset, test_dataset = random_split(yolov5_dataset, [train_size, test_size])

    save_dataset(train_dataset, image_train_dir, label_train_dir)
    save_dataset(test_dataset, image_test_dir, label_test_dir)

    # remove the dataset folder to save space, but keep the zip file
    for dataset in datasets:
        dataset.clean()