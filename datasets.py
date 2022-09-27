import os
import zipfile
import tarfile
import shutil
import cv2

import numpy as np
import pandas as pd
import wget
from PIL import Image
from tqdm import tqdm

import xml.etree.ElementTree as ET

from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader

DATASET_DIR = 'dataset'

FRUIT_NAME = ['apple', 'papaya', 'mango']

def download_dataset(name, url, output_dir, zip_file, replace=False):
    """Downloads the dataset from the source, or does nothing if replace=False and the dataset already exists"""

    if replace or not os.path.exists(zip_file):
        print(f'Downloading {name} dataset...')
        wget.download(url, out=output_dir)

def extract_dataset(name, filename, output_dir):
    """extracts the dataset from the source"""
    print(f'Extracting {name} dataset...')

    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif filename.endswith('.tar.gz'):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        raise ValueError(f'Cannot extract dataset file. Unknown file type: {filename}')

    print(f'{name} dataset extracted.')

def save_dataset(dataset, image_folder, label_folder):

    for i, (img_path, boxes) in tqdm(enumerate(dataset), total=len(dataset), desc=f'Saving dataset to {image_folder} and {label_folder}'):

        img_extension = img_path.split('.')[-1]
        img_dest = os.path.join(image_folder, f'image_{i}.{img_extension}')
        label_dest = os.path.join(label_folder, f'image_{i}.txt')

        if img_path.startswith('_UPPER_'):
            img_path = img_path[7:]

            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img = img[:h//2]
            cv2.imwrite(img_dest, img)
        elif img_path.startswith('_LOWER_'):
            img_path = img_path[7:]

            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img = img[h//2:]
            cv2.imwrite(img_dest, img)
        else:
            shutil.copy(img_path, img_dest)

        boxes.to_csv(label_dest, sep=' ', header=False, index=False)

class MultifruitDataset(Dataset):

    download_url = 'https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/acfr-multifruit-2016.zip'

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir

        self.multifruit_dir = os.path.join(self.dataset_dir, 'acfr-fruit-dataset')
        self.zip_file = os.path.join(self.multifruit_dir, 'acfr-multifruit-2016.zip')


        if not os.path.exists(self.multifruit_dir):
            download_dataset('Multifruit', self.download_url, self.dataset_dir, self.zip_file)
            extract_dataset('Multifruit', self.zip_file, self.dataset_dir)

        # sort the filenames so that the images are in the same order as the annotations
        self.apple_imgs = list(sorted(os.listdir(os.path.join(self.multifruit_dir, 'apples', 'images'))))
        self.apple_annotations = list(sorted(os.listdir(os.path.join(self.multifruit_dir, 'apples', 'annotations'))))
        self.apple_h = 202
        self.apple_w = 308

        self.mango_imgs = list(sorted(os.listdir(os.path.join(self.multifruit_dir, 'mangoes', 'images'))))
        self.mango_annotations = list(sorted(os.listdir(os.path.join(self.multifruit_dir, 'mangoes', 'annotations'))))
        self.mango_h = 500
        self.mango_w = 500

    def apple_annotation_to_bounding_box(self, filename):
        """Converts a circle annotation to a bounding box annotation.
        """
        circles = pd.read_csv(filename)
        bounding_boxes = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])
        for index, row in circles.iterrows():
            bounding_boxes.loc[index] = [0, row['c-x'], row['c-y'], 2 * row['radius'], 2 * row['radius']]

        # normalize x, y, w, h to 0-1
        bounding_boxes['x'] = bounding_boxes['x'] / self.apple_w
        bounding_boxes['y'] = bounding_boxes['y'] / self.apple_h
        bounding_boxes['w'] = bounding_boxes['w'] / self.apple_w
        bounding_boxes['h'] = bounding_boxes['h'] / self.apple_h

        return bounding_boxes

    def mango_annotation_to_bounding_box(self, filename):
        annotations = pd.read_csv(filename)
        bounding_boxes = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])

        for index, row in annotations.iterrows():
            # annotation in the form of [x,y,dx,dy], but we need x and y to be the center of the box
            bounding_boxes.loc[index] = [2, row['x'] + row['dx'] / 2, row['y'] + row['dy'] / 2, row['dx'], row['dy']]

        # normalize x, y, w, h to 0-1
        bounding_boxes['x'] = bounding_boxes['x'] / self.mango_w
        bounding_boxes['y'] = bounding_boxes['y'] / self.mango_h
        bounding_boxes['w'] = bounding_boxes['w'] / self.mango_w
        bounding_boxes['h'] = bounding_boxes['h'] / self.mango_h

        return bounding_boxes


    def __len__(self):
        return len(self.apple_imgs) + len(self.mango_imgs)

    def __getitem__(self, idx):
        if idx < len(self.apple_imgs):
            img_path = os.path.join(self.multifruit_dir, 'apples', 'images', self.apple_imgs[idx])
            annotation_path = os.path.join(self.multifruit_dir, 'apples', 'annotations', self.apple_annotations[idx])
            bounding_boxes = self.apple_annotation_to_bounding_box(annotation_path)
        else:
            img_path = os.path.join(self.multifruit_dir, 'mangoes', 'images', self.mango_imgs[idx - len(self.apple_imgs)])
            annotation_path = os.path.join(self.multifruit_dir, 'mangoes', 'annotations', self.mango_annotations[idx - len(self.apple_imgs)])
            bounding_boxes = self.mango_annotation_to_bounding_box(annotation_path)

        return img_path, bounding_boxes

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
            download_dataset('Minneapple', self.download_url, self.dataset_dir, self.zip_file)
            extract_dataset('Minneapple', self.zip_file, self.dataset_dir)

        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.minneapple_dir, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(self.minneapple_dir, 'masks'))))

        # this lets us save the bounding boxes as we load them so that we don't have to re-calculate them from images
        self.bounding_boxes = [None] * len(self.imgs) * 2
        self.h = 1280
        self.w = 720

    def mask_to_bounding_boxes(self, mask_path, idx):
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

            # for even indices, only keep bounding boxes in the upper half of the image
            if idx % 2 == 0:
                if ymin > 0.5:
                    continue
                elif ymax > 0.5:
                    ymax = 0.5

                ymin *= 2
                ymax *= 2
            # for odd indices, only keep bounding boxes in the lower half of the image
            else:
                if ymax < 0.5:
                    continue
                elif ymin < 0.5:
                    ymin = 0.5

                ymin = (ymin - .5) * 2
                ymax = (ymax - .5) * 2

            # it also expects the x and y values to represent the center of the bounding box
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            boxes.append({'class': 0, 'x': x, 'y': y, 'w': xmax - xmin, 'h': ymax - ymin})

        bounding_boxes = pd.DataFrame(boxes)

        return bounding_boxes

    def __len__(self):
        return len(self.imgs) * 2 # for an upper and lower half of the image

    def __getitem__(self, idx):
        image_index = idx // 2
        if idx % 2 == 0:
            half = '_UPPER_'
        else:
            half = '_LOWER_'

        img_name = self.imgs[image_index]
        mask_name = self.masks[image_index]

        img_path = os.path.join(self.minneapple_dir, 'images', img_name)
        img_path = half + img_path

        mask_path = os.path.join(self.minneapple_dir, 'masks', mask_name)

        bounding_box = self.bounding_boxes[idx]
        if bounding_box is None:
            bounding_box = self.mask_to_bounding_boxes(mask_path, idx)
            self.bounding_boxes[idx] = bounding_box

        return img_path, bounding_box

    def clean(self, remove_zip=False):
        shutil.rmtree(self.detection_dir)
        if remove_zip:
            os.remove(self.zip_file)

        print('Minneapple Dataset cleaned.')


class MangoYoloDataset(Dataset):

    download_url = 'https://acquire.cqu.edu.au/ndownloader/files/26220632'

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir

        self.mango_dir = os.path.join(self.dataset_dir, 'MangoYOLO')
        self.voc_dir = os.path.join(self.mango_dir, 'VOCdevkit', 'VOC2007')

        self.zip_file = os.path.join(self.dataset_dir, 'MangoYOLO.zip')

        if not os.path.exists(self.mango_dir):
            os.makedirs(self.mango_dir)
            download_dataset('MangoYOLO', self.download_url, self.dataset_dir, self.zip_file)
            extract_dataset('MangoYOLO', self.zip_file, self.mango_dir)

        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.voc_dir, 'JPEGImages'))))
        self.annotations = list(sorted(os.listdir(os.path.join(self.voc_dir, 'Annotations'))))

        self.h = 512
        self.w = 612

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        annotation_name = self.annotations[idx]

        img_path = os.path.join(self.voc_dir, 'JPEGImages', img_name)
        annotation_path = os.path.join(self.voc_dir, 'Annotations', annotation_name)

        return img_path, self.annotation_to_bounding_boxes(annotation_path)

    def annotation_to_bounding_boxes(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []

        for object in root.findall('object'):
            name = object.find('name').text
            if name != 'M':
                continue

            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text) / self.w
            xmax = int(bndbox.find('xmax').text) / self.w
            ymin = int(bndbox.find('ymin').text) / self.h
            ymax = int(bndbox.find('ymax').text) / self.h

            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            boxes.append({'class': 2, 'x': x, 'y': y, 'w': xmax - xmin, 'h': ymax - ymin})

        bounding_boxes = pd.DataFrame(boxes)

        return bounding_boxes

    def clean(self, remove_zip=False):
        shutil.rmtree(self.mango_dir)
        if remove_zip:
            os.remove(self.zip_file)

        print('MangoYOLO Dataset cleaned.')

class PapayaDataset(Dataset):
    """
    Dataset class that represents the Papaya dataset. Todo: find a place to host it
    """

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.papaya_dir = os.path.join(self.dataset_dir, 'papaya_object_detection')
        self.zip_file = os.path.join(self.dataset_dir, 'papaya_object_detection.zip')

        if not os.path.exists(self.papaya_dir):
            if not os.path.exists(self.zip_file):
                raise Exception('Papaya dataset not found. You must manually download it, rename it "papaya_object_detection.zip", and place it in the datasets folder.')

            extract_dataset('Papaya', self.zip_file, self.dataset_dir)

        self.imgs = list(sorted(os.listdir(os.path.join(self.papaya_dir, 'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(self.papaya_dir, 'labels'))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        img_path = os.path.join(self.papaya_dir, 'images', img_name)
        label_path = os.path.join(self.papaya_dir, 'labels', label_name)

        bounding_boxes = pd.read_csv(label_path, header=None, names=['class', 'x', 'y', 'w', 'h'], sep=' ')

        return img_path, bounding_boxes

    def clean(self, remove_zip=False):
        shutil.rmtree(self.papaya_dir)
        if remove_zip:
            os.remove(self.zip_file)

        print('Papaya Dataset cleaned.')


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

    #datasets = [MultifruitDataset(), MinneappleDataset()]
    datasets = [MangoYoloDataset(), PapayaDataset(), MultifruitDataset(), MinneappleDataset()]

    yolov5_dataset = ConcatDataset(datasets)

    # the datasets are ordered by default, and we can't use Dataloaders b/c they can't handle dataframes,
    # ... so we generate a random permutation of the indices and use that to split the dataset
    random_indices = np.random.permutation(len(yolov5_dataset))

    train_dataset = []
    test_dataset = []

    # (Wang, Tang, & Whitty 2022)
    num_training_objects = [4000, 1000, 4000]
    num_training_objects_per_fruit = [0, 0, 0]
    num_testing_objects_per_fruit = [0, 0, 0]
    empty_images_in_train = 0
    empty_images_in_test = 0

    for index in tqdm(random_indices, desc='Splitting dataset', total=len(random_indices)):
        img_path, bounding_boxes = yolov5_dataset[index]

        num_objects = len(bounding_boxes)

        if num_objects == 0:
            # send 20% of images with no objects to the test set and the rest to training
            if np.random.random() < 0.2:
                test_dataset.append((img_path, bounding_boxes))
                empty_images_in_test += 1
            else:
                train_dataset.append((img_path, bounding_boxes))
                empty_images_in_train += 1
            continue

        fruit = int(bounding_boxes.iloc[0, 0])

        if num_training_objects_per_fruit[fruit] > num_training_objects[fruit]:
            test_dataset.append((img_path, bounding_boxes))
            num_testing_objects_per_fruit[fruit] += num_objects
        else:
            train_dataset.append((img_path, bounding_boxes))
            num_training_objects_per_fruit[fruit] += num_objects

    # test_percent = .2
    # train_size = int(len(yolov5_dataset) * (1 - test_percent))
    # test_size = len(yolov5_dataset) - train_size

    # train_dataset, test_dataset = random_split(yolov5_dataset, [train_size, test_size])

    save_dataset(train_dataset, image_train_dir, label_train_dir)
    save_dataset(test_dataset, image_test_dir, label_test_dir)

    for i in range(len(FRUIT_NAME)):
        print(f'{FRUIT_NAME[i]}: {num_training_objects_per_fruit[i]} training objects, {num_testing_objects_per_fruit[i]} testing objects')

    print(f'{empty_images_in_train} empty images in training set, {empty_images_in_test} empty images in test set')

    # remove the dataset folder to save space, but keep the zip file
    # for dataset in datasets:
    #     dataset.clean()