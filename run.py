from re import T
# Debug flag 
# TODO # REMOVE
DEBUG = False

# Redownload data if true
DOWNLOAD_DATA = True

# Bool for changing between CV2 image preprocess and default preprocess
USING_CV2 = True

# Shape to resize the images after loading
IMAGE_SHAPE = (270, 480)

# Shape to resize the DNI after detecting it for the RCNN
DNI_IMAGE_SHAPE = (224,224)

# Minimum threshold for an image to be considered to have DNI in it
THRESHOLD_DNI_IN_IMAGE = 0.6

# Maximum threshold for a square with some DNI but not more than that %
THRESHOLD_GT_IN_MID_SQUARE = 0.7

# Load or save train images and labels
LOAD_TRAIN = True
SAVE_TRAIN = False

from typing import List, Tuple
import glob
import tqdm
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
plt.rcParams["figure.figsize"] = (20, 16)


classes = [
    "alb_id",
    "aze_passport",
    "esp_id",
    "est_id",
    "fin_id",
    "grc_passport",
    "lva_passport",
    "rus_internalpassport",
    "srb_passport",
    "svk_id",
]

def get_class(*, img_path: str, dataset: str):
    if dataset in ['midv500', 'midv2019']:
        dirname = img_path.split('/')[-4]
        return '_'.join(dirname.split('_')[1:3])
    else:
        dirname = img_path.split('/')[-2]
        return '_'.join(dirname.split('_')[:2])

def get_location(*, loc_path: str):
    return json.load(open(loc_path, 'r'))['quad']

def get_location_path(img_path: str, loc_dirname: str = 'ground_truth'):
    loc_path = img_path.replace('images', loc_dirname)
    loc_path = os.path.splitext(loc_path)[0]+'.json'
    return loc_path

def get_metadata(image_paths: List[str], dataset: str, gt_dirname: str = 'ground_truth'):
    location_paths = [get_location_path(img_path=path, loc_dirname=gt_dirname) for path in image_paths]
    locations = np.stack(
        [get_location(loc_path=path) for path in location_paths],
        axis=0
    )
    class_labels = np.array([get_class(img_path=path, dataset=dataset) for path in image_paths])

    return locations, class_labels
  
def divLocation(locations: Tuple[int, int]) -> Tuple[int, int]:
    x1, x2, x3, x4 = locations/4
    x1, y1 = x1
    x2, y2 = x2
    x3, y3 = x3
    x4, y4 = x4
    return [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]


def get_midv500_data(path='data/midv500'):
    image_paths = glob.glob(os.path.join(path, '*', '*', 'images', '*', '*'))
    locations, class_labels = get_metadata(image_paths, dataset='midv500')
    locations = np.array([divLocation(xi) for xi in locations])

    return image_paths, locations, class_labels

def get_midv2019_data(path='data/midv2019'):
    image_paths = glob.glob(os.path.join(path, '*', 'images', '*', '*'))
    locations, class_labels = get_metadata(image_paths, dataset='midv2019')

    return image_paths, locations, class_labels

def get_midv2020_data(path='data/midv2020'):
    gt_paths = glob.glob(os.path.join(path, 'photo', 'annotations', '*.json'))

    class_labels = []
    locations = []
    image_paths = []

    for gt_path in gt_paths:
        json_data = json.load(open(gt_path, 'r'))
        # class_name = json_data['_via_settings']['project']['name']
        class_name = os.path.splitext(os.path.basename(gt_path))[0]
        basedir = os.path.join(path, 'photo', 'images', class_name)

        for k, v in json_data['_via_img_metadata'].items():
            image_paths.append(os.path.join(basedir, v['filename']))
            for reg in v['regions']:
                if reg['shape_attributes']['name'] == 'polygon':
                    x = reg['shape_attributes']['all_points_x']
                    y = reg['shape_attributes']['all_points_y']
                    loc = np.stack([x, y], axis=1)
            locations.append(loc)
            class_labels.append(class_name)

    return image_paths, locations, class_labels

def show_image(image_path: str, location: np.ndarray = None, label: str = None):
    plt.imshow(load_img(image_path))

    if location is not None:
        x, y = location[:, 0], location[:, 1]
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), color=(1, 0, 0), linewidth=2.0)
    if label is not None:
        print(label)
    plt.show()


