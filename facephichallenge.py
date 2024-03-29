# -*- coding: utf-8 -*-
"""FacePhiChallenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UW_d_tBzcu2YwE40FbA9lnjgmlZQtBfM

#FacePhi Challenge2022

## Configuration
"""

from shapely.geometry import box, Polygon
from typing import List, Tuple
from typing import Tuple, List
import numpy as np
import os
import tqdm
import argparse
from urllib.request import urlretrieve
import tarfile
import zipfile
import os
from re import T
from typing import List, Tuple
import glob
import tqdm
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import ipdb

# Debug flag 
# TODO # REMOVE
DEBUG = False

# Redownload data if true
DOWNLOAD_DATA = False

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
LOAD_TRAIN = False
SAVE_TRAIN = True

"""##Data preparation

###Downloading and extraction
"""



midv500_links = [
    "ftp://smartengines.com/midv-500/dataset/01_alb_id.zip",
    "ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip",
    "ftp://smartengines.com/midv-500/dataset/22_est_id.zip",
    "ftp://smartengines.com/midv-500/dataset/24_fin_id.zip",
    "ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip",
    "ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/42_svk_id.zip",
]

midv2019_links = [
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/01_alb_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/05_aze_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/21_esp_id_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/22_est_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/24_fin_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/25_grc_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/32_lva_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/39_rus_internalpassport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/41_srb_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/42_svk_id.zip",
]

midv2020_links = ["ftp://smartengines.com//midv-2020/dataset/photo.tar"]

def extract(path):
    out_path, extension = os.path.splitext(path)

    if extension == ".tar":
        with tarfile.open(path, "r:") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, out_path)
    elif extension == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_path)
    else:
        raise NotImplementedError()

class tqdm_upto(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url: str, save_dir: str):
    # Creates save_dir if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Downloads the file
    with tqdm_upto(unit="B", unit_scale=True, miniters=1) as t: 
        urlretrieve(
            url,
            filename=os.path.join(save_dir, url.split("/")[-1]),
            reporthook=t.update_to,
            data=None,
        )

def download_and_extract(links_set, download_dir: str = './data'):
    out_path = os.path.join(download_dir)
    for i, link in enumerate(links_set):
        # download zip file
        link = link.replace("\\", "/")
        filename = os.path.basename(link)
        print()
        print(f"Downloading {i+1}/{len(links_set)}:", filename)
        download(link, out_path)

        # unzip zip file
        print("Unzipping:", filename)
        zip_path = os.path.join(out_path, filename)
        extract(zip_path)

        # remove zip file
        os.remove(zip_path)

if DOWNLOAD_DATA:
  download_and_extract(midv500_links, download_dir='data/midv500')
  #download_and_extract(midv2019_links, download_dir='data/midv2019')
  download_and_extract(midv2020_links, download_dir='data/midv2020')

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

"""###Preprocessing"""

def load_img(path: str, size: Tuple[int, int] = None):
    if USING_CV2 is True:
      img = cv2.imread(path)
    else:
      img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    if size is not None:
        img = cv2.resize(img, size)
    return img

def preprocess(
    image_paths: List[str],
    locations: np.ndarray,
    labels: List[str],
    image_size: Tuple[int, int],
    class_names: str = None
):
    if class_names is None:
        unique_labels, label_ids = np.unique(labels, return_inverse=True)
    else:
        unique_labels = class_names
        label_ids = np.array([class_names.index(l) for l in labels])

    images = np.zeros((len(image_paths), image_size[1], image_size[0], 3), dtype=np.uint8)
    for i, path in enumerate(tqdm.tqdm(image_paths)):
        images[i] = load_img(path, size=image_size)

    # TODO Remove if
    if USING_CV2 is False:
      # Normalize in range (-1, 1)
      images = images.astype(np.float32) / 127.5 - 1.0
    return images, label_ids, unique_labels


input_size = IMAGE_SHAPE
image_paths, locations, labels = get_midv500_data("./data/midv500")
train_images, train_label_ids, unique_labels = preprocess(image_paths, locations, labels, input_size)




"""### Function definition"""


Point2D = Tuple[int, int]

def expandDims(poly: List[Point2D]) -> List[Point2D]: 
  """
  Convert square from 2 points to polygon of 4 points.
  Use (xMin, yMin) and (xMax, yMax) from square to a polygon shape
  with 4 corners
  Input:
    poly -> list of 2 tuples of 2 points
  Returns:
    newPoly -> list of 4 tuples of 2 points each
  """
  xMin, yMin = poly[0]
  xMax, yMax = poly[1]
  newPoly = [(xMin, yMin), (xMax, yMin), (xMax, yMax), (xMin, yMax)]
  return newPoly
  

def getIOU(poly1: List[Point2D], poly2: List[Point2D]) -> float:
  """
  Extract Interection Over Union of two squares.
  Input:
    poly1 -> first square
    poly2 -> second square
  Output:
    IOU -> interection Over Union as float
  """

  poly1 = expandDims(poly1)
  poly2 = expandDims(poly2)

  # Define Each polygon 
  polygon1_shape = Polygon(poly1)
  polygon2_shape = Polygon(poly2)
  
  # Calculate Intersection and union, and tne IOU
  polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
  polygon_union = polygon1_shape.union(polygon2_shape).area
  IOU = polygon_intersection / polygon_union 

  return IOU

def location2poly(location: List[Point2D]) -> Polygon:
  """
  Function that converts a location to a polygon
  Input:
    location -> list of 4 tuples with the (x, y) locations of each corner
  Returns:
    Polygon -> Polygon with the location points as vertex
  """
  return Polygon([(location[0][0],location[0][1]),
                  (location[1][0],location[1][1]),
                  (location[2][0],location[2][1]),
                  (location[3][0],location[3][1])])
  
def poly2location(poly: Polygon) -> List[Point2D]:
  """
  Fucntion that converts a polygon to a location
  Input:
    poly -> Polygon to convert to a location
  Returns:
    location -> list of 4 tuples with the (x, y) locations of each corner 
  """
  x, y = poly.exterior.coords.xy
  # Last vertex is not used as it's only for closing the polygon
  return [xy for xy in zip(x[:-1],y[:-1])]

def imgShape2poly(imgShape: Tuple[int, int]) -> Polygon:
  """
  Function that converts an img shape to a polygon
  Input:
    imgShape -> tuple of 2 with the width and height
  Returns:
    Polygon -> Polygon with the image frame
  """
  return Polygon([(0, 0),
                  (0, imgShape[1]),
                  (imgShape[0], imgShape[1]),
                  (imgShape[0],0)])

def checkDNIInImage(location: List[Point2D], imgShape: Tuple[int,int]) -> float:
  """
  Function for getting the amount of DNI that there is inside an image so
  they can be discarded if there is a low amount of it 
  Input:
    location -> list of 4 tuples with the (x, y) locations of each corner
    imgShape -> tuple with the shape of the image
  Returns:
    float -> amount of DNI inside an image from 0 to 1
  """
  imgPoly = imgShape2poly(imgShape)
  DNIPoly = location2poly(location)
  
  intersection = DNIPoly.intersection(imgPoly)

  return intersection.area/DNIPoly.area

def fixLocation(location: List[Point2D], imgShape: List[int]) -> List[Point2D]:
  """
  Fixes DNI location to a new location so all the points are inside the visible
  image
  Input:
    location -> list of 4 tuples with the (x, y) locations of each corner
    imgShape -> tuple with the shape of the image
  Returns:
    newLocation -> New location with all its points inside the image
  """
  locPoly = location2poly(location)
  imgPoly = imgShape2poly(imgShape)
  insidePoly = locPoly.intersection(imgPoly)
  newLocation = poly2location(insidePoly)
  return newLocation

def location2square(location: List[Point2D], imgShape: List[int]) -> List[Point2D]:
  """
  Convert polygonal shape from dni image to square shape.
  Input:
    location -> list of 4 tuples with the (x, y) locations of each corner
  Returns:
    newLocation -> list of 2 tuples with the (x, y) locations of the new
    square corners
  """
  location = fixLocation(location, imgShape)

  x = [v for v, _ in location]
  y = [v for _, v in location]

  return [(int(min(x)), int(min(y))), (int(max(x)), int(max(y)))]

def square2poly(square: List[Point2D]) -> Polygon:
  """
  Function for converting a square into a polygon
  Input:
    square -> Square that will be converted
  Returns:
    poly -> Polygon with the vertex of the square
  """
  xMin, yMin = square[0]
  xMax, yMax = square[1]
  return Polygon([(xMin, yMin),(xMin, yMax),(xMax, yMax),(xMax, yMin)])

def percentageSquareInSquare(
    squareLittle: List[Point2D],
    squareBig: List[Point2D]) -> float:
  """
  Function that checks if a square is inside another
  Input:
    squareLittle -> Square to check if it's inside
    squareBig -> Square to check against
  Return:
    float -> percentage of square that is inside
  """
  polyLittle = square2poly(squareLittle)
  polyBig = square2poly(squareBig)
  intersection = polyBig.intersection(polyLittle)
  return intersection.area/polyLittle.area


"""### Calc IOU"""

# Use selective search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ssMapping = [ss.switchToSingleStrategy,
             ss.switchToSelectiveSearchFast,
             ss.switchToSelectiveSearchQuality]

trainImg = []
trainLabels = []

for idx, img in enumerate(tqdm.tqdm(train_images)):
  pInImage = checkDNIInImage(locations[idx], IMAGE_SHAPE)
  if pInImage < THRESHOLD_DNI_IN_IMAGE:
    continue
  xMax, yMax, wMax, hMax = (0,0,0,0)
  xMid, yMid, wMid, hMid = (0,0,0,0)
  xMin, yMin, wMin, hMin = (0,0,0,0)
  minIOU = 1.0
  maxIOU = 0.0
  midIOU = 1.0
  minFound = False
  midFound = False
  maxFound = False
  gtBb = location2square(locations[idx], IMAGE_SHAPE)
  for ss_iter in range(3):
    ss.setBaseImage(img)
    ssMapping[ss_iter]()
    ssResults = ss.process()
    imgCopy = img.copy()
    for result in ssResults:
      x, y, w, h = result
      resultBb = [(x, y), (x+w, y+h)]
      iou = getIOU(resultBb, gtBb)
      if iou > 0.85:
        if maxIOU < iou:
          maxFound = True
          maxIOU = iou
          xMax, yMax, wMax, hMax = x, y, w, h
      elif iou < 0.1:
        if minIOU > iou:
          minFound = True
          minIOU = iou
          xMin, yMin, wMin, hMin = x, y, w, h
      #elif 0.2 < iou and iou < 0.3:
        #if percentageSquareInSquare(gtBb, resultBb) > THRESHOLD_GT_IN_MID_SQUARE:
          #continue
        #if midIOU > iou:
          #midFound = True
          #midIOU = iou
          #xMid, yMid, wMid, hMid = x, y, w, h
    
    #if minFound is True and midFound is True and maxFound is True:
    if minFound is True and maxFound is True:
        break
  
  if maxFound is True:
    maxImg = imgCopy[yMax:yMax+hMax, xMax:xMax+wMax]
    maxImg = cv2.resize(maxImg, (224,224))
    # Append images with DNI as label 1
    trainImg.append(maxImg)
    trainLabels.append(1)

  if minFound is True:
    minImg = imgCopy[yMin:yMin+hMin, xMin:xMin+wMin]
    minImg = cv2.resize(minImg, (224,224))
    # Append images without DNI as label 0
    trainImg.append(minImg)
    trainLabels.append(0)

  #if midFound is True:
    #midImg = imgCopy[yMid:yMid+hMid, xMid:xMid+wMid]
    #midImg = cv2.resize(midImg, (224,224))
    # Append images with a small part of DNI as label 0
    #trainImg.append(midImg)
    #trainLabels.append(0)
    

trainLabels = [np.array([1.0,0.0]) if v == 1 else np.array([0.0,1.0]) for v in trainLabels]
trainLabels = np.array(trainLabels)
trainImg = np.array(trainImg)
trainImg = trainImg.astype(np.float32) / 255.0

"""### Save trainIMG and trainLabels"""

if SAVE_TRAIN is True:
  # Save files to disk in order to avoid execute the selective 
  # search every time the environment is disconnected
  # just load the files and done
  with open("./selectiveSearch/trainImg.npy", "wb") as f:
    np.save(f, trainImg)
  with open("./selectiveSearch/trainLabel.npy", "wb") as f:
    np.save(f, trainLabels)
elif LOAD_TRAIN is True:
  # Load trainImg and trainLabel files
  with open("./selectiveSearch/trainImg.npy", "rb") as f:
    trainImg = np.load(f)
  with open("./selectiveSearch/trainLabel.npy", "rb") as f:
    trainLabels = np.load(f)


