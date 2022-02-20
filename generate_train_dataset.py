import os
import cv2
import glob
import json
import numpy as np
from shapely.geometry import Polygon
from typing import TypeAlias, Tuple
from numpy.typing import NDArray
import tqdm
import ipdb

DatasetType: TypeAlias = str
ColorImage: TypeAlias = NDArray[np.uint8]
GroundTruth: TypeAlias = NDArray[np.int32]

class NotInImage(Exception):
    pass

class ImageCollision(Exception):
    def __init__(self, id_img: ColorImage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_img = id_img

def get_dataset_type(dataset_path: str) -> DatasetType:
    where = dataset_path.find("midv")
    dataset_type = dataset_path[where:]
    dataset_type = dataset_type.split("/")[0]
    return dataset_type

def get_all_image_paths(dataset_path: str) -> NDArray[np.str_]:
    dataset_type = get_dataset_type(dataset_path)
    if dataset_type == "midv500":
        all_img_paths = glob.glob(os.path.join(
            dataset_path,
            "*",
            "*",
            "images",
            "*",
            "*"
        ))
    elif dataset_type == "midv2019":
        all_img_paths = glob.glob(os.path.join(
            dataset_path,
            "*",
            "images",
            "*",
            "*"
        ))
    else:
        all_img_paths = glob.glob(os.path.join(
            dataset_path,
            "photo",
            "images",
            "*",
            "*"
        ))
    all_img_paths = np.vectorize(
        lambda s: s.replace("\\","/")
    )(all_img_paths)
    return all_img_paths

def get_all_gt_paths(image_paths: NDArray[np.str_]) -> NDArray[np.str_]:
    if image_paths.shape == ():
        raise RuntimeError("No-dimensional array")
    elif image_paths.shape[0] == 0:
        raise RuntimeError("Empty path array")
    if get_dataset_type(image_paths[0]) == "midv2020":
        raise RuntimeError("Trying to get paths from the wrong dataset")
    gt_paths = np.vectorize(
        lambda s: s.replace("images","ground_truth").replace(".tif",".json")
    )(image_paths)
    return gt_paths


def load_image(image_path: str) -> ColorImage:
    return cv2.imread(image_path)

def load_gt(gt_path: str) -> GroundTruth:
    with open(gt_path,"r") as f:
        data = json.load(f)["quad"]
    return np.array(data, dtype=np.int32)

def shape_to_poly(shape: Tuple[int, int]) -> Polygon:
    x_max, y_max = shape
    vertex_list = [[0,0],[x_max,0],[x_max,y_max],[0,y_max]]
    return Polygon(vertex_list)

def gt_to_poly(gt: GroundTruth) -> Polygon:
    xs = list(gt)
    xs = list(map(
        list,
        xs
    ))
    return Polygon(xs)

def process_image(image: ColorImage, gt: GroundTruth) -> Tuple[ColorImage, ColorImage]:
    # Extract ID whith 15px of margin
    # Extract something that IT'S NOT AN ID
    # with a similar shape
    img_poly = shape_to_poly((image.shape[1],image.shape[0]))
    orig_gt_poly = gt_to_poly(gt)
    gt_poly = orig_gt_poly.buffer(15)
    gt_poly = img_poly.intersection(gt_poly)
    if gt_poly.area == 0.0:
        raise NotInImage()
    if gt_poly.area / orig_gt_poly.area < 0.7:
        raise NotInImage()
    x, y = gt_poly.exterior.coords.xy #type: ignore
    x = list(map(int,x))
    y = list(map(int,y))
    x = x[:-1]
    y = y[:-1]
    true_img = image[min(y):max(y),min(x):max(x),:].copy()
    

    if min(y) > image.shape[0]/2:
        # Search above
        y = list(map(
            lambda y_val: y_val - min(y),
            y
        ))
    else:
        # Search below
        y = list(map(
            lambda y_val: y_val + (image.shape[0]-max(y)) - 1,
            y
        ))
    not_id_poly = Polygon(zip(x,y))
    inter_area = not_id_poly.intersection(gt_poly).area
    union_area = not_id_poly.union(gt_poly).area
    if inter_area/union_area > 0.005:
        raise ImageCollision(true_img)
    false_img = image[min(y):max(y),min(x):max(x),:].copy()
    return true_img, false_img

def get_output_paths(orig_path: str) -> Tuple[str, str]:
    where = orig_path.find("images")
    where = orig_path.find("/", where+1)
    base_path = orig_path[:where]
    base_path = base_path.replace("images","id")
    filename = os.path.basename(orig_path)
    true_path = os.path.join(base_path,filename)
    true_path = true_path.replace(".tif", ".png")

    dataset_type = get_dataset_type(orig_path)
    where = orig_path.find(dataset_type)
    where = orig_path.find("/",where)
    base_path = orig_path[:where]
    where_end = orig_path.find("/",where+1)
    country_name = orig_path[where+1:where_end]
    filename = country_name+"_"+os.path.basename(orig_path)
    false_path = os.path.join(base_path, "00_no_id", filename)
    false_path = false_path.replace(".tif", ".png")
    return true_path, false_path

def ensure_path_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path, 0o755, exist_ok=True)

def process_dataset_image(image_path, gt_path):
    img = load_image(image_path)
    gt = load_gt(gt_path)
    try:
        true_img, false_img = process_image(img, gt)
    except NotInImage:
        return
    except ImageCollision as e:
        true_path, _ = get_output_paths(image_path)
        ensure_path_exists(os.path.dirname(true_path))
        cv2.imwrite(true_path, e.id_img)
        return
    true_path, false_path = get_output_paths(image_path)
    ensure_path_exists(os.path.dirname(true_path))
    ensure_path_exists(os.path.dirname(false_path))
    cv2.imwrite(true_path, true_img)
    cv2.imwrite(false_path, false_img)


if __name__ == "__main__":
    img_paths = get_all_image_paths("./data/midv2019")
    gt_paths = get_all_gt_paths(img_paths)
    # count = 0
    for img_path, gt_path in tqdm.tqdm(zip(img_paths,gt_paths)):
        # count += 1
        # if count < 1715:
            # continue
        # print(count)
        # ipdb.set_trace()
        process_dataset_image(img_path,gt_path)
