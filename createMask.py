"""
Module to create mask images from orinal ones
"""

import cv2
from typing import List, Any, Tuple, Dict
from nptyping import NDArray
import numpy as np
import json
import os
import glob
import tqdm

def loadImg(path: List[str]) -> NDArray[(Any, Any, Any, 3), int]:
    """
    Funtion to load images.
    Input:
        path -> list with paths to ground_truth files
    Returns:
        images -> array with imagesof the complete dataset
    """
    newPath = imagesPath(path)
    images = np.zeros((len(newPath), 480, 270, 3), dtype=np.uint8)
    for idx, imgPath in enumerate(tqdm.tqdm(newPath)):
        auxImg = cv2.imread(imgPath)
        images[idx] = cv2.resize(auxImg, (270, 480))
    return images


def maskPath(paths: List[str]) -> List[str]:
    """
    Funtion to extract mask paths from ground_truth paths.
    Replaces ground_truth with mask folders to create them
    if they do not exist
    Input:
        path -> complete paths to ground_truth
    Returns:
        maskPaths ->  complete path to masks folders
    """
    maskPaths = []
    for path in paths:
        auxPath = os.path.dirname(path)
        auxPath = auxPath.replace("ground_truth", "mask")
        maskPaths.append(auxPath)
    return maskPaths


def imagesPath(paths: List[str]) -> List[str]:
    """
    Funtion to extract images paths from ground_truth paths.
    Replaces ground_truth with images folders and .json with
    .tif extension
    Input:
        path -> complete paths to ground_truth
    Returns:
        imgPaths ->  complete path to images
    """
    imgPaths = []
    for path in paths:
        auxPath = path.replace("ground_truth", "images")
        imgPaths.append(auxPath.replace(".json",".tif"))
    return imgPaths


def resize_location(locations: NDArray[float]) -> List[Tuple[int, int]]:
    """
    Function to resize the corner points when resizing the image.
    Input:
        locations -> list of 4 tuples with (x, y) points each
    Returns:
        list of 4 tuples with coordinates resized. Casted to int
        
    """
    x1, x2, x3, x4 = locations/4
    x1, y1 = x1
    x2, y2 = x2
    x3, y3 = x3
    x4, y4 = x4
    return [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]

def createMask(path: str) -> None:
    """
    Funtion to create mask from the whole dataset.
    Uses dni corner position to extract its location and create the mask.
    Stores the mask images in a new folder.
    Input:
        path -> path where the dataset is located
    Returns:
        None
    """

    completePath = glob.glob(os.path.join(path, "*", "*", "ground_truth", "*", "*"))
    images = loadImg(completePath)
    maskPaths = maskPath(completePath)

    for idx, jsonPath in enumerate(tqdm.tqdm(completePath)):
        blackImg = np.zeros((images.shape[1], images.shape[2], 1), np.uint8)

        try:
            os.makedirs(maskPaths[idx])
        except:
            pass
        with open(jsonPath, "r") as jsonFile:
            jsonObject = json.load(jsonFile)

        location = resize_location(np.asarray(jsonObject["quad"]))

        pts = np.asarray(location)
        pts = pts.reshape((-1,1,2))
        maskImg = cv2.fillPoly(blackImg, [pts], (255) )
        cv2.imshow("a",maskImg)

        maskSavePath = jsonPath.replace("ground_truth", "mask")
        maskSavePath = maskSavePath.replace(".json", ".png")
        cv2.imwrite(maskSavePath, maskImg)



if __name__ == "__main__":
    createMask("./data/midv500/")


