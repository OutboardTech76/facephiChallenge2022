"""
Module to test morphological operations in images
"""
import ipdb
import cv2
import tqdm
import time
import tensorflow as tf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from typing import List, Tuple, Any, Union, TypeAlias
from nptyping import NDArray

ImageSize = NDArray[(Any, Any), int]
Image = NDArray[(Any, Any, 3), int]
ImageBinary = NDArray[(Any, Any, 1), int]
ImageCollection = list[Image]
TensorFlowModel = tf.keras.Model


def loadModel(path: str) -> TensorFlowModel:
    """
    Function to load tensorflow model.
    Input:
        path -> str to models path using SaveModel tensorflow format
    Returns:
        model -> loaded model
    """
    return tf.keras.models.load_model(path)


def loadTestDataset(path: str, wholeDataset: bool = False) -> ImageCollection:
    """
    Function to load testing dataset.
    Input:
        path -> location to dataset
        wholeDataset -> boolean, if True get the whole dataset from path, else
                        get the first 10 images
    Returns:
        images -> list of images existing in dataset
    """
    partialPath = path.split("/")
    if partialPath[-1] == "":
        datasetType = partialPath[-2]
        joinPath = ""
    else:
        datasetType = partialPath[-1]
        joinPath = "/"

    if datasetType == "midv2020":
        path = path + joinPath
        completePath = glob.glob(os.path.join(path, "images", "*", "*"))
    else:
        completePath = glob.glob(os.path.join(path,  "*", "*", "images", "*", "*"))


    if wholeDataset is True:
        images = [cv2.imread(imgPath) for imgPath in tqdm.tqdm(completePath)]
    else:
        images = []
        for idx, imgPath in enumerate(completePath):
            images.append(cv2.imread(imgPath))
            if idx > 9:
                break
    return images


def openImg(img: Image) -> Image:
    """
    Function to realize close operation over an image.
    Input:
        img -> image where apply the transformation
    Returns:
        outImg -> image with the transformation applied
    """
    filterSize = (17, 17)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def dilateClose(img: Image) -> Image:
    """
    Function to realize dilate and close operations.
    Input:
        img -> input image where realize operations
    Returns:
        outImg -> image with the operations done
    """

    filterClose = (9, 9)
    filterDilate = (5,5)
    closeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterClose)
    dilateKernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterDilate)

    auxImg = cv2.dilate(img, dilateKernel, iterations=5)
    outImg = cv2.morphologyEx(auxImg, cv2.MORPH_CLOSE, closeKernel)
    return outImg


def predictImg(model: TensorFlowModel, image: Union[Image, ImageCollection]) -> Tuple[Image, Image, ImageBinary]:
    """
    Funtion to locate dni in image.
    With the model predicts the location of the DNI, after that, applies morphological 
    operation such as dilate and close to create a new mask, with this mask the DNI is
    then predicted again. The diference between the first and the second precition is 
    considerable. After that, open operations are done.
    Inputs:
        model -> tensorflow model used to make predictions
        image -> can be a single image or an array with multiple images
    Returns:
        origImg -> original image or list of images scaled
        segImage -> image or list of images resulted from segmentation
        mask -> mask or list of masks of the segmented image(s)
    """

    if type(image) is not list:
        img = cv2.resize(image, (256, 256)) # Resize to network working size

        imgNorm = img / 255.0
        imgNorm = np.expand_dims(imgNorm, 0)

        pred = model.predict(imgNorm)
        mask = np.argmax(pred[0], axis=2)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask * 255.0
        # Convert to uint8 to do morphological operations
        mask = mask.astype(np.uint8)
        mask = dilateClose(mask)

        img2 = cv2.bitwise_and(img, img, mask = mask)
        # Convert black points to brown in image with mask
        img2 = np.where(img2[:,:,:] > 1, img2[:,:,:], (19,69,139))
        img2 = img2.astype(np.uint8)
        img2Norm = img2 / 255.0
        img2Norm = np.expand_dims(img2Norm, 0)

        # Make precition of the new image
        pred = model.predict(img2Norm)
        mask = np.argmax(pred[0], axis=2)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask * 255.0
        # Convert to uint8 to use bitwise operations
        mask = mask.astype(np.uint8)
        mask = openImg(mask)

        # 4k images. Dont uncomment
        # mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = cv2.resize(mask, (540, 960))
        resizedImg = cv2.resize(image, (540, 960))
        segImg = cv2.bitwise_and(resizedImg, resizedImg, mask = mask)
        return resizedImg, segImg, mask
    else:
        binaryImages =[]
        resizedImages = []
        segmentedImages = []
        for currentImg in image:
            img = cv2.resize(currentImg, (256, 256)) # Resize to network working size

            imgNorm = img / 255.0
            imgNorm = np.expand_dims(imgNorm, 0)

            pred = model.predict(imgNorm)
            mask = np.argmax(pred[0], axis=2)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask * 255.0
            # Convert to uint8 to do morphological operations
            mask = mask.astype(np.uint8)
            mask = dilateClose(mask)

            img2 = cv2.bitwise_and(img, img, mask = mask)
            # Convert black points to brown in image with mask
            img2 = np.where(img2[:,:,:] > 1, img2[:,:,:], (19,69,139))
            img2 = img2.astype(np.uint8)
            img2Norm = img2 / 255.0
            img2Norm = np.expand_dims(img2Norm, 0)

            # Make precition of the new image
            pred = model.predict(img2Norm)
            mask = np.argmax(pred[0], axis=2)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask * 255.0
            # Conver to uint8 to use bitwise operations
            mask = mask.astype(np.uint8)
            mask = openImg(mask)

            # 4k images. Dont uncomment
            mask = cv2.resize(mask, (540, 960))
            resizedImg = cv2.resize(currentImg, (540, 960))
            segImg = cv2.bitwise_and(resizedImg, resizedImg, mask = mask)
            binaryImages.append(mask)
            resizedImages.append(resizedImg)
            segmentedImages.append(segImg)
        
        return resizedImages, segmentedImages, binaryImages


def extractContours(img: Image, mask: ImageBinary) -> Tuple[ImageCollection, int]:
    """
    Function to calculate the contour of each prediction in the image.
    The bounding box of each contour is extracted and used to extract
    only this part from the original images. This images will be used
    in the next model to identify the country or if there is no ID.
    Input:
        img -> original image where extract the location of the mask
        mask -> mask of the image with the predictions
    Returns:
        images -> list of images with the extracted contour
        numCount -> number of contours existing in the image
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    images = []
    for contour in enumerate(contours):
        auxMask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

        # Calculate min area within the contour, extract points and draw.
        rect = cv2.minAreaRect(contour[1])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(auxMask,[box],0,(255,255,255),-1)

        blackBkgd = np.where(auxMask > 1)
        # Get max and min arguments from black background list
        maxRowArg = np.argmax(blackBkgd[0])
        maxColArg = np.argmax(blackBkgd[1])
        minRowArg = np.argmin(blackBkgd[0])
        minColArg = np.argmin(blackBkgd[1])

        maxRow = blackBkgd[0][maxRowArg]
        maxCol = blackBkgd[1][maxColArg]
        minRow = blackBkgd[0][minRowArg]
        minCol = blackBkgd[1][minColArg]

        images.append(img[minRow:maxRow, minCol:maxCol, :])
    return images


if __name__ == "__main__":
    # model = loadModel("./bestModel_loss_50epochs_256x256")
    model = loadModel("./bestModel_acc_50epochs_256x256")
    images = loadTestDataset("./data/midv2020") #, True)

    t0 = time.time()
    origImg, img, mask = predictImg(model, images[0])
    ipdb.set_trace()

    contourImgs = extractContours(origImg, mask)
    # for idx1, image in enumerate(origImg):
        # contourImgs = extractContours(image, mask[idx1])
        # for idx2, contour in enumerate(contourImgs):
            # path = "./data/testContour/img_" + str(idx1) + "_contour_" + str(idx2) + ".png"
            # cv2.imwrite(path, contour)
    # contourImgs, _ = extractContours(origImg, mask)

    t1 = time.time()

    print("TIME consumed in seconds: {}".format(t1-t0))

    while True:
        cv2.imshow("b", img)
        cv2.imshow("asd",mask)
        for idx, image in enumerate(contourImgs):
            cv2.imshow(str(idx), image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
          break

    cv2.destroyAllWindows()
