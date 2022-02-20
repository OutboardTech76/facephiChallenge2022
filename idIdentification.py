import cv2
import tqdm
import ipdb
import numpy as np
import tensorflow as tf
import os
import glob
from typing import List, Tuple, TypeAlias, Any
from nptyping import NDArray, UInt8
import visionOperations as vo
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Input
from tensorflow.keras.callbacks import ModelCheckpoint


Label: TypeAlias = str
Image: TypeAlias = NDArray[(Any, Any, 3), UInt8]
ImageCollection: TypeAlias = list[Image]
LabelCollection: TypeAlias = list[Label]
TensorFlowModel: TypeAlias = tf.keras.Model

idDict = {"00_no_id": 0, "01_alb_id": 1, "05_aze_passport": 2, "21_esp_id_old": 3, 
           "22_est_id": 4, "24_fin_id": 5, "25_grc_passport": 6, "32_lva_passport":7,
           "39_rus_internalpassport": 8, "41_srb_passport": 9, "42_svk_id": 10}

def loadDatasetFromFolders(path: str, imgSize: Tuple[int, int] = (256, 256)) -> Tuple[ImageCollection, LabelCollection]:
    """
    Function to load dataset from folder.
    Input:
        path -> path to folders
    Returns:
        img -> list of images
        label -> list of labels corresponding to each image
    """
    images = []
    labels = []

    directory = os.listdir(path)
    for partialPath in tqdm.tqdm(directory):
        aux = path+partialPath + "/"
        datasetPath = (glob.glob(os.path.join(aux, "*")))
        for partialPath2 in datasetPath:
            labelsName = partialPath2.split("/")[-1]
            aux2 = partialPath2 + "/"
            completePath = (glob.glob(os.path.join(aux2, "*")))
            for idx, imgPath in enumerate(completePath):
                img = cv2.imread(imgPath)
                img = cv2.resize(img, imgSize)
                if img is not None:
                    images.append(img)
                    labels.append(idDict[labelsName])
        if idx > 1000:
            break

    return images, labels


def createModel(numClasses: int, imgSize: Tuple[int, int] = (256, 256)) -> TensorFlowModel:
    """
    Funtion to create model using tensorflow's sequential method.
    Input:
       numClasses -> number of classes to the final layer 
       imgSize -> image size and input model size. Defatul = 256 x 256
    Returns:
        model -> created model
    """
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, activation='relu',
                     input_shape=(imgSize[0], imgSize[1], 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(numClasses, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model
    




if __name__ == "__main__":
    # images, labels = loadDatasetFromFolders("./data/newNet/")
    modelNet1 = vo.loadModel("./bestModel_acc_50epochs_256x256")
    imagesNet1 = vo.loadTestDataset("./data/midv2020")
    img = cv2.imread("./data/midv2020/images/esp_id/01.jpg")

    # origImg, img, mask = vo.predictImg(modelNet1, img)
    origImg, img, mask = vo.predictImg(modelNet1, imagesNet1[0])
    contourImgs = vo.extractContours(origImg, mask)

    results = []
    # modelNet2= vo.loadModel("./secondNet/bestModel_acc")
    modelNet2= vo.loadModel("./resultTraining/weights_epoch_43_valAcc_0.99")
    for image in contourImgs:
        image = cv2.resize(image, (256,256))
        imageNorm = image / 255.0
        imageNorm = np.expand_dims(imageNorm, 0)
        pred = modelNet2.predict(imageNorm)
        # Extract name from the dictionary
        results.append([k for k, v in idDict.items() if v == np.argmax(pred)])
        # ipdb.set_trace()
    # img = cv2.imread("./data/testContour/img_0_contour_1.png")
    # img = cv2.resize(img, (256, 256))
    # imgNorm = img / 255.0
    # imgNorm = np.expand_dims(imgNorm, 0)

    # pred = model.predict(imgNorm)
    # print(np.argmax(pred))
    print(results)
    ipdb.set_trace()
    
    # ipdb.set_trace()
    cv2.destroyAllWindows()


