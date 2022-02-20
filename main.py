
"""
Module to tinker with models, data loading and trainings
"""
import cv2
import os
import tqdm
import glob
import ipdb
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from numba import njit, vectorize, prange
from typing import List, Any, Tuple,Dict
from nptyping import NDArray
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from ctypes import *


# User-defined modules
from models import *

# Definition of types for typing
DataSources = Dict[str,str]
Image = NDArray[(Any, Any, 3), int]
ImageSeg = NDArray[(Any, Any, 3), int]
ImageSegBinary = NDArray[(Any, Any, 1), int]
ImageCollection = NDArray[(Any), Image]
ImageSegCollection = NDArray[(Any), ImageSeg]
ImageSegBinaryCollection = NDArray[(Any), ImageSegBinary]

# For printing all nmumpy array values
np.set_printoptions(threshold=np.inf)
 


class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, x_data, y_data, batch_size):
    self.x, self.y = x_data, y_data
    self.batch_size = batch_size
    self.num_batches = np.ceil(len(x_data) / batch_size)
    self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

  def __len__(self):
    return len(self.batch_idx)

  def __getitem__(self, idx):
    batch_x = self.x[self.batch_idx[idx]]
    batch_y = self.y[self.batch_idx[idx]]
    return batch_x, batch_y



def loadImg(path: List[str]) -> NDArray[(Any, Any, Any, 3), int]:
    """
    Funtion to load images.
    Input:
        path -> list with paths to ground_truth files
    Returns:
        images -> array with imagesof the complete dataset
    """
    newPath = imagesPath(path)
    # images = np.zeros((len(newPath), 480, 270, 3), dtype=np.uint8)
    images = np.zeros((len(newPath), 224, 224, 3), dtype=np.int32)
    for idx, imgPath in enumerate(tqdm.tqdm(newPath)):
        auxImg = cv2.imread(imgPath)
        # images[idx] = cv2.resize(auxImg, (270, 480))
        images[idx] = cv2.resize(auxImg, (224, 224))
        # images[idx] = tf.keras.preprocessing.image.load_img(imgPath,target_size=(224,224))
    return images

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
        auxPath = path.replace("mask", "images")
        imgPaths.append(auxPath.replace(".png",".tif"))
    return imgPaths


 
def loadDataset(path: str) -> Tuple[List[Image], List[ImageSegBinary]]:

    completePath = glob.glob(os.path.join(path, "*", "*", "mask", "*", "*"))
    images = loadImg(completePath)
    masks = np.zeros((len(completePath), 224, 224, 1), dtype=np.int32)
    # masks = np.zeros((len(completePath), 480, 270, 1), dtype=np.uint8)

    for idx, maskPath in enumerate(tqdm.tqdm(completePath)):
        auxImg = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
        # auxImg = tf.keras.preprocessing.image.load_img(maskPath,color_mode="grayscale",target_size=(224,224))
        auxImg = cv2.resize(auxImg, (224, 224))
        masks[idx] = np.expand_dims(auxImg, 2)
        masks[idx] -= 1


    return images, masks

 
 

 
if __name__ == "__main__":

    # Clear gpu session
    tf.keras.backend.clear_session()
    
    # Limit gpu memory. Unncomment to set limit to 2GB
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_virtual_device_configuration(
                    # gpu,
                    # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        except RuntimeError:
            print("Invalid GPU configuration")

    # Set where the channels are specified
    tf.keras.backend.set_image_data_format("channels_last")


    trainImg, trainLabel = loadDataset("./data/midv500/")
    trainImg = trainImg / 255.0
    trainLabel = trainLabel / 255.0
    trainGenerator = DataGenerator(trainImg, trainLabel, 8)
    # ipdb.set_trace()

    numClasses = 2
    nEpochs = 10
    batchSize = 4

    net: UNetX = UNetX(img_size=(224,224,3),n_filters=[32,64,128,256,256,128,64,32], n_classes=numClasses)
    net.summary()

    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Create checkpoints to save different models
    path = "resultTraining/weightsEpoch_{epoch:02d}_valLoss_{val_loss:.2f}.hdf5"
    path2 = "resultTraining/bestModel.hdf5"
    checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
    checkpoint2 = ModelCheckpoint(path2, monitor='val_loss', verbose=1, save_best_only=True)
    callbackList = [checkpoint, checkpoint2]

    # history = net.fit(trainGenerator, epochs=nEpochs, batch_size=batchSize)
    history = net.fit(trainImg, trainLabel, validation_split=0.3, epochs=nEpochs, batch_size=batchSize, callbacks=callbackList)

    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    # save the losses figure
    plt.tight_layout()
    plt.savefig('resultTraining/losses.png')
    plt.close()

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'],'r',linewidth=3.0)
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    # save the accuracies figure
    plt.tight_layout()
    plt.savefig('resultTraining/accs.png')
    plt.close()

