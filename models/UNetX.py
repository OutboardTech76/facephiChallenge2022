"""
This module defines the class UNetX for working with a U-Net Xception-style model
"""

import tensorflow as _tf
from tensorflow import keras as _keras

from .types import *
from typing import List,Tuple

class UNetX(_keras.Model):
    """
    Class for creating and working with a U-Net Xception-style model
    Args:
        img_size: ColorImageSize with the shape of the images that will be fit in
            A None will be preapended for batch feeding
        n_filters: List of integers that contain the number of filters per
            convolutional layer, it has to have 8 elements
    """
    
    def __init__(self, img_size: ColorImageSize, n_filters: List[int], n_classes) -> None:
        super().__init__()
        if not isinstance(img_size,tuple) or not isinstance(img_size[0],int) or img_size[2] != 3:
            raise TypeError("img_size must be a ColorImageSize")
        if not isinstance(n_filters,list) or not isinstance(n_filters[0],int):
            raise TypeError("n_filters must be a list of ints")
        if not isinstance(n_classes, int):
            raise TypeError("n_classes must be of type int")
        if len(img_size) != 3:
            raise TypeError("img_size must be a ColorImageSize")
        if len(n_filters) != 8:
            raise ValueError("n_filters must have 8 elements")
        ## Imput of the network
        in_l = _keras.Input(shape=img_size)

        ## First half of the network
        ## Downsampling
        ## Block 0
        # Entry
        self._conv1 = _keras.layers.Conv2D(n_filters[0],3,strides=2,padding="same") # (x)
        self._bn1 = _keras.layers.BatchNormalization() # (x)
        self._act1 = _keras.layers.Activation("relu") # (x)


        ## Block 1
        self._act2 = _keras.layers.Activation("relu") # (x)
        self._conv2 = _keras.layers.SeparableConv2D(n_filters[1],3,padding="same") # (x)
        self._bn2 = _keras.layers.BatchNormalization() # (x)
        self._act3 = _keras.layers.Activation("relu") # (x)
        self._conv3 = _keras.layers.SeparableConv2D(n_filters[1],3,padding="same") # (x)
        self._bn3 = _keras.layers.BatchNormalization() # (x)
        self._mp1 = _keras.layers.MaxPooling2D(3,strides=2,padding="same") # (x)
        # Residual
        self._conv4_r = _keras.layers.Conv2D(n_filters[1],1,strides=2,padding="same") # (prev_block)


        ## Block 2
        self._act4 = _keras.layers.Activation("relu") # (x)
        self._conv5 = _keras.layers.SeparableConv2D(n_filters[2],3,padding="same") # (x)
        self._bn4 = _keras.layers.BatchNormalization() # (x)
        self._act5 = _keras.layers.Activation("relu") # (x)
        self._conv6 = _keras.layers.SeparableConv2D(n_filters[2],3,padding="same") # (x)
        self._bn5 = _keras.layers.BatchNormalization() # (x)
        self._mp2 = _keras.layers.MaxPooling2D(3,strides=2,padding="same") # (x)
        # Residual
        self._conv7_r = _keras.layers.Conv2D(n_filters[2],1,strides=2,padding="same") # (prev_block)


        ## Block 3
        self._act6 = _keras.layers.Activation("relu") # (x)
        self._conv8 = _keras.layers.SeparableConv2D(n_filters[3],3,padding="same") # (x)
        self._bn6 = _keras.layers.BatchNormalization() # (x)
        self._act7 = _keras.layers.Activation("relu") # (x)
        self._conv9 = _keras.layers.SeparableConv2D(n_filters[3],3,padding="same") # (x)
        self._bn7 = _keras.layers.BatchNormalization() # (x)
        self._mp3 = _keras.layers.MaxPooling2D(3,strides=2,padding="same") # (x)
        # Residual
        self._conv10_r = _keras.layers.Conv2D(n_filters[3],1,strides=2,padding="same") # (prev_block)


        ## Second half
        ## Upsampling
        ## Block 4
        self._act8 = _keras.layers.Activation("relu") # (x)
        self._conv11 = _keras.layers.Conv2DTranspose(n_filters[4],3,padding="same") # (x)
        self._bn8 = _keras.layers.BatchNormalization() # (x)
        self._act9 = _keras.layers.Activation("relu") # (x)
        self._conv12 = _keras.layers.Conv2DTranspose(n_filters[4],3,padding="same") # (x)
        self._bn9 = _keras.layers.BatchNormalization() # (x)
        self._us1 = _keras.layers.UpSampling2D(2) # (x)
        # Residual 
        self._us2_r = _keras.layers.UpSampling2D(2) # (prev_block)
        self._conv13_r = _keras.layers.Conv2D(n_filters[4],1,padding="same") # (residual)


        ## Block 5
        self._act10 = _keras.layers.Activation("relu") # (x)
        self._conv14 = _keras.layers.Conv2DTranspose(n_filters[5],3,padding="same") # (x)
        self._bn10 = _keras.layers.BatchNormalization() # (x)
        self._act11 = _keras.layers.Activation("relu") # (x)
        self._conv15 = _keras.layers.Conv2DTranspose(n_filters[5],3,padding="same") # (x)
        self._bn11 = _keras.layers.BatchNormalization() # (x)
        self._us3 = _keras.layers.UpSampling2D(2) # (x)
        # Residual 
        self._us4_r = _keras.layers.UpSampling2D(2) # (prev_block)
        self._conv16_r = _keras.layers.Conv2D(n_filters[5],1,padding="same") # (residual)


        ## Block 6
        self._act12 = _keras.layers.Activation("relu") # (x)
        self._conv17 = _keras.layers.Conv2DTranspose(n_filters[6],3,padding="same") # (x)
        self._bn12 = _keras.layers.BatchNormalization() # (x)
        self._act13 = _keras.layers.Activation("relu") # (x)
        self._conv18 = _keras.layers.Conv2DTranspose(n_filters[6],3,padding="same") # (x)
        self._bn13 = _keras.layers.BatchNormalization() # (x)
        self._us5 = _keras.layers.UpSampling2D(2) # (x)
        # Residual 
        self._us6_r = _keras.layers.UpSampling2D(2) # (prev_block)
        self._conv19_r = _keras.layers.Conv2D(n_filters[6],1,padding="same") # (residual)


        ## Block 7
        self._act14 = _keras.layers.Activation("relu") # (x)
        self._conv20 = _keras.layers.Conv2DTranspose(n_filters[7],3,padding="same") # (x)
        self._bn14 = _keras.layers.BatchNormalization() # (x)
        self._act15 = _keras.layers.Activation("relu") # (x)
        self._conv21 = _keras.layers.Conv2DTranspose(n_filters[7],3,padding="same") # (x)
        self._bn15 = _keras.layers.BatchNormalization() # (x)
        self._us7 = _keras.layers.UpSampling2D(2) # (x)
        # Residual 
        self._us8_r = _keras.layers.UpSampling2D(2) # (prev_block)
        self._conv22_r = _keras.layers.Conv2D(n_filters[7],1,padding="same") # (residual)


        ## Block 8
        ## Output
        self._outputs = _keras.layers.Conv2D(n_classes,3,activation="softmax",padding="same")#(x)
        self.out = self.call(in_l)
        super().__init__(inputs=in_l,outputs=self.out)

    def call(self,inputs,training=None,mask=None):
        """
        Funcion para construir la red utilizando las capas creadas en __ini__
        Esta funcion permite multi input y multi output
        Args:
            inputs: input placeholder para pode construir la red
            training: ?
            mask: ?
        Returns:
            Salida de la red ya construida

        Examples:
            Multi-input/multi-output
            ```
            def call(self, inputs, training=None, mask=None):
                x1 = inputs[0]
                x2 = inputs[1]
                # Model construction
                return x1,x2
            ```
        """
        x = inputs
        ## Block 0
        # Entry
        x = self._conv1(x)
        x = self._bn1(x)
        x = self._act1(x)
        prev_block = x


        ## Block 1
        x = self._act2(x)
        x = self._conv2(x)
        x = self._bn2(x)
        x = self._act3(x)
        x = self._conv3(x)
        x = self._bn3(x)
        x = self._mp1(x)
        # Residual
        residual = self._conv4_r(prev_block)
        x = _keras.layers.add([x,residual])
        prev_block = x


        ## Block 2
        x = self._act4(x)
        x = self._conv5(x)
        x = self._bn4(x)
        x = self._act5(x)
        x = self._conv6(x)
        x = self._bn5(x)
        x = self._mp2(x)
        # Residual
        residual = self._conv7_r(prev_block)
        x = _keras.layers.add([x,residual])
        prev_block = x

        
        ## Block 3
        x = self._act6(x)
        x = self._conv8(x)
        x = self._bn6(x)
        x = self._act7(x)
        x = self._conv9(x)
        x = self._bn7(x)
        x = self._mp3(x)
        # Residual
        residual = self._conv10_r(prev_block)
        x = _keras.layers.add([x,residual])
        prev_block = x


        ## Block 4
        x = self._act8(x)
        x = self._conv11(x)
        x = self._bn8(x)
        x = self._act9(x)
        x = self._conv12(x)
        x = self._bn9(x)
        x = self._us1(x)
        # Residual 
        residual = self._us2_r(prev_block)
        residual = self._conv13_r(residual)
        x = _keras.layers.add([x,residual])
        prev_block = x
        

        ## Block 5
        x = self._act10(x)
        x = self._conv14(x)
        x = self._bn10(x)
        x = self._act11(x)
        x = self._conv15(x)
        x = self._bn11(x)
        x = self._us3(x)
        # Residual 
        residual = self._us4_r(prev_block)
        residual = self._conv16_r(residual)
        x = _keras.layers.add([x,residual])
        prev_block = x


        ## Block 6
        x = self._act12(x)
        x = self._conv17(x)
        x = self._bn12(x)
        x = self._act13(x)
        x = self._conv18(x)
        x = self._bn13(x)
        x = self._us5(x)
        # Residual 
        residual = self._us6_r(prev_block)
        residual = self._conv19_r(residual)
        x = _keras.layers.add([x,residual])
        prev_block = x


        ## Block 7
        x = self._act14(x)
        x = self._conv20(x)
        x = self._bn14(x)
        x = self._act15(x)
        x = self._conv21(x)
        x = self._bn15(x)
        x = self._us7(x)
        # Residual 
        residual = self._us8_r(prev_block)
        residual = self._conv22_r(residual)
        x = _keras.layers.add([x,residual])


        ## Block 8
        ## Output
        x = self._outputs(x)
        return x


if __name__ == "__main__":
    pass
