"""
Module for unifying the typing system
"""
from typing import Any as _Any
from tensorflow import keras as _keras

__all__=["ColorImageSize","ImageSize","SepConv2DKeras",
         "BatchNormKeras", "Conv2DKeras", "ActivationKeras",
         "TranspConv2DKeras", "UpsampligKeras", "PoolingKeras"]

ColorImageSize = (_Any,_Any,3)
"""
Color image shaping
ColorImageSize, aka (Any, Any, 3)
"""

ImageSize = (_Any,_Any)
"""
One-chanel image shaping
ImageSize, aka (Any, Any)
"""

"""
Objects for keras typing
"""
SepConv2DKeras = _keras.layers.SeparableConv2D
BatchNormKeras = _keras.layers.BatchNormalization
Conv2DKeras = _keras.layers.Conv2D
ActivationKeras  = _keras.layers.Activation
TranspConv2DKeras = _keras.layers.Conv2DTranspose
UpsampligKeras = _keras.layers.UpSampling2D
PoolingKeras = _keras.layers.MaxPooling2D
