#-----------------------------
# tensorflow
#-----------------------------
import tensorflow as tf
from tensorflow import keras, GradientTape
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History, LambdaCallback
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.backend import eval as tfeval
from tensorflow.keras.backend import function as tffunction

#-----------------------------
# numpy
#-----------------------------
import numpy as np
from numpy import size, unique, empty, where, asarray, amax, amin, zeros, absolute
from numpy import mean, std, sqrt, ceil, reshape, corrcoef
from numpy.random import choice

#-----------------------------
# matplotlib
#-----------------------------
from matplotlib.pyplot import figure, show, subplot, xticks, yticks, grid, imshow, xlabel, subplots, show, xlim
from matplotlib.pyplot import ylim, title, suptitle, tight_layout, ylabel, tight_layout, style, close, axis, rcParams
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------
# images and videos
#-----------------------------
import sys
from PIL import Image
from os import remove
from os.path import exists
from cv2 import imread, VideoWriter, VideoWriter_fourcc

#-----------------------------
# turn off warnings
#-----------------------------
import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()