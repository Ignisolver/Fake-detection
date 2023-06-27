from PIL import Image
from abc import ABC, abstractmethod
import sklearn

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class GenericModelWrapper(ABC):

    def __init__(self, model):
        self.model = model
        self.image_size = 20

    def classify_image(self, im: Image) -> bool:
        raise TypeError("Not implemented")


class SVM(GenericModelWrapper):

    def __init__(self):
        self.model = sklearn.svm()
        GenericModelWrapper.__init__(model=self.model)

    def model_init_restart(self):
