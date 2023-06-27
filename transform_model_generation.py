import models as cl
from filters import *
import random
from models import SVM

TRANSFORMS = {
    'canny': canny,
    'sobel': sobel,
    'harris': harris,
    # 'susan': ,
    'shi&tomasi': shi_tomas,
    # 'level_curve_curvature': ,
    'fast': fast,
    'laplacian_of_gaussian': laplacian_of_gaussian,
    # 'difference_of_gaussian': ,
    # 'determinant_of_hessian': ,
    # 'hessian_strength_feature_measures': ,
    # 'MSER': ,
    # 'principal_curvature_ridges': ,
    # 'grey-level-blobs': ,
    'ridge': ridge,
    'fourier': fourier
}

MODELS = {
    'svm': SVM
}
class TransformGen:

    def __init__(self, n_sets: int = 5):
        self.tf = []
        for i in range(n_sets):
            self.tf.append(self.generate_transformation_set())

    @staticmethod
    def generate_transformation_set(self):
        transform_set = {}
        while random.randint(0, 100) < 75:
            transform_set.append(random.choice(list(TRANSFORMS)))
        return transform_set


class ModelGen:

    @staticmethod
    def generate_radom_model(self):
        model = None
        return model
