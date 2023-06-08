# Remember about Duck typing!
from PIL import Image
from numpy import copy

"""
https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
"""
# TODO: think of implementation of "image transformation" wrapper that will allow for easy grid search over TF's and params
# TODO: find above solution in opencv/sklearn/???


class Transformers:
    def __init__(self):
        self.transformation_list = []
        self.transformation_list.append(self.generic_transformation())

    @staticmethod
    def generic_transformation(input_image: Image) -> Image:
        output_image = copy(Image)
        return output_image
