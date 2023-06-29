from PIL import Image, ImageStat
from typing import List
from random import randint
import numpy as np

class ImageSampler:

    @staticmethod
    def sample_image(img: Image, sample_size: int = 20, n_samples: int = 5):
        sample_list: List[Image]
        w, h = img.size
        w_bound = w - sample_size
        h_bound = h - sample_size

        for i in range(n_samples):
            sample_position = [randint(0, w_bound),
                               randint(0, h_bound)]
            region = (sample_position[0], sample_position[1],
                      sample_position[0] + sample_size, sample_position[0] + sample_size)
            img_t = img.transform((sample_size, sample_size), Image.EXTENT,
                                  data=region)
            sample_list.append(img_t)
        return sample_list

    @staticmethod
    def sample_high_variance(img: Image, sample_size: int = 20, n_samples: int = 5, top_n: int = 5):
        if top_n > n_samples:
            top_n = n_samples
        all_samples = ImageSampler.sample_image(img=img, sample_size=sample_size, n_samples=n_samples)

        def sort_by_variance(val):
            pass

        samples_sorted_by_variance = all_samples.sort(key=sort_by_variance)
        return samples_sorted_by_variance[0:top_n-1]


class TransformHandler:
    def __init__(self, transform_method):
        self.transform_method = transform_method

    def transform_image(self, im: Image) -> Image:
        return self.transform_method(im)

    def transform_image_array(self, X: np.ndarray):
        res = []
        for i in range(X.shape[0]):
            image = X[i, :, :, :]
            image_transformed = self.transform_method(Image.fromarray(image, mode="RGB"))
            res.append(image_transformed)
        return res


class TransformPipe:

    def __init__(self, transform_list: List[TransformHandler]):
        self.transform_list = transform_list

    def transform_image(self, im: Image):
        im_t = im.deepcopy()
        for t in self.transform_list:
            im_t = t(im_t)
        return im_t

    def transform_image_array(self, X: np.ndarray) -> np.ndarray:
        X_transformed = np.ndarray(shape=[X.shape])
        for i in range(len(X)):
            X_transformed[i] = self.transform_image(Image.fromarray(X[i]))
        return X_transformed
