from PIL import Image
from typing import List
from random import randint


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


class TransformHandler:
    def __init__(self, transform_method):
        self.transform_method = transform_method

    def transform_image(self, im: Image) -> Image:
        return self.transform_method(im)


class TransformPipe:

    def __init__(self, transform_list: List[TransformHandler]):
        self.transform_list = transform_list

    def transform_image(self, im: Image):
        im_t = im.deepcopy()
        for t in self.transform_list:
            im_t = t(im_t)
        return im_t
