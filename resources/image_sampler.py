from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


# Generator for obtaining samples from given image
# TODO implement random sampling. Use external lib?
def random_sample(image: Image, n_samples: int = 1, sample_size: int = None):
    if sample_size is None:
        sample_size = 100
    # Generator for obtaining samples from given image
    raise Exception("Image sampling not implemented!")  # TODO sample random elements of image
    for i in range(n_samples):
        yield "sample"