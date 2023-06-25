from typing import List
from image_transforms import TransformPipe, ImageSampler
from PIL import Image
from classifier_wrappers import GenericWrapper


class SyntheticImageDetector:

    def __init__(self, transforms: TransformPipe, classifier: GenericWrapper, sample_size: int = 10, n_samples: int = 5):
        self.transforms = transforms
        self.n_samples = n_samples
        self.sample_size = sample_size

    def test_image_synthetic(self, im: Image) -> bool:  # Detection pipeline
        im_transformed = self.transforms.transform_image(im)
        samples = ImageSampler.sample_image(im_transformed, sample_size=self.sample_size, n_samples=self.n_samples)
        fake_votes = 0
        for sample in samples:
            if self.classifier.classify_image(sample):
                fake_votes += 1
            else:
                fake_votes -= 1

        return fake_votes >= 0
