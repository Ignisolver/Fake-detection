from typing import List
from image_transforms import TransformPipe, ImageSampler
from PIL import Image
from models import GenericModelWrapper
from data_loader import DataLoader
import random


class SyntheticImageDetector:

    def __init__(self, transforms: TransformPipe, classifier: GenericModelWrapper, sample_size: int = 10, n_samples: int = 5):
        self.transforms = transforms
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.classifier = classifier

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

    def train_model(self, data_loader: DataLoader):
        train, test = data_loader.load()

        # Shuffle two lists with same order
        # Using zip() + * operator + shuffle()
        temp = list(zip(train, test))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        res1, res2 = list(res1), list(res2)
        pass