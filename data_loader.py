import numpy as np
import os

from PIL import Image

from constans_and_types import FAKE, REAL
from utils import get_dataset_path


class DataLoader:
    def __init__(self, data_set_nr, samples_amount,
                 shuffle=False, compressed=False):
        self.dataset_path = get_dataset_path(data_set_nr, compressed)
        self.samples_amount = int(samples_amount/2)
        self.shuffle = shuffle
        self.compressed = compressed

    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)
        array_im = np.asarray(image)
        return array_im

    def load(self):
        fake, real = self._load_fake_real()
        x, y = self._mix_fake_real(fake, real)
        return np.asarray(x), np.asarray(y)

    def _load_fake_real(self):
        fake_path = self.dataset_path.joinpath(FAKE)
        real_path = self.dataset_path.joinpath(REAL)
        self._check_dataset_size(fake_path)
        self._check_dataset_size(real_path)
        fake = self._load_from_folder(fake_path)
        real = self._load_from_folder(real_path)
        return fake, real

    @staticmethod
    def _mix_fake_real(fake, real):
        x = []
        y = []
        for f, r in zip(fake, real):
            x.append(f)
            y.append(0)
            x.append(r)
            y.append(1)
        return x, y

    def _load_from_folder(self, folder_path):
        data = []
        os.chdir(folder_path)
        for file_name in self._get_names(folder_path):
            im = self.read_image(file_name)
            data.append(im)
        return data

    def _get_names(self, folder_path):
        if self.shuffle:
            return self._get_shuffle_names(folder_path)
        else:
            return self._get_first_names(folder_path)

    def _get_first_names(self, folder):
        return os.listdir(folder)[:self.samples_amount]

    def _get_shuffle_names(self, folder):
        all_names = os.listdir(folder)
        names = np.random.choice(all_names, self.samples_amount)
        return names

    def _check_dataset_size(self, f):
        assert len(os.listdir(f)) >= self.samples_amount, "Dataset is to small"








