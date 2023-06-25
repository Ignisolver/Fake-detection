from PIL import Image
from abc import ABC, abstractmethod


class GenericWrapper(ABC):  # TODO create wrappers for cv models

    def classify_image(self, im: Image) -> bool:
        return True
