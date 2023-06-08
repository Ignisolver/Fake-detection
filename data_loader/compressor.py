import os
from pathlib import Path
from constans_and_types import DEF_COMPR_QUA
from PIL import Image


class Compressor:
    @staticmethod
    def compress(src_image_path, dst_file_path, quality=DEF_COMPR_QUA):
        image = Image.open(src_image_path)
        image.save(dst_file_path, "JPEG", optimize=True, quality=quality)

    def compress_folder(self, src_folder_path, dst_folder_path):
        if not os.path.exists(dst_folder_path):
            os.mkdir(dst_folder_path)
        for file_name in os.listdir(src_folder_path):
            src_file_path = Path(src_folder_path).joinpath(file_name)
            dst_file_path = Path(dst_folder_path).joinpath(file_name)
            self.compress(src_file_path, dst_file_path)


if __name__ == "__main__":
    from constans_and_types import DATA_PATH, COMPRESSED

    compressed_folder_path = DATA_PATH.joinpath(COMPRESSED)
    os.mkdir(compressed_folder_path)
    for dataset_folder_name in os.listdir(DATA_PATH):
        if dataset_folder_name == COMPRESSED:
            continue
        dataset_folder = DATA_PATH.joinpath(dataset_folder_name)
        new_dataset_folder_path = \
            compressed_folder_path.joinpath(dataset_folder_name)
        os.mkdir(new_dataset_folder_path)
        for fake_or_real_name in os.listdir(dataset_folder):
            src = dataset_folder.joinpath(fake_or_real_name)
            dst = new_dataset_folder_path.joinpath(fake_or_real_name)
            c = Compressor()
            c.compress_folder(src, dst)
