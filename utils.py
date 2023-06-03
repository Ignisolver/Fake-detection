from constans_and_types import DATA_PATH, COMPRESSED


def get_dataset_path(dataset_nr: int, compressed):
    str_nr = str(dataset_nr)
    path = DATA_PATH
    if compressed:
        path = path.joinpath(COMPRESSED)
    return path.joinpath(str_nr)

