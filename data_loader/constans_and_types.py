from pathlib import Path
ROOT_PATH = Path(__file__).parent.absolute()
DATA_PATH = ROOT_PATH.joinpath("../datasets")
FAKE = "FAKE"
REAL = "REAL"
COMPRESSED = "COMPRESSED"
DEF_COMPR_QUA = 90
