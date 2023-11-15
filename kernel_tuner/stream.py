import argparse
import numpy as np
from kernel_tuner import tune_kernel
from kernel_tuner.accuracy import TunablePrecision


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--size", help="Size of the arrays (in elements)", required=True, type=int
    )
    parser.add_argument("--float", help="Use single precision", action="store_true")
    parser.add_argument("--half", help="Use half precision", action="store_true")
    return parser.parse_args()


def tune_copy(size: int, type: str):
    with open("stream.cu", "r") as file:
        source = file.read()

    n = np.int32(size)
    a = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)

    args = [TunablePrecision("TYPE", a), TunablePrecision("TYPE", c), n]
    answer = [None, a, None]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["TYPE"] = [type]

    tune_kernel("copy", source, size, args, tune_params, answer=answer, lang="cupy")


arguments = parse_cli()
if arguments.float:
    type = "float"
elif arguments.half:
    type = "half"
else:
    type = "double"
print("Tuning copy")
tune_copy(arguments.size, type)
