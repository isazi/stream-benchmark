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


def tune_copy(size: int, type: str, elem_size: int):
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

    metrics = dict()
    metrics["GB/s"] = lambda p: (2 * elem_size * size / 10**9) / (p["time"] / 10**3)

    tune_kernel(
        "copy",
        source,
        size,
        args,
        tune_params,
        answer=answer,
        lang="cupy",
        metrics=metrics,
    )


arguments = parse_cli()
if arguments.float:
    type = "float"
    elem_size = 4
elif arguments.half:
    type = "half"
    elem_size = 2
else:
    type = "double"
    elem_size = 8
print("Tuning copy")
tune_copy(arguments.size, type, elem_size)
