import argparse
import numpy as np
from kernel_tuner import tune_kernel
from kernel_tuner.accuracy import TunablePrecision


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--size", help="Size of the arrays (in elements)", required=True
    )
    return parser.parse_args()


def tune_copy(size: int):
    with open("stream.cu", "r") as file:
        source = file.read()

    n = np.int32(size)
    a = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)

    args = [TunablePrecision("TYPE", a), TunablePrecision("TYPE", c), n]
    answer = [None, a, None]

    results, env = tune_kernel("copy", source, size, args, answer=answer, lang="cupy")


arguments = parse_cli()
print("Tuning copy")
tune_copy(arguments.size)
