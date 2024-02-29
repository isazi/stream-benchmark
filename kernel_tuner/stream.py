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

    args = [TunablePrecision("T", a), TunablePrecision("T", c), n]
    answer = [None, a, None]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["T"] = [type]

    metrics = dict()
    metrics["GB/s"] = lambda p: (2 * elem_size * size / 10**9) / (p["time"] / 10**3)

    tune_kernel(
        f"copy<{type}>",
        source,
        size,
        args,
        tune_params,
        answer=answer,
        lang="cupy",
        metrics=metrics,
    )


def tune_scale(size: int, type: str, elem_size: int):
    with open("stream.cu", "r") as file:
        source = file.read()

    n = np.int32(size)
    scalar = np.float64(3.0)
    c = np.random.randn(size).astype(np.float64)
    b = np.zeros(size).astype(np.float64)

    args = [
        TunablePrecision("T", scalar),
        TunablePrecision("T", b),
        TunablePrecision("T", c),
        n,
    ]
    answer = [None, c * scalar, None, None]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["T"] = [type]

    metrics = dict()
    metrics["GFLOP/s"] = lambda p: (size / 10**9) / (p["time"] / 10**3)
    metrics["GB/s"] = lambda p: (2 * elem_size * size / 10**9) / (p["time"] / 10**3)

    tune_kernel(
        f"scale<{type}>",
        source,
        size,
        args,
        tune_params,
        answer=answer,
        lang="cupy",
        metrics=metrics,
    )


def tune_add(size: int, type: str, elem_size: int):
    with open("stream.cu", "r") as file:
        source = file.read()

    n = np.int32(size)
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)

    args = [
        TunablePrecision("T", a),
        TunablePrecision("T", b),
        TunablePrecision("T", c),
        n,
    ]
    answer = [None, None, a + b, None]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["T"] = [type]

    metrics = dict()
    metrics["GFLOP/s"] = lambda p: (size / 10**9) / (p["time"] / 10**3)
    metrics["GB/s"] = lambda p: (3 * elem_size * size / 10**9) / (p["time"] / 10**3)

    tune_kernel(
        f"add<{type}>",
        source,
        size,
        args,
        tune_params,
        answer=answer,
        lang="cupy",
        metrics=metrics,
    )


def tune_triad(size: int, type: str, elem_size: int):
    with open("stream.cu", "r") as file:
        source = file.read()

    n = np.int32(size)
    scalar = np.float64(3.0)
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)

    args = [
        TunablePrecision("T", scalar),
        TunablePrecision("T", a),
        TunablePrecision("T", b),
        TunablePrecision("T", c),
        n,
    ]
    answer = [None, b + (scalar * c), None, None, None]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["T"] = [type]

    metrics = dict()
    metrics["GFLOP/s"] = lambda p: (2 * size / 10**9) / (p["time"] / 10**3)
    metrics["GB/s"] = lambda p: (3 * elem_size * size / 10**9) / (p["time"] / 10**3)

    tune_kernel(
        f"triad<{type}>",
        source,
        size,
        args,
        tune_params,
        answer=answer,
        lang="cupy",
        metrics=metrics,
    )


def tune_stream(size: int, type: str, elem_size: int):
    with open("stream.cu", "r") as file:
        source = file.read()

    n = np.int32(size)
    scalar = np.float64(3.0)
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
    c = np.random.randn(size).astype(np.float64)

    args = [
        TunablePrecision("T", scalar),
        TunablePrecision("T", a),
        TunablePrecision("T", b),
        TunablePrecision("T", c),
        n,
    ]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["T"] = [type]

    metrics = dict()
    metrics["GFLOP/s"] = lambda p: (4 * size / 10**9) / (p["time"] / 10**3)
    metrics["GB/s"] = lambda p: (10 * elem_size * size / 10**9) / (
        p["time"] / 10**3
    )

    tune_kernel(
        f"stream<{type}>",
        source,
        size,
        args,
        tune_params,
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
print("Tuning scale")
tune_scale(arguments.size, type, elem_size)
print("Tuning add")
tune_add(arguments.size, type, elem_size)
print("Tuning Triad")
tune_triad(arguments.size, type, elem_size)
print("Tuning Stream")
tune_stream(arguments.size, type, elem_size)
