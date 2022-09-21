import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve2d, medfilt2d

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", nargs="+", required=True, type=str, help="HDF5 file(s) to plot"
)
parser.add_argument(
    "--labels", nargs="+", required=True, type=str, help="Title of plots"
)
parser.add_argument("--var", help="Variable to plot", default="jx", type=str)
parser.add_argument(
    "--passes", nargs="+", help="Number of passes with filter", default=0, type=int
)
parser.add_argument(
    "--to_filter", nargs="+", help="Which of the plots to filter", type=int
)
parser.add_argument("--title", help="Figure title", type=str)
parser.add_argument(
    "--filter",
    help="Filter to apply",
    choices=["gaussian", "median", "uniform"],
    type=str,
)
parser.add_argument("--flip", nargs="+", help="Which of the plots to invert", type=int)
parser.add_argument(
    "--no_flip",
    action="store_true",
    help="Don't invert any of the plots",
)
args = parser.parse_args()


n = len(args.files)
fig, ax = plt.subplots(1, n, figsize=(6 + 5 * n, 4))
if n == 1:
    ax = [ax]

if args.to_filter is not None:
    to_filter = args.to_filter
else:
    to_filter = list(range(1, n))

if args.flip is not None:
    to_flip = args.flip
else:
    to_flip = list(range(1, n))

if args.no_flip:
    to_flip = []

if len(args.passes) == 1:
    passes = args.passes * n
else:
    assert (
        len(args.passes) == n
    ), "Number of passes must be 1 or equal to number of files"
    passes = args.passes


for i, fname in enumerate(args.files):
    f = h5py.File(fname, "r")
    current = f[args.var][:]
    if current.ndim == 3:
        current = current[0, :, :]
    if i in to_filter:
        for _ in range(passes[i]):
            if args.filter == "median":
                current = medfilt2d(current, kernel_size=3)
            else:
                if args.filter == "gaussian":
                    filter = np.matrix("1 2 1; 2 4 2; 1 2 1") / 16.0
                elif args.filter == "uniform":
                    filter = np.ones((3, 3)) / 9.0
                current = convolve2d(current, filter, mode="same")
    if i in to_flip:
        current *= -1
    im = ax[i].imshow(current)
    ax[i].set_title(
        f"{args.labels[i]} {'+ ' + str(passes[i]) + ' ' + args.filter + ' passes' if i in to_filter and passes[i] > 0 else ''}"
    )
    plt.colorbar(im, ax=ax[i])
    f.close()

plt.suptitle(args.title)
plt.show()
