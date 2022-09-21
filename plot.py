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
    "--passes", help="Number of passes with filter", default=0, type=int
)
parser.add_argument(
    "--to_filter", nargs="+", help="Which of the plots to filter", type=int
)
parser.add_argument(
    "--filter", help="Filter to apply", choices=["gaussian", "median"], type=str
)
parser.add_argument("--flip", nargs="+", help="Which of the plots to flip", type=int)
args = parser.parse_args()


n = len(args.files)
fig, ax = plt.subplots(1, n, figsize=(6 + 5 * n, 5))
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

for i, fname in enumerate(args.files):
    f = h5py.File(fname, "r")
    current = f[args.var][:]
    if current.ndim == 3:
        current = current[0, :, :]
    if i in to_filter:
        for _ in range(args.passes):
            if args.filter == "gaussian":
                filter = np.matrix("1 2 1; 2 4 2; 1 2 1") / 16.0
                current = convolve2d(current, filter, mode="same")
            elif args.filter == "median":
                current = medfilt2d(current, kernel_size=3)
    if i in to_flip:
        current *= -1
    im = ax[i].imshow(current)
    ax[i].set_title(
        f"{args.labels[i]} {'+ ' + str(args.passes) + ' passes' if i in to_filter else ''}"
    )
    plt.colorbar(im, ax=ax[i])
    f.close()
plt.show()
