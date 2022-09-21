import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve2d, medfilt2d
from scipy.ndimage import gaussian_filter
from skimage import filters

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
    "--show_original", action="store_true", help="Also original unfiltered plots"
)
parser.add_argument(
    "--colorbar", action="store_true", help="Add colorbar to each panel"
)
parser.add_argument(
    "--show_gradients", action="store_true", help="Also plot Sobel filtered plots"
)
parser.add_argument(
    "--no_flip",
    action="store_true",
    help="Don't invert any of the plots",
)
args = parser.parse_args()


n = len(args.files)

nrows = 1
if args.show_original:
    nrows += 1
if args.show_gradients:
    nrows += 1

fig, ax = plt.subplots(
    nrows,
    n,
    figsize=(6 + 5 * n, 2 + nrows * 3),
)
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

    if args.show_original:
        im = ax[0, i].imshow(current)

    if i in to_filter:
        for _ in range(passes[i]):
            if args.filter == "median":
                current = medfilt2d(current, kernel_size=3)
            elif args.filter == "gaussian":
                current = gaussian_filter(current, sigma=1)
            elif args.filter == "uniform":
                filter = np.ones((3, 3)) / 9.0
                current = convolve2d(current, filter, mode="same")

    if i in to_flip:
        current *= -1

    fulltitle = (
        args.labels[i]
        + f"{'+' + str(passes[i]) + ' ' + args.filter + ' passes' if i in to_filter and passes[i] > 0 else ''}"
    )

    if args.show_original:
        ax[0, i].set_title(args.labels[i])

        im = ax[1, i].imshow(current)
        ax[1, i].set_title(
            f"{str(passes[i]) + ' ' + args.filter + ' passes' if i in to_filter and passes[i] > 0 else ''}"
        )
        if args.show_gradients:
            im = ax[2, i].imshow(filters.sobel(current))
            ax[2, i].set_title("sobel filtered")
    else:
        if args.show_gradients:
            im = ax[0, i].imshow(current)
            ax[0, i].set_title(fulltitle)
            im = ax[1, i].imshow(filters.sobel(current))
            ax[1, i].set_title("sobel filtered")
        else:
            im = ax[i].imshow(current)
            ax[i].set_title(fulltitle)

    if args.colorbar:
        plt.colorbar(im, ax=ax[i])
    f.close()

plt.suptitle(args.title)
plt.show()
