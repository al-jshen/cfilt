import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve2d, medfilt2d
from scipy.ndimage import gaussian_filter
from skimage import filters
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
import time

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
    nargs="+",
    help="Filter to apply",
    choices=["gaussian", "median", "uniform", "tv", "wavelet"],
    type=str,
)
parser.add_argument("--flip", nargs="+", help="Which of the plots to invert", type=int)
parser.add_argument("--cmap", help="Colormap of the plots", type=str, default="viridis")
parser.add_argument(
    "--show_original", action="store_true", help="Also original unfiltered plots"
)
parser.add_argument(
    "--colorbar", action="store_true", help="Add colorbar to each panel"
)
parser.add_argument(
    "--gradients",
    choices=["sobel", "prewitt"],
    help="Also show gradient operator filtered plots",
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
if args.gradients is not None:
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

if args.gradients is not None:
    if args.gradients == "sobel":
        grad = filters.sobel
    elif args.gradients == "prewitt":
        grad = filters.prewitt

if len(args.filter) == 1:
    filters = [args.filter[0]] * n
else:
    assert len(args.filter) == n
    filters = args.filter

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
        im = ax[0, i].imshow(current, cmap=args.cmap)

    start = time.time()
    if i in to_filter:
        for _ in range(passes[i]):
            if filters[i] == "median":
                current = medfilt2d(current, kernel_size=3)
            elif filters[i] == "gaussian":
                current = gaussian_filter(current, sigma=1)
            elif filters[i] == "uniform":
                filter = np.ones((3, 3)) / 9.0
                current = convolve2d(current, filter, mode="same")
            elif filters[i] == "tv":
                current = denoise_tv_chambolle(current)
            elif filters[i] == "wavelet":
                current = denoise_wavelet(current)

    if i in to_flip:
        current *= -1

    end = time.time()

    fulltitle = (
        args.labels[i]
        + f"{'+' + str(passes[i]) + ' ' + filters[i] + ' passes' if i in to_filter and passes[i] > 0 else ''}"
        + str(end - start)
        + "s"
    )

    if args.show_original:
        ax[0, i].set_title(args.labels[i])

        im = ax[1, i].imshow(current, cmap=args.cmap)
        ax[1, i].set_title(
            f"{str(passes[i]) + ' ' + filters[i] + ' passes' if i in to_filter and passes[i] > 0 else ''}"
            + str(end - start)
            + "s"
        )
        if args.gradients is not None:
            im = ax[2, i].imshow(grad(current))
            ax[2, i].set_title(f"{args.gradients} filtered")
    else:
        if args.gradients is not None:
            im = ax[0, i].imshow(current, cmap=args.cmap)
            ax[0, i].set_title(fulltitle)
            im = ax[1, i].imshow(grad(current))
            ax[1, i].set_title(f"{args.gradients} filtered")
        else:
            im = ax[i].imshow(current, cmap=args.cmap)
            ax[i].set_title(fulltitle)

    if args.colorbar:
        plt.colorbar(im, ax=ax[0, i])

    f.close()

plt.suptitle(args.title)
plt.show()
