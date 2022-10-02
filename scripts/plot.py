import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve2d, medfilt2d
from scipy.ndimage import gaussian_filter
from skimage import filters
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_wavelet,
    estimate_sigma,
    denoise_nl_means,
    denoise_bilateral,
)
import time
import torch
import torch.nn as nn
import sys

sys.path.append("/scratch/gpfs/js5013/programs/cfilt/")
from cfilt.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    choices=[
        "gaussian",
        "median",
        "uniform",
        "tv",
        "wavelet",
        "nlmeans",
        "bilateral",
        "nn",
    ],
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
parser.add_argument("--row", type=int, help="Take a slice along a row")
parser.add_argument("--col", type=int, help="Take a slice along a column")
parser.add_argument("--save_path", type=str, help="Path to save the image to")
parser.add_argument("--no_show", action="store_true", help="Don't show the plot")
parser.add_argument(
    "--nn_layers", type=int, help="Number of resblocks in neural network"
)
parser.add_argument(
    "--nn_hidden", type=int, help="Number of hidden channels in neural network"
)
parser.add_argument(
    "--nn_path", type=str, help="Path to load neural network state_dict from"
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
if args.row is not None:
    nrows += 1
if args.col is not None:
    nrows += 1
if args.gradients is not None:
    nrows += 1

fig, ax = plt.subplots(
    nrows,
    n,
    figsize=(4 + 3 * n, 4 + nrows * 3),
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


if "nn" in filters:
    model = nn.Sequential(
        ConvXCoder(
            (125, 133),
            1,
            args.nn_hidden,
            args.nn_hidden,
            args.nn_layers,
            device,
        ),
        ConvXCoder(
            (125, 133),
            args.nn_hidden,
            1,
            args.nn_hidden,
            args.nn_layers,
            device,
        ),
    )
    model.load_state_dict(remove_data_parallel(torch.load(args.nn_path)))
    model.eval()


def filter(current, kind, passes):
    if kind == "nn":
        imsize = current.shape
        current = current.reshape(1, 1, *imsize)
        current = model(torch.from_numpy(current).to(device)).cpu().detach().numpy()
        current = current.reshape(imsize)
    else:
        for _ in range(passes):
            if kind == "median":
                current = medfilt2d(current, kernel_size=3)
            elif kind == "gaussian":
                current = gaussian_filter(current, sigma=1)
            elif kind == "uniform":
                filter = np.ones((3, 3)) / 9.0
                current = convolve2d(current, filter, mode="same")
            elif kind == "tv":
                current = denoise_tv_chambolle(current)
            elif kind == "wavelet":
                current = denoise_wavelet(current)
            elif kind == "nlmeans":
                sigma_est = np.mean(estimate_sigma(current))
                current = denoise_nl_means(
                    current,
                    h=0.8 * sigma_est,
                    sigma=sigma_est,
                    patch_size=5,
                    patch_distance=6,
                    fast_mode=False,
                )
            elif kind == "bilateral":
                sigma_est = np.mean(estimate_sigma(current))
                shift = current.min() - 1
                current = denoise_bilateral(
                    current - shift, sigma_color=sigma_est, sigma_spatial=15
                )
                current += shift
    return current


for i, fname in enumerate(args.files):

    f = h5py.File(fname, "r")
    current = f[args.var][:]

    if current.ndim == 3:
        current = current[0, :, :]

    if i in to_flip:
        current *= -1

    ctr = 0
    if args.show_original:
        im = ax[ctr, i].imshow(current, cmap=args.cmap)
        ax[ctr, i].set_title(args.labels[i])
        ctr += 1

    start = time.time()

    if i in to_filter:
        current = filter(current, filters[i], passes[i])

    end = time.time()

    title = (
        f"{str(passes[i]) + ' ' + filters[i] + ' passes' if i in to_filter and passes[i] > 0 else ''}"
        + f" {end - start:.2f}s"
        if i in to_filter
        else ""
    )
    fulltitle = args.labels[i] + " " + title

    fim = ax[ctr, i].imshow(current, interpolation="none", cmap=args.cmap)
    ax[ctr, i].set_title(fulltitle if ctr == 0 else title)
    fctr = ctr
    ctr += 1

    if args.row is not None:
        ax[ctr, i].plot(current[args.row, :])
        ax[ctr, i].set_title(f"Row {args.row}")
        ctr += 1

    if args.col is not None:
        ax[ctr, i].plot(current[:, args.col])
        ax[ctr, i].set_title(f"Column {args.col}")
        ctr += 1

    if args.gradients is not None:
        ax[ctr, i].imshow(grad(current), interpolation="none")
        ax[ctr, i].set_title(f"{args.gradients} filtered")
        ctr += 1

    assert ctr == nrows

    if args.colorbar:
        if args.show_original:
            fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
        fig.colorbar(fim, ax=ax[fctr, i], fraction=0.046, pad=0.04)

    f.close()

plt.suptitle(args.title)

if args.save is not None:
    plt.savefig(args.save, bbox_inches="tight", dpi=300)

if not args.no_show:
    plt.show()
