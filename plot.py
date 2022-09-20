import h5py
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", nargs="+", required=True, type=str, help="HDF5 file(s) to plot"
)
parser.add_argument("--norm", help="vmin/vmax", default=0.001, type=float)
parser.add_argument("--var", help="Variable to plot", default="jx", type=str)
args = parser.parse_args()

n = len(args.files)
fig, ax = plt.subplots(1, n, figsize=(6 + 4 * n, 5))

for i, fname in enumerate(args.files):
    f = h5py.File(fname, "r")
    im = ax[i].imshow(f[args.var], vmin=-args.norm, vmax=args.norm)
    plt.colorbar(im, ax=ax[i])
    f.close()
plt.show()
