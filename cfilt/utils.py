import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from tqdm.auto import tqdm
import h5py
import hdf5plugin
import numpy as np
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology


crop_fn = lambda x: x[3:-1, 3:-2]


def erode_image(image, **kwargs):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    filled = morphology.reconstruction(seed, image, method='erosion', **kwargs)
    return filled


def dilate_image(image, **kwargs):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    rec = morphology.reconstruction(seed, image, method='dilation', **kwargs)
    return rec

def discretize_image(image, bins, quantile=False):
    im_copy = np.copy(image)
    if quantile:
        cbins = np.quantile(im_copy, np.linspace(0, 1, bins + 1))
    else:
        cbins = np.linspace(im_copy.min(), im_copy.max(), bins + 1)
    for c in range(cbins.size - 1):
        im_copy[np.where((im_copy >= cbins[c]) & (im_copy <= cbins[c + 1]))] = (cbins[c] + cbins[c + 1]) / 2.0
    return im_copy

class CDS(Dataset):
    def __init__(self, ppcs, j, out_dir, crop=True, normalize=True, transform=None):

        super().__init__()
        self.ppcs = ppcs
        self.transform = transform
        self.mean = {}
        self.std = {}

        self.images = {}

        nfiles = [
            [
                int(i.split(".")[-1])
                for i in os.listdir(out_dir)
                if i.startswith(f"out-{ppc}.")
            ]
            for ppc in ppcs
        ]
        setfiles = [set(i) for i in nfiles]
        assert (
            len(np.unique(setfiles)) == 1
        ), "not all ppcs have the same number of files"
        self.total_size = len(nfiles[0])

        for ppc in tqdm(ppcs):
            self.images[ppc] = {}
            js = []
            for i in range(1, self.total_size + 1):
                with h5py.File(f"{out_dir}/out-{ppc}.{str(i).zfill(3)}", "r") as f:
                    if crop:
                        js.append(crop_fn(f[j][:]))
                    else:
                        js.append(f[j][:])
            self.images[ppc] = np.stack(js)
            if normalize:
                self.mean[ppc] = self.images[ppc].mean()
                self.std[ppc] = self.images[ppc].std()
                self.images[ppc] = (self.images[ppc] - self.mean[ppc]) / self.std[ppc]

    def __len__(self):
        return self.total_size

    def __getitem__(self, i):
        ims = [self.images[ppc][i] for ppc in self.ppcs]

        if self.transform:
            ims = [self.transform(im) for im in ims]

        return ims


def unpack(x):
    xs = list(zip(*x))
    xs = [torch.stack(i) for i in xs]
    return xs


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ResidualBlock(nn.Module):
    def __init__(self, channels):

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        out = F.gelu(out)
        return out


class ConvXCoder(nn.Module):
    def __init__(
        self,
        im_shape,
        in_channels,
        out_channels,
        hidden_channels,
        num_res_blocks,
        device,
    ):
        super().__init__()
        self.im_shape = im_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, hidden_channels, (3, 3), padding=1).to(
            device
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels).to(device) for _ in range(num_res_blocks)]
        )
        self.bn = nn.BatchNorm2d(hidden_channels).to(device)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, (3, 3), padding=1).to(
            device
        )

        if im_shape is not None:
            self.view = View(-1, self.in_channels, *self.im_shape)
        else:
            self.view = None

    def forward(self, x):
        if self.view:
            x = self.view(x)

        x = self.conv_in(x)
        x = F.gelu(self.bn(x))

        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.conv_out(x)
        return x


class MS_SSIM_L1_Loss(nn.Module):
    def __init__(
        self,
        gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
        data_range=1.0,
        K=(0.01, 0.03),
        alpha=0.025,
        compensation=200.0,
        cuda_dev=0,
    ):
        super().__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction="none")  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(
            loss_l1,
            self.g_masks.narrow(dim=0, start=-3, length=3),
            groups=1,
            padding=self.pad,
        ).mean(
            1
        )  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


class Cascade(nn.Module):
    def __init__(self, base_model, load_paths):
        super().__init__()
        self.models = [copy.deepcopy(base_model) for _ in range(len(load_paths))]
        for i, m in tqdm(enumerate(self.models), total=len(load_paths)):
            state_dict = torch.load(load_paths[i])
            state_dict = remove_data_parallel(state_dict)
            m.load_state_dict(state_dict)
        self.models = nn.ModuleList(self.models)
        self.intermediates = []

    def get_intermediates(self):
        return self.intermediates

    def forward(self, x):
        self.intermediates.clear()
        for m in self.models:
            x = m(x)
            self.intermediates.append(x)
        return x



def qplot(ims, fn=None):
    if fn is not None:
        ims = list(map(fn, ims))
    n = len(ims)
    fig, ax = plt.subplots(1, n, figsize=(2 + 3 * n, 3.5))
    for i in range(n):
        ax[i].pcolormesh(ims[i])
    return fig, ax


fft2d = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
ifft2d = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x))).real
