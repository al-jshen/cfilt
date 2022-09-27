import os
import h5py
import hdf5plugin
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


class CDS(Dataset):
    def __init__(self, low_ppc, high_ppc, j, out_dir, normalize=True, transform=None):

        super().__init__()
        self.low_ppc = low_ppc
        self.high_ppc = high_ppc
        self.transform = transform
        self.mean = {}
        self.std = {}

        self.images = {}

        nfiles_low = [
            int(i.split(".")[-1])
            for i in os.listdir("./out/")
            if i.startswith(f"out-{low_ppc}.")
        ]
        nfiles_high = [
            int(i.split(".")[-1])
            for i in os.listdir("./out/")
            if i.startswith(f"out-{high_ppc}.")
        ]
        assert set(nfiles_low) == set(nfiles_high)
        self.total_size = len(nfiles_low)

        for ppc in tqdm([low_ppc, high_ppc]):
            self.images[ppc] = {}
            js = []
            for i in tqdm(range(1, self.total_size + 1)):
                with h5py.File(f"{out_dir}/out-{ppc}.{str(i).zfill(3)}", "r") as f:
                    js.append(f[j][:])
            self.images[ppc] = np.stack(js)
            if normalize:
                self.mean[ppc] = self.images[ppc].mean()
                self.std[ppc] = self.images[ppc].std()
                self.images[ppc] = (self.images[ppc] - self.mean[ppc]) / self.std[ppc]

    def __len__(self):
        return self.total_size

    def __getitem__(self, i):
        lr = self.images[self.low_ppc][i]
        hr = self.images[self.high_ppc][i]

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr


def unpack(x):
    x1, x2 = list(zip(*x))
    x1, x2 = torch.stack(x1), torch.stack(x2)
    return x1, x2


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
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out = out + residual
        out = F.relu(out)
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
        x = F.relu(self.bn(x))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--epochs", type=int, default=2500, help="number of epochs of training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.8, help="weight for SSIM loss")
    parser.add_argument(
        "--im_channels", type=int, default=1, help="number of image channels"
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=2, help="number of feature maps"
    )
    parser.add_argument(
        "--num_res_blocks", type=int, default=2, help="number of residual blocks"
    )
    parser.add_argument(
        "--low_ppc", type=int, default=4, help="low resolution particles per cell"
    )
    parser.add_argument(
        "--high_ppc", type=int, default=4, help="low resolution particles per cell"
    )
    parser.add_argument("--sim_dir", type=str, help="simulation output directory")
    parser.add_argument("--var", type=str, help="variable to train on")
    parser.add_argument("--im_size", nargs="+", type=int, help="image size")
    parser.add_argument("--save_path", type=str, help="path to save model")
    parser.add_argument("--load_path", type=str, help="path to load model from")
    parser.add_argument("--model_name", type=str, help="name of saved model")
    args = parser.parse_args()

    osize = tuple(args.im_size)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize(tuple((np.array(osize) * 1.1).astype(int))),
            # transforms.TenCrop(osize),
            # transforms.Lambda(lambda crops: torch.stack(crops)),
        ]
    )
    ds = CDS(
        args.low_ppc,
        args.high_ppc,
        args.var,
        args.sim_dir,
        normalize=True,
        transform=tfm,
    )

    train_len = int(len(ds) * 0.95)
    train_ds, test_ds = random_split(ds, (train_len, len(ds) - train_len))
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=unpack
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True, collate_fn=unpack
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = ConvXCoder(
        osize,
        args.im_channels,
        args.hidden_channels,
        args.hidden_channels,
        args.num_res_blocks,
        device,
    )
    decoder = ConvXCoder(
        osize,
        args.hidden_channels,
        args.im_channels,
        args.hidden_channels,
        args.num_res_blocks,
        device,
    )
    autoencoder = nn.Sequential(encoder, decoder).to(device)

    if args.load_path is not None:
        autoencoder.load_state_dict(torch.load(args.load_path))
    else:

        loss_fn = lambda x, y: MS_SSIM_L1_Loss()(x, y) * args.alpha + nn.L1Loss()(
            x, y
        ) * (1 - args.alpha)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)

        losses = []

        for e in (pbar := tqdm(range(args.epochs))):

            for i, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)

                x = x.reshape(-1, 1, *osize)
                y = y.reshape(-1, 1, *osize)

                pred = autoencoder(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                nloss = loss.item() / x.shape[0]

                losses.append(nloss)

                if i % 50 == 0:
                    pbar.set_description(f"loss: {nloss:.2e}")

        if args.save_dir is not None:
            torch.save(autoencoder.state_dict(), f"{args.save_dir}/{args.model_name}")

        plt.plot(losses)
        plt.yscale("log")
        plt.savefig(f"{args.save_path}/loss.png", bbox_inches="tight")

    autoencoder.eval()
    x, y = next(iter(test_dl))
    x = x.reshape(-1, 1, *osize)
    pred = autoencoder(x.to(device))
    x = x.reshape(y.shape)
    pred = pred.reshape(y.shape)

    fig, ax = plt.subplots(8, 3, figsize=(15, 30))
    for i in range(8):
        ax[i, 0].imshow(x[i][0].cpu().detach().numpy())
        ax[i, 1].imshow(pred[i][0].cpu().detach().numpy())
        ax[i, 2].imshow(y[i][0].cpu().detach().numpy())
    ax[0, 0].set_title(f"{args.low_ppc} ppc")
    ax[0, 1].set_title("denoised image")
    ax[0, 2].set_title(f"{args.high_ppc} ppc")

    plt.savefig(f"{args.save_path}/results.png", bbox_inches="tight")
