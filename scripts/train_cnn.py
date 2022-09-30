import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import sys
sys.path.insert(0, '/scratch/gpfs/js5013/programs/cfilt')
from cfilt.utils import *

VERBOSE = 0


def dbg(x):
    if args.verbose:
        print(x)


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
    parser.add_argument("--train", action="store_true", help="keep training models")
    parser.add_argument("--verbose", action="store_true", help="print stuff")
    parser.add_argument(
        "--mixed", action="store_true", help="train with mixed precision"
    )
    parser.add_argument("--model_name", type=str, help="name of saved model")
    parser.add_argument(
        "--tencrop", action="store_true", help="do 10-crop data augmentation"
    )
    args = parser.parse_args()
    dbg("Args parsed")

    osize = tuple(args.im_size)
    tfm = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(tuple((np.array(osize) * 1.1).astype(int))),
                transforms.TenCrop(osize),
                transforms.Lambda(lambda crops: torch.stack(crops)),
            ]
        )
        if args.tencrop
        else transforms.ToTensor()
    )

    dbg(
        f"Making dataset with low_ppc={args.low_ppc}, high_ppc={args.high_ppc}, and with tencrop={str(args.tencrop)}"
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
    dbg(f"training set has {len(train_ds)} samples")

    dl_cfg = dict(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=unpack,
        num_workers=16,
        pin_memory=True,
    )
    train_dl = DataLoader(train_ds, **dl_cfg)
    test_dl = DataLoader(test_ds, **dl_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dbg(f"Using device {device}")

    dbg("Making model")
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
    autoencoder = nn.DataParallel(
        autoencoder, device_ids=list(range(torch.cuda.device_count()))
    )

    if args.load_path is not None:
        autoencoder.load_state_dict(torch.load(args.load_path))
        dbg(f"Loaded saved model from {args.load_path}")
    elif os.path.isfile(f"{args.save_path}/{args.model_name}"):
        autoencoder.load_state_dict(torch.load(f"{args.save_path}/{args.model_name}"))
        dbg(f"Model already exists, loading from {args.save_path}/{args.model_name}")
    if args.train:
        dbg("Beginning training loop")
        loss_fn = lambda x, y: MS_SSIM_L1_Loss(alpha=args.alpha)(x, y)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.mixed)

        losses = []

        for e in (pbar := tqdm(range(args.epochs))):

            for i, (x, y) in enumerate(train_dl):
                optimizer.zero_grad()

                with torch.cuda.device(0):
                    x, y = x.to(device), y.to(device)
                x = x.reshape(-1, 1, *osize)
                y = y.reshape(-1, 1, *osize)

                with torch.cuda.amp.autocast(enabled=args.mixed):
                    pred = autoencoder(x)
                    pred_denorm = pred * ds.std[args.low_ppc] + ds.mean[args.low_ppc]
                    y_denorm = y * ds.std[args.high_ppc] + ds.mean[args.high_ppc]
                    loss = loss_fn(pred_denorm, y_denorm)
                    loss /= x.shape[0]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())

                pbar.set_description(f"loss: {loss.item():.2e}")

        dbg(f"Saving model to {args.save_path}/{args.model_name}")
        torch.save(autoencoder.state_dict(), f"{args.save_path}/{args.model_name}")

        dbg("Plotting losses")
        plt.plot(losses)
        plt.yscale("log")
        plt.savefig(
            f"{args.save_path}/{args.model_name}-loss.png",
            bbox_inches="tight",
        )

    dbg("Making test images")
    autoencoder.eval()
    x, y = next(iter(test_dl))
    x = x.reshape(-1, 1, *osize)
    pred = autoencoder(x.to(device))
    x = x.reshape(y.shape)
    pred = pred.reshape(y.shape)

    fig, ax = plt.subplots(8, 4, figsize=(18, 30))
    index = lambda x, i, crop: x[i][0][0] if crop else x[i][0]
    for i in range(8):
        ax[i, 0].imshow(index(x, i, args.tencrop).cpu().detach().numpy())
        ax[i, 1].imshow(index(pred, i, args.tencrop).cpu().detach().numpy())
        ax[i, 2].imshow(index(y, i, args.tencrop).cpu().detach().numpy())
        ax[i, 3].imshow(index(y - pred.cpu(), i, args.tencrop).cpu().detach().numpy())
    ax[0, 0].set_title(f"{args.low_ppc} ppc")
    ax[0, 1].set_title("denoised image")
    ax[0, 2].set_title(f"{args.high_ppc} ppc")
    ax[0, 3].set_title("residual")

    plt.savefig(
        f"{args.save_path}/{args.model_name}-results.png",
        bbox_inches="tight",
    )
