import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import os
import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment


ngf = 64
nz = 256

# input params
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"
im_size = 256
bs = 4
checkpoint = "train_results/potsdam_cars/models/all_45000.pth"


ckpt = torch.load(checkpoint)

netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
netG.to(device)
netG.load_state_dict(ckpt['g'])
avg_param_G = ckpt['g_ema']
load_params(netG, avg_param_G)

noise_a = torch.randn(bs, nz).to(device)
noise_b = torch.randn(bs, nz).to(device)


def get_early_features(net, noise):
    feat_4 = net.init(noise)
    feat_8 = net.feat_8(feat_4)
    feat_16 = net.feat_16(feat_8)
    feat_32 = net.feat_32(feat_16)
    feat_64 = net.feat_64(feat_32)
    return feat_8, feat_16, feat_32, feat_64


def get_late_features(net, im_size, feat_64, feat_8, feat_16, feat_32):
    feat_128 = net.feat_128(feat_64)
    feat_128 = net.se_128(feat_8, feat_128)

    feat_256 = net.feat_256(feat_128)
    feat_256 = net.se_256(feat_16, feat_256)
    if im_size == 256:
        return net.to_big(feat_256)

    feat_512 = net.feat_512(feat_256)
    feat_512 = net.se_512(feat_32, feat_512)
    if im_size == 512:
        return net.to_big(feat_512)

    feat_1024 = net.feat_1024(feat_512)
    return net.to_big(feat_1024)


feat_8_a, feat_16_a, feat_32_a, feat_64_a = get_early_features(netG, noise_a)
feat_8_b, feat_16_b, feat_32_b, feat_64_b = get_early_features(netG, noise_b)

images_b = get_late_features(
    netG, im_size, feat_64_b, feat_8_b, feat_16_b, feat_32_b)
images_a = get_late_features(
    netG, im_size, feat_64_a, feat_8_a, feat_16_a, feat_32_a)

imgs = [torch.ones(1, 3, im_size, im_size)]
imgs.append(images_b.cpu())
for i in range(bs):
    imgs.append(images_a[i].unsqueeze(0).cpu())

    gimgs = get_late_features(netG, im_size, feat_64_a[i].unsqueeze(
        0).repeat(bs, 1, 1, 1), feat_8_b, feat_16_b, feat_32_b)
    imgs.append(gimgs.cpu())

imgs = torch.cat(imgs)
vutils.save_image(imgs.add(1).mul(0.5), os.path.join(os.path.split(
    os.path.split(checkpoint)[0])[0], 'style_mix_1.jpg'), nrow=bs+1)
