#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from datasets.cifar10 import load_CIFAR
from datasets.control import load_control
from datasets.mnist import load_MNIST
from datasets.celebA import load_CELEB_A

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_mmd(X, Y, sigma_list, const_diagonal=False, biased=True):

    assert X.size(0) == Y.size(0), "Batch sizes of X and Y must match"
    m = X.size(0)

    Z = torch.cat((X, Y), dim=0)  # Shape: (2m, feature_dim)
    ZZT = torch.mm(Z, Z.t())       # Gram matrix (2m x 2m)
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)

    # Squared Euclidean distances matrix:
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    K_XX, K_XY, K_YY = K[:m, :m], K[:m, m:], K[m:, m:]

    m = K_XX.size(0)  # Batch size (assumed same for X and Y)

    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)
        diag_Y = torch.diag(K_YY)
        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = (
            Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m)
        )

    return mmd2


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif 'Linear' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


def get_dataloader(dataset_name="cifar10", batch_size=64, image_size=64, control_data=None):
    if dataset_name.lower() == "cifar10":
        trainloader, _, nc = load_CIFAR(batch_size=batch_size)
    elif dataset_name.lower() == "mnist":
        trainloader, _, nc = load_MNIST(batch_size=batch_size)
    elif dataset_name.lower() == "celeba":
        trainloader, _, _, nc = load_CELEB_A(batch_size=batch_size)
    elif dataset_name.lower() == "control":
        trainloader, _, _, _ = load_control(control_data, batch_size=1000)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainloader, nc

def plot_losses(losses_D, losses_G, iterations, save_path):
    plt.figure(figsize=(10,5))
    plt.plot(iterations, losses_D, label="Discriminator Loss")
    plt.plot(iterations, losses_G, label="Generator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def smooth_curve(values, window_size=100):
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

def plot_mmd2(mmd2_values, iterations, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, mmd2_values, label="MMD²", color='blue')
    plt.xlabel("Generator Iterations")
    plt.ylabel("MMD²")
    plt.title("MMD² Distance over Training")
    plt.yscale('log')  
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


