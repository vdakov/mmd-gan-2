#!/usr/bin/env python
# encoding: utf-8

import torch

def compute_mmd(X, Y, sigma_list, const_diagonal=False, biased=True):

    X = X.view(X.size(0), -1)
    Y = Y.view(Y.size(0), -1)

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