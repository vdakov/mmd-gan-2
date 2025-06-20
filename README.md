# Reproduction Project: MMD-GAN — Towards Deeper Understanding of Moment Matching Networks

This repository contains the code for our reproduction of the paper:

**[MMD-GAN: Towards Deeper Understanding of Moment Matching Networks](https://arxiv.org/abs/1705.08584)**  
by Chun-Liang Li, Wei-Cheng Chang, Yu Cheng, Yiming Yang, and Barnabás Póczos.

## Running the MMD-GAN Model

To install the dependencies and run the model, follow the steps below:

```bash
pip install -r requirements.txt
python run_mmd_gan.py --dataset mnist  # Options: mnist | cifar10 | celeba
