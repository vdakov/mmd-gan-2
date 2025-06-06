import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import amp
import argparse

from modules import Generator, Discriminator, OneSidedHingeLoss
from utils import compute_mmd, weights_init, get_dataloader, plot_losses, smooth_curve, plot_mmd2, grad_norm



def main(dataset_name="cifar10"):


    batch_size = 64
    image_size = 64
    
    if dataset_name.lower() == "mnist":
        nz = 16
        image_size = 32
    elif dataset_name.lower() == "celeba":
        nz = 64
    elif dataset_name.lower() in ["cifar10", "lsun"]:
        nz = 128
        image_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    ngf = 64
    ndf = 64
    lr = 0.00005
    max_iter = 10000
    experiment_dir = os.path.join('./mmdgan_experiment', 'mnist16')
    sigma_list = [1, 2, 4, 8, 16]

    losses_D = []
    losses_G = []
    mmd2_values = []
    iterations = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(experiment_dir, exist_ok=True)

    # Data loader
    dataloader, nc = get_dataloader(dataset_name=dataset_name, batch_size=batch_size, image_size=image_size)

    # Create models
    netG = Generator(image_size, nc, nz, ngf).to(device)
    netD = Discriminator(image_size, nc, nz, ndf, ngf).to(device)
    hinge_loss = OneSidedHingeLoss().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)
    hinge_loss.apply(weights_init)

    # Optimizers
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)

    one = torch.tensor(1.0, device=device)
    mone = one * -1

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    print("Starting training...")

    gen_iterations = 0
    start_time = time.time()

    for epoch in range(max_iter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):

            # (1) Update D network
            for p in netD.parameters():
                p.requires_grad = True

            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = 5

            for _ in range(Diters):

                if i == len(dataloader):
                    break
                
                for p in netD.encoder.parameters():
                    p.data.clamp_(-0.01, 0.01)

                data = next(data_iter)
                i += 1

                netD.zero_grad()

                real_images = data[0].to(device)
                batch_size_cur = real_images.size(0)

                # Forward pass real images through D
                f_enc_real, f_dec_real = netD(real_images)

                # Sample noise and generate fake images
                noise = torch.randn(batch_size_cur, nz, 1, 1, device=device)
                fake_images = netG(noise).detach()
                f_enc_fake, f_dec_fake = netD(fake_images)

                # Compute MMD loss between real and fake features

                mmd2 = compute_mmd(f_enc_real, f_enc_fake, sigma_list)
                mmd2 = nn.functional.relu(mmd2)

                # Autoencoder losses (L2 between input images and decoded output)
                L2_real = nn.functional.mse_loss(f_dec_real.view(batch_size_cur, -1), real_images.view(batch_size_cur, -1))
                L2_fake = nn.functional.mse_loss(f_dec_fake.view(batch_size_cur, -1), fake_images.view(batch_size_cur, -1))


                # Full discriminator loss (adjust weights as per paper)
                lambda_ae_x = 8.0
                lambda_ae_y = 8.0
                lambda_rg = 16.0

                hinge = hinge_loss(f_enc_real.mean(0), f_enc_fake.mean(0))
                loss_D = torch.sqrt(mmd2) + lambda_rg * hinge - lambda_ae_x * L2_real - lambda_ae_y * L2_fake

                loss_D.backward(mone)
                optimizerD.step()

                last_loss_D = loss_D.detach()

            # (2) Update G network

            for p in netD.parameters():
                p.requires_grad = False

            for _ in range(1):

                if i == len(dataloader):
                    break
                data = next(data_iter)
                i += 1

                netG.zero_grad()

                real_images = data[0].to(device)
                batch_size_cur = real_images.size(0)

                f_enc_real, _ = netD(real_images)

                noise = torch.randn(batch_size_cur, nz, 1, 1, device=device)
                fake_images = netG(noise)

                f_enc_fake, _ = netD(fake_images)

                mmd2 = compute_mmd(f_enc_real, f_enc_fake, sigma_list)
                mmd2 = nn.functional.relu(mmd2)

                mmd2_values.append(mmd2.item())

                hinge_g = hinge_loss(f_enc_real.mean(0), f_enc_fake.mean(0))
                loss_G = torch.sqrt(mmd2) + lambda_rg * hinge_g

                loss_G.backward(one)
                optimizerG.step()

                losses_D.append(last_loss_D.item())
                losses_G.append(loss_G.item())
                iterations.append(gen_iterations)

                gen_iterations += 1

            # Logging
            if gen_iterations % 500 == 0:
                elapsed = (time.time() - start_time) / 60

                gD = grad_norm(netD)
                gG = grad_norm(netG)

                print(
                    f"[{epoch}/{max_iter}][{i}/{len(dataloader)}] Iter: {gen_iterations} "
                    f"Loss_D: {loss_D.item():.6f} Loss_G: {loss_G.item():.6f} "
                    f"|gD|: {gD:.4e} |gG|: {gG:.4e} Time(min): {elapsed:.2f}"
                )


            # Save samples every 500 iters
            if gen_iterations % 1000 == 0:
                with torch.no_grad():
                    fake_samples = netG(fixed_noise).detach().cpu()
                    fake_samples = (fake_samples + 1) / 2  # rescale [-1,1] to [0,1]
                    vutils.save_image(fake_samples, os.path.join(experiment_dir, f"fake_samples_{gen_iterations}.png"), nrow=8)

            # Save models every 1000 iters
            if gen_iterations % 5000 == 0:
                torch.save(netG.state_dict(), os.path.join(experiment_dir, f"netG_{gen_iterations}.pth"))
                torch.save(netD.state_dict(), os.path.join(experiment_dir, f"netD_{gen_iterations}.pth"))

                smoothed_mmd2 = smooth_curve(mmd2_values)
                plot_mmd2(smoothed_mmd2, iterations[-len(smoothed_mmd2):], os.path.join(experiment_dir, f"mmd2_plot_{gen_iterations}.png"))

                plot_losses(losses_D, losses_G, iterations, os.path.join(experiment_dir, f"loss_plot_{gen_iterations}.png"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use: cifar10 | mnist | celeba')
    args = parser.parse_args()

    main(dataset_name=args.dataset)
