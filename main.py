import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from classes import Generator, Discriminator, weights_init
from utils import compute_mmd


def get_dataloader(batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def main():
    # Hyperparameters (adjust as needed)
    batch_size = 64
    image_size = 64
    nc = 3  # number of channels
    nz = 100  # latent dimension
    ngf = 64
    ndf = 64
    lr = 0.00005
    max_iter = 10000
    experiment_dir = './mmdgan_experiment'
    sigma_list = [1, 2, 4, 8, 16]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(experiment_dir, exist_ok=True)

    # Create models
    netG = Generator(image_size, nc, nz, ngf).to(device)
    netD = Discriminator(image_size, nc, nz, ndf, ngf).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Optimizers
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)

    # Data loader
    dataloader = get_dataloader(batch_size, image_size)

    one = torch.tensor(1.0, device=device)
    mone = one * -1

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    print("Starting training...")

    gen_iterations = 0
    start_time = time.time()

    for epoch in range(max_iter):
        data_iter = iter(dataloader)
        i = 0
        print("Epoch: ", epoch)
        while i < len(dataloader):

            # (1) Update D network
            for p in netD.parameters():
                p.requires_grad = True

            Diters = 5 if gen_iterations >= 25 else 100
            for a in range(Diters):

                if i == len(dataloader):
                    break
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

                # One-sided loss: to push real > fake in feature space (requires your own one-sided loss function)
                # For brevity, omitted here. Add as needed.

                # Full discriminator loss (adjust weights as per paper)
                lambda_ae_x = 8.0
                lambda_ae_y = 8.0
                lambda_rg = 16.0
                loss_D = torch.sqrt(mmd2) - lambda_ae_x * L2_real - lambda_ae_y * L2_fake

                loss_D.backward(mone)
                optimizerD.step()

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

                loss_G = torch.sqrt(mmd2)
                loss_G.backward(one)
                optimizerG.step()

                gen_iterations += 1

            # Logging
            if gen_iterations % 100 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"[{epoch}/{max_iter}][{i}/{len(dataloader)}] Iter: {gen_iterations} Loss_D: {loss_D.item():.6f} Loss_G: {loss_G.item():.6f} Time(min): {elapsed:.2f}")

            # Save samples every 500 iters
            if gen_iterations % 500 == 0:
                with torch.no_grad():
                    fake_samples = netG(fixed_noise).detach().cpu()
                    fake_samples = (fake_samples + 1) / 2  # rescale [-1,1] to [0,1]
                    vutils.save_image(fake_samples, os.path.join(experiment_dir, f"fake_samples_{gen_iterations}.png"), nrow=8)

            # Save models every 1000 iters
            if gen_iterations % 1000 == 0:
                torch.save(netG.state_dict(), os.path.join(experiment_dir, f"netG_{gen_iterations}.pth"))
                torch.save(netD.state_dict(), os.path.join(experiment_dir, f"netD_{gen_iterations}.pth"))


if __name__ == "__main__":
    main()
