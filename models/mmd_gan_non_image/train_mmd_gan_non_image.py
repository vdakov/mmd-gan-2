import os
import time
import torch
from tqdm import tqdm

from models.mmd_gan_non_image.mmd_gan_vector import Discriminator, Generator
from modules import OneSidedHingeLoss
from utils import compute_mmd, grad_norm, plot_losses, plot_mmd2, smooth_curve, weights_init
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils


def train_mmd_gan_vector(dataloader, data_dim, latent_dim=100, hidden_dims_enc=[256, 128], hidden_dims_dec=[128, 256], batch_size=64, max_iter=10000, experiment_name='mmdgan_experiment', sigma_list=[1, 2, 4, 8, 16]):
    max_iter = max_iter
    lr = 0.00005
    experiment_dir = os.path.join(f'./mmdgan_experiment/{experiment_name}')

    losses_D = []
    losses_G = []
    mmd2_values = []
    iterations = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(f"{experiment_dir}/{experiment_name}", exist_ok=True)

    # Data load

    # Create models
    netG = Generator(data_dim, latent_dim, hidden_dims_dec).to(device)
    netD = Discriminator(data_dim, latent_dim, hidden_dims_enc, hidden_dims_dec).to(device)
    hinge_loss = OneSidedHingeLoss().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)
    hinge_loss.apply(weights_init)

    # Optimizers
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)

    one = torch.tensor(1.0, device=device)
    mone = one * -1

    fixed_noise = torch.randn(batch_size, latent_dim, device=device)
    
    if os.path.exists(os.path.join(experiment_dir, f"netG_{max_iter}.pth")):
        save_path = os.path.join(experiment_dir, experiment_name)
        print("Loading existing models...")
        checkpoint = torch.load(save_path)
        
        
    else:

        print("Starting training...")

        gen_iterations = 0
        start_time = time.time()

        for epoch in tqdm(range(max_iter), desc="Training Epochs"):
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
                    noise = torch.randn(batch_size_cur, latent_dim, device=device)
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

                    noise = torch.randn(batch_size_cur, latent_dim, device=device)
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

                # Save models every 1000 iters
                if gen_iterations % 5000 == 0:
                    torch.save(netG.state_dict(), os.path.join(experiment_dir, f"netG_{gen_iterations}.pth"))
                    torch.save(netD.state_dict(), os.path.join(experiment_dir, f"netD_{gen_iterations}.pth"))

                    smoothed_mmd2 = smooth_curve(mmd2_values)
                    plot_mmd2(smoothed_mmd2, iterations[-len(smoothed_mmd2):], os.path.join(experiment_dir, f"mmd2_plot_{gen_iterations}.png"))

                    plot_losses(losses_D, losses_G, iterations, os.path.join(experiment_dir, f"loss_plot_{gen_iterations}.png"))
                
    return netG, netD, losses_D, losses_G, mmd2_values, iterations
