import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=100, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, latent_dim))  # Output layer
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim=100, hidden_dims=[128, 256]):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU(True))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))  # Output layer
        layers.append(nn.Tanh())  # Optional: use based on your data range
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class OneSidedHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, real_feat_mean, fake_feat_mean):
        return -self.relu(real_feat_mean - fake_feat_mean).mean()


class Generator(nn.Module):
    def __init__(self, output_dim, latent_dim=100, hidden_dims=[128, 256]):
        super().__init__()
        self.decoder = Decoder(output_dim, latent_dim, hidden_dims)

    def forward(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim=100, enc_dims=[256, 128], dec_dims=[128, 256]):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, enc_dims)
        self.decoder = Decoder(input_dim, latent_dim, dec_dims)

    def forward(self, x):
        f_enc = self.encoder(x)               # (batch, latent_dim)
        f_dec = self.decoder(f_enc)           # (batch, input_dim)
        return f_enc, f_dec
