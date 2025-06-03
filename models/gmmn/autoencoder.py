import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, n_inp, n_encoded):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(n_inp, 1024)
        self.encoder_fc2 = nn.Linear(1024, n_encoded)


        self.decoder_fc1 = nn.Linear(n_encoded, 1024)
        self.decoder_fc2 = nn.Linear(1024, n_inp)

    def forward(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return e, d

    def encode(self, x):
        x = torch.sigmoid(self.encoder_fc1(x))
        x = torch.sigmoid(self.encoder_fc2(x))
        return x

    def decode(self, x):
        x = torch.sigmoid(self.decoder_fc1(x))
        x = torch.sigmoid(self.decoder_fc2(x))  # Output activation depends on your data distribution
        return x
