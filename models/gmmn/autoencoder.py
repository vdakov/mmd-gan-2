import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, n_inp, n_encoded, dropout_prob=0.5):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(n_inp, 64)
        self.encoder_fc2 = nn.Linear(64, n_encoded)
        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob)


        self.decoder_fc1 = nn.Linear(n_encoded, 64)
        self.decoder_fc2 = nn.Linear(64, n_inp)

    def forward(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return e, d

    def encode(self, x):
        x = self.encoder_fc1(x)
        x = self.dropout1(x)
        x = torch.sigmoid(self.encoder_fc2(x))
  
        return x

    def decode(self, x):
        x = torch.sigmoid(self.decoder_fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.decoder_fc2(x))
        return x
