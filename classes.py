import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, isize, nc, k=100, ndf=64):
        super().__init__()
        assert isize % 16 == 0, "image size has to be a multiple of 16"

        layers = []
        # Initial conv + LeakyReLU (no BatchNorm)
        layers.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = isize // 2, ndf

        # Downsample until 4x4 spatial size
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.append(nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            cndf = out_feat
            csize = csize // 2

        # Final conv to latent vector (k channels, 1x1 spatial)
        layers.append(nn.Conv2d(cndf, k, 4, 1, 0, bias=False))

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super().__init__()
        assert isize % 16 == 0, "image size has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf *= 2
            tisize *= 2

        layers = []
        # Initial conv transpose + BatchNorm + ReLU
        layers.append(nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(cngf))
        layers.append(nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            layers.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(cngf // 2))
            layers.append(nn.ReLU(True))
            cngf //= 2
            csize *= 2

        # Final conv transpose to output image channels + Tanh
        layers.append(nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    """
    Generator wraps the Decoder.
    Input: noise tensor (batch_size, k, 1, 1)
    Output: generated image tensor (batch_size, nc, isize, isize)
    """
    def __init__(self, isize, nc, nz=100, ngf=64):
        super().__init__()
        self.decoder = Decoder(isize, nc, k=nz, ngf=ngf)

    def forward(self, input):
        return self.decoder(input)


class Discriminator(nn.Module):
    """
    Discriminator wraps Encoder + Decoder.
    Input: image tensor (batch_size, nc, isize, isize)
    Outputs:
        f_enc: latent feature (batch_size, k, 1, 1)
        f_dec: reconstructed image (batch_size, nc, isize, isize)
    """
    def __init__(self, isize, nc, nz=100, ndf=64, ngf=64):
        super().__init__()
        self.encoder = Encoder(isize, nc, k=nz, ndf=ndf)
        self.decoder = Decoder(isize, nc, k=nz, ngf=ngf)

    def forward(self, input):
        f_enc = self.encoder(input)              # (batch, nz, 1, 1)
        f_dec = self.decoder(f_enc)              # (batch, nc, isize, isize)
        return f_enc, f_dec


#TODO: change this implementations
def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


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

