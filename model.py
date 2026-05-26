import math

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.encoder(x).flatten(1)


class CrossAttention(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.scale = math.sqrt(latent_dim)

    def forward(self, f_u, f_d):
        q = self.query(f_d)
        k = self.key(f_u)
        v = self.value(f_u)
        weights = torch.softmax(torch.matmul(q, k.transpose(0, 1)) / self.scale, dim=-1)
        return torch.matmul(weights, v)


class Gate(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())

    def forward(self, f_u, f_d):
        return self.fc(torch.cat([f_u, f_d], dim=1))


class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 512, 4, 4)
        return self.decoder(z)


class MDOC(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.en_uf = ConvEncoder(in_channels, latent_dim)
        self.en_ds = ConvEncoder(in_channels, latent_dim)
        self.attention = CrossAttention(latent_dim)
        self.gate = Gate(latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x, noise_std=0.4472136):
        if noise_std > 0:
            x_u = x + torch.randn_like(x) * noise_std
        else:
            x_u = x
        f_u = self.en_uf(x_u)
        f_d = self.en_ds(x)
        f_fused = self.attention(f_u, f_d)
        gate = self.gate(f_u, f_d)
        f_combined = gate * f_fused + (1 - gate) * f_u
        recon = self.decoder(f_combined)
        return recon, f_u, f_d, f_fused, gate


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)
