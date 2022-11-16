import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(200, 512, 4, 2, 0),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),

            nn.Conv3d(256, 512, 4, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),

            nn.Conv3d(512, 1, 4, 2, 0),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.main(x).view(-1)
