import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def set_same_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(x):
    a, b = x.min(), x.max()
    return (x - a) / (b - a)


def generate_images(generator):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(100, 100, 1, 1, device=device)
        images = generator(noise).moveaxis(1, 3)
        images = normalize(images.cpu().numpy())
    return images


def output_generate_images(generator, output_dir):
    for i in range(10):
        images = generate_images(generator)
        for j in range(100):
            plt.imsave(f'{output_dir}/{str(i * 100 + j + 1).zfill(4)}.png', images[j])


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, X):
        return self.main(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='output generated images directory path', type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator = Generator()
    generator = torch.load('generator.pth')
    generator = generator.to(device)

    output_dir = args.output
    set_same_seed(48)
    output_generate_images(generator, output_dir)
