import argparse
import os

import imageio
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16


class en_de(nn.Module):
    def __init__(self):
        super(en_de, self).__init__()
        self.encoder = vgg16(weights=None).features[:31]
        self.decoder = nn.Sequential(nn.Conv2d(512, 512, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(512, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(256, 128, 3, padding='same'), nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(128, 64, 3, padding='same'), nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(64, 7, 3, padding='same'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def class2color(classs):
    color_list = np.array([[0, 255, 255], [255, 255, 0], [255, 0, 255],
                           [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]])
    color_img = np.empty((512, 512, 3))
    for i in range(7):
        color_img[classs == i] = color_list[i]
    color_img = torch.tensor(np.transpose(color_img, (2, 0, 1)))
    return color_img


def model_pred(model, input_dir, output_dir, transform=None):
    t = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.uint8)])
    img_list = [file for file in os.listdir(input_dir) if file.endswith('.jpg')]
    img_list.sort()
    mask_list = img_list.copy()
    for i, m in enumerate(mask_list):
        mask_list[i] = m.replace('.jpg', '.png')
    for i, f in enumerate(img_list):
        img = imageio.v2.imread(f'{input_dir}/{f}')
        img = t(img)
        if transform:
            img = transform(img)
        img = torch.unsqueeze(img.to(torch.float), 0)
        img = img.to(device)
        pred_lebel = model(img)
        pred_class = torch.squeeze(pred_lebel.argmax(1), 0)
        pred_color = class2color(pred_class.to('cpu'))
        pred_mask = np.transpose(pred_color.to(torch.uint8), (1, 2, 0))
        save_path = f'{output_dir}/{mask_list[i]}'
        imageio.imsave(save_path, pred_mask)


if __name__ == '__main__':
    print('hello')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='test images directory path', type=str, required=True)
    parser.add_argument('-o', '--output', help='output masks directory path', type=str, required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_path = 'model_hw1_2.pth'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = en_de()
    model = torch.load(model_path)
    model.to(device)

    model_pred(model, input_dir, output_dir)
