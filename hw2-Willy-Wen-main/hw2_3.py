import os
import csv
import random
import argparse
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def set_same_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def csv2list(csv_path):
    gt = []
    with open(csv_path, newline='') as file:
        rows = csv.reader(file)
        for row in rows:
            gt.append(row)
    gt.pop(0)
    return gt


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = [file for file in os.listdir(img_dir) if file.endswith('.png')]
        self.transform = transform
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = f'{self.img_dir}/{img_name}'
        image = np.array(Image.open(img_path))
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image, img_name


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


def model_pred(test_dataloader, feature_extractor, label_predictor):
    result = list()
    feature_extractor.eval()
    label_predictor.eval()
    with torch.no_grad():
        for img, name in test_dataloader:
            img = img.to(device)
            pred = feature_extractor(img)
            pred = label_predictor(pred).argmax(1)
            for i, n in enumerate(name):
                result.append([n, str(pred[i].item())])
    result.sort(key=lambda s: s[0])
    return result


def csv_output(test_gt, csv_path):
    test_gt.insert(0, ['image_name', 'label'])
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='test images directory path', type=str)
    parser.add_argument('-o', '--output', help='output prediction csv path', type=str)
    args = parser.parse_args()

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((32, 32))])

    test_img_dir = args.input
    csv_path = args.output
    if 'svhn' in test_img_dir:
        feature_extractor_path = 'feature_extractor_mnistm2svhn.pth'
        label_predictor_path = 'label_predictor_mnistm2svhn.pth'
        print('load mnistm2svhn')
    elif 'usps' in test_img_dir:
        feature_extractor_path = 'feature_extractor_mnistm2usps.pth'
        label_predictor_path = 'label_predictor_mnistm2usps.pth'
        print('load mnistm2usps')

    test_data = TestDataset(img_dir=test_img_dir, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    feature_extractor = FeatureExtractor()
    feature_extractor = torch.load(feature_extractor_path)
    feature_extractor = feature_extractor.to(device)

    label_predictor = LabelPredictor()
    label_predictor = torch.load(label_predictor_path)
    label_predictor = label_predictor.to(device)

    result = model_pred(test_dataloader, feature_extractor, label_predictor)
    csv_output(result, csv_path)
