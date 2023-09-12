import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b2


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
        return image, img_name


class efficientnet_ft(nn.Module):
    def __init__(self):
        super(efficientnet_ft, self).__init__()
        base_model = efficientnet_b2(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(1408, 50))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def model_pred(model, dataloader):
    result = list()
    with torch.no_grad():
        for img, name in dataloader:
            img = img.to(device)
            pred = model(img).argmax(1)
            for i, n in enumerate(name):
                result.append([n, str(pred[i].item())])
    result.sort(key=lambda s: s[0])
    return result


def csv_output(test_gt, csv_path):
    test_gt.insert(0, ['filename', 'label'])
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_gt)


def same_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='test images directory path', type=str)
    parser.add_argument('-o', '--output', help='output prediction csv path', type=str)
    args = parser.parse_args()

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.ConvertImageDtype(torch.uint8),
                                         transforms.Resize((288, 288)),
                                         transforms.ConvertImageDtype(torch.float),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    batch_size = 256

    test_img_dir = args.input
    csv_path = args.output
    model_path = 'model_hw1_1.pth'

    test_data = TestDataset(img_dir=test_img_dir, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = efficientnet_ft()
    model = torch.load(model_path)
    model.to(device)

    same_seeds(1000)
    result = model_pred(model, test_dataloader)
    csv_output(result, csv_path)
