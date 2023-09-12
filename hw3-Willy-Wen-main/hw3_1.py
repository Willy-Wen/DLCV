import os
import csv
import json
import tqdm
import random
import argparse
import numpy as np
import PIL.Image as Image

import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms


def set_same_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def csv_output(test_gt: list, csv_path: str):
    test_gt_copy = test_gt.copy()
    test_gt_copy.insert(0, ['filename', 'label'])
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_gt_copy)
        
        
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
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='test images directory path', type=str)
    parser.add_argument('-j', '--json', help='id2label json path', type=str)
    parser.add_argument('-o', '--output', help='output prediction csv path', type=str)
    args = parser.parse_args()

    test_dir_path = args.input
    json_path = args.json
    csv_path = args.output
    
    set_same_seed(0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    test_dataset = TestDataset(test_dir_path, transform=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    with open(json_path, newline='') as file:
        json_dict = json.load(file)
    json_values = list(json_dict.values())
    json_keys = list(json_dict.keys())
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in json_values]).to(device)
    total = len(test_dataloader)
    test_gt = []
    model.eval()
    with tqdm.tqdm(total=total) as pbar:
        with torch.no_grad():
            for i, (images, names) in enumerate(test_dataloader):
                #print(i, end='\r')
                image_input = images.to(device)
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                for j, s in enumerate(similarity):
                    test_gt.append([names[j], s.argmax().item()])
                pbar.update(1)
            
    csv_output(test_gt, csv_path)