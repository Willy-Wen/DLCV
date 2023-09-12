import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from torchvision import models, datasets, transforms

import numpy as np
import timm
import sys
import os
import json
import tqdm
from PIL import Image

from models import utils, caption
from datasets import coco
from configuration import Config
import re

MAX_DIM = 299


class p2Data(Dataset):
    def __init__(self, fnames, transform=None):
        self.transform = transform
        self.fnames = fnames
        self.file_list = [file for file in os.listdir(fnames) if file.endswith('.jpg')]
        self.file_list.sort()
        self.num_samples = len(self.file_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        filepath = os.path.join(self.fnames, fname)
        img = Image.open(filepath)
        img = self.transform(img)
        return img, fname


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((imgs.shape[0], max_length), dtype=torch.long)
    # print('caption_template = ', caption_template)
    mask_template = torch.ones((imgs.shape[0], max_length), dtype=torch.bool)
    # print('mask_template = ', mask_template)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template







def main(config):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_image_dir = 'D:/NTU/DLCV/hw3/hw3_data/p2_data/images/val'
    test_set = p2Data(test_image_dir, transform=test_transform)
    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False)
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, criterion = caption.build_model(config)
    model.to(device)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of params: {n_parameters}")

    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
    #     "lr": config.lr_backbone,},
    # ]

    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        # checkpoint = torch.load(config.checkpoint, map_location='cpu')
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model'])

    # tokenizer = Tokenizer.from_file("../../../hw3_data/caption_tokenizer.json")
    print(f"Valid: {len(test_set)}")
    print("Start Evaluating..")
    tokenizer = Tokenizer.from_file("D:/NTU/DLCV/hw3/hw3_data/caption_tokenizer.json")
    model.eval()
    total = len(test_dataloader)
    result_dict = {}
    with tqdm.tqdm(total=total) as pbar:
        for k, (imgs, fnames) in enumerate(test_dataloader):
            imgs = imgs.to(device)
            # print(images.shape)
            # imgs = imgs.unsqueeze(0)
            print(imgs.shape)
            # samples = utils.NestedTensor(images, masks).to(device)
            # caps = caps.to(device)
            # cap_masks = cap_masks.to(device)
            # predictions = model(samples, caps[:, :-1], cap_masks[:, :-1])

            # cap, cap_mask = create_caption_and_mask(
            #     start_token, config.max_position_embeddings)

            start_token = 2
            end_token = 3

            cap22, cap_mask22 = create_caption_and_mask(
                start_token, config.max_position_embeddings)
            # cap22, cap_mask22 = create_caption_and_mask(
            #     start_token, 60)
            cap22 = cap22.to(device)
            cap_mask22 = cap_mask22.to(device)
            print(cap22.shape)

            # @torch.no_grad()
            # def evaluate():
            #     # model.eval()
            with torch.no_grad():
                for i in range(config.max_position_embeddings - 1):
                    # for i in range(60 - 1):
                    predictions = model(imgs, cap22, cap_mask22)
                    # print("predictions = ", predictions)
                    predictions = predictions[:, i, :]
                    # print("predictions = ", predictions)
                    predicted_id = torch.argmax(predictions, axis=-1)
                    # print("predictions = ", predicted_id)
                    # if predicted_id[0] == 3:
                    #     return caption
                    for j in range(imgs.shape[0]):
                        if predicted_id[j] != 3:
                            # return caption
                            cap22[j, i + 1] = predicted_id[j]
                            cap_mask22[j, i + 1] = False
            print('captions = ', cap22)
            # return cap22

            # output2 = evaluate()
            # print('output = ', cap22)
            result = []
            for r in range(imgs.shape[0]):
                temp = tokenizer.decode(cap22[r].tolist(), skip_special_tokens=True)
                result.append(temp)
                # result = tokenizer.decode(output[0], skip_special_tokens=True)
                temp2 = temp.capitalize()
                # print("temp2 = ", temp2)
                # temp2 = re.split(r'.', temp2)
                temp2 = temp2.split('.')[0]
                print("temp2 = ", temp2)
                result_dict[fnames[r][:-4]] = temp2[:-1] + "."
            print(result_dict)
            pbar.update(1)

    json_object = json.dumps(result_dict, indent=4)
    with open("p2_output.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    config = Config()
    main(config)

# import os
# import csv
# import json
# import random
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# # import clip
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, datasets, transforms