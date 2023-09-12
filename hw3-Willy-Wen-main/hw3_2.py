import os
import json
import tqdm
import warnings
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
from tokenizers import Tokenizer

from catr.models import utils, caption
from catr.configuration import Config


class TestData(Dataset):
    def __init__(self, fnames, transform=None):
        self.transform = transform
        self.fnames = fnames
        self.file_list = [file for file in os.listdir(fnames) if (file.endswith('.jpg') or file.endswith('.png'))]
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
    
class to_dim:
    def __init__(self):
        self.dim = 3

    def __call__(self, x):
        if x.shape[0] == 1:
            x = x.repeat(3,1,1)
        return x

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((imgs.shape[0], max_length), dtype=torch.long)
    mask_template = torch.ones((imgs.shape[0], max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='test images directory path', type=str)
    parser.add_argument('-o', '--output', help='output generated json path', type=str)
    args = parser.parse_args()

    test_img_dir = args.input
    json_path = args.output
    
    config = Config()
    warnings.filterwarnings("ignore")
    
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         to_dim(),    
                                         transforms.Resize((384, 384)),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_dataset = TestData(test_img_dir, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _, criterion = caption.build_model(config)
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=False)
    model.to(device)
    model.eval()
    
    checkpoint = torch.load('checkpointV3.pth')
    model.load_state_dict(checkpoint['model'])
    
    tokenizer = Tokenizer.from_file('caption_tokenizer.json')
    
    total = len(test_dataloader)
    
    start_token = 2
    end_token = 3
    max_len = 30
    result_dict = {}
    with tqdm.tqdm(total=total) as pbar:
        with torch.no_grad():
            for k, (imgs, fnames) in enumerate(test_dataloader):
                imgs = imgs.to(device)

                cap, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
                cap = cap.to(device)
                cap_mask = cap_mask.to(device)

                for i in range(max_len):
                    predictions = model(imgs, cap, cap_mask)[:, i, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
                    for j in range(imgs.shape[0]):
                        if predicted_id[j] != end_token:
                            cap[j, i + 1] = predicted_id[j]
                            cap_mask[j, i + 1] = False

                for r in range(imgs.shape[0]):
                    s = tokenizer.decode(cap[r].tolist(), skip_special_tokens=True).capitalize().split('.')[0]
                    s = s[:-1]+'.'
                    name = fnames[r][:-4]
                    #print(f'{name}: {s}')
                    result_dict[name] = s
                pbar.update(1)
    
    
    json_object = json.dumps(result_dict, indent=4)
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
