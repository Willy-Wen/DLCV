import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader



def same_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
class Test_Dataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.gt = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx_row = self.gt.loc[idx]
        img_name = idx_row['filename']
        img_path = f'{self.img_dir}/{img_name}'
        image = Image.open(img_path)
        csv_id = idx_row['id']
        if self.transform:
            image = self.transform(image)
        return image, csv_id
        
        
class model_ft(nn.Module):
    def __init__(self):
        super(model_ft, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.classifier = nn.Sequential(nn.BatchNorm1d(1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(1000, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(512, 65)
                                       )
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    
def test_loop(dataloader, model):
    model.eval()
    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)
            pred_labels = model(images)
            pred_labels = pred_labels.argmax(1)
            for j, i in enumerate(ids):
                dataloader.dataset.gt.loc[i.item(),'label'] = label2class_dict[pred_labels[j].item()] 
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input test csv path', type=str, required=True)
    parser.add_argument('-d', '--dir', help='test images dir path', type=str, required=True)
    parser.add_argument('-o', '--output', help='output pred csv path', type=str, required=True)
    args = parser.parse_args()
    
    office_test_csv_path = args.input
    office_test_img_dir = args.dir
    office_pred_csv_path = args.output
    
    same_seeds(777)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    label2class_dict = {0: 'TV', 1: 'Desk_Lamp', 2: 'Speaker', 3: 'Hammer', 4: 'Folder', 5: 'Laptop', 6: 'Pen', 7: 'Postit_Notes',
    8: 'Mop', 9: 'Mug', 10: 'Radio', 11: 'File_Cabinet', 12: 'Eraser', 13: 'Ruler', 14: 'Couch', 15: 'Trash_Can', 16: 'Webcam', 
    17: 'Backpack', 18: 'Bucket', 19: 'Kettle', 20: 'Batteries', 21: 'Telephone', 22: 'Chair', 23: 'Toys', 24: 'Refrigerator',
    25: 'Clipboards', 26: 'Fork', 27: 'Push_Pin', 28: 'Marker', 29: 'Candles', 30: 'Flipflops', 31: 'Helmet', 32: 'Pencil',
    33: 'Calendar', 34: 'Monitor', 35: 'Shelf', 36: 'Sneakers', 37: 'Soda', 38: 'Bottle', 39: 'Flowers', 40: 'Drill', 41: 'Table',
    42: 'Knives', 43: 'Computer', 44: 'Alarm_Clock',  45: 'Sink', 46: 'Exit_Sign', 47: 'Bed', 48: 'Oven', 49: 'Keyboard', 
    50: 'Paper_Clip', 51: 'Lamp_Shade', 52: 'Scissors', 53: 'Curtains', 54: 'Fan', 55: 'Spoon', 56: 'Screwdriver', 57: 'Glasses',
    58: 'Pan', 59: 'ToothBrush', 60: 'Mouse', 61: 'Printer', 62: 'Calculator', 63: 'Notebook', 64: 'Bike'}
    
    batch_size = 128
    transform_test = transforms.Compose([transforms.Resize(128),
                                        transforms.CenterCrop((128,128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    
    office_test_dataset = Test_Dataset(office_test_img_dir, office_test_csv_path, transform=transform_test)
    office_test_dataloader = DataLoader(office_test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    model = model_ft()
    model = torch.load('best_model_C.pt')
    model.to(device)
    
    test_loop(office_test_dataloader, model)
    office_test_dataloader.dataset.gt.to_csv(office_pred_csv_path, index=False)
    
    
    
    
