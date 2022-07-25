import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import os

class Chexpert_dataset(Dataset):
    def __init__(self,path_to_csv,root_dir,columns,transform=None):

        super().__init__()
        self.csv = pd.read_csv(path_to_csv,low_memory=False)
        self.labels = self.csv[columns].values
        self.group = self.csv['Sex'].values
        self.root_dir = root_dir
        self.image_paths = self.csv['Path']
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.image_paths[idx])
        image = read_image(img_name)
        image = image.repeat((3, 1, 1))
        image = image.float()
        label = torch.tensor(self.labels[idx]).float()
        
        if self.transform:
            image = self.transform.forward(image)
        
        return image, label