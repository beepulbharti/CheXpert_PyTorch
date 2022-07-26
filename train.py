import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm
from dataset import Chexpert_dataset
from utils import get_splits
import numpy as np
import os

# Enable GPU use
os.environ['CUDE_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# Set parameters to load training dataset
path_to_csv = 'updated_train.csv'

# Root dir (needs to be changed depending on if I am using remote or gaon)
root_dir = '/Users/beepulbharti/Desktop'
root_dir_gaon = '/export/gaon1/data/bbharti1'

# Additional specifications
columns = ['Sex']
transform = transforms.Resize((320,320))
all_data = Chexpert_dataset(path_to_csv,root_dir_gaon,columns,transform = transform)

# Load Densenet121
model = densenet121(weights = DenseNet121_Weights.DEFAULT)

# Fix last layer of densenet to reflect number of classes
input_num = model.classifier.in_features
model.classifier = nn.Linear(input_num,len(columns))

# Set criterion and optimizer
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04,betas=(0.9,0.999))

# K-Fold Cross-validation
splitter = get_splits(all_data, n_splits=5)

# Set the dataloader
train_loader = DataLoader(all_data,batch_size=16,shuffle=True)

# Train the model for number of epochs  
num_epochs = 25
k = 0

for train_ind, val_ind in splitter:

    print('Fold = ', k + 1)
    train_set = Subset(all_data,train_ind)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_set = Subset(all_data,val_ind)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    for epoch in range(num_epochs):

        print(f"Started epoch {epoch + 1}")
        model.train()
        torch.set_grad_enabled(True)
        running_train_loss = 0

        # Training step
        for i, data in enumerate(tqdm(train_loader)):
            image, label, _ = data
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            output = torch.sigmoid(output)
            train_batch_loss = criterion(output, label)
            # print(batch_loss.item())
            running_train_loss += train_batch_loss.item()

            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        running_val_loss = 0
        for i, data in enumerate(tqdm(val_loader)):
            image, label, image_paths = data
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            output = torch.sigmoid(output)
            val_batch_loss = criterion(output, label)
            running_val_loss += val_batch_loss.item()
    
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        print(f"Train loss: {epoch_train_loss:.4f}")
        print(f"Val loss: {epoch_val_loss:.4f}")