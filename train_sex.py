import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm
from dataset import Chexpert_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import wandb

# Initialize wandb
wandb.init(project="sex_classifier_project")

# Enable GPU use
os.environ['CUDE_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# Load csv and create dataframe
path_to_csv = 'csvs/sex_model_train.csv'
all_data = pd.read_csv(path_to_csv,low_memory=False)

# Root dir (needs to be changed depending on if I am using remote or gaon)
root_dir = '/Users/beepulbharti/Desktop'
root_dir_gaon = '/export/gaon1/data/bbharti1'

# Additional specifications
columns = ['Sex']
transform = transforms.Resize((320,320))
data_csv = pd.read_csv(path_to_csv)
all_data = Chexpert_dataset(all_data,root_dir_gaon, columns, transform = transform)

# Load Densenet121
sex_model = densenet121(weights = DenseNet121_Weights.DEFAULT)

# Fix last layer of densenet to reflect number of classes
input_num = sex_model.classifier.in_features
sex_model.classifier = nn.Linear(input_num,len(columns))

# Set criterion and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(sex_model.parameters(), lr=1e-04, betas=(0.9,0.999))

# Put model to device
sex_model = sex_model.to(device)

# Train the model for number of epochs  
num_epochs = 3

# Split data in training and validation set
indices = np.arange(len(all_data))
train_ind, val_ind = train_test_split(indices, test_size=0.1, stratify = data_csv['Sex'])
train_data = Subset(all_data,train_ind)
val_data = Subset(all_data,val_ind)

# Create train and val loaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

best_acc = 0
for epoch in range(num_epochs):

    print(f"Started epoch {epoch + 1}")
    sex_model.train()
    torch.set_grad_enabled(True)
    running_train_loss = 0
    running_train_acc = 0

    # Training step
    for i, data in enumerate(tqdm(train_loader)):
        image, label, image_paths = data
        image = image.to(device)
        label = label.to(device)

        output = sex_model(image)
        output = torch.sigmoid(output)
        train_batch_acc = torch.sum((output >=0.5) == label)/len(label)
        train_batch_loss = criterion(output, label)
        running_train_loss += train_batch_loss.item()
        running_train_acc += train_batch_acc

        optimizer.zero_grad()
        train_batch_loss.backward()
        optimizer.step()
    
    # Validation step
    sex_model.eval()
    torch.set_grad_enabled(False)
    running_val_loss = 0
    running_val_acc = 0

    for i, data in enumerate(tqdm(val_loader)):
        image, label, image_paths = data
        image = image.to(device)
        label = label.to(device)

        output = sex_model(image)
        output = torch.sigmoid(output)
        val_batch_acc = torch.sum((output >=0.5) == label)/len(label)
        val_batch_loss = criterion(output, label)
        running_val_loss += val_batch_loss.item()
        running_val_acc += val_batch_acc

    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_train_acc = running_train_acc / len(train_loader)
    epoch_val_acc = running_val_acc / len(val_loader)
    epoch_val_loss = running_val_loss / len(val_loader)
    wandb.log({'train_loss': epoch_train_loss,'val_loss': epoch_val_loss, 'val_accuracy':epoch_val_acc})

    print(f"Train loss: {epoch_train_loss:.4f}")
    print(f"Val loss: {epoch_val_loss:.4f}")
    print(f"Val acc: {epoch_val_acc:.4f}")

    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        best_sex_model_state_dict = copy.deepcopy(sex_model.state_dict())
        torch.save(best_sex_model_state_dict, os.path.join('pretrained_sex_model', 'model.pt'))