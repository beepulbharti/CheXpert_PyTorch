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
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import copy
import wandb

# Set condition and initialize wandb
columns = ['grouped_condition']
condition = columns[0]
project_name = condition + '_classifier'
wandb.init(project=project_name)

# Enable GPU use
os.environ['CUDE_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# Set parameters to load training dataset
path_to_csv = 'csvs/train/train_' + condition + '.csv'

# Root dir (needs to be changed depending on if I am using remote or gaon)
root_dir = '/Users/beepulbharti/Desktop'
root_dir_gaon = '/export/gaon1/data/bbharti1'

# Additional specifications
transform = transforms.Resize((320,320))
all_data = pd.read_csv(path_to_csv,low_memory=False)
# all_data = Chexpert_dataset(path_to_csv,root_dir_gaon,columns,transform = transform)

# Load Densenet121
model = densenet121(weights = DenseNet121_Weights.DEFAULT)

# Fix last layer of densenet to reflect number of classes
input_num = model.classifier.in_features
model.classifier = nn.Linear(input_num,len(columns))

# Set criterion and optimizer
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, betas=(0.9,0.999))

# K-Fold Cross-validation
# splitter = get_splits(all_data, n_splits=5)

# Set the dataloader
# train_loader = DataLoader(all_data,batch_size=16,shuffle=True)

# Train the model for number of epochs  
num_epochs = 3
k = 0
best_auc = 0

# for train_ind, val_ind in splitter:

# Print fold
# print('Fold = ', k + 1)
# train_ind = train_ind[0:10000]
# val_ind = val_ind[0:5000]

# Create split for training and validation data
x_ind = np.linspace(0,len(all_data)-1,len(all_data)).astype('int')
group = all_data['group']
train_ind, val_ind = train_test_split(x_ind,test_size=0.25,stratify=group)

# Create training data
train_data = all_data.iloc[train_ind]
train_set = Chexpert_dataset(train_data,root_dir_gaon,columns,transform=transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

# Create validation data
val_data = all_data.iloc[val_ind]
val_set = Chexpert_dataset(val_data,root_dir_gaon,columns,transform=transform)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

for epoch in range(num_epochs):

    print(f"Started epoch {epoch + 1}")
    model.train()
    torch.set_grad_enabled(True)
    running_train_loss = 0

    # Training step
    for i, data in enumerate(tqdm(train_loader)):
        image, label, image_paths = data
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
    output_list = []
    model.eval()
    torch.set_grad_enabled(False)
    running_val_loss = 0
    for i, data in enumerate(tqdm(val_loader)):
        image, label, image_paths = data
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        output = torch.sigmoid(output)
        val_batch_loss = criterion(output, label)
        running_val_loss += val_batch_loss.item()
        output_list.append(output)

    y_probs = torch.cat(output_list).cpu().ravel()
    y = val_set.labels.ravel()
    epoch_auc = roc_auc_score(y,y_probs)
    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_val_loss = running_val_loss / len(val_loader)
    print(f"Train loss: {epoch_train_loss:.4f}")
    print(f"Val loss: {epoch_val_loss:.4f}")
    print(f"Val AUC: {epoch_auc:.4f}")
    wandb.log({'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss,
               'val_auc': epoch_auc, 'epoch': epoch + 1})

    if epoch_auc > best_auc:
        print('saving new model!')
        best_auc = epoch_auc
        best_model_state_dict = copy.deepcopy(model.state_dict())
        # Save best model
        torch.save(best_model_state_dict, os.path.join('pretrained_classifier', condition + '_model.pt'))

'''
# Re-evaluate using the best model
model.load_state_dict(best_model_state_dict)
model.to(device)

# Final evaluation
output_list = []
for i, data in enumerate(tqdm(val_loader)):
        image, label, image_paths = data
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        output = torch.sigmoid(output)
        val_batch_loss = criterion(output, label)
        running_val_loss += val_batch_loss.item()
        output_list.append(output)

paths = val_set.image_paths
a = val_set.group
y = val_set.labels.ravel()
y_probs = torch.cat(output_list).cpu().ravel()
d = {'Paths': paths, 'a': a, 'y':y, 'y_probs':y_probs }
df = pd.DataFrame(d)
df.to_csv('results_Atelectasis.csv')
'''
