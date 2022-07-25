import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from dataset import Chexpert_dataset
from utils import get_splits
import os

# Enable GPU use
os.environ['CUDE_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# Set parameters to load training dataset
path_to_csv = 'train.csv'
root_dir = '/Users/beepulbharti/Desktop'
columns = ['Sex']
transform = transforms.Resize((320,320))
data = Chexpert_dataset(path_to_csv,root_dir,columns,transform = transform)

# Load Densenet121
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

# Fix last layer of densenet to reflect number of classes
input_num = model.classifier.in_features
model.classifier = nn.Linear(input_num,len(columns))

# Set criterion and optimizer
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04,betas=(0.9,0.999))

# K-Fold Cross-validation
splitter = get_splits(data, n_splits=5)

# Set the dataloader
train_loader = DataLoader(data,batch_size=16,shuffle=True)

# Train the model for number of epochs  
num_epochs = 25
k = 0

for train_ind, val_ind in splitter:

    print('Fold = ', k + 1)
    train_sampler = SubsetRandomSampler(train_ind)
    train_loader = DataLoader(data, batch_size=16, sampler = train_sampler, shuffle=False)

    for epoch in range(num_epochs):

        print(f"Started epoch {epoch + 1}")
        model.train()
        torch.set_grad_enabled(True)
        running_loss = 0

        # Training step
        for i, data in enumerate(tqdm(train_loader)):
            image, label = data
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            output = torch.sigmoid(output)
            batch_loss = criterion(output, label)
            print(batch_loss.item())
            running_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_ind)
        print(f"Train loss: {epoch_loss:.4f}")