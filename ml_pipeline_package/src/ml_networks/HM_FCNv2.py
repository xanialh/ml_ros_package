import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import math
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import yaml
import torchmetrics
import time
import cv2
import PIL

# Load configuration
try:
  with open("config/config_HM_FCNv2.yaml", "r") as f:
    config = yaml.safe_load(f)
except FileNotFoundError:
  print("Error: Configuration file 'config.yaml' not found!")
  # Handle the error or use default values

file_path_input = config["file_path_input"]
file_path_output = config["file_path_output"]
training_image_size = config["training_image_size"]
criterion = config["criterion"]
optimizer = config["optimiser"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
dict = config["dict"]
lower_bound_threshold = config["lower_bound_threshold"]
upper_bound_threshold = config["upper_bound_threshold"]
learning_rate = config["learning_rate"]
momentum = config["momentum"]
betas = config["betas"]
alpha = config["alpha"]
rho = config["rho"]

class SocialHeatMapFCN(nn.Module):
    def __init__(self):
        super(SocialHeatMapFCN, self).__init__()
        
        # Encoder (feature extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),  # Add dropout with probability 0.5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)  # Add dropout with probability 0.5
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)
        
        # Forward pass through the decoder
        x = self.decoder(x)

        x = self.softmax(x)
        
        return x
    
class HMDataset(Dataset):
    def __init__(self) -> None:
        self.data = []
        self.labels = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        label3Channels = one_hot_encode(label.squeeze(0))

        return sample, label3Channels
    
    def addData(self,data,labels,row_index,column_index):
        data = np.array(data)
        labels = np.array(labels)

        data = data.astype(np.float32)
        labels = labels.astype(np.float32)

        reshapeData = data.reshape(row_index,column_index)
        newReshapeData = cv2.resize(reshapeData, (128, 128), interpolation=cv2.INTER_AREA)
        dataTensor = torch.from_numpy(newReshapeData)
        

        reshapeLabels = labels.reshape(row_index,column_index)
        newReshapeLabels = cv2.resize(reshapeLabels, (128, 128), interpolation=cv2.INTER_AREA)
        labelsTensor = torch.from_numpy(newReshapeLabels)

        self.data.append(dataTensor.unsqueeze(0))
        self.labels.append(labelsTensor)

    def getData(self):
        return self.data
    
    def getLabels(self):
        return self.labels

def socialMapToLabels(socialGridMap):
    low_bound = 0.01
    high_bound = 0.6

    length = len(socialGridMap)

    for i in range(length):
        value = float(socialGridMap[i])
        if value < low_bound:
            socialGridMap[i] = 0.0
        elif value > high_bound:
            socialGridMap[i] = 2.0
        else:
            socialGridMap[i] = 1.0

    return socialGridMap

def one_hot_encode(data_tensor):
  if data_tensor.dim() != 2:
    raise ValueError("Input tensor should have 2 dimensions (height and width).")
  
  num_channels = 3

  one_hot = torch.zeros((num_channels, data_tensor.shape[0], data_tensor.shape[1]))

  one_hot[0] = torch.where(data_tensor == 0, 1.0, 0.0)  # Channel 0: Low activity
  one_hot[1] = torch.where(data_tensor == 1, 1.0, 0.0)  # Channel 1: Medium activity
  one_hot[2] = torch.where(data_tensor == 2, 1.0, 0.0)  # Channel 2: High activity
  print(torch.sum(one_hot[2]))

  return one_hot

def loadFromTxt(sgmFilename,ogmFilename):
    pairs = []
    with open(sgmFilename,"r") as sgmFile, open(ogmFilename,"r") as ogmFile:

        sgmReader = csv.reader(sgmFile)
        ogmReader = csv.reader(ogmFile)

        ogmLines = list(ogmReader)

        for line1 in sgmReader:
            for line2 in ogmLines:
                if line1[1] == line2[1]:
                    pairs.append((line1,line2))
                    break     
    
    return pairs

def loadIntoDataset(pairs,dataset):
    for pair in pairs:
        sgm = pair[0]
        ogm = pair[1]

        assert int(float(sgm[1])) == int(float(ogm[1]))

        row_index = int(float(sgm[2]))
        column_index = int(float(sgm[3]))

        data = ogm[4:]
        labels = socialMapToLabels(sgm[4:])

        dataset.addData(data,labels,row_index,column_index)

def show_array_as_image(array):
  plt.imshow(array, cmap='gray')
  plt.colorbar()
  plt.show()

def find_matching_files(folder_path):
    matching_files = {}
    for file in os.listdir(folder_path):
        split_file = file.split("_")
        if len(split_file) == 4 and split_file[3].endswith(".txt"):
            file_number = split_file[0]
            map_type = split_file[3].split(".")[0]
            if map_type in ["socialGridMap", "obstacleGridMap"]:
                if file_number not in matching_files:
                    matching_files[file_number] = {}
                matching_files[file_number][map_type] = os.path.join(folder_path, file)
    return matching_files

def addFilesToDataset(matching_files,dataset):
    for file_number, files in matching_files.items():
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            pairs = loadFromTxt(sgmFilename, ogmFilename)
            print(f"Pairs for files with number {file_number}:")
            loadIntoDataset(pairs,dataset)

if __name__ == "__main__":

    model = SocialHeatMapFCN() # Instantiate the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SocialHeatMapFCN().to(device)

    accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
    accuracy = accuracy.to(device)

    if criterion == 1:
        criterion = nn.CrossEntropyLoss()
    elif criterion == 2:
        criterion = nn.BCELoss()
    else:
        raise ValueError("Invalid criterion value for loss function")

    if optimizer == 1:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == 2:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
    elif optimizer == 3:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha)
    elif optimizer == 4:
        optimizer = optim.Adadelta(model.parameters(), rho=rho)
    elif optimizer == 5:
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid value for optimiser")

    matchingFiles = find_matching_files(file_path_input)

    tolerance = 0.0000000000000000001  # Threshold for loss function change (adjust as needed)
    prev_loss = float('inf')  # Initialize with a high value
    plateau_tolerance = 25

    for file_number, files in matchingFiles.items():
        plateau_count = 0
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            newDataset = HMDataset()
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            pairs = loadFromTxt(sgmFilename, ogmFilename,)
            print(f"file number: {file_number}")
            loadIntoDataset(pairs,newDataset)

            newDataLoader = DataLoader(newDataset,batch_size=batch_size,shuffle=True)

            stop = False

            for epoch in range(num_epochs):
            # Iterate over the dataset
                if stop != True:
                    for inputs, labels in newDataLoader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    # Forward pass
                        outputs = model(inputs)
                    # Compute the loss
                        loss = criterion(outputs, labels)
                        print(loss.item())

                        if abs(prev_loss - loss.item()) > tolerance:
                            prev_loss = loss.item()  # Update previous loss
                            plateau_count = 0  # Reset plateau counter if improvement detected
                        else:
                            plateau_count += 1  # Increment counter if loss plateaus
                        # Move to next file if plateau_tolerance is reached
                        if plateau_count >= plateau_tolerance:
                            print(f"Loss plateaued for {plateau_tolerance} epochs. Moving to next file.")
                            stop = True
                    # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        accuracy.update(outputs,labels)

            del newDataset
            del newDataLoader

    time.sleep(5)

    accuracy = accuracy.compute()
    print(f"Evaluation Accuracy: {accuracy}")
    torch.save(model.state_dict(), file_path_output + "weightedfixedclassesChanged.pt")
