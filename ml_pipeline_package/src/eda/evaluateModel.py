import torch
from torch import nn
from torch.utils.data import DataLoader

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
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Add another convolutional layer
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),  # Reduce to 32 filters before final layer
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

def load_model(model_path):
  model = SocialHeatMapFCN()
  model.load_state_dict(torch.load(model_path))
  model.eval()  # Set the model to evaluation mode
  return model

class HMDataset(Dataset):
    def __init__(self, transform=None) -> None:
        self.data = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)
            label = self.transform(label)
            label = torch.round(label)

        label3Channels = one_hot_encode(label.squeeze(0))

        return sample, label3Channels
    
    def addData(self,data,labels,row_index,column_index):
        data = np.array(data)
        labels = np.array(labels)

        data = data.astype(float)
        labels = labels.astype(int)

        reshapeData = data.reshape(row_index,column_index)
        dataTensor = torch.tensor(reshapeData)

        reshapeLabels = labels.reshape(row_index,column_index)
        labelsTensor = torch.Tensor(reshapeLabels)

        self.data.append(dataTensor)
        self.labels.append(labelsTensor)

    def getData(self):
        return self.data
    
    def getLabels(self):
        return self.labels

def socialMapToLabels(socialGridMap):
    low_bound = 0.01
    high_bound = 0.4

    length = len(socialGridMap)

    for i in range(length):
        value = float(socialGridMap[i])
        if value < low_bound or math.isnan(value):
            socialGridMap[i] = 0
        elif value >= high_bound:
            socialGridMap[i] = 2
        else:
            socialGridMap[i] = 1

    return socialGridMap

def one_hot_encode(data_tensor):
  if data_tensor.dim() != 2:
    raise ValueError("Input tensor should have 2 dimensions (height and width).")
  
  num_channels = 3

  one_hot = torch.zeros((num_channels, data_tensor.shape[0], data_tensor.shape[1]))

  one_hot[0] = torch.where(data_tensor == 0, 1.0, 0.0)  # Channel 0: Low activity
  one_hot[1] = torch.where(data_tensor == 1, 1.0, 0.0)  # Channel 1: Medium activity
  one_hot[2] = torch.where(data_tensor == 2, 1.0, 0.0)  # Channel 2: High activity

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

# Define transformations
transform = transforms.Compose([
transforms.ToPILImage(),
transforms.Resize((128,128), interpolation=transforms.InterpolationMode.NEAREST),
transforms.ToTensor()
])

model = load_model("/home/danielhixson/socNavProject/ml_ros_package/ml_pipeline_package/data/trained_models/office/crop_fix_model_steven.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=3)
accuracy = accuracy.to(device)

mse = torchmetrics.MeanSquaredError()
mse = mse.to(device)

mae = torchmetrics.MeanAbsoluteError()
mae = mae.to(device)

iou = torchmetrics.JaccardIndex(task= "multiclass",num_classes=3)
iou = iou.to(device)

precision = torchmetrics.Precision(task= "multiclass",num_classes=3)
precision = precision.to(device)

f1 = torchmetrics.F1Score(task= "multiclass",num_classes=3)
f1 = f1.to(device)

recall = torchmetrics.Recall(task= "multiclass",num_classes=3)
recall = recall.to(device)

confusionMatrix = torchmetrics.ConfusionMatrix(task= "multiclass",num_classes=3)
confusionMatrix = confusionMatrix.to(device)

criterion = nn.CrossEntropyLoss()

matchingFiles = find_matching_files("/home/danielhixson/socNavProject/ml_ros_package/ml_pipeline_package/data/office/eval/without_coords/cropped_maps")

total_loss = 0.0
num_samples = 0

for file_number, files in matchingFiles.items():
    if 'socialGridMap' in files and 'obstacleGridMap' in files:
        newDataset = HMDataset(transform=transform)
        sgmFilename = files['socialGridMap']
        ogmFilename = files['obstacleGridMap']
        pairs = loadFromTxt(sgmFilename, ogmFilename,)
        print(f"file number: {file_number}")
        loadIntoDataset(pairs,newDataset)

        newDataLoader = DataLoader(newDataset,batch_size=12,shuffle=True)

        stop = False
        with torch.no_grad():
            for inputs, labels in newDataLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                # Compute the loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_samples += 1
                mse.update(outputs,labels)
                mae.update(outputs,labels)
                iou.update(outputs,labels)
                precision.update(outputs,labels)
                accuracy.update(outputs,labels)
                f1.update(outputs,labels)
                recall.update(outputs,labels)
                confusionMatrix.update(outputs,labels)

    del newDataset
    del newDataLoader

time.sleep(5)

accuracy = accuracy.compute()
mse = mse.compute()
mae = mae.compute()
iou = iou.compute()
precision = precision.compute()
f1 = f1.compute()
recall = recall.compute()
confusionMatrix = confusionMatrix.compute()

avg_loss = total_loss/num_samples

print(f"iou: {iou}")
print(f"average loss: {avg_loss}")
print(f"mse: {mse}")
print(f"mae: {mae}")
print(f"Evaluation Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"f1: {f1}")
print(f"recall: {recall}")
print(f"confusion matrix: {confusionMatrix}")


