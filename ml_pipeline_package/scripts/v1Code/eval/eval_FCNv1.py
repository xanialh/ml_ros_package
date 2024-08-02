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
import cv2
   
def load_model(model_path):
  model = SocialHeatMapFCN()
  model.load_state_dict(torch.load(model_path))
  model.eval()  # Set the model to evaluation mode
  return model

class SocialHeatMapFCN(nn.Module):
    def __init__(self):
        super(SocialHeatMapFCN, self).__init__()
        
        # Encoder (feature extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)
        
        # Forward pass through the decoder
        x = self.decoder(x)
   
        return x
   
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
        
        label = label.long()

        return sample, label
    
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

        labelsTensor = torch.round(labelsTensor)

        self.data.append(dataTensor.unsqueeze(0))
        self.labels.append(labelsTensor)

    def getData(self):
        return self.data
    
    def getLabels(self):
        return self.labels

def socialMapToLabels(socialGridMap):
    low_bound = 0.1
    high_bound = 0.5

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
        if len(split_file) == 5 and split_file[4].endswith(".txt"):
            file_number = split_file[0]
            map_type = split_file[4].split(".")[0]
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

def train():
    model = load_model("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/data/trained_models/bookstore/FCNv1.pt")

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

    confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3).to(device)
    confusion_matrix = confusion_matrix.to(device)

    criterion = nn.CrossEntropyLoss()

    matchingFiles = find_matching_files("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/data/bookstore/eval/without_coords/final_maps")

    total_loss = 0.0
    num_samples = 0

    newDataset = HMDataset()

    for file_number, files in matchingFiles.items():
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            pairs = loadFromTxt(sgmFilename, ogmFilename,)
            print(f"file number: {file_number}")
            loadIntoDataset(pairs,newDataset)

    newDataLoader = DataLoader(newDataset,batch_size=4,shuffle=True)

    stop = False
    with torch.no_grad():
        for inputs, labels in newDataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            print(outputs.size())
            #print("**************")
            pred_classes = torch.argmax(outputs, dim=1)
            print(labels.size())
            # Compute the loss
            loss = criterion(outputs, labels)
            print(loss.item())
            total_loss += loss.item()
            num_samples += 1
            mse.update(pred_classes ,labels)
            mae.update(pred_classes ,labels)
            iou.update(pred_classes ,labels)
            precision.update(pred_classes ,labels)
            accuracy.update(pred_classes ,labels)
            f1.update(pred_classes ,labels)
            recall.update(pred_classes ,labels)
            confusion_matrix.update(pred_classes , labels)

    time.sleep(5)

    accuracy = accuracy.compute()
    mse = mse.compute()
    mae = mae.compute()
    iou = iou.compute()
    precision = precision.compute()
    f1 = f1.compute()
    recall = recall.compute()
    confusion_matrix = confusion_matrix.compute()

    avg_loss = total_loss/num_samples

    print(f"iou: {iou}")
    print(f"average loss: {avg_loss}")
    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"Evaluation Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"f1: {f1}")
    print(f"recall: {recall}")
    print(f"Confusion Matrix:\n{confusion_matrix}")

if __name__ == "__main__":
    torch.set_printoptions(threshold=float('inf'), precision=4, edgeitems=10)
    train()


