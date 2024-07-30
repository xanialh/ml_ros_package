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
  model = SocialHeatMapCombined()
  model.load_state_dict(torch.load(model_path))
  model.eval()  # Set the model to evaluation mode
  return model

class SocialHeatMapCombined(nn.Module):
    def __init__(self):
        super(SocialHeatMapCombined, self).__init__()

        # Define input channels for robot position and obstacle grid map
        robot_pos_channels = 2  # Assuming x and y coordinates as separate channels
        obstacle_map_channels = 1  # Assuming a single channel for obstacle grid map

        # Feature extraction for robot position
        self.robot_pos_encoding = nn.Sequential(
            nn.Linear(robot_pos_channels, 32),  # Adjust hidden units as needed
            nn.ReLU(inplace=True)
        )

        # Feature extraction for obstacle grid map
        self.obstacle_map_encoding = nn.Sequential(
            nn.Conv2d(in_channels=obstacle_map_channels, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Combined input channels for the encoder after concatenation of robot and obstacle features
        combined_input_channels = 32 + 8  # 32 from robot_features and 8 from obstacle_map_features

        # Shared feature extraction (encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=combined_input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder with three channels for social heat map
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, robot_position, obstacle_grid_map):
        # Encode robot position
        robot_features = self.robot_pos_encoding(robot_position)
        # Expand dimensions to match the 4D shape of obstacle_map_features
        robot_features = robot_features.unsqueeze(-1).unsqueeze(-1)  # Change shape to [32, 32, 1, 1]
        robot_features = robot_features.expand(-1, -1, 128, 128)  # Change shape to [32, 32, 128, 128]

        # Encode obstacle grid map
        obstacle_map_features = self.obstacle_map_encoding(obstacle_grid_map)  # Expected shape: [32, 8, 128, 128]

        # Concatenate along the channel dimension
        concat_features = torch.cat((robot_features, obstacle_map_features), dim=1)  # New shape: [32, 40, 128, 128]

        # Forward pass through the shared encoder
        encoded_features = self.encoder(concat_features)  # This layer now correctly handles 40 input channels

        # Forward pass through the decoder with three channels for social heat map
        social_heatmap = self.decoder(encoded_features)

        return social_heatmap

class HMDataset(Dataset):
    def __init__(self, transform=None) -> None:
        self.data = []
        self.labels = []
        self.coords = []
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        coords = self.coords[index]

        label = label.long()

        return sample, label, coords

    def addData(self,data,labels,row_index,column_index,coord):

      data = np.array(data)
      labels = np.array(labels)
      coordArray = np.array(coord)

      data = data.astype(np.float32)
      labels = labels.astype(np.float32)
      coordArray = coordArray.astype(np.float32)

      reshapeData = data.reshape(row_index,column_index)
      newReshapeData = cv2.resize(reshapeData, (128, 128), interpolation=cv2.INTER_AREA)
      dataTensor = torch.from_numpy(newReshapeData)
        
      reshapeLabels = labels.reshape(row_index,column_index)
      newReshapeLabels = cv2.resize(reshapeLabels, (128, 128), interpolation=cv2.INTER_AREA)
      labelsTensor = torch.from_numpy(newReshapeLabels)

      coordTensor = torch.Tensor(coordArray)

      labelsTensor = torch.round(labelsTensor)

      self.data.append(dataTensor.unsqueeze(0))
      self.labels.append(labelsTensor)
      self.coords.append(coordTensor)


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

def loadFromTxt(sgmFilename,ogmFilename,density=False):
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

        data = ogm[6:]
        labels = socialMapToLabels(sgm[6:])
        coords = [ogm[4],ogm[5]]

        dataset.addData(data,labels,row_index,column_index,coords)

def show_array_as_image(array):
  plt.imshow(array, cmap='gray')
  plt.colorbar()
  plt.show()

def pad_array_to_shape(array, target_shape, pad_value=0):
        # Calculate the padding amounts for each dimension
    pad_width = [(target_shape[0] - array.shape[0], 0), (target_shape[1] - array.shape[1], 0)]

    # Pad the array using np.pad
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=pad_value)

    return padded_array

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
            pairs = loadFromTxt(sgmFilename, ogmFilename,False)
            print(f"Pairs for files with number {file_number}:")
            loadIntoDataset(pairs,dataset)

def loadConfig():
        # Load configuration
    try:
        with open("ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def train():
    model = load_model("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/data/trained_modelsonecfg.pt")

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


    matchingFiles = find_matching_files(file_path_input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SocialHeatMapCombined().to(device)

    accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
    accuracy = accuracy.to(device)
    
    criterion = nn.CrossEntropyLoss()

    matchingFiles = find_matching_files(file_path_input)

    total_loss = 0.0
    num_samples = 0

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

            with torch.no_grad():
                for inputs, labels in newDataLoader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    coord = coord.to(device)
                    # Forward pass
                    outputs = model(coord,inputs)
                    pred_classes = torch.argmax(outputs, dim=1)
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

    time.sleep(2)



if __name__ == "__main__":
    torch.set_printoptions(threshold=float('inf'), precision=4, edgeitems=10)
    train()
