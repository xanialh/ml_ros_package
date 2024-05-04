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

import torch
import torch.nn as nn

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

        self.softmax = nn.Softmax(dim=1)

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

        social_heatmap = self.softmax(social_heatmap)

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

        if self.transform:
            sample = self.transform(sample)
            label = self.transform(label)
            label = torch.round(label)

        label3Channels = one_hot_encode(label.squeeze(0))

        return sample, label3Channels, coords

    def addData(self,data,labels,row_index,column_index,coord):

      data = np.array(data)
      labels = np.array(labels)
      coordArray = np.array(coord)

      data = data.astype(float)
      labels = labels.astype(int)
      coordArray = coordArray.astype(float)

      reshapeData = data.reshape(row_index,column_index)
      dataTensor = torch.tensor(reshapeData)

      reshapeLabels = labels.reshape(row_index,column_index)
      labelsTensor = torch.Tensor(reshapeLabels)

      coordTensor = torch.Tensor(coordArray)

      self.data.append(dataTensor)
      self.labels.append(labelsTensor)
      self.coords.append(coordTensor)


    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels



def socialMapToLabels(socialGridMap):
    low_bound = 0.5
    high_bound = 30

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
  # Ensure the input tensor has the expected number of dimensions (2 for height and width)
  if data_tensor.dim() != 2:
    raise ValueError("Input tensor should have 2 dimensions (height and width).")

  # Define the number of channels (one for each class)
  num_channels = 3

  # Create an empty tensor for the one-hot encoded representation
  one_hot = torch.zeros((num_channels, data_tensor.shape[0], data_tensor.shape[1]))

  # Use conditional statements to fill each channel with 1.0 based on the class label
  one_hot[0] = torch.where(data_tensor == 0, 1.0, 0.0)  # Channel 0: Low activity
  one_hot[1] = torch.where(data_tensor == 1, 1.0, 0.0)  # Channel 1: Medium activity
  one_hot[2] = torch.where(data_tensor == 2, 1.0, 0.0)  # Channel 2: High activity

  return one_hot

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

    if density == True:
        largest_header_pair = max(pairs, key=lambda p: float(p[0][1]))
        largest_social_grid, largest_obstacle_grid = largest_header_pair
        largest_height, largest_width = int(float(largest_social_grid[2])), int(float(largest_social_grid[3]))
        largest_header = float(largest_social_grid[1])
        np_social_grid = np.array(largest_social_grid[4:])
        reshape_final_social_grid = np_social_grid.reshape((largest_height, largest_width))
        reshape_final_social_grid = reshape_final_social_grid.astype(float)
        reshape_final_social_grid = np.nan_to_num(reshape_final_social_grid,nan=0)

        #test_ogm = np.array(largest_obstacle_grid[4:])
        #test_ogm = test_ogm.reshape((largest_height, largest_width))
        #test_ogm = test_ogm.astype(float)
        #show_array_as_image(test_ogm)

        for pair in pairs:
            if pair[0][1] != largest_header:
                socialGridMap = np.array(pair[0][4:])
                height,width = int(float(pair[0][2])),int(float(pair[0][3]))
                reshape_socialGridMap = socialGridMap.reshape(height,width)
                reshape_socialGridMap = reshape_socialGridMap.astype(float)
                padded_array = pad_array_to_shape(reshape_socialGridMap,(largest_height, largest_width),0)
                padded_array = padded_array.astype(float)
                padded_array = np.nan_to_num(padded_array,nan=0)
                #show_array_as_image(padded_array)
                reshape_final_social_grid = np.add(reshape_final_social_grid,padded_array)

        reshape_final_social_grid = np.ravel(reshape_final_social_grid)
        sgmFront = np.array([1,largest_header,largest_height,largest_width])
        sgm = np.concatenate((sgmFront,reshape_final_social_grid))
        sgmList = sgm.tolist()

        pairs = [(sgmList,largest_obstacle_grid)]

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

def main():
    pass

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

model = SocialHeatMapCombined() # Instantiate the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

newDataset = HMDataset(transform=transform)

matchingFiles = find_matching_files("/home/danielhixson/ros/noetic/system/src/ML/ml/dataCollection/mapsWithCoords/MAPSSPLITBYINDEX")

addFilesToDataset(matchingFiles,newDataset)

#create dataloader
batch_size = 32
dataLoader = DataLoader(newDataset,batch_size=batch_size,shuffle=True)

num_epochs = 200
#EXAMPLE TRAINING (CHANGE FOR LATER SINCE FILES NEED TO BE READ AND CLASSED)
for epoch in range(num_epochs):
    # Iterate over the dataset
    for inputs, labels, coord in dataLoader:
        # Forward pass

        outputs = model(coord,inputs)

        # Compute the loss
        loss = criterion(outputs, labels)
        print(loss.item())
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    test_social_grid = [float("nan"),25,2,4,89,99,0.4,0.1,0.1,0,float("nan")]

    print(socialMapToLabels(test_social_grid))





