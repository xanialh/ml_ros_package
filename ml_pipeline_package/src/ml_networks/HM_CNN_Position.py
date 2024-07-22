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

# Load configuration
try:
  with open("config/config_HM_CNN_Position.yaml", "r") as f:
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

      self.data.append(dataTensor)
      self.labels.append(labelsTensor)
      self.coords.append(coordTensor)


    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels

def socialMapToLabels(socialGridMap):
    low_bound = 0.1
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

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

matchingFiles = find_matching_files(file_path_input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SocialHeatMapCombined().to(device)

accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
accuracy = accuracy.to(device)

criterion = 1
optimizer = 2

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
        newDataset = HMDataset(transform=transform)
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
                for inputs, labels, coord in newDataLoader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    coord = coord.to(device)
                # Forward pass
                    outputs = model(coord,inputs)
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
torch.save(model.state_dict(), file_path_output + "my_model.pt")




