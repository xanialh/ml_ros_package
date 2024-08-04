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
import matplotlib.pyplot as plt
import copy

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        
        # Encoder (feature extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
        
    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)
        
        # Forward pass through the decoder
        x = self.decoder(x)

        return x
   
class HM_Dataset(Dataset):
    def __init__(self) -> None:
        self.data = []
        self.labels = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        
        label = label.long()

        return sample, label
    
    def add_data(self,data,labels,row_index,column_index,training_image_size):
        data = np.array(data)
        labels = np.array(labels)

        data = data.astype(np.float32)
        labels = labels.astype(np.float32)

        reshape_data = data.reshape(row_index,column_index)
        new_reshape_data = cv2.resize(reshape_data, tuple(training_image_size), interpolation=cv2.INTER_AREA)
        data_tensor = torch.from_numpy(new_reshape_data)
        
        reshape_labels = labels.reshape(row_index,column_index)
        new_reshape_labels = cv2.resize(reshape_labels, tuple(training_image_size), interpolation=cv2.INTER_AREA)
        labels_tensor = torch.from_numpy(new_reshape_labels)

        labels_tensor = torch.round(labels_tensor)

        self.data.append(data_tensor.unsqueeze(0))
        self.labels.append(labels_tensor)

    def getData(self):
        return self.data
    
    def getLabels(self):
        return self.labels

def social_map_to_labels(social_GridMap):
    low_bound = 0.33
    high_bound = 0.66

    length = len(social_GridMap)

    for i in range(length):
        value = float(social_GridMap[i])
        if value < low_bound or math.isnan(value):
            social_GridMap[i] = 0
        elif value > high_bound:
            social_GridMap[i] = 2
        else:
            social_GridMap[i] = 1

    return social_GridMap

def load_from_txt(sgm_filename,ogm_filename):
    pairs = []
    with open(sgm_filename,"r") as sgmFile, open(ogm_filename,"r") as ogmFile:

        sgm_reader = csv.reader(sgmFile)
        ogm_reader = csv.reader(ogmFile)

        ogm_lines = list(ogm_reader)

        for line1 in sgm_reader:
            for line2 in ogm_lines:
                if line1[1] == line2[1]:
                    pairs.append((line1,line2))
                    break
    
    return pairs

def load_into_dataset(pairs,dataset,training_image_size):
    for pair in pairs:
        sgm = pair[0]
        ogm = pair[1]

        assert int(float(sgm[1])) == int(float(ogm[1]))

        row_index = int(float(sgm[2]))
        column_index = int(float(sgm[3]))

        data = ogm[4:]
        labels = social_map_to_labels(sgm[4:])
        dataset.add_data(data,labels,row_index,column_index,training_image_size)

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

def load_config():
        # Load configuration
    try:
        with open("ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values
    

def train():
    config_full = load_config()
    config = config_full["FCN"]

    file_path_input = config["file_path_input"]
    file_path_output = config["file_path_output"]
    training_image_size = config["training_image_size"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    weighted = config["weighted"]
    if weighted == True:
        class_matrix = config["class_matrix"]
    model_name = config["model_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCN().to(device)

    accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=3)
    accuracy = accuracy.to(device)

    val_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=3)
    val_accuracy = accuracy.to(device)

    if weighted:
            # Example confusion matrix
        confusion_matrix = np.array(class_matrix)
        # Calculate class frequencies
        total_samples_per_class = confusion_matrix.sum(axis=1)
        weights = total_samples_per_class.sum() / (len(total_samples_per_class) * total_samples_per_class)
        misclassification_rate = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        weights = weights * (1 + misclassification_rate / total_samples_per_class)
        weights = weights / weights.sum()

        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

        print(weights_tensor)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    matching_files = find_matching_files(file_path_input)

    if len(matching_files) == 0:
        raise FileNotFoundError("No files found")
    
    total_files = len(matching_files)
    train_split = int(0.8 * total_files)
    val_split = total_files - train_split

    train_files, val_files = list(matching_files.items())[:train_split], list(matching_files.items())[train_split:]

    # Create separate datasets for training and validation
    train_dataset = HM_Dataset()
    val_dataset = HM_Dataset()
    
    for file_number, files in train_files:
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            pairs = load_from_txt(sgmFilename, ogmFilename,)
            print(f"file number: {file_number}")
            load_into_dataset(pairs,train_dataset,training_image_size)

    for file_number, files in val_files:
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            pairs = load_from_txt(sgmFilename, ogmFilename,)
            print(f"file number: {file_number}")
            load_into_dataset(pairs,val_dataset,training_image_size)

    train_dataLoader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_dataLoader = DataLoader(val_dataset,batch_size=batch_size)

    patience = config.get("patience", 5)  # Number of epochs to wait for improvement
    best_loss = float('inf')  # Initial best loss (set to a high value)
    best_model_wts = copy.deepcopy(model.state_dict())  # Best model weights
    previous_val_loss = float('inf')  # Track previous validation loss

    for epoch in range(num_epochs):
    # Iterate over the dataset
        print(f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in train_dataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        # Forward pass
            outputs = model(inputs)
        # Compute the loss
            loss = criterion(outputs, labels)
            print(loss.item())
        # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy.update(outputs,labels)
        
            # Print training loss
    time.sleep(1)

    print(f"Evaluation Accuracy: {accuracy}")
    print(f"best loss:{best_loss}")
    torch.save(model.state_dict(), file_path_output + model_name + "_FCN.pt")

def show_tensor_as_image(tensor, title='Image', cmap='viridis',index=0):
    """Display a tensor as an image."""
    # Ensure the tensor is on the CPU and converted to numpy
    tensor = tensor.cpu().numpy()
    tensor = tensor[0]

    if tensor.ndim == 4:  # Batch dimension exists
        tensor = tensor[index]
    
    # Remove singleton dimensions if any
    if tensor.ndim > 2:
        tensor = tensor.squeeze()

    plt.imshow(tensor, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    train()

