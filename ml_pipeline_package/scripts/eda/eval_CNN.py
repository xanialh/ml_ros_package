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
  model = CNN()
  model.load_state_dict(torch.load(model_path))
  model.eval()  # Set the model to evaluation mode
  return model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

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

        # Encoder (feature extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=combined_input_channels, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
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

class HM_Dataset(Dataset):
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

    def add_data(self,data,labels,row_index,column_index,coord,training_image_size):
        data = np.array(data)
        labels = np.array(labels)
        coord_array = np.array(coord)

        data = data.astype(np.float32)
        labels = labels.astype(np.float32)
        coord_array = coord_array.astype(np.float32)

        reshape_data = data.reshape(row_index,column_index)
        new_reshape_data = cv2.resize(reshape_data, tuple(training_image_size), interpolation=cv2.INTER_AREA)
        data_tensor = torch.from_numpy(new_reshape_data)

        reshape_labels = labels.reshape(row_index,column_index)
        new_reshape_labels = cv2.resize(reshape_labels, tuple(training_image_size), interpolation=cv2.INTER_AREA)
        labels_tensor = torch.from_numpy(new_reshape_labels)

        coord_tensor = torch.Tensor(coord_array)

        labels_tensor = torch.round(labels_tensor)

        self.data.append(data_tensor.unsqueeze(0))
        self.labels.append(labels_tensor)
        self.coords.append(coord_tensor)

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

        data = ogm[6:]
        labels = social_map_to_labels(sgm[6:])
        coords = [ogm[4],ogm[5]]

        dataset.add_data(data,labels,row_index,column_index,coords,training_image_size)

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
    config = config_full["eval_CNN"]

    model_location = config["model_location"]
    file_path_input = config["eval_data_location"]
    training_image_size = config["training_image_size"]
    batch_size = config["batch_size"]
    
    model = load_model(model_location)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=3)
    accuracy = accuracy.to(device)

    per_class_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=3, average=None).to(device)

    mse = torchmetrics.MeanSquaredError()
    mse = mse.to(device)

    mae = torchmetrics.MeanAbsoluteError()
    mae = mae.to(device)

    iou = torchmetrics.classification.JaccardIndex(task= "multiclass",num_classes=3)
    iou = iou.to(device)

    precision = torchmetrics.classification.Precision(task= "multiclass",num_classes=3,average=None)
    precision = precision.to(device)

    f1 = torchmetrics.classification.F1Score(task= "multiclass",num_classes=3,average=None)
    f1 = f1.to(device)

    recall = torchmetrics.classification.Recall(task= "multiclass",num_classes=3,average=None)
    recall = recall.to(device)

    confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3).to(device)
    confusion_matrix = confusion_matrix.to(device)

    total_loss = 0.0
    num_samples = 0

    criterion = nn.CrossEntropyLoss()
    matching_files = find_matching_files(file_path_input)
    
    if len(matching_files) == 0:
        raise FileNotFoundError("No files found")

    new_dataset = HM_Dataset()

    for file_number, files in matching_files.items():
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            pairs = load_from_txt(sgmFilename, ogmFilename)
            print(f"file number: {file_number}")
            load_into_dataset(pairs,new_dataset,training_image_size)

    new_dataLoader = DataLoader(new_dataset,batch_size=batch_size,shuffle=True)

    with torch.no_grad():
        for inputs, labels, coord in new_dataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            coord = coord.to(device)
            # Forward pass
            outputs = model(coord,inputs)
            #print("**************")
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
            per_class_accuracy.update(outputs, labels)

    time.sleep(1)

    accuracy = accuracy.compute()
    mse = mse.compute()
    mae = mae.compute()
    iou = iou.compute()
    precision = precision.compute()
    f1 = f1.compute()
    recall = recall.compute()
    confusion_matrix = confusion_matrix.compute()
    class_accuracies = per_class_accuracy.compute()

    avg_loss = total_loss/num_samples
    print("-----------------------------------------------------------------------------------------------------")
    print(f"Model name: {model_location}")

    print(f"Evaluation Accuracy: {accuracy}")
    print(f"Per class accuracy:{class_accuracies}")
    print(f"iou: {iou}")
    print(f"average loss: {avg_loss}")
    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"Precision: {precision}")
    print(f"f1: {f1}")
    print(f"recall: {recall}")
    print(f"Confusion Matrix:\n{confusion_matrix}")

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


