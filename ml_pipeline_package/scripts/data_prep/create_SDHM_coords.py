import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

def pad_array_to_shape(array, target_shape, pad_value=0):
        # Calculate the padding amounts for each dimension
    pad_width = [(target_shape[0] - array.shape[0], 0), (target_shape[1] - array.shape[1], 0)]

    # Pad the array using np.pad
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=pad_value)

    return padded_array

def show_array_as_image(array):
  plt.imshow(array, cmap='gray')
  plt.colorbar()
  plt.show()

def min_max_norm(arr, new_min=0, new_max=1):
  """
  Normalizes a NumPy array using min-max scaling.

  Args:
      arr (np.ndarray): The array to normalize.
      new_min (float, optional): The minimum value in the normalized range. Defaults to 0.
      new_max (float, optional): The maximum value in the normalized range. Defaults to 1.

  Returns:
      np.ndarray: The normalized array.
  """

  # Find minimum and maximum values
  min_val = np.min(arr)
  max_val = np.max(arr)

  # Normalize the data (avoid division by zero)
  if min_val != max_val:
    scaled_arr = (arr - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
  else:
    scaled_arr = np.ones_like(arr) * new_min  # All elements are equal, return new_min

  return scaled_arr

def loadFromTxt(sgmFilename,ogmFilename,aggregation_method="sum"):
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

    #change to cropping with final heatmap
    largest_header_pair = max(pairs, key=lambda p: float(p[0][1]))
    largest_social_grid, largest_obstacle_grid = largest_header_pair
    largest_height, largest_width = int(float(largest_social_grid[2])), int(float(largest_social_grid[3]))
    largest_header = float(largest_social_grid[1])
    largest_x = float(largest_social_grid[4])
    largest_y = float(largest_social_grid[5])
    np_social_grid = np.array(largest_social_grid[6:])
    reshape_final_social_grid = np_social_grid.reshape((largest_height, largest_width))
    reshape_final_social_grid = reshape_final_social_grid.astype(float)
    reshape_final_social_grid = np.nan_to_num(reshape_final_social_grid,nan=0)
    
    for pair in pairs:
        if pair[0][1] != largest_header:
            socialGridMap = np.array(pair[0][6:])
            height,width = int(float(pair[0][2])),int(float(pair[0][3]))
            reshape_socialGridMap = socialGridMap.reshape(height,width)
            reshape_socialGridMap = reshape_socialGridMap.astype(float)
            padded_array = pad_array_to_shape(reshape_socialGridMap,(largest_height, largest_width),0)
            padded_array = padded_array.astype(float)
            padded_array = np.nan_to_num(padded_array,nan=0)
            
            # Apply chosen aggregation method
            if aggregation_method == "sum":
                reshape_final_social_grid += padded_array
            elif aggregation_method == "average":
                reshape_final_social_grid += padded_array / len(pairs)  # Average over all grids
            elif aggregation_method == "max":
                reshape_final_social_grid = np.maximum(reshape_final_social_grid, padded_array)
            else:
                raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    
    reshape_final_social_grid = min_max_norm(reshape_final_social_grid)
    reshape_final_social_grid = np.ravel(reshape_final_social_grid)
    
    new_pairs = []

    for pair in pairs:
        if pair[1][1] != largest_header:
            ogm_header = pair[1][1]
            obstacleGridMap = np.array(pair[1][6:])
            height,width = int(float(pair[1][2])),int(float(pair[1][3]))
            x, y = int(float(pair[0][4])),int(float(pair[0][5]))
            reshape_obstacleGridMap = obstacleGridMap.reshape(height,width)
            reshape_obstacleGridMap = reshape_obstacleGridMap.astype(float)
            padded_ogm_array = pad_array_to_shape(reshape_obstacleGridMap,(largest_height, largest_width),0)
            padded_ogm_array = padded_ogm_array.astype(float)

            cropped_social_grid = reshape_final_social_grid[:height, :width]

            padded_ogm_array = np.ravel(padded_ogm_array)
            ogmFront = np.array([0,ogm_header,largest_height,largest_width,x,y])
            ogm = np.concatenate((ogmFront,padded_ogm_array))
            ogmList = ogm.tolist()

            sgmFront = np.array([1,ogm_header,largest_height,largest_width,largest_x,largest_y])
            sgm = np.concatenate((sgmFront,cropped_social_grid.ravel()))
            sgmList = sgm.tolist()

            new_pairs.append((sgmList,ogmList))
    return new_pairs

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

def addFilesToDataset(matching_files, folderPathOutput):
    for file_number, files in matching_files.items():
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']

            sgm_filename_only = os.path.basename(sgmFilename)
            ogm_filename_only = os.path.basename(ogmFilename)

            print(f"Pairs for files with number {file_number}:")
            pairs = loadFromTxt(sgmFilename, ogmFilename,"average")
            # Create folder path with variable
            folder_path = os.path.join(folderPathOutput, "")  # Ensure trailing slash

            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)  # Handles existing folders

            # Construct file paths with folder path variable
            social_file_path = os.path.join(folder_path, sgm_filename_only)
            obstacle_file_path = os.path.join(folder_path, ogm_filename_only)

            for pair in pairs:
                sgm = np.array(pair[0]).astype(float)
                ogm = np.array(pair[1]).astype(float)
                # Open files in append mode
                with open(social_file_path, "a") as social_file:
                    np.savetxt(social_file,[sgm],fmt="%f",delimiter=",",newline='\n')

                with open(obstacle_file_path, "a") as obstacle_file:
                    np.savetxt(obstacle_file, [ogm],fmt="%f",delimiter=",",newline='\n')

            # Print or process remaining pairs (optional)
            # print(f"Remaining pairs: {remaining_pairs}")
        else:
            print("No pairs found in files.")

        
def socialHeatDensityCreate(folderPathInput,folderPathOutput):
    matching_pairs = find_matching_files(folderPathInput)
    addFilesToDataset(matching_pairs,folderPathOutput)

def loadConfig():
    # Load configuration
    try:
        with open("ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def main():
    configFull = loadConfig()
    config = configFull["create_SDHM_coords"]
    folderPathInput = config["folderPathInput"]
    folderPathOutput = config["folderPathOutput"]
    socialHeatDensityCreate(folderPathInput,folderPathOutput)
    

if __name__ == "__main__":
    main()