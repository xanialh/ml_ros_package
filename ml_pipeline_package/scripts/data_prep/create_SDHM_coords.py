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
            social_GridMap = np.array(pair[0][6:])
            height,width = int(float(pair[0][2])),int(float(pair[0][3]))
            reshape_socialGridMap = social_GridMap.reshape(height,width)
            reshape_socialGridMap = reshape_socialGridMap.astype(float)
            padded_array = pad_array_to_shape(reshape_socialGridMap,(largest_height, largest_width),0)
            padded_array = padded_array.astype(float)
            padded_array = np.nan_to_num(padded_array,nan=0)
          
            reshape_final_social_grid += padded_array / len(pairs) 
           
    reshape_final_social_grid = min_max_norm(reshape_final_social_grid)
    
    new_pairs = []

    for pair in pairs:
        if pair[1][1] != largest_header:
            ogm_header = pair[1][1]
            obstacle_GridMap = np.array(pair[1][6:])
            height,width = int(float(pair[1][2])),int(float(pair[1][3]))
            x, y = int(float(pair[0][4])),int(float(pair[0][5]))
            reshape_obstacleGridMap = obstacle_GridMap.reshape(height,width)
            reshape_obstacleGridMap = reshape_obstacleGridMap.astype(float)

            cropped_social_grid = reshape_final_social_grid[:height, :width]

            reshape_obstacleGridMap = np.ravel(reshape_obstacleGridMap)
            ogm_front = np.array([0,ogm_header,height,width,x,y])
            ogm = np.concatenate((ogm_front,reshape_obstacleGridMap))
            ogm_list = ogm.tolist()

            sgm_front = np.array([1,ogm_header,height,width,x,y])
            sgm = np.concatenate((sgm_front,cropped_social_grid.ravel()))
            sgm_list = sgm.tolist()

            new_pairs.append((sgm_list,ogm_list))
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

def add_files_to_dataset(matching_files, folder_path_output):
    for file_number, files in matching_files.items():
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgm_filename = files['socialGridMap']
            print(sgm_filename)
            ogm_filename = files['obstacleGridMap']

            sgm_filename_only = os.path.basename(sgm_filename)
            ogm_filename_only = os.path.basename(ogm_filename)

            print(f"Pairs for files with number {file_number}:")
            pairs = load_from_txt(sgm_filename, ogm_filename)
            # Create folder path with variable
            folder_path = os.path.join(folder_path_output, "")  # Ensure trailing slash

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

        
def social_heat_density_create(folderPathInput,folder_path_output):
    matching_pairs = find_matching_files(folderPathInput)
    add_files_to_dataset(matching_pairs,folder_path_output)

def load_config():
    # Load configuration
    try:
        with open("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def main():
    config_full = load_config()
    config = config_full["create_SDHM_coords"]
    folderPathInput = config["folder_path_input"]
    folder_path_output = config["folder_path_output"]
    social_heat_density_create(folderPathInput,folder_path_output)
    

if __name__ == "__main__":
    main()