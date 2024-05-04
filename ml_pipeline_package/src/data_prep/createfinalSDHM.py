import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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
    np_social_grid = np.array(largest_social_grid[4:])
    reshape_final_social_grid = np_social_grid.reshape((largest_height, largest_width))
    reshape_final_social_grid = reshape_final_social_grid.astype(float)
    reshape_final_social_grid = np.nan_to_num(reshape_final_social_grid,nan=0)

    test_ogm = np.array(largest_obstacle_grid[4:])
    test_ogm = test_ogm.reshape((largest_height, largest_width))
    test_ogm = test_ogm.astype(float)
    show_array_as_image(test_ogm)
    
    for pair in pairs:
        if pair[0][1] != largest_header:
            socialGridMap = np.array(pair[0][4:])
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
            elif aggregation_method == "weighted_average":
                pass
            else:
                raise ValueError(f"Invalid aggregation method: {aggregation_method}")
            
            
    show_array_as_image(reshape_final_social_grid)
    reshape_final_social_grid = np.ravel(reshape_final_social_grid)
    sgmFront = np.array([1,largest_header,largest_height,largest_width])
    sgm = np.concatenate((sgmFront,reshape_final_social_grid))
    sgmList = sgm.tolist()

    pair = (sgmList,largest_obstacle_grid)

    return pair

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

def addFilesToDataset(matching_files, folderPathOutput):
    for file_number, files in matching_files.items():
        if 'socialGridMap' in files and 'obstacleGridMap' in files:
            sgmFilename = files['socialGridMap']
            ogmFilename = files['obstacleGridMap']
            print(f"Pairs for files with number {file_number}:")
            pair = loadFromTxt(sgmFilename, ogmFilename,"average")
            print(len(pair))

            # Separate the first pair
            first_pair = np.array(pair[0]).astype(float)
            remaining_pair = np.array(pair[1]).astype(float)

            # Create folder path with variable
            folder_path = os.path.join(folderPathOutput, "")  # Ensure trailing slash

            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)  # Handles existing folders

            # Construct file paths with folder path variable
            social_file_path = os.path.join(folder_path, sgmFilename + "_final_social_data.txt")
            obstacle_file_path = os.path.join(folder_path, ogmFilename + "_final_obstacle_data.txt")

            # Open files in append mode
            with open(social_file_path, "a") as social_file:
                np.savetxt(social_file,[first_pair],fmt="%f",delimiter=",",newline='\n')

            with open(obstacle_file_path, "a") as obstacle_file:
                np.savetxt(obstacle_file, [remaining_pair],fmt="%f",delimiter=",",newline='\n')

            # Print or process remaining pairs (optional)
            # print(f"Remaining pairs: {remaining_pairs}")
        else:
            print("No pairs found in files.")

        
def socialHeatDensityCreate(folderPathInput,folderPathOutput):
    matching_pairs = find_matching_files(folderPathInput)
    addFilesToDataset(matching_pairs,folderPathOutput)

if __name__ == "__main__":
    folderPathInput = "/home/danielhixson/ros/noetic/system/src/ML/ml/dataCollection/maps/MAPSSPLITBYINDEX"
    folderPathOutput = "/home/danielhixson/ros/noetic/system/src/ML/ml/dataCollection/maps/MAPS_DENSITY"
    socialHeatDensityCreate(folderPathInput,folderPathOutput)