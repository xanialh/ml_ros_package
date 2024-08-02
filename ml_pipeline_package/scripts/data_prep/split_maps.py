import os
import csv
import yaml

def load_config():
    # Load configuration
    try:
        with open("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def split():
    config_full = load_config()
    config = config_full["split_maps"]

    directory_path_input = config["directory_path_input"]
    directory_path_output = config["directory_path_output"]

    txt_file_count = 0

    social_GridMap = "_socialGridMap.txt"
    obstacle_GridMap = "_obstacleGridMap.txt"

    for filename in os.listdir(directory_path_input):
        if filename.endswith(".txt"):
            current_filePath = os.path.join(directory_path_input, filename)

            new_SGM_File = filename.replace(".txt",social_GridMap )
            new_OGM_File = filename.replace(".txt",obstacle_GridMap)

            output_file_path_SGM = os.path.join(directory_path_output, new_SGM_File)
            output_file_path_OGM = os.path.join(directory_path_output, new_OGM_File)

            with open(output_file_path_SGM,"w") as SGM_file, open(output_file_path_OGM,"w") as OGM_file:
                with open(current_filePath,"r") as read_file:
                    reader = csv.reader(read_file)
                    for line in read_file:
                        if int(line[0]) == 1:
                            SGM_file.write(line)
                        elif int(line[0]) == 0:
                            OGM_file.write(line)

    SGM_file.close()
    OGM_file.close()
    read_file.close()

if __name__ == "__main__":
    split()