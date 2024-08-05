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

    socialGridMap = "_socialGridMap.txt"
    obstacleGridMap = "_obstacleGridMap.txt"

    for filename in os.listdir(directory_path_input):
        if filename.endswith(".txt"):
            currentFilePath = os.path.join(directory_path_input, filename)

            new_SGM_file = filename.replace(".txt",socialGridMap )
            new_OGM_file = filename.replace(".txt",obstacleGridMap)

            outputFilePath_SGM = os.path.join(directory_path_output, new_SGM_file)
            outputFilePath_OGM = os.path.join(directory_path_output, new_OGM_file)

            with open(outputFilePath_SGM,"w") as SGMFile, open(outputFilePath_OGM,"w") as OGMFile:
                with open(currentFilePath,"r") as readFile:
                    print(currentFilePath)
                    reader = csv.reader(readFile)
                    for line in readFile:
                        try:
                            value = int(line[0])
                            if value == 1:
                                SGMFile.write(line)
                            elif value == 0:
                                OGMFile.write(line)
                        except (ValueError, IndexError):
                            pass

    SGMFile.close()
    OGMFile.close()
    readFile.close()

if __name__ == "__main__":
    split()