import os
import csv
import yaml

def loadConfig():
    # Load configuration
    try:
        with open("config/config_split_maps.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def split():
    config = loadConfig()

    directory_path_input = config["directory_path_input"]
    directory_path_output = config["directory_path_output"]

    txt_file_count = 0

    socialGridMap = "_socialGridMap.txt"
    obstacleGridMap = "_obstacleGridMap.txt"

    for filename in os.listdir(directory_path_input):
        if filename.endswith(".txt"):
            currentFilePath = os.path.join(directory_path_input, filename)

            newSGMFile = filename.replace(".txt",socialGridMap )
            newOGMFile = filename.replace(".txt",obstacleGridMap)

            outputFilePath_SGM = os.path.join(directory_path_output, newSGMFile)
            outputFilePath_OGM = os.path.join(directory_path_output, newOGMFile)

            with open(outputFilePath_SGM,"w") as SGMFile, open(outputFilePath_OGM,"w") as OGMFile:
                with open(currentFilePath,"r") as readFile:
                    reader = csv.reader(readFile)
                    for line in readFile:
                        if int(line[0]) == 1:
                            SGMFile.write(line)
                        elif int(line[0]) == 0:
                            OGMFile.write(line)

    SGMFile.close()
    OGMFile.close()
    readFile.close()