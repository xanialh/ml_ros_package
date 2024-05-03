import os
import csv

directory_path_output = "/home/danielhixson/ros/noetic/system/src/ML/ml/dataCollection/maps/MAPSSPLITBYINDEX"
directory_path_input = "/home/danielhixson/ros/noetic/system/src/ML/ml/dataCollection/maps"

txt_file_count = 0

socialGridMap = "_socialGridMap.txt"
obstacleGridMmap = "_obstacleGridMap.txt"

for filename in os.listdir(directory_path_input):
    if filename.endswith(".txt"):
        currentFilePath = os.path.join(directory_path_input, filename)

        newSGMFile = filename.replace(".txt","_socialGridMap.txt")
        newOGMFile = filename.replace(".txt","_obstacleGridMap.txt")

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