import csv
import numpy as np
trueFileName = "045-manualLabelling.txt"
createdFileName = "045-manualLabelling - angle to vertical.txt"
methodName = "matching to reprojected features"

##trueFile = open(trueFileName, "r")
createdFile = open(createdFileName, "r")

##trueFeatures = trueFile.readlines()

trueFeatures = []
with open(trueFileName, mode='r') as trueFile:
    reader = csv.reader(trueFile, delimiter="\t")
    for r in reader:
        trueFeatures.append((r[1],r[2],r[3]))

labelledFeatures = []
with open(createdFileName, mode='r') as createdFile:
    reader = csv.reader(createdFile, delimiter="\t")
    for r in reader:
        labelledFeatures.append((r[1],r[2],r[3]))



matches = 0
for labelledFeature in labelledFeatures:
    for trueFeature in trueFeatures:
        if(labelledFeature == trueFeature):
            matches = matches+1
            print(labelledFeature,trueFeature)
    #feature_locations = {r[0]: np.array([r[1], r[2], r[3]]) for r in reader}
#print(feature_locations)

#print(featureLabelArray[2])

print(methodName+":", "Correctly found", matches, "features out of", len(trueFeatures), ",",(matches / len(labelledFeatures)) *100,"% correct.")
