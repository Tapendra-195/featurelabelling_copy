import csv
import numpy as np
trueFileName = "he_bolts239.txt"
createdFileName = "239-imageProcessing - angle to vertical.txt"
methodName = "angle to first bolt"

##trueFile = open(trueFileName, "r")
createdFile = open(createdFileName, "r")

##trueFeatures = trueFile.readlines()
trueFeatures = []
with open(trueFileName, mode='r') as trueFile:
    reader = csv.reader(trueFile, delimiter="\t")
    for r in reader:
        r[1] = int(float(r[1]))
        r[2] = int(float(r[2]))
        #print(r[1])
        trueFeatures.append((r[0],r[1],r[2]))
        

labelledFeatures = []
with open(createdFileName, mode='r') as createdFile:
    reader = csv.reader(createdFile, delimiter="\t")
    for r in reader:
        r[1] = r[1][-2:]
        r[2] = int(r[2])
        r[3] = int(r[3])
        labelledFeatures.append((r[1],r[2],r[3]))
        #print(r)



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
