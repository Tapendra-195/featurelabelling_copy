from IPython.core.display import display, HTML
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import pg_fitter_tools as fit
import sk_geo_tools as sk
import os
import csv
import math




####################Specify File Inputs###############################################
filename = "OUTPUT.jpg"
textFilename = "379 (BJ).txt"
# Output file will contain these initials next to coordinates:
initials = "MS"
######################################################################################

####################Settings##########################################################
plotReprojectedPMTs = True
plotReprojectedFeatures = False
plotManualFeatures = True
inputFileType = "manualLabelling"
#inputFileType = "imageProcessing"

offset =250
suppressInput = False
labellingStats = True
######################################################################################

####################Internal Camera Parameters########################################
focal_length = [2.760529621789217e+03, 2.767014510543478e+03]
principle_point = [1.914303537872458e+03, 1.596386868474348e+03]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]
######################################################################################

####################External Camera Parameters########################################
rotation_vector = np.array([[1.52593994],[-0.71901074],[0.60290209]])
translation_vector = np.array([[-100.74973094],[1606.91543897],[-916.79105257]])
######################################################################################



######################################################################################





####################Plot points#######################################################
def plot_pmts(coordinates, imageIdentifier, off_set=0, color=(0,255,0)):
    counter = 0
    for i in coordinates:
        #print(i[0],i[1])
        if np.abs(int(i[0]))<=4000 and np.abs(int(i[1])-int(off_set))<=2750:
            plotx = int(i[0])
            ploty = int(i[1])-int(off_set)
            cv2.circle(imageIdentifier,(plotx,ploty),5,color,-1)
            counter=counter+1
            cv2.putText(img, f'{i[2]}', (plotx,ploty), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

######################################################################################

####################Obtain Reprojected Points#########################################
def obtain_reprojected_points(features):
    nfeatures = len(features)

    seed_feature_locations = np.zeros((nfeatures, 3))
    feature_index = {}
    index_feature = {}
    f_index = 0
    for f_key, f in features.items():
        feature_index[f_key] = f_index
        index_feature[f_index] = f_key
        seed_feature_locations[f_index] = f
        f_index += 1

    
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    transformed_positions = (rotation_matrix @ seed_feature_locations.T).T + translation_vector.T
    indices = np.where(transformed_positions[:,2]>0)[0]

    camera_matrix = build_camera_matrix(focal_length, principle_point)
    distortion = build_distortion_array(radial_distortion, tangential_distortion)


    reprojected = cv2.projectPoints(seed_feature_locations[indices], rotation_vector, translation_vector,camera_matrix, distortion)[0].reshape((indices.size, 2))

    reprojected_points = {}
    reprojected_points[filename_no_extension] = dict(zip([index_feature[ii] for ii in indices], reprojected))

    return reprojected_points
######################################################################################


def build_camera_matrix(focal_length, principle_point):
    return np.array([
        [focal_length[0], 0, principle_point[0]],
        [0, focal_length[1], principle_point[1]],
        [0, 0, 1]], dtype=float)


def build_distortion_array(radial_distortion, tangential_distortion):
    return np.concatenate((radial_distortion, tangential_distortion)).reshape((4, 1))


################################Read Points from Text File##############################
def read_image_feature_locations(filename, delimiter="\t", offset=np.array([0., 0])):

    if(inputFileType == "manualLabelling"):
        image_feature_locations = {}
        
        coordinates = []
        

        with open(filename, mode='r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for r in reader:
                image_feature_locations.setdefault(r[0],{}).update({r[1]: np.array([r[2], r[3]]).astype(float) + offset})
                coordinates.append([r[2],int(float(r[3])),r[1]])


        coordinates = np.stack(list(coordinates))
        return coordinates
######################################################################################

    if(inputFileType == "imageProcessing"):
        image_feature_locations = {}
        
        coordinates = []
        

        with open(filename, mode='r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for r in reader:
                image_feature_locations.setdefault(r[0],{}).update({r[0]: np.array([r[1], r[2]]).astype(float) + offset})
                coordinates.append(            [          int(float(r[1])),int(float(r[2]))              ]                 )


        coordinates = np.stack(list(coordinates))
        return coordinates
######################################################################################   








# Create output text file:
filename_no_extension = os.path.splitext(filename)[0]
camera_rotations = np.zeros((1, 3))
camera_translations = np.zeros((1, 3))
camera_rotations[0, :] = rotation_vector.ravel()
camera_translations[0, :] = translation_vector.ravel()

outputTextFilename = os.path.join(filename_no_extension+"-"+inputFileType+" - angle to vertical.txt")
print("Creating file for writing output:",outputTextFilename)
outputFile = open(outputTextFilename,"w")

# Read all 2D locations from input file
input_feature_locations = read_image_feature_locations(textFilename, offset=np.array([0, offset]))


cv2.namedWindow(filename,cv2.WINDOW_NORMAL)
cv2.moveWindow(filename, 500, 0)
img = cv2.imread(filename)



if(plotManualFeatures==True):
    plot_pmts(input_feature_locations, img, 0, (0,255,255))

   
cv2.imwrite("OUTPUT.jpg",img)
cv2.imshow(filename,img)
outputFile.close()














