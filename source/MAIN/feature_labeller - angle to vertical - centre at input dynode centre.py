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

import comparingLabelling




####################Specify File Inputs###############################################
filename = "045.jpg"
textFilename = "045-manualLabelling.txt"
# Output file will contain these initials next to coordinates:
initials = "MS"
######################################################################################

####################Settings##########################################################
plotReprojectedPMTs = True
plotReprojectedFeatures = False
plotManualFeatures = True
inputFileType = "manualLabelling"
#inputFileType = "imageProcessing"

offset = 0
suppressInput = False

#Compare labels from script to original labels (if they exist) - Specify file locations at end of program.
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
            cv2.circle(imageIdentifier,(plotx,ploty),7,color,-1)
            counter=counter+1
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
                coordinates.append([r[2],int(float(r[3]))])


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


# Read all 3D PMT locations
all_pmt_locations = fit.read_3d_feature_locations("parameters/SK_all_PMT_locations.txt")

all_bolt_locations = sk.get_bolt_locations_barrel(all_pmt_locations)


# Obtain 2D reprojected feature points
pmt_reprojected_points = obtain_reprojected_points(all_pmt_locations)

feature_reprojected_points = obtain_reprojected_points(all_bolt_locations)

pmt_repro_coords = np.stack(list(pmt_reprojected_points[filename_no_extension].values()))

feature_repro_coords = np.stack(list(feature_reprojected_points[filename_no_extension].values()))


# Read all 2D locations from input file
input_feature_locations = read_image_feature_locations(textFilename, offset=np.array([0, offset]))


cv2.namedWindow(filename,cv2.WINDOW_NORMAL)
cv2.moveWindow(filename, 500, 0)
img = cv2.imread(filename)


# Plot reprojected features, 
if(plotReprojectedFeatures==True):
    plot_pmts(feature_repro_coords,img,0,(0,255,0))

if(plotReprojectedPMTs==True):
    for item in pmt_reprojected_points[filename_no_extension]:
        if(item[-2:] == "00"):
            PMT_ID = item[:-3]
            PMT_ID_Coordinates = pmt_reprojected_points[filename_no_extension][item]
            plotx = int(PMT_ID_Coordinates[0])
            ploty = int(PMT_ID_Coordinates[1])
        if abs(plotx)<=4000 and abs(ploty)<=2750:
                cv2.circle(img,(plotx,ploty),10,(0,255,255),-1)
                PMT_ID = PMT_ID.lstrip("0")
                cv2.putText(img, f'{PMT_ID}', (plotx-50, ploty+50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (128,128,255), 3)

if(plotManualFeatures==True):
    plot_pmts(input_feature_locations, img, 0, (255,0,0))





################################Find nearest feature to assign##############################
##Take an input point from the input file
for inputPoints in input_feature_locations:
    minDistance = 100000
    closestItemPMT = False


    #nodes = np.asarray(pmt_repro_coords)
    #dist = np.sqrt(np.sum((nodes - [int(inputPoints[0]),int(inputPoints[1])])**2, axis=1))
    #closest = np.argmin(dist)
    #print(pmt_repro_coords[closest])
            
    ## Is that point a PMT centre? Match PMT centre to point
    for item in pmt_reprojected_points[filename_no_extension]:        
        pmtPoints = pmt_reprojected_points[filename_no_extension][item]
        if abs(pmtPoints[0])<=4000 and abs(pmtPoints[1]-offset)<2750:
            distance =  math.sqrt(     ( int(inputPoints[0])-int(pmtPoints[0]) )**2 + ( int(inputPoints[1])-int(pmtPoints[1]) )**2)

            if(minDistance>distance):
                minDistance = distance
                closestItem = item
                closestItemPMT = True

    ## Is that point a feature? Make sure point is not a feature
    for item in feature_reprojected_points[filename_no_extension]:
        if(closestItemPMT == True):
            featurePoints = feature_reprojected_points[filename_no_extension][item]
            if abs(featurePoints[0])<=4000 and abs(featurePoints[1])<2750:
                distance =  math.sqrt(     ( int(inputPoints[0])-int(featurePoints[0]) )**2 + ( int(inputPoints[1])-int(featurePoints[1]) )**2)
                if(minDistance>distance):
                    minDistance = distance
                    closestItem = item
                    closestItemPMT = False
            
    ## If that point is a PMT centre
    if closestItemPMT == True:
        bolts = []
        reprojected_points = []
        
        cv2.line(img, (int(inputPoints[0]),int(inputPoints[1])), (int(pmt_reprojected_points[filename_no_extension][closestItem][0])
              ,int(pmt_reprojected_points[filename_no_extension][closestItem][1])), (0,0,255), thickness=3, lineType=8, shift=0)

        ## Save it to the output file with a PMT-ID
        outputFile.write("%s\t%s\t%d\t%d\t%s\n" %(filename_no_extension,closestItem[:],int(inputPoints[0]),int(inputPoints[1]),initials))

        ## Find matching bolts to that PMT
        for boltPoints in input_feature_locations:
            distanceFromCentre = math.sqrt(( int(inputPoints[0])-int(boltPoints[0]) )**2 + ( int(inputPoints[1])-int(boltPoints[1]) )**2)
            if distanceFromCentre > 200 or (inputPoints[0]==boltPoints[0] and inputPoints[1]==boltPoints[1]) or distanceFromCentre < 5:
                continue
            
            bolts.append([int(boltPoints[0]),int(boltPoints[1])])

        ## Create a list of associated reprojected bolt features for that PMT
        for item in feature_reprojected_points[filename_no_extension]:
            if(closestItem[:-3] == item[:-3]):
                reprojected_points.append((int(feature_reprojected_points[filename_no_extension][item][0]),int(feature_reprojected_points[filename_no_extension][item][1])))

        ## Determining the search radius
        counter = 0
        average_distance = 0
        for point in reprojected_points:
            distance = math.sqrt(( int(point[0])-int(inputPoints[0]) )**2 + ( int(point[1])-int(inputPoints[1]))**2)

            average_distance = distance+average_distance
            counter = counter+1
            cv2.circle(img,(point[0],point[1]),5,(0,100,0),-1)
            
        searchRadius = int(average_distance / counter)

######################Ordering the identified input bolts##########################
        ordered_input_bolts = []
        
        #circle parameters
        centre_x = int(inputPoints[0])
        centre_y = int(inputPoints[1])
        radius = int((searchRadius - 30)*1)

        
        #bolt locations found beforehand
        nodes = np.asarray(bolts)
        if(len(nodes) != 0):        
            #loop over 2pi in (60 for now) steps finding closest bolts
            for i in np.linspace(0, 2*np.pi, num=60) : 
                #point on circle (scaled by radius of hough transform) to search near
                search_x = int(centre_x + int(radius*np.sin(i)))
                search_y = int(centre_y - int(radius*np.cos(i)))
                #cv2.line(img, (int(inputPoints[0]),int(inputPoints[1])), (int(search_x),int(search_y)), (0,120,0), thickness=2, lineType=8, shift=0)            
                #find bolt closest to point
                dist = np.sqrt(np.sum((nodes - [search_x,search_y])**2, axis=1))
                closest = np.argmin(dist)
                #only append each bolt coordinate once
                if bolts[closest] not in ordered_input_bolts :
                    ordered_input_bolts.append(bolts[closest])
    #                cv2.line(img, (int(inputPoints[0]),int(inputPoints[1])), (int(bolts[closest][0])
    #                      ,int(bolts[closest][1])), (0,0,255), thickness=3, lineType=8, shift=0)

        #print("    ", arcCircumference)
        
        #for i in np.linspace(0, 2*np.pi, num=24): 
            #point on circle (scaled by radius of hough transform) to search near
        #    search_x = int(centre_x + int((radius+300)*np.sin(i)))
        #    search_y = int(centre_y - int((radius+300)*np.cos(i)))
        #    cv2.line(img, (int(inputPoints[0]),int(inputPoints[1])), (int(search_x),int(search_y)), (0,0,120), thickness=2, lineType=8, shift=0)             
#########################Saving the bolts ##########################################
        #draw bolt locations and number them 
        bolt_no = 0
        #arcCircumference = (2*np.pi/24) * searchRadius
        idealAngle = 360.0 / 24.0
        idealAngleHalf = idealAngle / 2.0
        PMTID = closestItem[:-3]
        ######################### Automatic bolt number labelling on mask ##############################
        averageAngle = 0
        angleCounter = 0
#        print("///////////////////////////////////////////////////////////////\n///////////////////////////////////////////////////////////////")
        
#        print("Looking at PMT", PMTID)
#        print("There are",len(ordered_input_bolts), "points to look at.")

        prevBolt = 24
        for bolt_no in range(len(ordered_input_bolts)) :
            #print("--------------------------------")
            #print("Looking at bolt index: ",bolt_no)                           

            currentLineX = float(ordered_input_bolts[bolt_no][0])-float(inputPoints[0])
            currentLineY = float(ordered_input_bolts[bolt_no][1])-float(inputPoints[1])

            currentLineLength = math.sqrt(     ( currentLineX )**2 + ( currentLineY )**2   )

            horizontalLength = currentLineX
            #print("Y length = ", currentLineY, "X length = ", currentLineX)
            if(currentLineY == 0):
                if(currentLineX >= 0):
                    angle = 90.0
                else:
                    angle = 270.0
            else:
                angle = math.degrees(math.atan( (currentLineX / currentLineY)))

                if(currentLineX > 0 and currentLineY > 0):
                    angle = 180.0 - angle
                if(currentLineX > 0 and currentLineY < 0):
                    angle = abs(angle)
                if(currentLineX < 0 and currentLineY > 0):
                    angle = 180.0 + abs(angle)
                if(currentLineX < 0 and currentLineY < 0):
                    angle = 360.0 - abs(angle)

            ordered_bolt_no = int( (angle+idealAngleHalf) / idealAngle + 1)
            if(ordered_bolt_no == 25):
                ordered_bolt_no = 1

            #print(currentLineX,currentLineY,"Angle with the vertical is: ", angle, "Bolt number is: ", ordered_bolt_no)
            #print("The angle between these bolts is: ",angle * 180/np.pi)
            #print("This bolt will be labelled as", closestItem[:-3],ordered_bolt_no)
            
            printBoltNumber = f'{ordered_bolt_no:02}'

            outputFile.write("%s\t%s-%s\t%s\t%s\t%s\n" %(filename_no_extension,closestItem[:-3],printBoltNumber, str(ordered_input_bolts[bolt_no][0]), str(ordered_input_bolts[bolt_no][1]),initials))

            if(currentLineX==0):
                textAngle=3.1415926/2
            else:
                textAngle = np.arctan(currentLineY/currentLineX)
                #Calculate text location based on this angle and the location of the bolt along the circle.
                
            #bolt 1
            if(ordered_bolt_no==1):
                textx = int(ordered_input_bolts[bolt_no][0])
                texty = int(ordered_input_bolts[bolt_no][1]+(30))
            #bolts 2 through 14 (on the right side of the circle/the 1st and 4th quadrant) 
            elif(ordered_bolt_no<=12):
                textx = int(ordered_input_bolts[bolt_no][0]-(30)*np.cos(textAngle))
                texty = int(ordered_input_bolts[bolt_no][1]-(30)*np.sin(textAngle))
            #bolt 13
            elif(ordered_bolt_no==13):
                textx = int(ordered_input_bolts[bolt_no][0]+(30)*np.cos(textAngle))
                texty = int(ordered_input_bolts[bolt_no][1]+(30)*np.sin(textAngle))
               #bolts 14 through 19 (on the 3rd quadrant)
            elif(ordered_bolt_no<=19):
                textx = int(ordered_input_bolts[bolt_no][0]+(30)*np.cos(textAngle))
                texty = int(ordered_input_bolts[bolt_no][1]+(30)*np.sin(textAngle))
            #bolts 20 through 24 (on the 2nd quadrant)
            else:
                textx = int(ordered_input_bolts[bolt_no][0]+(30)*np.cos(textAngle))
                texty = int(ordered_input_bolts[bolt_no][1]+(30)*np.sin(textAngle))
            pointerx = int((textx+textx+20)/2)
            pointery = int((texty+texty-20)/2)


            cv2.line(img, (ordered_input_bolts[bolt_no][0],ordered_input_bolts[bolt_no][1]), (textx,texty), (0,140,140), thickness=1, lineType=8, shift=0)

            cv2.putText(img, f'{ordered_bolt_no}', (textx,texty), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            bolt_no += 1

#############################################################################################


#########################Obtaining statistics on incorrectly labelled features##########################################


if(labellingStats == True):
    incorrectlyMatchedFeatures = comparingLabelling.compare(inputFileType, outputTextFilename, textFilename)


    for i in incorrectlyMatchedFeatures:
        cv2.circle(img,(int(float(i[1])),int(float(i[2]))),5,(255,255,255),1)
        text = i[0][-2:]
        text = text.lstrip("0")
        cv2.putText(img, f'{text}', (int(float(i[1])),int(float(i[2]))), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,180), 1)
      
cv2.imwrite("OUTPUT.jpg",img)
cv2.imshow(filename,img)
outputFile.close()














