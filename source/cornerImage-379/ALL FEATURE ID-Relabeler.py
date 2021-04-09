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
filename = "379.jpg"
textFilename = "379.txt"
sourceTextFilename = "379 (BJ).txt"
initials = "MS"
######################################################################################

plotReprojectedPMTs = False
plotReprojectedFeatures = False
plotManualFeatures = True
inputFileType = "manualLabelling"
#inputFileType = "imageProcessing"

offset = 0
suppressInput = False
labellingStats = False



####################Plot points#######################################################
def plot_pmts(coordinates, imageIdentifier, off_set=0, color=(0,255,0)):
    counter = 0
    for i in coordinates:
        #print(i[0],i[1])
        if np.abs(int(i[0]))<=4000 and np.abs(int(i[1])-int(off_set))<=2750 and i[2][-2:]!="00":
            plotx = int(i[0])
            ploty = int(i[1])-int(off_set)
            cv2.circle(imageIdentifier,(plotx,ploty),4,color,-1)
            counter=counter+1
######################################################################################


################################Read Points from Text File##############################
def read_image_feature_locations(filename, delimiter="\t", offset=np.array([0., 0])):
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


# Create output text file:
filename_no_extension = os.path.splitext(filename)[0]


outputTextFilename = os.path.join(filename_no_extension+"-correctedTapendra-ALLFEATURES.txt")
print("Creating file for writing output:",outputTextFilename)
outputFile = open(outputTextFilename,"w")




# Read all 2D locations from input file
input_feature_locations = read_image_feature_locations(textFilename, offset=np.array([0, offset]))

source_feature_locations = read_image_feature_locations(sourceTextFilename, offset=np.array([0, offset]))

#print(input_feature_locations)
#print(source_feature_locations)
cv2.namedWindow(filename,cv2.WINDOW_NORMAL)
cv2.moveWindow(filename, 500, 0)
img = cv2.imread(filename)


#if(plotReprojectedFeatures==True):
#    plot_pmts(feature_repro_coords,img,0,(0,255,0))

#for item in source_feature_locations:
#        if(item[2][-2:] == "00"):
#            PMT_ID = item[2][:-3]
#            plotx = int(item[0])
#            ploty = int(item[1])
#        if abs(plotx)<=4000 and abs(ploty)<=2750:
#                cv2.circle(img,(plotx,ploty),6,(220,220,0),-1)
#                PMT_ID = PMT_ID.lstrip("0")
#                cv2.putText(img, f'{PMT_ID}', (plotx-50, ploty+50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (128,128,255), 3)

#if(plotManualFeatures==True):
#    plot_pmts(input_feature_locations, img, 0, (255,0,0))
#    plot_pmts(source_feature_locations, img, 0, (220,220,0))
    



################################Find nearest feature to assign##############################
##Take an input point from the input file
for inputPoints in input_feature_locations:
    minDistance = 100000
    closestItemPMT = False

    ## Is that point a PMT centre? Match PMT centre to point
    for item in source_feature_locations:
        
        pmtPoints = (int(item[0]),int(item[1]))
        if abs(pmtPoints[0])<=4000 and abs(pmtPoints[1]-offset)<2750:
            distance =  math.sqrt(     ( int(inputPoints[0])-int(pmtPoints[0]) )**2 + ( int(inputPoints[1])-int(pmtPoints[1]) )**2)
            if(minDistance>distance):
                minDistance = distance
                closestItem = item
                #closestItemPMT = True
    if closestItem[2][-2:] == "00":
        if minDistance < 100:
            inputPoints[2] = closestItem[2]
            outputFile.write("%s\t%s\t%d\t%d\t%s\n" %(filename_no_extension,closestItem[2],int(inputPoints[0]),int(inputPoints[1]),initials))
            cv2.circle(img,(int(inputPoints[0]),int(inputPoints[1])),6,(255,0,0),-1)
            cv2.circle(img,(int(closestItem[0]),int(closestItem[1])),6,(220,220,0),-1)
            cv2.line(img, (  int(inputPoints[0]),int(inputPoints[1])), (int(closestItem[0]),int(closestItem[1]) ), (0,0,255), thickness=3, lineType=8, shift=0)
            PMT_ID = closestItem[2][:-3].lstrip("0")
            cv2.putText(img, f'{PMT_ID}', (int(closestItem[0])-50, int(closestItem[1])+50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (128,128,255), 3)
    else:
        if minDistance < 10:
            inputPoints[2] = closestItem[2]
            cv2.circle(img,(int(inputPoints[0]),int(inputPoints[1])),6,(255,0,0),-1)
            cv2.circle(img,(int(closestItem[0]),int(closestItem[1])),6,(220,220,0),-1)
            cv2.line(img, (  int(inputPoints[0]),int(inputPoints[1])), (int(closestItem[0]),int(closestItem[1]) ), (0,0,255), thickness=3, lineType=8, shift=0)
            outputFile.write("%s\t%s\t%d\t%d\t%s\n" %(filename_no_extension,closestItem[2],int(inputPoints[0]),int(inputPoints[1]),initials))


#    print(inputPoints[2], closestItem[2])
#    inputPoints[2] = closestItem[2]
    #if closestItem[2][-2:] == "00":
        #print(closestItem)
    #    closestItemPMT = True
        
    ## Is that point a feature? Make sure point is not a feature

for inputPoints in input_feature_locations:
    
    if inputPoints[2][-2:] == "00":
        bolts = []
        PMTID = closestItem[2][:-3]
        for boltPoints in input_feature_locations:
            if(boltPoints[2][:-3] == inputPoints[2][:-3] and boltPoints[2][-2:] != "00"):
                bolts.append((int(boltPoints[0]),int(boltPoints[1]),boltPoints[2]))


        for item in bolts:
            ordered_bolt_no = int(item[2][-2:])
                #Find the angle at the centre of the PMT between the x axis and 
                #the line pointing from the centre of the PMT to the centre of the bolt
            lengthx = item[0]-int(inputPoints[0])
            lengthy = item[1]-int(inputPoints[1])
            length = math.sqrt(lengthx**2+lengthy**2)
            if(lengthx==0):
                angle=3.1415926/2
            else:
                angle = np.arctan(lengthy/lengthx)
               #Calculate text location based on this angle and the location of the bolt along the circle.
                 
                #bolt 1
            if(ordered_bolt_no==1):
                textx = int(item[0])
                texty = int(item[1]+(30))
                #bolts 2 through 14 (on the right side of the circle/the 1st and 4th quadrant) 
            elif(ordered_bolt_no<=12):
                textx = int(item[0]-(30)*np.cos(angle))
                texty = int(item[1]-(30)*np.sin(angle))
            #bolt 13
            elif(ordered_bolt_no==13):
                textx = int(item[0]+(30)*np.cos(angle))
                texty = int(item[1]+(30)*np.sin(angle))
               #bolts 14 through 19 (on the 3rd quadrant)
            elif(ordered_bolt_no<=19):
                textx = int(item[0]+(30)*np.cos(angle))
                texty = int(item[1]+(30)*np.sin(angle))
             #bolts 20 through 24 (on the 2nd quadrant)
            else:
                textx = int(item[0]+(30)*np.cos(angle))
                texty = int(item[1]+(30)*np.sin(angle))
            pointerx = int((textx+textx+20)/2)
            pointery = int((texty+texty-20)/2)





            cv2.line(img, (item[0],item[1]), (textx,texty), (0,150,150), thickness=2, lineType=8, shift=0)
                     #cv2.rectangle(img,(textx,texty),(textx+15,texty-15),(0,0,0), thickness=-1, lineType=8, shift=0)
                     #cv2.circle(img,(textx,texty),10,(0,0,0),-1)
                    #cv2.circle(img, (item[1], item[0]), 5, (0,0,255), -1)
            cv2.putText(img, f'{ordered_bolt_no}', (textx,texty), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)



    #############################################################################################
    if(labellingStats == True):
        incorrectlyMatchedFeatures = comparingLabelling.compare(inputFileType, outputTextFilename, "379 (BJ).txt")


        for i in incorrectlyMatchedFeatures:
            cv2.circle(img,(int(float(i[1])),int(float(i[2]))),5,(255,255,255),1)
            text = i[0][-2:]
            text = text.lstrip("0")
            cv2.putText(img, f'{text}', (int(float(i[1])),int(float(i[2]))), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,180), 1)
  
      
cv2.imwrite("OUTPUT.jpg",img)
cv2.imshow(filename,img)
outputFile.close()


