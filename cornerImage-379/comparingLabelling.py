import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

#compare("manualLabelling","379 (BJ).txt","379-correctedTapendra-ALLFEATURES.txt")


def compare(fileName1, fileName2, imageName):
    print("\nRunning matching script to determine label accuracy.\n")

    features1 = []
    with open(fileName1, mode='r') as file1:
        reader = csv.reader(file1, delimiter="\t")
        for r in reader:
            features1.append((r[2],r[3],r[1]))

    features2 = []
    with open(fileName2, mode='r') as file2:
        reader = csv.reader(file2, delimiter="\t")
        for r in reader:
            features2.append((r[2],r[3],r[1]))


    
    matches = 0
    correctFeatures = []
    errors = []
    points = []
    for feature1 in features1:
        for feature2 in features2:
            if(feature2[2] == feature1[2]):
                error = np.sqrt( float((float(feature1[0])-float(feature2[0]))**2) + float((float(feature1[1])-float(feature2[1]))**2) )
                if(error<10):
                    errors.append(error)
                    points.append([error,feature1[0],feature1[1],feature2[0],feature2[1]])
                else:
                    print(error)

    errorAverage = sum(errors)/len(errors)
    print(errorAverage)

    maxerror = max(errors)
    print(maxerror)


    n, bins, patches = plt.hist(errors, 20, density=True, facecolor='g')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Probability')
    plt.title('Error - Distance Between Points for Manual Labelling, Image 379')
    printAverage = "Average = " + str(round(errorAverage,2))
    plt.text(7, .8, printAverage,fontsize=12)
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()






        
    cv2.namedWindow(imageName,cv2.WINDOW_NORMAL)
    cv2.moveWindow(imageName, 500, 0)
    img = cv2.imread(imageName)
    
    for point in points:
        color = point[0]/maxerror * 255
        #print(color)
        cv2.circle(img,(int(point[1]),int(point[2])),4,(0,255,0),-1)
        cv2.circle(img,(int(point[3]),int(point[4])),4,(255-color,0,color),-1)
        #cv2.line(img, (  int(point[1]),int(point[2])), (int(point[3]),int(point[4]) ), (0,220,220), thickness=2, lineType=8, shift=0)
        
    cv2.imwrite("errors.jpg",img)
    cv2.imshow(imageName,img)
    #print("Correctly found", len(correctFeatures), "features out of", len(features2), ",",(len(errors) / len(features2)) *100,"% correct.")

compare("379 (BJ).txt","379-correctedTapendra-ALLFEATURES.txt", "379.JPG")
