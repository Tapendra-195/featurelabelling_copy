import cv2
import os
import numpy as np
import shutil

global COLOR
COLOR = 255

global RED
RED = COLOR,0,0

global BLUE
BLUE = 0,0,COLOR

global fileSaved
fileSaved = True



#Set up callbacks for drawing circles on click and drag, bound to left and middle mouse 
def draw_circle(event,x,y,flags,param):
    if(inputting==False):
        global mouseX,mouseY
        global ix, iy, drawing, rdrawing, mode
        global count

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
            cv2.circle(img, (x, y), small_size,BLUE,-1)
            print(ix, "x  ", iy,"y")
          
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
 

        if event == cv2.EVENT_MBUTTONDOWN:
            rdrawing = True
            ix,iy = x,y
            cv2.circle(img, (x, y), large_size,RED,-1)

            print(ix, "x", iy,"y")

        elif event == cv2.EVENT_MBUTTONUP:
            rdrawing = False
                     


        
##Set up for a single image
#function: annotate_img
# Inputs
#    img_path - name and location of the image, example "B.jpg"
#    size1 - size of the 1st brush circle, in px
#    size2 - size of the 2nd brush circle, in px
#    initials - First name and last name initials of the person doing the labelling. Recorded next to every PMT feature ID line in the text file.
# Outputs
#    {image_name}.png - A map of all the features in binary - features labelled in blue are recorded as RGB (1,1,1), and features in red are recorded as RGB (2,2,2).
#    {image_name}.txt - A text file. Each line contains a PMT feature ID and its pixel coordinates on the image. The text file has the same name as the image.

##User keypresses (make sure the image window is selected):
# click on image to grab pixel coordinates:
#	 Left-click to label with red 
#	 middle-click to label with blue
# s to close image and exit program
# r to write coordinates (you will be prompted for the PMT feature ID if a PMT has not been selected yet)
# f to select new PMT
# n to increment the featureID number
def annotate_img(img_path, initials, size1=1, size2=1) :
    print("image path is: ",img_path)
    global saveToText
    saveToText = True
    global count
    global inputting
    inputting = False
    global pmtSelected
    pmtSelected = False
    global nextPMT
    nextPMT = False
    count = 0
    global coords
    coords = [[],[]]

    #Extract the name of the image from the inputted path to the image.
    base = os.path.basename(img_path)
    filename = os.path.splitext(base)[0]
    saveFile("empty",os.path.join(filename+".txt"),"text")
    if(saveToText == True):
        file = open("%s.txt" %filename,"w")
    
    #Create window and put it in top left corner of screen
    
    cv2.namedWindow(filename,cv2.WINDOW_NORMAL) ####################### Added cv2.WINDOW_NORMAL flag to allow to resize window.
    ##cv2.moveWindow(filename, 40, 30) ##40 and 30 are x and y coordinates on the screen
    cv2.moveWindow(filename, 500, 0)
    cv2.resizeWindow(filename, 1200, 900)
    global drawing, rdrawing, large_size, small_size, img
    large_size=size2
    small_size=size1
    
    img = cv2.imread(img_path) ##read the image specified in the input
    cv2.setMouseCallback(filename,draw_circle) ##Link mouse position and button states to the draw_circle function.

    #Drawing callbacks
    drawing=False
    rdrawing=False

    print("\n Click on image to grab pixel coordinates.\n Press s to close image and exit program.\n Press r to write coordinates to file. \n    (you will be prompted for the PMT feature ID)\n Press left mouse button to label in red, and middle click to label in blue.")

    while(1):
        cv2.imshow(filename,img)     ##keep displaying the image even if the user exits.
        k = cv2.waitKey(20) & 0xFF
        if(k == ord('s')):          ##if the s key is pressed, exit the loop.
            print("Saving to: ",filename+".txt\n")
            for i in range(len(coords[0])):
                if(coords[0][i]!="N"):
                    writestartingVal = f'{i:02}'

                    print(filename+"\t"+writepmtID+"-"+writestartingVal+"\t"+str(coords[0][i])+"\t"+str(coords[1][i])+"\t"+initials+"\n")
                    if(saveToText==True):
                        file.write("%s\t%s-%s\t%s\t%s\t%s\n" %(filename,writepmtID,writestartingVal, str(coords[0][i]), str(coords[1][i]),initials))
                    coords[0][i]="N"
                    coords[1][i]="N"  
            break

###############################Add line to text output###############################
        if(k==ord('f')):
           nextPMT = True
           pmtSelected = False
           print("Saving to: ",filename+".txt\n")
           
           
           for i in range(len(coords[0])):
                if(coords[0][i]!="N"):
                    writestartingVal = f'{i:02}' 
                    print(filename+"\t"+writepmtID+"-"+writestartingVal+"\t"+str(coords[0][i])+"\t"+str(coords[1][i])+"\t"+initials+"\n")
                    if(saveToText==True):
                        file.write("%s\t%s-%s\t%s\t%s\t%s\n" %(filename,writepmtID,writestartingVal, str(coords[0][i]), str(coords[1][i]),initials))
                    coords[0][i]="N"
                    coords[1][i]="N"  
           print("Ended recording for PMT ",str(pmtID)+". Record first feature for another by selecting a point and pressing r.")
        elif(k==ord('n')):
           startingVal = startingVal+1
           print("Ready to record", startingVal)

        elif(k==ord('b')):
            if(startingVal==0):
                print("Error, can't decrement feature ID below 0.")
            else:
                startingVal = startingVal-1
                print("Ready to record",startingVal)

        if(k==ord('r')):
            if(pmtSelected == True):
                

                writestartingVal = f'{startingVal:02}'
                writepmtID = pmtID.zfill(5)   
                print("Recording ", writepmtID+"-"+writestartingVal, ix, "x", iy,"y\n")
                    
                
                
                while(startingVal>=len(coords[0])):
                    coords[0].append("N")
                    coords[1].append("N")

                coords[0][startingVal] = ix
                coords[1][startingVal] = iy


                startingVal = startingVal+1
                print("Ready to record",startingVal)
                
                

            else:
                inputting = True ##used to pause the draw_circle function from recording more coordinates
                pmtID = input("Input PMT number to add it to the list, or input 'd' to not register:\n-->")
                if(pmtID != 'd'):
                    startingFeature = input("Input starting feature number, or press enter to start from 0.\n-->")
                    if(not startingFeature):
                        startingVal = 0
                    else:
                        startingVal = int(startingFeature)
                      
                    writepmtID = pmtID.zfill(5)
                    writestartingVal = f'{startingVal:02}'   
                    print("Recording ", writepmtID+"-"+writestartingVal, ix, "x", iy,"y\n")
                    print("Record another selected point by pressing r.\nPress n to increment feature number.\nPress b to decrement feature number.\nPress f to finish recording features for this PMT.")
                    
                    while(startingVal>=len(coords[0])):
                        coords[0].append("N")
                        coords[1].append("N")

                    coords[0][startingVal] = ix
                    coords[1][startingVal] = iy

                    
                    startingVal = startingVal+1
                    print("Ready to record",startingVal)

                    pmtSelected = True
                    nextPMT = True
                print("\n")
                inputting = False
###############################Add line to text output###############################

    file.close   

###############################Image Output##########################################
    #Make mask same colour as drawing and output binarised image
    #Make additional visible mask with white background for viewing
    #Create a dictionary listing all drawing colors
    colorDictionary = {"color1": RED,"color2": BLUE}

    train_labels = img[:,:,:3]
    counter = 1
    #Create the empty mask
    mask = 0
    visibleMask = 255
    #Add drawings from every color in the dictionary to the mask 
    for color in colorDictionary:
        color_thresh = np.array(colorDictionary[color], dtype = "uint16")
        mask_color =  cv2.inRange(train_labels, color_thresh, color_thresh) 
        mask_color[mask_color < COLOR] = 0
        mask_color[mask_color!=0] = counter
        counter = counter+1
        mask = np.add(mask, mask_color)
        visibleMask = np.add(visibleMask,mask_color)


    maskName = os.path.join(filename+'.png')
    visibleMaskName = os.path.join(filename+'-visible.png')

    saveFile(mask,maskName,"mask")

    saveFile(visibleMask,visibleMaskName,"mask")

###############################Image Output##########################################

    cv2.destroyWindow(filename)

    return
            









#Set up for directory of images with file structure for image segmentation
#Function: annotate_dir
# Inputs
#    img_dir - image directory. Images are contained inside this folder.
#    initials - First name and last name initials of the person doing the labelling. Recorded next to every PMT feature ID line in the text file.
#    size1 - size of the 1st brush circle, in px.
#    size2 - size of the 2nd brush circle, in px.
# Outputs (for every image in {img_dir})
#    {image_name}.png - A map of all the features in binary - features labelled in blue are recorded as RGB (1,1,1), and features in red are recorded as RGB (2,2,2).
#    {image_name}.txt - A text file. Each line contains a PMT feature ID and its pixel coordinates on the image. The text file has the same name as the image.
# Folder Structure
#    {img_dir}
#        - Source images are stored here.
#    {img_dir}_texts
#        - Output text files are stored here.
#    {img_dir}_masks
#        - Output mask files are stored here. 
def annotate_dir(img_dir, initials, size1=1, size2=1) :
    #Create window and put it in top left corner off screen
    global fileSaved

####### Creating text and mask save directories #######
    text_save_path = os.path.join(img_dir+"_texts")
    if not os.path.exists(text_save_path):
        os.mkdir(text_save_path)

    mask_save_path = os.path.join(img_dir+"_masks")    
    if not os.path.exists(mask_save_path):
        os.mkdir(mask_save_path)

    print("Saving texts to: ",text_save_path)
    print("Saving masks to: ",mask_save_path)

    #Array of names in directory to iterate over
    f = []
    for (dirpath, dirnames, filenames) in os.walk(f'{img_dir}'):
        filenames.sort()
        f.extend(filenames)
        break
    
    print("Found: ",f,"\n\n")
    
    for i in f :
        skip = False

        base = os.path.basename(i)
        imageName = os.path.splitext(base)[0]
        imageLocation = os.path.join(img_dir,i)

        annotate_img(imageLocation, initials, size1, size2)


        textName = os.path.join(imageName+".txt")
        maskName = os.path.join(imageName+".png")
        visibleMaskName = os.path.join(imageName+"-visible.png")

        fileSave=True
        saveFile("empty",os.path.join(text_save_path,textName),"text")

        if(fileSaved == True):
            print("Moving",textName, "to", text_save_path)        
            shutil.move(textName,text_save_path)


        print("\n")


        fileSaved=True
        saveFile("empty",os.path.join(mask_save_path,maskName),"mask-test")
        if(fileSaved == True):
            print("Moving",maskName, "to", mask_save_path)        
            shutil.move(maskName,mask_save_path)
  

        fileSaved=True
        saveFile("empty",os.path.join(mask_save_path,visibleMaskName),"mask-test")
        if(fileSaved == True):
            print("Moving",visibleMaskName, "to", mask_save_path)        
            shutil.move(visibleMaskName,mask_save_path)
        
         

def saveFile(file,name,text_or_mask):
    global fileSaved
    #mask_save_path = os.path.join(location)  ##temp  

    location_name = name

    
    fileSaved=True
    if(os.path.exists(location_name)):
        print("File", location_name, "already exists. Would you like to overwrite it?")
        overwriteFile = input("(y/n)\n-->")
        if(overwriteFile.lower() == "y"):
            fileSaved=True
            os.remove(location_name)
        else:
            fileSaved=False
            if(text_or_mask=="text"):
                saveToText = False
                print("Will proceed without recording text file.\n")
    if(fileSaved == True):
        if(text_or_mask=="mask"):
            cv2.imwrite(location_name, file)
