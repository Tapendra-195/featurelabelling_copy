from IPython.core.display import display, HTML
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import pg_fitter_tools as fit
import sk_geo_tools as sk
import csv



filename = "045.jpg"
textFilename = "045.txt"
plotReprojectedPMTs = True
plotReprojectedFeatures = False
plotManualFeatures = True
offset = 0

####################Specify Internal Camera Parameters################################
focal_length = [2.760529621789217e+03, 2.767014510543478e+03]
principle_point = [1.914303537872458e+03, 1.596386868474348e+03]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]
######################################################################################

def plot_pmts(coordinates, imageIdentifier, off_set=0, color=(0,255,0)):
    counter = 0
    for i in coordinates:

        if abs(i[0])<=4000 and abs(i[1]-off_set)<=2750:
            plotx = int(i[0])
            ploty = int(i[1]-off_set)
            cv2.circle(imageIdentifier,(plotx,ploty),7,color,-1)
            counter=counter+1




all_pmt_locations = fit.read_3d_feature_locations("parameters/SK_all_PMT_locations.txt")

#print("all_pmt_locations=",all_pmt_locations)
#get the location of a PMT:
for x in all_pmt_locations:
    if x=="00813-00":
        print(x,all_pmt_locations[x])


        

image_feature_locations = {}
image_feature_locations.update(fit.read_image_feature_locations(textFilename, offset=np.array([0, offset])))
#image_feature_locations.update(fit.read_image_feature_locations("2-PMT-test.txt", offset=np.array([0, offset])))


# swap out -25 and -00 for features in images where both exist
## For some PMTs only the 00 feature exists and not the 25 feature. The 00 feature, the reflection in the centre of the PMT, was initially thought to be the centre of the dynode, but the 25 feature that was added later is a better location for the dynode centre. So initially the notebook just used the 00 feature of each PMT, then when the 25 feature was added for some PMTs the code was modified to use that instead, where it is available.
for im in image_feature_locations.values():
    for feat, loc in im.items():
        
        if feat[-2:] == "00" and feat[:-2]+"25" in im:
            tmp = loc
            im[feat] = im[feat[:-2]+"25"]
            im[feat[:-2]+"25"] = tmp



# choose features that appear 2+ times
feature_counts = Counter([f for i in image_feature_locations.values() for f in i.keys()])
common_features = [f for f in feature_counts if feature_counts[f] >= 1] ###  Set this to >= 1 to try.

## Real pmt_locations of all PMTs in the labelled images.
#pmt_locations = {k: p for k, p in all_pmt_locations.items() if k in common_features}
pmt_locations = all_pmt_locations 

# generate bolt locations from PMT locations
bolt_locations = sk.get_bolt_locations_barrel(pmt_locations)

common_feature_locations = {**pmt_locations, **bolt_locations}
common_image_pmt_locations = {
    k: {j: f for j, f in i.items() if j in common_features and j in pmt_locations}
    for k, i in image_feature_locations.items()}
common_image_feature_locations = {
    k: {j: f for j, f in i.items() if j in common_features and j in common_feature_locations}
    for k, i in image_feature_locations.items()}
common_image_bolt_locations = {
    k: {j: f for j, f in i.items() if j in common_features and j in bolt_locations}
    for k, i in image_feature_locations.items()}
nimages = len(common_image_feature_locations)
nfeatures = len(common_feature_locations)






fitter_pmts = fit.PhotogrammetryFitter(common_image_pmt_locations, pmt_locations,focal_length, principle_point, radial_distortion)
#fitter_bolts = fit.PhotogrammetryFitter(common_image_bolt_locations, bolt_locations,focal_length, principle_point, radial_distortion)
fitter_all = fit.PhotogrammetryFitter(common_image_feature_locations, common_feature_locations,
                                       focal_length, principle_point, radial_distortion)
print("fitter_all=",fitter_all)
camera_rotations, camera_translations, reprojected_points = fitter_all.estimate_camera_poses(flags=cv2.SOLVEPNP_EPNP)

#NEw
camera_orientations, camera_positions = fit.camera_world_poses(camera_rotations, camera_translations)

fig = plt.figure(figsize=(12,9))
#pmt_array = np.stack(list(pmt_locations.values()))
#feat_array = np.stack(list(common_feature_locations.values()))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(feat_array[:,0], feat_array[:,1], feat_array[:,2], marker='.', label="seed positions", zorder=2)
#for i, f in enumerate(pmt_locations.keys()):
#    ax.text(pmt_array[i,0], pmt_array[i,1], pmt_array[i,2], f[:5], size=4, zorder=4, color='k') 
#ax.scatter(camera_positions[0], camera_positions[1], camera_positions[2], marker='*', label="camera estimate", zorder=1)
#plt.legend(loc=0)
#fig.tight_layout()


#print("\n\n\ncamera_orientations=\n",camera_orientations)
#print("camera_positions=\n",camera_positions)


test_image = 0
coords = np.stack(list(common_image_feature_locations[fitter_pmts.index_image[test_image]].values()))
repro_coords = np.stack(list(reprojected_points[fitter_pmts.index_image[test_image]].values()))
ids_array = np.stack(list(reprojected_points[fitter_all.index_image[test_image]]))
print(coords)

cv2.namedWindow(filename,cv2.WINDOW_NORMAL) ####################### Added cv2.WINDOW_NORMAL flag to allow to resize window.
cv2.moveWindow(filename, 500, 0)
img = cv2.imread(filename)

if(plotReprojectedFeatures==True):
    plot_pmts(repro_coords,img,offset,(0,255,0))
    
if(plotReprojectedPMTs==True):
    for item in reprojected_points[fitter_pmts.index_image[test_image]]:
        if(item[-2:] == "00"):
        #print(item, reprojected_points[fitter_pmts.index_image[test_image]][item])
            PMT_ID = item[:-3]
            PMT_ID_Coordinates = reprojected_points[fitter_pmts.index_image[test_image]][item]
            plotx = int(PMT_ID_Coordinates[0])
            ploty = int(PMT_ID_Coordinates[1])
        if abs(plotx)<=4000 and abs(ploty-offset)<=2750:
                cv2.circle(img,(plotx,ploty-offset),10,(0,255,255),2)
                PMT_ID = PMT_ID.lstrip("0")
                cv2.putText(img, f'{PMT_ID}', (plotx-50, ploty+50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (128,128,255), 3)

if(plotManualFeatures==True):
    plot_pmts(coords, img, offset, (0,255,0))
    



#print("------------------------------")#################################Next Image######################################################
#test_image = 0
#coords1 = np.stack(list(common_image_feature_locations[fitter_pmts.index_image[test_image]].values()))
#repro_coords1 = np.stack(list(reprojected_points[fitter_pmts.index_image[test_image]].values()))
#print(coords1)
#filename2 = "test-image2"
#cv2.namedWindow(filename2,cv2.WINDOW_NORMAL)
#cv2.resizeWindow(filename2, 1200, 900)
#img2 = cv2.imread("046.jpg")


#plot_pmts(coords1, img2, offset, (255,0,0))
#plot_pmts(repro_coords1,img2,offset,(0,255,0))

        


            
cv2.imwrite("OUTPUT.jpg",img)
#cv2.imwrite("OUTPUT2.jpg",img2)

        


cv2.imshow(filename,img)
#cv2.imshow(filename2,img2)


