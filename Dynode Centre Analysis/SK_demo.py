from IPython.core.display import display, HTML
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import pg_fitter_tools as fit
import sk_geo_tools as sk
focal_length = [2.760529621789217e+03, 2.767014510543478e+03]
principle_point = [1.914303537872458e+03, 1.596386868474348e+03]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]

all_pmt_locations = fit.read_3d_feature_locations("parameters/SK_all_PMT_locations.txt")
#offset = np.array([0, 250])
offset = np.array([0, 0])
image_feature_locations = {}
image_feature_locations.update(fit.read_image_feature_locations("BarrelSurveyFar_TopInjector_PD3/BarrelSurveyFar_TopInjector_median_texts/045.txt", offset=offset))
image_feature_locations.update(fit.read_image_feature_locations("BarrelSurveyFar_TopInjector_PD3/BarrelSurveyFar_TopInjector_median_texts/046.txt", offset=offset))
image_feature_locations.update(fit.read_image_feature_locations("BarrelSurveyFar_TopInjector_PD3/BarrelSurveyFar_TopInjector_median_texts/047.txt", offset=offset))
image_feature_locations.update(fit.read_image_feature_locations("BarrelSurveyFar_TopInjector_PD3/BarrelSurveyFar_TopInjector_median_texts/048.txt", offset=offset))
# swap out -25 and -00 for features in images where both exist
for im in image_feature_locations.values():
    for feat, loc in im.items():
        if feat[-2:] == "00" and feat[:-2]+"25" in im:
            tmp = loc
            im[feat] = im[feat[:-2]+"25"]
            im[feat[:-2]+"25"] = tmp


# choose features that appear in 2+ 
feature_counts = Counter([f for i in image_feature_locations.values() for f in i.keys()])
common_features = [f for f in feature_counts if feature_counts[f] > 1]
pmt_locations = {k: p for k, p in all_pmt_locations.items() if k in common_features}

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
print(nimages, nfeatures)

#print("----------", common_feature_locations)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
bolt_array = np.stack(list(bolt_locations.values()))
pmt_array = np.stack(list(pmt_locations.values()))
ax.scatter(bolt_array[:,0], bolt_array[:,1], bolt_array[:,2], marker='.', label="bolt (seed position)")
ax.scatter(pmt_array[:,0], pmt_array[:,1], pmt_array[:,2], marker='^', label="pmt (seed position)")
for i, f in enumerate(pmt_locations.keys()):
    ax.text(pmt_array[i,0], pmt_array[i,1], pmt_array[i,2], f[:5], size=8, zorder=4, color='k') 
plt.legend(loc=0)
fig.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(9,9))
bolt_array = np.stack(list(bolt_locations.values()))
pmt_array = np.stack(list(pmt_locations.values()))
ax.scatter(bolt_array[:,0], bolt_array[:,1], marker='.', label="bolt (seed position)")
ax.scatter(pmt_array[:,0], pmt_array[:,1], marker='^', label="pmt (seed position)")
ax.set_ylim((500,1050))
ax.set_xlim((1200,1800))
plt.legend(loc=0)
fig.tight_layout()


fitter_pmts = fit.PhotogrammetryFitter(common_image_pmt_locations, pmt_locations,
                                       focal_length, principle_point, radial_distortion)
#fitter_bolts = fit.PhotogrammetryFitter(common_image_bolt_locations, common_feature_locations,
                                       #focal_length, principle_point, radial_distortion)
fitter_all = fit.PhotogrammetryFitter(common_image_feature_locations, common_feature_locations,
                                       focal_length, principle_point, radial_distortion)


camera_rotations, camera_translations, reprojected_points = fitter_all.estimate_camera_poses(flags=cv2.SOLVEPNP_EPNP)
#print(camera_rotations,"\n",camera_translations)






test_image = 1
fig, ax = plt.subplots(figsize=(12,9))
coords = np.stack(list(common_image_feature_locations[fitter_pmts.index_image[test_image]].values()))
repro_coords = np.stack(list(reprojected_points[fitter_pmts.index_image[test_image]].values()))
ax.scatter(coords[:,0], 3000-coords[:,1], marker='.', label='detected')
ax.scatter(repro_coords[:,0], 3000-repro_coords[:,1], marker='.', label='reprojected')
for t, f in common_image_feature_locations[fitter_pmts.index_image[test_image]].items():
    ax.text(f[0], 3000-f[1], t, size=6, zorder=4, color='k')
for t, f in reprojected_points[fitter_pmts.index_image[test_image]].items():
    ax.text(f[0], 3000-f[1], t, size=6, zorder=4, color='gray')
ax.set_title("Image {}".format(fitter_pmts.index_image[test_image]))
ax.set_ylim(0, 3000)
ax.set_xlim(0, 4000)
plt.legend(loc=0)
fig.tight_layout()


camera_orientations, camera_positions = fit.camera_world_poses(camera_rotations, camera_translations)
fig = plt.figure(figsize=(12,9))
pmt_array = np.stack(list(pmt_locations.values()))
feat_array = np.stack(list(common_feature_locations.values()))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feat_array[:,0], feat_array[:,1], feat_array[:,2], marker='.', label="seed positions", zorder=2)
for i, f in enumerate(pmt_locations.keys()):
    ax.text(pmt_array[i,0], pmt_array[i,1], pmt_array[i,2], f[:5], size=4, zorder=4, color='k') 
ax.scatter(camera_positions[:,0], camera_positions[:,1], camera_positions[:,2], marker='*', label="camera estimate", zorder=1)
plt.legend(loc=0)
fig.tight_layout()









###############################################################

print(camera_rotations, camera_translations)

print("started")
camera_rotations, camera_translations, reco_locations = fitter_all.bundle_adjustment(camera_rotations, camera_translations)
print("Done.")
#print("1=",camera_rotations, "2=",camera_translations,"3=", reco_locations)



#print("common_feature_locations=",common_feature_locations)
print("reco_locations=", reco_locations)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
bolt_array = np.stack(list(bolt_locations.values()))
pmt_array = np.stack(list(reco_locations.values()))
ax.scatter(bolt_array[:,0], bolt_array[:,1], bolt_array[:,2], marker='.', label="bolt (reprojected position)")
ax.scatter(pmt_array[:,0], pmt_array[:,1], pmt_array[:,2], marker='*', label="pmt (seed position)")
#for i, f in enumerate(reco_locations.keys()):
#    ax.text(pmt_array[i,0], pmt_array[i,1], pmt_array[i,2], f[:5], size=8, zorder=4, color='k') 
plt.legend(loc=0)
fig.tight_layout()
plt.show()
###########################################################################

errors, reco_transformed, scale, R, translation, location_mean = fit.kabsch_errors(common_feature_locations, reco_locations)
#print("mean reconstruction error:", linalg.norm(errors, axis=1).mean())
#print("max reconstruction error:", linalg.norm(errors, axis=1).max())

print("reco_transformed = ",reco_transformed)



















camera_orientations, camera_positions = fit.camera_world_poses(camera_rotations, camera_translations)
camera_orientations = np.matmul(R, camera_orientations)
camera_positions = camera_positions - translation
camera_positions = scale*R.dot(camera_positions.transpose()).transpose() + location_mean



outputdir = "results/"
fitter_all.save_result(outputdir+"SK_demo2_features.txt", outputdir+"SK_demo2_cameras.txt")


true_array = np.stack(list(common_feature_locations.values()))
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
rot_theta = (90-27.261)*np.pi/180
ax.scatter(reco_transformed[:,0]*np.cos(rot_theta)-reco_transformed[:,1]*np.sin(rot_theta), reco_transformed[:,0]*np.sin(rot_theta)+reco_transformed[:,1]*np.cos(rot_theta), marker='.', label="reconstructed")
ax.scatter(true_array[:,0]*np.cos(rot_theta)-true_array[:,1]*np.sin(rot_theta), true_array[:,0]*np.sin(rot_theta)+true_array[:,1]*np.cos(rot_theta), marker='.', label="expected", s=100)
#ax.set_ylim((500,1050))
#ax.set_xlim((1320,1600))
circle = plt.Circle((0, 0), 1690-15, color='black', fill=False)
ax.add_artist(circle)
#for i, f in enumerate(pmt_locations.keys()):
#    ax.text(reco_transformed[i,0], reco_transformed[i,1], f, size=4, zorder=1, color='k') 
#ax.scatter(camera_positions[:,0], camera_positions[:,1], camera_positions[:,2], marker='*', label="camera")
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
plt.legend(loc=0)
fig.tight_layout()

plt.show()












# Sort the images/features by highest reprojection errors, for manual checking
#print(np.array(np.unravel_index(np.argsort(linalg.norm(fitter_all.reprojected_locations()-fitter_all.image_feature_locations, axis=2).ravel()), (nimages, nfeatures))))
#print(linalg.norm(fitter_all.reprojected_locations()-fitter_all.image_feature_locations, axis=2)[1, 90])
#print(fitter_all.index_feature[323])





fig, ax = plt.subplots(figsize=(8,6))
ax.hist(linalg.norm(errors, axis=1), bins='auto')
ax.set_title("Reconstructed position distance from expected ({} images, {} features), mean = {:.2f} cm".format(
    nimages, nfeatures, linalg.norm(errors, axis=1).mean()))
fig.tight_layout()



bolt_dict = {b: reco_transformed[fitter_all.feature_index[b]] for b in common_feature_locations.keys()}
#print("bolt_dict = ", bolt_dict)
bolt_dists = sk.get_bolt_distances(bolt_dict)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(bolt_dists, bins='auto')
ax.set_title("Reconstructed distance between adjacent bolts (cm)")
ax.axvline(linewidth=2, color='r', x=sk.bolt_distance)
fig.tight_layout()



bolt_radii = sk.get_bolt_ring_radii(bolt_dict)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(bolt_radii, bins='auto')
ax.set_title("Reconstructed distance between bolts and centre of bolt ring (cm)")
ax.axvline(linewidth=2, color='r', x=sk.bolt_ring_radius)
fig.tight_layout()


planes = sk.get_bolt_ring_planes(bolt_dict)
#print("planes=", planes)
# flip planes if they're facing wrong direction (all normals should point towards tank centre)
plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='3d')
NN = 100
for pmt in sk.get_unique_pmt_ids(bolt_dict):
    pmt_bolt_coords = np.array([l for b, l in bolt_dict.items() if b[:5] == pmt])
    icolor = 'b' if int(pmt)>607 else 'orange' 
    ax.scatter(pmt_bolt_coords[:,0], pmt_bolt_coords[:,1], pmt_bolt_coords[:,2], color=icolor)
    p, n = planes[pmt]
    X,Y,Z = [[p[i]-n[i]*NN, p[i]+n[i]*NN] for i in range(3)]
    ax.plot(X,Y,Z,color=icolor)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(1200,1800)
ax.set_ylim(500, 1100)
ax.set_zlim(1000, 1500)
ax.view_init(50, 240)




####### calculate the mean normal angles
###### mean normal direction of left 12 and right 12
normals1 = np.array([n for pmt, (p, n) in planes.items() if int(pmt) <= 607])
normals2 = np.array([n for pmt, (p, n) in planes.items() if int(pmt) > 607])
N1 = np.mean(normals1, axis=0)
N2 = np.mean(normals2, axis=0)
print ('norm (avg of 12 angles)', N1, N2)

##### normal for fitting all 12 at once
N1_all = sk.get_supermodule_plane(bolt_dict, 451, 606)[1]
N2_all = sk.get_supermodule_plane(bolt_dict, 655, 810)[1]
print ('norm (simultanous fit 12 PMTs)', N1_all, N2_all)

###### difference
print ('diff (degrees)', np.degrees(np.arccos(np.dot(N1, N1_all))), np.degrees(np.arccos(np.dot(N1, N2_all))))



cos1=np.dot(normals1, N1_all)
cos2=np.dot(normals2, N2_all)
theta1=np.degrees(np.arccos(cos1))
theta2=np.degrees(np.arccos(cos2))
diff_2walls = np.degrees(np.arccos(np.dot(N1_all, N2_all)))

# print (cos1)
# print (cos2)
print ('Angle between 2 walls (using normals fitted to 12 PMTs simultaneously):',diff_2walls)
print (theta1)
print (theta2)





