import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv


filename = "plotFile.txt"
reconstructed_bolts_array = []
reconstructed_average_bolt_array = []
reconstructed_dynode_centre_array = []
projected_reconstructed_dynode_centre_array = []





recordingMode = "."
#Reading
with open(filename, mode='r') as file:
    reader = csv.reader(file, delimiter="\t")
    
    skip = False
    for r in reader:
        #print(r[0])
        if r[0] == "reconstructed_bolts_array:" or r[0] == "reconstructed_average_bolt_array:" or r[0]=="reconstructed_dynode_centre_array:" or r[0]=="projected_reconstructed_dynode_centre_array:":
            recordingMode = r[0]
            
            continue

        if recordingMode ==  "reconstructed_bolts_array:":
            reconstructed_bolts_array.append((r[0],r[1],r[2]))
            
        if recordingMode == "reconstructed_average_bolt_array:":
            reconstructed_average_bolt_array.append((r[0],r[1],r[2]))
        if recordingMode == "reconstructed_dynode_centre_array:":
            reconstructed_dynode_centre_array.append((r[0],r[1],r[2]))
        if recordingMode == "projected_reconstructed_dynode_centre_array:":
            projected_reconstructed_dynode_centre_array.append((r[0],r[1],r[2],r[3]))

        skip = False



def plotter(points, color, marker, label):
    plotLegend = False
    for i in points:
        #print(i[0],i[1],i[2])
        if(plotLegend == False):
            ax.plot([float(i[0])],[float(i[1])],[float(i[2])],color = color, marker = marker, label=label)
            plotLegend = True
        else:
            ax.plot([float(i[0])],[float(i[1])],[float(i[2])],color = color, marker = marker)
    
def plotter_labels(points):
    for i in points:
        #ax.text(float(i[0]),float(i[1]),float(i[2]),"thing")
        ax.text(float(i[0]),float(i[1]),float(i[2])+10, i[3][-3:],size=12, zorder=4, color='#5158fc')
#Plotting
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

plotter(reconstructed_bolts_array, '#ff9696', '.', "reconstructed bolt points")
plotter(reconstructed_average_bolt_array, "orange", '^', "average reconstructed bolt location")
plotter(reconstructed_dynode_centre_array, "red", "*", "reconstructed dynode centre")
plotter(projected_reconstructed_dynode_centre_array, "red", ".", "projected reconstructed dynode point")
plotter_labels(projected_reconstructed_dynode_centre_array)





#title = str(chosenPMT)+", distance = "+str(round(pmt_shift_distance,3))+" px"
#print(str(chosenPMT)+", distance = "+str(round(pmt_shift_distance,3))+" px")



plt.legend(loc=0)

plt.show()






