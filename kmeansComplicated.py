from collections import defaultdict
from matplotlib import pyplot as plt
from tensorflow.keras import datasets 
from tabulate import tabulate
import numpy as np
from LRA import LRA
from kmeansCommon import *

#Load the CIFAR-10 Dataset, 32x32 3 channel images showing various images
(x1, y1), (x2, y2) = datasets.cifar10.load_data()

#Shuffle and use only use the first 2,000 images to save compute time
XDataComplex = np.concatenate((x1, x2))
targetComplex = np.concatenate((y1, y2))
XDataComplex, targetComplex = shuffle(XDataComplex, targetComplex)

XDataComplex = XDataComplex[:2000]
targetComplex = targetComplex[:2000]

targetComplex = targetComplex.reshape(len(targetComplex))
XDataComplex = np.array([page.reshape(3072) for page in XDataComplex])

# Define the percentiles desired for the LRA
percentiles = np.percentile(np.arange(1, len(XDataComplex[0]) + 1), np.arange(5, 101, 5))
# Round these percentiles, np.percentile returns float
percentiles = np.round(percentiles).astype(int)
#Add in extremely low ranks to show plateau of learning
percentiles = np.array([0, 1, 2, 3, 4]+ list(percentiles))
#percentiles = np.array([0, 1, 2, 3, 4])

# Create empty list for the training accuracies
accuracy_train_complex = []

# Create empty list for the test accuracies
accuracy_test_complex = []

# Loop through the percentile values
for rank in percentiles:
    # Reshape training images from 3072x1 images to the original 32x32x3 images
    X_data_reshaped = XDataComplex.reshape(-1, 32, 32, 3)

    # Create empty array for the LRA images within the loop for each percentile
    X_data_LRA = []

    # Perform LRA for each image at the requested percentile rank in the loop progression
    for j in range(len(X_data_reshaped)):
        
        # Compute the LRA on the reshaped training images at the given percentile in percentiles
        X_data_LRA.extend(LRA(X_data_reshaped[j,:,:], rank))
   
    # Reshape the training data to the original 36000 images of size 3072x1
    X_data_LRA = np.array(X_data_LRA, dtype=np.float64).reshape(len(XDataComplex),3072)

    accuraciesForCurrentPercentileTraining = []
    accuraciesForCurrentPercentileTesting = []
    
    #Run three realizations at each rank
    for j in range (3):
        train_data, train_target, test_data, test_target = partition(X_data_LRA, targetComplex, .6)
        #Run KMeans on the data using random initalization
        IndexSet, centroids = KMeansClusters(randomInit, train_data)

        """
        Check the accuracy of the training data. Assume the greatest proportion of the target data in each cluster is the
        correct classification for that cluster.  

        Example: In Cluster 0 we have 100 data points, 60 of these points resolve to a "3" in terms of their target. We
                 assume that "3" is the correct classification for Cluster 0. Accuracy of this cluster would then be 60%. 
        """

        #Map the cluster value to the "correct" classification
        clusterIMapping = {}
        totalCorrect = 0
        for i in range(k):
            #Find the data assigned to cluster i
            clusterI = np.where(IndexSet == i)[0]
            maps = defaultdict(int)
        
            for idx, val in enumerate(clusterI):
                maps[train_target[val]] += 1

            #Assign the classification to this cluster by checking which key has the most points assigned 
            clusterIValue = [key for key, value in maps.items() if value == max(maps.values())]
        
            #Check the accuracy of the current cluster if there are any data within 
            if (len(clusterI)):
                correctInCluster = 0
                for val in clusterI:
                    
                    #correctInCluster += 1 if clusterIValue == train_target[val] else 0
                    correctInCluster += 1 if all(clusterIValue == train_target[val]) else 0 # modified this
                totalCorrect += correctInCluster
    
                clusterIMapping[i] = clusterIValue
        
        
        # Accuracy of training data for the current realization
        accuracy_trained = totalCorrect/len(train_data)
        # Append accuracy to be averaged of three realizations later
        accuraciesForCurrentPercentileTraining.append(accuracy_trained)
        
        #Accuracy of Testing Data
        closestCluster = findClosestCluster(test_data, centroids)
        numCorrect = 0
        for i in range(len(test_data)):
        
            if clusterIMapping.get(int(closestCluster[i]), [-1])[0] == test_target[i]:
                numCorrect += 1

        # Accuracy of testing data for the current realization      
        accuracy_tested = numCorrect/len(test_data)
        # Append accuracy to be averaged of three realizations later
        accuraciesForCurrentPercentileTesting.append(accuracy_tested)
    
    # Append accuracies of three realizations 
    accuracy_train_complex.append(accuraciesForCurrentPercentileTraining)
    accuracy_test_complex.append(accuraciesForCurrentPercentileTesting)

#Convert accuracy lists to np arrays and average the three realizations for each rank
accuracy_train = np.array(accuracy_train_complex)
avg_accuracy_train = np.array([np.average(inner) for inner in accuracy_train])

accuracy_test = np.array(accuracy_test_complex)
avg_accuracy_test = np.array([np.average(inner) for inner in accuracy_test])

#Add one to show the true rank of the image
percentiles += 1

# Show Data Table
data = {
    "Rank (32x32, 3 channel images)": percentiles,
    "Training Accuracy (Average)": np.round(avg_accuracy_train, 2),
    "Testing Accuracy (Average)": np.round(avg_accuracy_test, 2)
}

print(tabulate(data, headers="keys", tablefmt="grid"))

#Plotting Requirements
plt.rc('font', size=18)         
plt.rc('axes', titlesize=18)  
plt.rc('axes', labelsize=16)  
plt.rc('xtick', labelsize=18)  
plt.rc('ytick', labelsize=18)   
plt.rc('legend', fontsize=16)    
plt.rc('figure', titlesize=20)  
plt.rc('lines',markersize=10) 
plt.rc('lines',linewidth=4)  

plt.plot(percentiles, avg_accuracy_train, label="Training Accuracy")
plt.plot(percentiles, avg_accuracy_test, label="Testing Accuracy")

#plt.xticks(percentiles)

plt.xlabel("Image Rank (32x32 \"Complicated\" Images - 3 Channels)")
plt.ylabel("Accuracy (Average of Three Realizations)")
plt.title("Accuracy of K Means Clustering Algorithm at Different Rank Approximations")
plt.legend()
plt.show()
