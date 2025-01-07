from tensorflow.keras import datasets 
from tabulate import tabulate
import numpy as np
from LRA import LRA
import matplotlib.pyplot as plt
from SVMCommon import *

## Load in cifar dataset
(x1, y1), (x2, y2) = datasets.cifar10.load_data()

## Shuffle and use only use the first 2,000 images to save compute time
XDataComplex = np.concatenate((x1, x2))
targetComplex = np.concatenate((y1, y2))
XDataComplex, targetComplex = shuffle(XDataComplex, targetComplex)

XDataComplex = XDataComplex[:2000]
targetComplex = targetComplex[:2000]

## Reshape the data
targetComplex = targetComplex.reshape(len(targetComplex))
XDataComplex = np.array([page.reshape(3072) for page in XDataComplex])

## Set the p value for the percentage of training data wanted
p = .6

# define the percentiles desired for the LRA
percentiles = np.percentile(np.arange(1, len(XDataComplex[0]) + 1), np.arange(5, 101, 5))
percentiles = np.round(percentiles).astype(int)


# Add lowest ranks in. 
percentiles = np.array([0, 1, 2, 3, 4] + list(percentiles))
# percentiles = np.array([0, 1, 2, 3, 4])

# create empty list for the training accuracies
accuracy_train_complex = []

# create empty list for the test accuracies
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
    
    ## Run through the SVC 3 times in order to shuffle the split and average
    for j in range(3):
        
        ## Select the training and testing data plus labels of each
        train_data, train_target, test_data, test_target = partition(X_data_LRA, targetComplex, p)
        
        ## Solve for the acurracy of the train and test for this rank approx
        accuracy_trained, accuracy_tested = svc_analysis(train_data, 
                                                     train_target, test_data, 
                                                     test_target)
        
        ## Append the accuracies for both in order for the avg to be taken
        accuraciesForCurrentPercentileTraining.append(accuracy_trained)      
        accuraciesForCurrentPercentileTesting.append(accuracy_tested)

    # append the train and test accuracies for each rank approximation
    accuracy_train_complex.append(accuraciesForCurrentPercentileTraining)
    accuracy_test_complex.append(accuraciesForCurrentPercentileTesting)

# Find the avg accuracy for each rank approx    
accuracy_train = np.array(accuracy_train_complex)
avg_accuracy_train = np.array([np.average(inner) for inner in accuracy_train_complex])

accuracy_test = np.array(accuracy_test_complex)
avg_accuracy_test = np.array([np.average(inner) for inner in accuracy_test_complex])

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

# plt.xticks(percentiles)

plt.xlabel("Image Rank (32x32 \"Complicated\" Images - 3 Channels)")
plt.ylabel("Accuracy (Average of Three Realizations)")
plt.title("Accuracy of SVM Classifier at Different Rank Approximations")
plt.legend()
plt.show()

