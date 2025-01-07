import numpy as np
from sklearn import datasets
from tabulate import tabulate
from LRA import LRA
import matplotlib.pyplot as plt
from SVMCommon import *

digits = datasets.load_digits()

## Set the p value for the percentage of training data wanted
p = .6

# define the percentiles desired for the LRA.
percentiles = np.percentile(np.arange(1, len(digits.data[0]) + 1), np.arange(5, 101, 5))
percentiles = np.round(percentiles).astype(int)

# Add lowest ranks in
percentiles = np.array([0, 1, 2, 3, 4] + list(percentiles))
#percentiles = np.array([0, 1, 2, 3, 4])

# create empty list for the training accuracies
accuracy_train = []

# create empty list for the test accuracies
accuracy_test = []

# loop through the percentile values
for rank in percentiles:
    
    # Reshape training images from 64x1 images to the original 8x8 images
    X_data_reshaped = digits.data[:,].reshape(-1, 8, 8)

    # create empty array for the LRA images within the loop for each percentile
    X_data_LRA = []

    # Perform LRA for each image at the requested percentile rank in the loop progression
    for j in range(len(X_data_reshaped)):
        
        # Compute the LRA on the reshaped training images at the given percentile in percentiles
        X_data_LRA.extend(LRA(X_data_reshaped[j,:,:], rank))
   
    # Reshape the training data to the original 1078 images of size 64x1
    X_data_LRA = np.array(X_data_LRA, dtype=np.float64).reshape(len(digits.data),64)

    accuraciesForCurrentPercentileTraining = []
    accuraciesForCurrentPercentileTesting = []
    
    #Run three realizations at each rank
    for j in range(3):
        train_data, train_target, test_data, test_target = partition(X_data_LRA, digits.target, .6)

        ## Solve for the acurracy of the train and test for this rank approx
        accuracy_trained, accuracy_tested = svc_analysis(train_data, 
                                                     train_target, test_data, 
                                                     test_target)
        
        ## Append the accuracies for both in order for the avg to be taken
        accuraciesForCurrentPercentileTraining.append(accuracy_trained)      
        accuraciesForCurrentPercentileTesting.append(accuracy_tested)

    # append the train and test accuracies for each rank approximation
    accuracy_train.append(accuraciesForCurrentPercentileTraining)
    accuracy_test.append(accuraciesForCurrentPercentileTesting)

# Find the avg accuracy for each rank approx    
accuracy_train = np.array(accuracy_train)
avg_accuracy_train = np.array([np.average(inner) for inner in accuracy_train])

accuracy_test = np.array(accuracy_test)
avg_accuracy_test = np.array([np.average(inner) for inner in accuracy_test])

percentiles += 1

#Show Data Table
data = {
    "Rank (8x8, 1 channel images)": percentiles,
    "Training Accuracy (Average)": np.round(avg_accuracy_train, 2),
    "Testing Accuracy (Average)": np.round(avg_accuracy_test, 2)
}

print(tabulate(data, headers="keys", tablefmt="grid"))
print("PERCENTILES: ", percentiles)
print("Training Averages", np.round(avg_accuracy_train, 2))
print("Testing Averages", np.round(avg_accuracy_test, 2))

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

plt.xlabel("Image Rank (8x8 \"Simple\" Images - 1 Channel)")
plt.ylabel("Accuracy (Average of Three Realizations)")
plt.title("Accuracy of SVM Classifier at Different Rank Approximations")
plt.legend()
plt.show()


