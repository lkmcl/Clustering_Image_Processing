from sklearn.utils import shuffle
import numpy as np

# Set the number of clusters (k)
k = 10

def partition(data, target, p):
    """
    :param data: training data
    :param target: target data
    :param p: a percentage of the data to be used for training
    """
    #https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
    data, target = shuffle(data, target)

    idx = round(p * len(data))
        
    train_data = data[:idx]
    train_target = target[:idx]

    test_data = data[idx:]
    test_target = target[idx:]

    return train_data, train_target, test_data, test_target

def randomInit(vectorLength):
    """
    :param vectorLength: length of the representative vectors to be created
    """
    c = np.vstack([np.random.uniform(0,1,vectorLength) for _ in range(0,k)])
    return c

def findClosestCluster(data, centroids):
    """
    Code provided by Hangjie Ji
    :param data: data vectors to check what is the closest centroid
    :param centroids: centroids calculated using KMeans Clustering
    """

    closestCluster = np.zeros(len(data))

    # Reassign each data vector to the new, closest cluster
    for d in range(len(data)):
        
        # Store the coordinates of the current data vector
        xD = data[d, :]

        # Set the minimum distance tracker to be a very large number
        sqDistMin = 1e16

        # Find the closest representative vector (cluster) to the current data vector
        for i in range(k):
            sqDist = np.linalg.norm(centroids[i, :] - xD, ord=2)
            
            # If the distance is less than the current min, assign the
            # current data vector to this cluster
            if sqDist < sqDistMin:
                closestCluster[d] = i
                sqDistMin = sqDist

    return closestCluster

def KMeansClusters(initalizationTechnique, train_data):
    """
    Original code provided by Hangjie Ji, modified by project team.
    :param initalizationTechnique: function definition that returns initial represenative vectors
    :param train_data: data to be clusters
    """
    c = initalizationTechnique(len(train_data[0]))

    # Create a data structure to store closest representative vector for each data point
    closestCluster = findClosestCluster(train_data, c)

    # Update the assignments of the data vectors to their new clusters
    IndexSet = closestCluster.astype(int)

    # Create data structures to store the representative vectors from the previous iteration (cPrev)
    cPrev = np.copy(c)

    # The Alternating Minimization Scheme
    doneFlag = False

    # Keep alternating updates to representative vectors and cluster assignments until representative vectors no longer change their locations
    while not doneFlag:
        # Update the representative vectors in each cluster via the centroid formula
        for i in range(k):
            
            # Find the indices for all data vectors currently in cluster i
            ClusterIndices = np.where(IndexSet == i)[0]

            # Find the number of data vectors currently in cluster i
            NumVecsInCluster = len(ClusterIndices)

            # Create a data structure to store representative vector for the current cluster
            c[i, :] = np.zeros(len(train_data[0]))

            # Update cluster vector using the centroid formula
            for j in range(NumVecsInCluster):
                c[i, :] += train_data[ClusterIndices[j], :] / NumVecsInCluster

        # Now reassign all data vectors to the closest representative vector (cluster)
        # Create a data structure to store closest representative vector for each data point
        closestCluster = findClosestCluster(train_data, c)

        # Update the assignments of the data vectors to their new clusters
        IndexSet = closestCluster.astype(int)

        # Terminate the alternating scheme if the representative vectors are unaltered
        # relative to the previous iteration
        if np.array_equal(c, cPrev):
            doneFlag = True
        else:
            cPrev = np.copy(c)

    return IndexSet, c