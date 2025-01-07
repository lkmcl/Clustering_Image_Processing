from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def svc_analysis(train_data, train_target, test_data, test_target):

    """
    Use a linear svc classifer scheme to match handwritten images to target keys.
    Requires split of training and testing data which can be found by running
    the partition function

    Returns train and test accuracy.
    
    """
    
    ## Initialized the Linear SVC
    svc = SVC(kernel='linear')
    ## Fitting the svc
    svc.fit(train_data, train_target)
    
    ## Find predictions based upon the test data and solve for accuracy
    train_pred = svc.predict(train_data)
    test_pred = svc.predict(test_data)
    
    ## Find the accuracy of the model
    train_accuracy = accuracy_score(train_target, train_pred)
    test_accuracy = accuracy_score(test_target, test_pred)
    
    return train_accuracy, test_accuracy