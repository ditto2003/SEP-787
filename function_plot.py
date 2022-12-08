"""System packages that comparison functions used"""
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import time

def Load_mat_single(data_path):
    """Load the .mat data"""
    mat_contents = sio.loadmat(data_path)
    for i, key in enumerate(mat_contents):
        print(i, key)
    return mat_contents

def mat_to_array(mat_contents):
    """Convert the .mat data into Numpy array"""
    mat_name = []
    mat_data = []
    for i, (k, v) in enumerate(mat_contents.items()):
        mat_name.append(k)
        mat_data.append(v)
    vibration_signal_all = np.array(mat_data[3])
    return vibration_signal_all


def plot_confusion_matrix(Y_test, prediction, clf):
    
    cm = confusion_matrix(Y_test, prediction, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    
    disp.plot()
    disp.ax_.set_title('{}'.format(clf))
    plt.show()


def train_test(X_train,Y_train, X_test,Y_test,clf, show_time =False):
    """Train the data with AI model and display the execution time"""
    # Enable time flag
    print("The experiment is %s \n" % (clf))
    print("The shape of X_train is {} \n".format(X_train.shape))
    
    #start training classifier
    start_time = time.time()
    clf.fit(X_train,Y_train)
    train_time = time.time()-start_time
    
    #testing the fitted model
    start_time_test = time.time()
    # Predict the test set
    prediction = clf.predict(X_test)
    test_time = time.time()-start_time_test

    if show_time == True:
        print("The train time is --- %.8f seconds ---" % (train_time))
        print("The test time is --- %.8f seconds ---" % (test_time))
    
    #compute error
    error = sum(prediction != Y_test)

    return error, prediction, train_time, test_time

def roc(Y_test, prediction, clf=str):
    fpr, tpr, _ = roc_curve(Y_test, prediction)
    roc_auc = auc(fpr, tpr)

    plt.title(f"ROC: {clf}")
    plt.plot(fpr, tpr, 'b', label=f'AUC={roc_auc * 100}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

def roc_comp(Y_test, prediction, clf=str):
    for i in range(len(clf)):
        fpr, tpr, _ = roc_curve(Y_test, prediction[i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, 'b', label=f'AUC={roc_auc * 100}')

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')

    plt.ioff()
    plt.show()