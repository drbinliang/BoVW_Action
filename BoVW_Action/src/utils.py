'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import config
from sklearn.metrics.metrics import confusion_matrix
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
import os

def getCategoryIdByName(categoryName):
    """ Get category id by category name """
    return config.actionCateogory.index(categoryName)


def plotConfusionMatrix(trueLabels, testPredLabels, 
                        saveFilename, normalization = False):
    """ Plot confusion matrix using true labels and prediction labels 
        normalization: True, accuracy of each class
                       False, number of results
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(trueLabels, testPredLabels)
    if normalization:  
        # If normalization
        cm = cm.astype(float) / LA.norm(cm, ord = 1, axis = 1)
    
    labels = []
    for item in trueLabels:
        if item not in labels:
            labels.append(item)
    
    
    # Plot confusion matrix
    font = {'size' : 12}
    mplib.rc('font', **font)
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.set_aspect('equal', adjustable='box')
    
    
    height, width = cm.shape
    
    for x in xrange(width):
        for y in xrange(height):
            
            if normalization:
                # If normalization
                floatNum = cm[x, y]
                annotation = "%.2f" % floatNum
                
                ax.annotate(annotation, xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
            else:
                intNum = cm[x, y]
                annotation = str(intNum)
                
                ax.annotate(annotation, xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
    
    heatmap = ax.imshow(np.array(cm), cmap=plt.cm.jet, 
                    interpolation='nearest')       
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(height), minor=False)
    ax.set_yticks(np.arange(width), minor=False)
    ax.set_xticklabels(labels, minor=False, rotation=45)
    ax.set_yticklabels(labels, minor=False)
    ax.set_xlabel('Predicted Labels', fontsize=18)
    ax.set_ylabel('True Labels', fontsize=18)
    
    plt.show()
    fig.savefig(saveFilename)


def removeall(path):
    ''' remove all files and folders in the path '''
    if not os.path.isdir(path):
        return
    
    files = os.listdir(path)

    for x in files:
        fullpath = os.path.join(path, x)
        if os.path.isfile(fullpath):
            os.remove(fullpath)
        elif os.path.isdir(fullpath):
            removeall(fullpath)
            os.rmdir(fullpath)