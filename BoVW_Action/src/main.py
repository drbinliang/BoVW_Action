'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import os
import config
from domain.action import ActionSequence
import numpy as np
from bovw.bag_of_words import BagOfWords
from validation.cross_validation import crossValidate
from recognition.svm_tool import SvmTool

def main():
    actionDatabaseDir = config.actionDatabaseDir
    categories = config.actionCateogory
    
    ## Step.1 Data loading and features extraction 
    # Get features from training data over all categories
    print "Data loading and feature extraction ..."
    
    allFeatures = np.array([])
    trainActionSequence = []
    testActionSequence = []
    
    if os.path.exists('allFeatures.npy') \
        and os.path.exists('trainActionSequence.npy') \
        and os.path.exists('testActionSequence.npy'):
        
        allFeatures = np.load('allFeatures.npy')
        trainActionSequence = np.load('trainActionSequence.npy')
        testActionSequence = np.load('testActionSequence.npy')
    else:
        for category in categories:
            categoryPath = os.path.join(actionDatabaseDir, category)
            allData = os.listdir(categoryPath)
            
            
            # Train data and test data loading
            for data in allData:
                filePath = os.path.join(categoryPath, data)
                actionSequence = ActionSequence(filePath)
                actionSequence.extractStip()    # extract STIP
                
                subject = actionSequence.subject
                if subject in config.trainDataSubjects:
                    trainActionSequence.append(actionSequence)
                    
                    if allFeatures.size == 0:
                        allFeatures = actionSequence.stipFeatures
                    else:
                        allFeatures = np.vstack((allFeatures, 
                                                 actionSequence.stipFeatures))
                        
                elif subject in config.testDataSubjects:
                    testActionSequence.append(actionSequence)
            
        
        np.save('allFeatures', allFeatures)        
        np.save('trainActionSequence', trainActionSequence)
        np.save('testActionSequence', testActionSequence)
            
    ## Step.2 Codebook generation
    print "Codebook generation ..."
    bovw = BagOfWords()
    if os.path.exists('codebook.npy'):
        codebook = np.load('codebook.npy')
        bovw.codebook = codebook
    else:
        bovw.generateCodebook(allFeatures)
        np.save('codebook', bovw.codebook)
        
    ## Step.3 Feature encoding for train data
    train_y = []
    train_X = []
    
    for actionSequence in trainActionSequence:
        # Feature encoding, pooling, normalization
        actionSequence.generateFinalFeatures(bovw)
        
        # Format train data
        train_y.append(actionSequence.categoryId)
        train_X.append(actionSequence.finalFeatures)
    
    # Cross validation    
    if config.is_cv:
        # cross validation
        crossValidate(train_y, train_X)
    
    ## Step.4 Classification
    # Learning using SVM
    svmTool = SvmTool()
    print "Model learning ..."
    svmTool.learnModel(train_y, train_X)
    
    # Feature encoding for test data and classify data using learned model
    print "Predicating ..."
    numCorrect = 0
    for actionSequence in testActionSequence:
        # Feature encoding, pooling, normalization
        actionSequence.generateFinalFeatures(bovw)
        
        # Format train data
        test_y = [actionSequence.categoryId]
        test_X = [actionSequence.finalFeatures]
        
        p_label, _ = svmTool.doPredication(test_y, test_X)
        predCagtegoryId = int(p_label[0])
        predCategoryName = categories[predCagtegoryId]
        
        # Write predicated action to predicated category
        predCategoryPath = os.path.join(config.predctDir, predCategoryName)
        if not os.path.exists(predCategoryPath):
            os.makedirs(predCategoryPath)
        
        predFilePath = os.path.join(predCategoryPath, actionSequence.filename)
        
        f = open(predFilePath, 'w')
        f.close()
        
        if predCagtegoryId == actionSequence.categoryId:
            numCorrect += 1 
    
    # Calculate results
    accuracy = numCorrect / float(len(testActionSequence))
    print "accuracy: ", accuracy
        

if __name__ == '__main__':
    main()