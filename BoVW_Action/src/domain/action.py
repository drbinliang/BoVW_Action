'''
Created on 28/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import os
from utils import getCategoryIdByName
import config
from stip.stip_tool import StipTool
import numpy as np

class ActionSequence(object):
    
    def __init__(self, filePath):
        self._filePath = filePath
        
        self._filename = os.path.split(self._filePath)[1]
        self._subject = self.filename.split('_')[0]
        self._categoryName = self.filename.split('_')[1]
        self._categoryId = getCategoryIdByName(self._categoryName)
        self._scenario = self.filename.split('_')[2]
        
        self._stipFilePath = ''
        self.stipFeatures = None
        self._finalFeatures = None
    
    
    @property
    def filePath(self):
        return self._filePath
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def subject(self):
        return self._subject
    
    @property
    def categoryName(self):
        return self._categoryName
    
    @property
    def categoryId(self):
        return self._categoryId
    
    @property
    def scenario(self):
        return self._scenario
    
    @property
    def finalFeatures(self):
        return self._finalFeatures
    
    def extractStip(self):
        """ Extract STIP """
        featureFilename = os.path.splitext(self._filename)[0] + '_stip'
        self._stipFilePath = os.path.join(config.stipFeatureOutDir, featureFilename)
        
        stipTool = StipTool()
        if not os.path.exists(self._stipFilePath):
            # if not exists the feature file
            stipTool.extractStip(self._filePath)    
        
        self._getFeaturesFromFile(self._stipFilePath)
    
    def _getFeaturesFromFile(self, stipFilePath):
        """ Get features from stip file """
        f = open(stipFilePath, 'r')
        features = np.array([])
        
        # Omit the first 3 line
        f.readline(), f.readline(), f.readline()
        for line in f:
            featureStr = line.split('\t')[7:-1]
            featureVec = []
            
            for item in featureStr:
                featureVec.append(float(item)) 
                
            if features.size == 0:
                features = np.array(featureVec)
            else:
                features = np.vstack((features, featureVec))
        f.close()
        
        self.stipFeatures = features
    
    def generateFinalFeatures(self, bovw):
        """ Generate final features using given bag-of-words """
        # 1. Do feature encoding
        encodedFeatures = bovw.doFeatureEncoding(self.stipFeatures)
        
        # 2. Do feature pooling
        pooledFeatures = bovw.doFeaturePooling(encodedFeatures)
        
        # 3. Do normalization
        self._finalFeatures = bovw.doFeatureNormalization(pooledFeatures)