'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from scipy.cluster.vq import whiten, kmeans2, ClusterError, vq
import config
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import MiniBatchKMeans

class BagOfWords(object):
    
    def __init__(self, 
                 codebookGenerateMethod = 'k-means', 
                 featureEncodingMethod = 'hard-assignment',
                 poolingMethod = 'sum-pooling',
                 normalizationMethod = 'L1-norm'):
        
        self._codebookSize = config.codebookSize    # codebook size
        self._codebook = None
        self._codebookGenerateMethod = codebookGenerateMethod
        self._featureEncodingMethod = featureEncodingMethod
        self._poolingMethod = poolingMethod
        self._normalizationMethod = normalizationMethod
    
    @property
    def codebook(self):
        return self._codebook
    
    @codebook.setter
    def codebook(self, value):
        self._codebook = value
    
    def generateCodebook(self, features):
        """ Generate codebook using extracted features """
        codebook = None
        
        if self._codebookGenerateMethod == 'k-means':
            whitenedFeatures = whiten(features)
            kmeans = MiniBatchKMeans(n_clusters = config.codebookSize, 
                                     init_size = 3 * config.codebookSize,
                                     n_init = 10, )
            kmeans.fit(whitenedFeatures)
            codebook = kmeans.cluster_centers_
        else:
            pass
        
        self._codebook = codebook
        
    
    def doFeatureEncoding(self, features):
        """ do feature encoding to original features"""
        encodedFeatures = None
        
        if self._featureEncodingMethod == 'hard-assignment':
            # Hard assignment
            whitenedFeatures = whiten(features)
            index, _ = vq(whitenedFeatures, self._codebook)
            row, _ = features.shape
            col = config.codebookSize
            encodedFeatures = np.zeros((row, col))
            
            for i in xrange(len(index)):
                encodedFeatures[i, index[i]] = 1
        else:
            pass
                
        return encodedFeatures
    
    
    def doFeaturePooling(self, encodedFeatures):
        """ Do feature pooling to encoded features """
        pooledFeatures = None
        if self._poolingMethod == 'sum-pooling':
            # Sum pooling
            pooledFeatures = np.sum(encodedFeatures, axis = 0)
        else:
            pass
        
        return pooledFeatures
    
    
    def doFeatureNormalization(self, pooledFeatures):
        """ Do feature normalization to pooled features """
        normFeatures = None
        if self._normalizationMethod == 'L1-norm':
            # L1-normalization
            normFeatures = pooledFeatures / LA.norm(pooledFeatures, ord = 1)
        else:
            pass
        
        return normFeatures