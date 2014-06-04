'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from scipy.cluster.vq import whiten, vq
import config
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import MiniBatchKMeans
import spams
from scipy.sparse.csc import csc_matrix

class BagOfWords(object):
    """ Bag-of-Words class
        Codebook generate method: 'k-means'(default)
        
        Feature encoding method: 'vector-quantization'(default)
                                 'sparse-coding'
                                 
        Pooling method: 'sum-pooling'(default), 
                        'max-pooling'
                        
        Normalization method: 'L1-norm'(default) 
                              'L2-norm'
    """
    def __init__(self, 
                 codebookGenerateMethod = 'k-means', 
                 featureEncodingMethod = 'vector-quantization',
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
        whitenedFeatures = whiten(features)
        
        if self._featureEncodingMethod == 'vector-quantization':
            # Vector quantization
            # each row is a feature vector
            index, _ = vq(whitenedFeatures, self._codebook)
            row, _ = features.shape
            col = config.codebookSize
            encodedFeatures = np.zeros((row, col))
            
            for i in xrange(len(index)):
                encodedFeatures[i, index[i]] = 1
                
        elif self._featureEncodingMethod == 'sparse-coding':
            # Sparse coding
            # each column is a feature vector
            X = np.asfortranarray(whitenedFeatures.transpose())
            X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),
                                              (X.shape[0],1)),
                                  dtype= X.dtype)
            D = np.asfortranarray(self._codebook.transpose())
            D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),
                                              (D.shape[0],1)), 
                                  dtype = D.dtype)
            
            # Parameters of the optimization are chosen
            param = {
                'lambda1': 0.15, 
                'numThreads': -1,
                'mode': 0    
                }
            
            alpha = spams.lasso(X, D, **param)   # alpha is sparse matrix
            
            alphaShape = (D.shape[1], X.shape[1])
            denseMatrix = csc_matrix(alpha, shape = alphaShape).todense()
            encodedFeatures = np.asarray(denseMatrix).transpose()
                
        return encodedFeatures
    
    
    def doFeaturePooling(self, encodedFeatures):
        """ Do feature pooling to encoded features """
        pooledFeatures = None
        if self._poolingMethod == 'sum-pooling':
            # Sum pooling
            pooledFeatures = np.sum(encodedFeatures, axis = 0)
        elif self._poolingMethod == 'max-pooling':
            # Max pooling
            pooledFeatures = np.max(encodedFeatures, axis = 0)
        
        return pooledFeatures
    
    
    def doFeatureNormalization(self, pooledFeatures):
        """ Do feature normalization to pooled features """
        normFeatures = None
        if self._normalizationMethod == 'L1-norm':
            # L1-normalization
            normFeatures = pooledFeatures / LA.norm(pooledFeatures, ord = 1)
        elif self._normalizationMethod == 'L2-norm':
            # L2-normalization
            normFeatures = pooledFeatures / LA.norm(pooledFeatures, ord = 2)
        
        return normFeatures