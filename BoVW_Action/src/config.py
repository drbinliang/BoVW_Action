'''
Created on 28/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''

actionDatabaseDir = 'C:\\KTH_Action_Database\\'

## STIP
stipBinDir = '.\\stip\\bin'
stipFeatureOutDir = '.\\stip_features\\'

## Codebook
codebookSize = 3000

# Experiment setting
actionCateogory = ['boxing', 'handclapping', 'handwaving', 
               'jogging', 'running', 'walking']
trainDataSubjects = ['person11', 'person12', 'person13', 'person14', 
                     'person15', 'person16', 'person17', 'person18']
validationDataSubjects = ['person19', 'person20', 'person21', 'person23', 
                          'person24', 'person25', 'person01', 'person04']
testDataSubjects = ['person22', 'person02', 'person03', 'person05', 
                    'person06', 'person07', 'person08', 'person09', 'person10']

## Cross validation
is_cv = False
svm_c = 32.0
svm_g = 0.0001220703125

## Predication
predctDir = '.\\pred'


## Experimental Results:
#    K-means, Vector Quantization, Sum Pooling, L1-Normalization
#    codebookSize    svm_c    svm_g            cv_acc    final_acc
#    500             2.0      0.03125          91.0995   0.842592592593
#    1000            512.0    0.0001220703125  93.1937   0.898148148148
#    2000            512.0    0.0001220703125  94.2408   0.888888888889

#    K-means, Sparse coding, Max Pooling, L2-Normalization
#    codebookSize    svm_c    svm_g            cv_acc    final_acc
#    500             32.0     0.001953125      91.623    0.87037037037
#    1000            32.0     0.0001220703125  93.1937   0.856481481481
#    2000            128.0    0.0001220703125  93.7173   0.898148148148
#    3000            32.0     0.0001220703125  92.6702
