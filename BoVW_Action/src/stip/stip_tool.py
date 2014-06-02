'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import os
import config
from subprocess import Popen, PIPE

class StipTool(object):
    
    def __init__(self):
        self._binDir = config.stipBinDir   # stip bin for windows
        self._stipdev = os.path.join(self._binDir, 'stipdet.exe')
        self._stipshow = os.path.join(self._binDir, 'stipshow.exe')
        
        assert os.path.exists(self._stipdev), "stipdet.exe not found"
        assert os.path.exists(self._stipshow), "stipshow.exe not found"
        
    def extractStip(self, filePath):
        """ Extract STIP from given file """
        filename = os.path.split(filePath)[1]
        featureFilename = os.path.splitext(filename)[0] + '_stip'
        stipFeatureOutDir = config.stipFeatureOutDir
        
        if not os.path.exists(stipFeatureOutDir):
            os.makedirs(stipFeatureOutDir)
        
        featureFilePath = os.path.join(stipFeatureOutDir, featureFilename)
        
        cmd = "{0} -f {1} -o {2}".format(self._stipdev, filePath, featureFilePath)
        print filename, "STIP extracting ... "
        Popen(cmd, shell = True, stdout = PIPE).communicate()
        
        