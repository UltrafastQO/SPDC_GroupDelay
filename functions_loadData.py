import os
import numpy as np

from zipfile import ZipFile


class dataObj:
    
    def __init__(self):
        self.data = None # numpy array
        self.dataType = None # string
        self.HWP1_angle = None # float
        self.HWP2_angle = None # float
            
        
        

def find_between( s, first, last ):
    '''In string s, find sub-string between str first and str last'''
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def loadData(zipFileName):

    dataContainer = []
    
    with ZipFile(zipFileName) as zf:
        for filename in zf.namelist():
            if filename.endswith('.txt'): # only look at txt files
                
                dataType = find_between(filename, '/', '.txt')
                HWP1_angle = find_between(filename, 'HWP1_', '_HWP2')
                HWP2_angle = find_between(filename, 'HWP2_', '/')
                                
                if dataType in ['AE_BL', 'BE_BL', 'AE_AL', 'BE_AL', 'clicks', 'pumpSpectrum']:
                
                    dataPoint = dataObj()
                    dataPoint.dataType = dataType
                    dataPoint.HWP1_angle = float(HWP1_angle)
                    dataPoint.HWP2_angle = float(HWP2_angle)
                    
                    with zf.open(filename) as f:
                        #dataPoint.data = f.read() # this is much faster than numpy loadtxt
                        # but I dont know how to then format it into a numpy array
                        dataPoint.data = np.loadtxt(f, delimiter=',')
                    
                    
             
                    dataContainer.append(dataPoint)
        

    return dataContainer
