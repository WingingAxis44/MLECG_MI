
from cProfile import label
import wfdb
import pandas as pd
import numpy as np

import glob 
import scipy

from PIL import Image

import random
#import preprocessing

from sklearn.model_selection import (train_test_split, cross_val_score, KFold)



abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']

# list of patients
# Eventually something more modular will be implemented than a hardcoded list
# like this.
pts = ['100', '101', '102', '103', '104', '105', '106', '107',
       '108', '109', '111', '112', '113', '114', '115', '116',
       '117', '118', '119', '121', '122', '123', '124', '200',
       '201', '202', '203', '205', '207', '208', '209', '210',
       '212', '213', '214', '215', '217', '219', '220', '221',
       '222', '223', '228', '230', '231', '232', '233', '234']


#gets all IDs from a given directory and gets annotation and record
def test_symbols(data_path):

    for file in glob.iglob(f'{data_path}/*.hea'):
        thing = file[0:9]
        if thing == 'mitdb/201':
            record = wfdb.rdrecord(thing)
            annotation = wfdb.rdann(thing,'atr')
            sym = annotation.symbol
            print(len(annotation.aux_note))

            # values, counts = np.unique(sym, return_counts=True)
            # df_sub = pd.DataFrame({'sym':values, 'val':counts, 'pt':file[6:9]})
            # df = pd.concat([df,df_sub],axis=0)


#Reads a signal ECG and gets the physical signal (raw data) and associated annotations and symbols
def load_ecg(file):
    record = wfdb.rdrecord(file)
    
    ann = wfdb.rdann(file, 'atr')
    p_signal = record.p_signal 
    aux_note = ann.aux_note
    
    sym = ann.symbol
    samp = ann.sample
    
    
    return p_signal, sym, samp, aux_note


def make_dataset(data_path,num_sec, fs, samples, downsample=False, resample_freq=250):
    
    num_cols = (int) (2*num_sec * fs)
    if downsample==True and fs!=resample_freq:
        num_cols = (int) (2*num_sec * resample_freq)
    X_all = np.zeros((1,num_cols))
   
    rhythm_all = []
    max_rows = []
    limit = 0
    flag = True
    
    
    for patient in samples:

        file = data_path + patient 
        limit = limit+1
   
        p_signal,_,samp, aux_note = load_ecg(file)
        
       
        p_signal = p_signal[:,0]
  
        df_ann = pd.DataFrame({'atr_sample':samp, 'aux_note':aux_note})
        #labels = make_labels(df_ann, p_signal)
        
        
   
       # X, rhythm = build_XY(p_signal,df_ann, num_cols, num_sec, fs, labels, downsample, resample_freq)
        X, rhythm = build_XY(p_signal,df_ann, num_cols, num_sec, fs,  downsample, resample_freq)
    
        #JUST FOR MEMORY OPTIMISATION - TAKE OUT IF HAVING ERRORS
       # del labels


        rhythm_all = rhythm_all+rhythm

        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
        

    # drop the first zero row
    X_all = X_all[1:,:]
   
    #print(np.unique(rhythm_all, return_counts=True))
     # check sizes make sense
 
    assert np.sum(max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert X_all.shape[0] == len(rhythm_all), 'number of X, Y rows messed up'
   
    return X_all,  rhythm_all  


#def build_XY(p_signal, df_ann, num_cols, num_sec, fs, labels, downsample = False, resample_freq = 250):
def build_XY(p_signal, df_ann, num_cols, num_sec, fs, downsample = False, resample_freq = 250):
    
    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))
  
    rhythm = []

    labelSet = [(note,index) for note,index in zip(df_ann.aux_note.values, df_ann.atr_sample.values) if note != '']
   
  
    # keep track of rows
    max_row = 0
    
    for atr_sample in df_ann.atr_sample.values:

            
        left = (int) (max([0,(atr_sample - num_sec*fs) ]))
        right = (int) (min([len(p_signal),(atr_sample + num_sec*fs) ]))
      
        x = p_signal[left: right]
        
         

        if downsample==True and fs!=resample_freq:
            x = scipy.signal.resample(x, num_cols)

        if len(x) == num_cols:
            X[max_row,:] = x

            segmentLabels = getSegmentLabels(left,right,labelSet)
           
            rhythmClass = setClass(segmentLabels)
       
            rhythm.append(rhythmClass)
            max_row += 1
   
    X = X[:max_row,:]
   
    return X, rhythm

def get_label(labels):
    if 1 in labels:
        return 1
    elif all_same(labels) and labels[0]==0:
        return 0 
    else:
        return 2 
def all_same(items):
    return all(x == items[0] for x in items)

def getAbnormalIndex(atr_sym,atr_sample):
    ab_index = [b for a,b in zip(atr_sym,atr_sample) if a in abnormal][:10]
    return ab_index

def getSegmentLabels(left,right,labelset):

   
    if(len(labelset)==1):
       
        return labelset
    
    start, end = 0, 0
  
    for i in range(len(labelset)-1):
         
        current, next = labelset[i][1], labelset[i+1][1]
        
        if (left >= current) and (left <= next):
           
            start = i
            break
      
    for i in range(start,len(labelset)-1):
        current, next = labelset[i][1], labelset[i+1][1]

        if (right >= current) and (right <= next):
           
            end = i
            break

    if(end==0):
        end = start
    #Add 1 to end because end point in list slicing is exclusive

    return labelset[start:end+1]

def setClass(segmentLabels):

    isOther = False
    for labelPair in segmentLabels:
        
        rhythm = labelPair[0]
        rhythmClass = getRhythmClass(rhythm)

        if(rhythmClass == 1):
            return 1
        elif(rhythmClass == 2):
            isOther = True

    if(isOther):
        return 2
    else:
        return 0     

def majorityVote(segmentLabels, left,right):

    #The position of the values represent the rhythm
    #i.e: classCounts[0] refers to the normal count
    #   : classCounts[1] refers to the disease count
    #   : classCounts[2] refers to the other count

    classCounts = [0,0,0]
    
    if(len(segmentLabels)==1):
        
        return getRhythmClass(segmentLabels[0][0])

    
    for i in range(len(segmentLabels)):

        rhythmClass = getRhythmClass(segmentLabels[i][0])

        if(i==0):
            start = left
        else:
            start = segmentLabels[i][1]   

        if(i==(len(segmentLabels)-1)):
            end = right
        else:
        
            end = segmentLabels[i+1][1]

        classCounts[rhythmClass] = classCounts[rhythmClass] + (end - start)

    return np.argmax(classCounts)


def getRhythmClass(aux_note):

    
    #Normal
    if ( ('(N' in aux_note) and ('(NOD' not in aux_note)):
      
        return 0 
    
    #Afib
    elif ('(AFIB' in aux_note):

        return 1

        #Other
    else:
        
        return 2   
          
def make_labels(df_ann, p_signal):
    labels = []
    aux_note_index = []
    aux_note = []
    count = 0 
    
    for atr_sample, aux_note_loop in zip(df_ann.atr_sample.values, df_ann.aux_note.values):
        if aux_note_loop!='':
            aux_note.append(aux_note_loop)
            aux_note_index.append(atr_sample)
    current = "(N"
    for i in range(len(p_signal)):
        
        if i in aux_note_index:
            pos_of_label_name = aux_note_index.index(i)
            current = aux_note[pos_of_label_name]
        if i<aux_note_index[0]:
            labels.append(0)
        else:
            if '(N' in current and '(NOD' not in current: 
                labels.append(0)
            elif "(AFIB" in current:
                labels.append(1)
            else:
                labels.append(2)
    return labels


    
  
