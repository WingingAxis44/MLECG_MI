import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import cv2
from sklearn.preprocessing import (MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler)
from imblearn.under_sampling import (TomekLinks, EditedNearestNeighbours,RandomUnderSampler, NeighbourhoodCleaningRule)
from imblearn.over_sampling import (SMOTE, ADASYN, RandomOverSampler)



def band_pass(signal_data, fs=500):

  
    sos = signal.butter(2, [0.5,100.0], btype='bandpass', output='sos', fs = fs)
    
  
    row = 0
    for sample in signal_data:
        signal_data[row] = np.asarray(signal.sosfiltfilt(sos,sample), dtype=np.float16)
        row = row + 1

    
    return signal_data  


def band_stop(signal_data, fs=500):

    sos = signal.butter(2, [59.0,61.0], btype='bandstop', output='sos', fs = fs)
    
    row = 0
    for sample in signal_data:
       
        signal_data[row] = np.asarray(signal.sosfiltfilt(sos,sample), dtype=np.float16)
        row = row + 1

    return signal_data  


## Discreet Wavelet Transform (DWT)
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')



#This undersamples the Majority class (regardless of the type of class)

def undersample(X, Y):

  
    undersamp = RandomUnderSampler()

    dim2 = X.shape[1]
    dim3 = X.shape[2]
    X = X.reshape(len(X),-1)
    
    Y = Y if isinstance(Y,list) else Y.tolist()
    X,Y = undersamp.fit_resample(X,Y)
  
    X = X.reshape(int(X.shape[0]), dim2, dim3)

    
    return X, Y

#This oversamples the disease class by default
#If it proves useful, it can be adapted to allow for oversampling either class

def oversample(X,Y):
  
    #oversamp = SMOTE()
    oversamp = RandomOverSampler()
    dim2 = X.shape[1]
    dim3 = X.shape[2]
    X = X.reshape(len(X),-1)

    Y = Y if isinstance(Y,list) else Y.tolist()
    X,Y = oversamp.fit_resample(X,Y)
  
    X = X.reshape(int(X.shape[0]), dim2, dim3)
    

    return X, np.asarray(Y, dtype=np.int8)


def dropOtherLite(X_all, rhythm_all):

    X_all = X_all[np.nonzero(rhythm_all!=2)]
 
    rhythm_all = rhythm_all[rhythm_all!=2]
   
    return X_all, rhythm_all



def dropOther(X_train, rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test):



    X_train = X_train[np.nonzero(rhythm_train!=2)]
 
    rhythm_train = rhythm_train[rhythm_train!=2]

    X_valid = X_valid[np.nonzero(rhythm_valid!=2)]
    rhythm_valid = rhythm_valid[rhythm_valid!=2]
  
    X_test = X_test[np.nonzero(rhythm_test!=2)]
    rhythm_test = rhythm_test[rhythm_test!=2]
   
    return X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test


def normalizeSegment(X):

    row = 0
    scaler = MaxAbsScaler(copy=False)
    for sample in X:
        X[row] = scaler.fit_transform(sample)
        row += 1

def normalizeWholeSet(X_train, X_valid, X_test):

   

    for lead in range(X_train.shape[2]):
        scaler = MaxAbsScaler(copy=False)
        #scaler = StandardScaler(copy=False)
        X_train[:,:,lead] = scaler.fit_transform(X_train[:,:,lead])
        X_valid[:,:,lead] = scaler.transform(X_valid[:,:,lead] )
        X_test[:,:,lead] = scaler.transform(X_test[:,:,lead] )

#Applies bandstop filter for powerline interference removal
#Also applies bandpass filter to remove baseline wander and some high frequency noise
def preprocess_all_fn(X, fs = 500):

   # order = int(0.3 * fs)
    order = 4
    sos_stop = signal.butter(order, [49.0,51.0], btype='bandstop', output='sos', fs = fs)
    
    
    sos_pass = signal.butter(order, [0.5,100.0], btype='bandpass', output='sos', fs = fs)
    
    row = 0

    for sample in X:
        
        for lead in range(X.shape[2]):

            X[row,:,lead] = signal.sosfiltfilt(sos_stop,sample[:,lead])
            X[row,:,lead] = signal.sosfiltfilt(sos_pass,sample[:,lead])
           
            # X[row,:,lead] = (panda_series.rolling(10, min_periods=1, center=True).mean()).to_numpy()
       
        row = row + 1
        