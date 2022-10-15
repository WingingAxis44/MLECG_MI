
import pandas as pd
import numpy as np
import os

import wfdb
from sklearn.model_selection import train_test_split
import preprocessing
import ast
from sklearn.model_selection import StratifiedKFold
from biosppy.signals import ecg

leads = ['II']
#leads = ['I','II']

sampling_filename = {'100':'filename_lr', '500':'filename_hr'} 

#Sample ECG signal data from WFDB data files
def load_ecg(df, fs, path):
   
    
    if(fs==100):
       
        data = [wfdb.rdsamp(path+f, channel_names=leads) for f in df.filename_lr]

    else:
      
        data = [wfdb.rdsamp(path+f, channel_names=leads) for f in df.filename_hr]
      
        
    data = np.asarray([signal for signal,_ in data])
    return data


def make_dataset(data_path,num_sec, fs, patient_split = True):

    print('Preparing Data...')
    print('Please note: This can take 5~10 minutes to finish\n if creating everything from scratch')
    # load and convert annotation data
    Y = pd.read_csv(data_path+'ptbxl_database.csv', index_col='ecg_id')

    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(data_path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
  
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    #Convert diagnostic superclass to class label
    convertToLabel(Y.diagnostic_superclass.values)

  
    isLoad_ECG = False
    # Load raw signal data
    # If it doesn't exist, create it using wfdb
    if (os.path.exists('./trained_models/processedX.npz')):
        data = np.load('./trained_models/processedX.npz', allow_pickle = True)
        X = data['X']
        data.close()
        if (X.shape[2] != len(leads)) : isLoad_ECG = True
        else:
            print('Filtered data successfully loaded froom \"./trained_models/processedX.npz\"')
    
    else:
        isLoad_ECG = True

    if (isLoad_ECG):
        print("Loading data from wfdb signal data. Please wait...")
        X = load_ecg(Y, fs, data_path)
        print("WFDB data loading done.")
        #Drop "other" labelled ECG data from X
        X = X[np.nonzero(Y.diagnostic_superclass.values!=2)]

        #Perform Signal preprocessing steps on raw ECG signals
        print('Filtering Data. Please wait...')
        preprocessing.preprocess_all_fn(X,fs)    
        print('Fitlering done.')

        #Save processed ECG data
        np.savez_compressed('./trained_models/processedX', X=X)
        print('Filtered data saved to \"./trained_models/processedX\"')

    #Drop "other" labelled ECG data from Y
    Y.drop(Y[Y.diagnostic_superclass==2].index, inplace=True)
    
    #Now, split data into train, validation and test sets for cross validation
    
    #Approach 1 (this is the main approach)
    #Split by patient for cross-validation (i.e. for inter-patient classification)
    #This ensures patient-specific data is only ever in one of training, validation or test sets
    #i.e. patient data leakage is prevented.
    #This is used in the primary contribution of this study (i.e. inter-patient MI classifier)

    if (patient_split):

        #Manually selected currently
        #This fold selection could also be done automatically when pairing this experimental platform
        #With a script that could cycle through all ten rounds

        valid_fold = 9
        test_fold = 10

        X_train = X[np.where( (Y.strat_fold != test_fold) & (Y.strat_fold != valid_fold))]

        X_valid = X[np.where(Y.strat_fold == valid_fold)]

        X_test = X[np.where(Y.strat_fold == test_fold)]


        rhythm_train = Y[(Y.strat_fold != test_fold) & (Y.strat_fold != valid_fold)].diagnostic_superclass.values


        rhythm_valid = Y[(Y.strat_fold == valid_fold)].diagnostic_superclass.values


        rhythm_test = Y[Y.strat_fold == test_fold].diagnostic_superclass.values

        #Segment data at QRS complex (this is the windowing step)
        print("Segmenting Data, please wait")
        X_train, rhythm_train = segmentByQRS(X_train, rhythm_train,num_sec,fs)
        print('Finished Segmenting Training Splits')

        X_valid, rhythm_valid = segmentByQRS(X_valid, rhythm_valid ,num_sec,fs)
        print('Finished Segmenting Validation Splits')

        X_test, rhythm_test = segmentByQRS(X_test, rhythm_test,num_sec,fs)
        print('Finished Segmenting Test Splits')
  
    #Approach 2 (for beat splitting experiments)    
    #Split by beat (i.e. for intra-patient classification)
    #This is for experimenting with patient-specific data leakage

    else:
        createNewSplits = False
        
        #Check to see if there are already premade beat splits (including the fold indicies)
        if (os.path.exists('./trained_models/beatSplitX.npz')):
            
            #Load pre-existing data from disc
            X,rhythm_all, folds, n_sec = load_beatSplit_data()

            #Create new splits if the user has changed the number of ECG leads required
            #Or if they have changed their window size (i.e. number of seconds left & right of R peak)

            if (X.shape[2] != len(leads) or n_sec != num_sec) : createNewSplits = True
    
        else:
            createNewSplits = True

        if (createNewSplits):
            rhythm_all = Y.diagnostic_superclass.values

            #Segment data at QRS complex (this is the windowing step)
            print("Segmenting Data, please wait")
            
            X, rhythm_all = segmentByQRS(X,rhythm_all,num_sec,fs)
            print('Finished Segmenting Data')

            #Create training and testing indices for stratified 10-fold cross validation
            skf = StratifiedKFold(n_splits=10)
       
            folds = {}
            fold = 1
            for train_index, test_index in skf.split(X, rhythm_all):
              
                folds[fold] = train_index, test_index
                fold += 1

            #Save data (including the fold information)

            np.savez_compressed('./trained_models/beatSplitX', X=X, rhythm_all=rhythm_all,
            folds=folds, num_sec=num_sec)
            print('Beat split data and folds saved to \"./trained_models/beatSplitX\"')

        #Fold number has to be manually selected (1,2,3 .. 10)
        #This refers to one of the 10 rounds of k fold cross validation
        #This fold selection could also be done automatically when pairing this experimental platform
        #With a script that could cycle through all ten rounds
        
        fold_selected = 1

        print('Fold selected: ', fold_selected)
        train_indices = folds[fold_selected][0]
        test_indices = folds[fold_selected][1]

        #Extract training and testing datasets for this round of cross validation
        X_train, X_test = X[train_indices], X[test_indices]
        rhythm_train, rhythm_test = rhythm_all[train_indices], rhythm_all[test_indices]

        #Validation dataset is a random subset of the training data, for that cross-validation round
        # By default it is 10% of the training dataset    
        X_train, X_valid, rhythm_train, rhythm_valid = train_test_split(X_train, rhythm_train,test_size=0.1)
            
    
    return X_train, rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test

#Function to load beatSplit datasets from disc   
def load_beatSplit_data():

    data = np.load('./trained_models/beatSplitX.npz', allow_pickle = True)
    X = data['X']
    rhythm_all = data['rhythm_all']
    folds = data['folds'][()]
    n_sec = data['num_sec']
    data.close()

    return X, rhythm_all, folds, n_sec



#This function converts the PTB-XL diagnostic superclass annotations into class labels according to the following rules:
# --- If the ECG signal contains the "MI" label, consider whole ECG signal to be MI. Label = 1

# --- If the ECG signal has no "MI" label, but has any other cardiovascular disease or no annotation at all, 
#     consider whole ECG signal to be other. Label = 2

# --- If the ECG signal is only labelled "NORM", consider whole ECG signal to be Normal Sinus Rhythm
#     (i.e. healthy control). Label = 0

def convertToLabel(Y):

    
    for i in range(len(Y)):

        if('MI' in Y[i]):
          Y[i] = 1
        elif(any ( ('CD' in Y[i], 'STTC' in Y[i], 'HYP' in Y[i], len(Y[i]) == 0 ) ) ):
          Y[i] = 2
        else:
          Y[i] = 0

#Segment ECG record based on a rectangular window.
#This function creates non-overlapping windows.
#This was experimented with, but not used in the final version of this study.

def simpleSegment(X,Y, window_size):

    n_row = int(X.shape[1]/window_size)
   
    X_new = np.empty((X.shape[0]*n_row, window_size, X.shape[2]))
    
    Y_new = []
    
    for record in range(X.shape[0]):
            
     
        current_row = 0
        
        for window in range(0,(X.shape[1]), window_size):
         
            X_new[record*current_row] = X[record][window:window+window_size]
            Y_new.append(Y[record])
            current_row += 1
  
    
    return X_new, Y_new


#Segment ECG record based on a sliding window.
#The amount of overlap between successive windows is a function of the shift parameter
#This was experimented with, but not used in the final version of this study.

def segment(X,Y, window_size, shift):


    n_row = int( (X.shape[1]/shift) - (window_size/shift - 1))
   
    X_new = np.empty((X.shape[0]*n_row, window_size, X.shape[2]))
    Y_new = []
    
    for record in range(X.shape[0]):
            
     
        current_row = 0
        
        for slide in range(0,(X.shape[1]-window_size+shift), shift):
         
            X_new[record*current_row] = X[record][slide:slide+window_size]
            Y_new.append(Y[record])
            current_row += 1

   
    return X_new, Y_new

#This function segments the ECG signals at the QRS complex.
#This technique is used in this study

def segmentByQRS(X,Y,num_sec,fs):

    from itertools import repeat
    n_col = int(2*num_sec*fs)
    X_all = []
    Y_all = []


    #Run through all ECG records passed in
    for record in range(X.shape[0]):
        
        rhythmClass = Y[record]
        
        #Detect QRS complexes (i.e. R peaks) in ECG signal
        #Uses the hamilton segmenting algorithm

        r_peaks = ecg.hamilton_segmenter(signal=X[record,:,0], sampling_rate=fs)[0]
        
        #Correct R-peak locations
        #This is done because there is always a phase delay when detecting the R-peaks
        #This ensures the signals are segmented exactly at the QRS complex's peak.

        r_peaks = ecg.correct_rpeaks(
        signal=X[record,:,0], rpeaks=r_peaks, sampling_rate=fs, tol=0.05)[0]
       
        

        max_row = len(r_peaks)
     
        X_new = np.empty((max_row,n_col, X.shape[2]))
    
        for lead in range(X.shape[2]):
           
            if(lead!=0):
                
                r_peaks = ecg.hamilton_segmenter(signal=X[record,:,lead], sampling_rate=fs)[0]
     
                r_peaks = ecg.correct_rpeaks(signal=X[record,:,lead], rpeaks=r_peaks, sampling_rate=fs, tol=0.05)[0]
           
            current_row = 0
            
            for sample in r_peaks:
            
                if(current_row==max_row):
                    break
              
                #Take the same amount of data left and right of the QRS complex.
                #The amount of data to be taken is a function of the sampling frequency and number of seconds
                #that the user has requested to take.

                left = (int) (max([0,(sample - num_sec*fs) ]))
                right = (int) (min([X.shape[1],(sample + num_sec*fs) ]))

                x = X[record,left:right,lead]
            
                if(len(x) == n_col):
                    X_new[current_row,:,lead] = x
                    
                    current_row += 1

            max_row = min(max_row, current_row)

            X_new=X_new[:max_row,:]
      
        for samp in X_new: 
            
            X_all.append(samp)
       
        #Add on as many of the relevant class labels to Y.
        #This is done to match up the rhythm labels with however many new samples have
        #been extracted from a particular ECG record

        Y_all.extend(repeat(rhythmClass, X_new.shape[0]))
        
    X_all = np.asarray(X_all)
    Y_all = np.asarray(Y_all, dtype=np.int8)
    
    assert X_all.shape[0] == len(Y_all), 'number of X, Y rows messed up'

    return X_all, Y_all
 