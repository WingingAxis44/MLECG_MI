import wfdb
import pandas as pd
import numpy as np

import scipy

abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']


data_path = 'mitdbaf/'
# Exclude 00735 and 03665
#.atr = rhythm annotation files
#.qrs = beat annotation files 
pts = ['04015','04043','04048','04126',
    '04746','04908','04936','05091',
    '05121','05261','06426','06453'
    ,'06995','07162','07859','07879'
    ,'07910','08215','08219','08378'
    ,'08405','08434','08455']

def load_ecg(file):
    signals, _ = wfdb.rdsamp(file)
    ann = wfdb.rdann(file, 'atr')
    beats = wfdb.rdann(file, 'qrs') 
    beat_annotations = beats.symbol
    

    aux_note = ann.aux_note
    sym = ann.symbol
    beat_location = beats.sample
    
    aux_note_index = ann.sample


    return signals, beat_annotations, beat_location, sym, aux_note_index, aux_note



def make_dataset(data_path,num_sec, fs, samples, downsample=False, resample_freq=250):
    
    num_cols = 2*num_sec * fs
    if downsample==True and fs!=resample_freq:
        num_cols = 2*num_sec * resample_freq
    X_all = np.zeros((1,num_cols))

    rhythm_all = []
    max_rows = []
    limit = 0
    flag = True
    
    for patient in samples:

        file = data_path + patient 
        limit = limit+1
        
        p_signal, beat_annotations,beat_location,  sym,aux_note_index, aux_note = load_ecg(file)
        
        
        #Extract lead 1 
        p_signal = p_signal[:,0]
        
        df_ann = pd.DataFrame({'atr_sym':beat_annotations, 'atr_sample':beat_location})

        # 
        labels = get_correct_data_structure_for_labelling(p_signal,aux_note_index,aux_note)
      
        X, rhythm = build_XY(p_signal,df_ann, num_cols, num_sec, fs,  labels, downsample, resample_freq)
        #X, rhythm = build_XY(p_signal,df_ann, num_cols, num_sec, fs, downsample, resample_freq)
        
        
        #JUST FOR MEMORY OPTIMISATION - TAKE OUT IF HAVING ERRORS
        del labels

   
        rhythm_all = rhythm_all+rhythm
        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
   
    X_all = X_all[1:,:]

  
    assert np.sum(max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert X_all.shape[0] == len(rhythm_all), 'number of X, Y rows messed up'

    return X_all, rhythm_all
        
def build_XY(p_signal, df_ann, num_cols, num_sec, fs,  labels, downsample = False, resample_freq = 250):
#def build_XY(p_signal, df_ann, num_cols, num_sec, fs, downsample = False, resample_freq = 250):
    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))

    rhythm = []
  
    max_row = 0


    for atr_sample, aux_note in zip(df_ann.atr_sample.values, df_ann.aux_note.values):
        left = max([0,(atr_sample - num_sec*fs) ])
        right = min([len(p_signal),(atr_sample + num_sec*fs) ])
        
        x = p_signal[left: right]
    

        if downsample==True and fs!=resample_freq:
            x = scipy.signal.resample(x, num_cols)
        if len(x) == num_cols:
            X[max_row,:] = x
            label_set = labels[left:right]
            rhythm.append(get_label(label_set))
  
            max_row += 1
    X = X[:max_row,:]
 
    return X,rhythm
        
def get_label(labels):
    if 1 in labels:
        return 1
    elif all_same(labels) and labels[0]==0:
        return 0 
    else:
        return 2 

def labelFix(aux_note):

    
    #Normal
    if ( ('(N' in aux_note) and ('(NOD' not in aux_note)):
      
        return 0 
    
    #Afib
    elif ('(AFIB' in aux_note):

        return 1

        #Other
    else:
        
        return 2


def all_same(items):
    return all(x == items[0] for x in items)

def get_correct_data_structure_for_labelling(p_signal, aux_note_index, aux_note):
    labels= []
    count = 0 
    current = "(N"
    for i in range(len(p_signal)):
        
        if i in aux_note_index:
            pos_of_label_name = np.where(aux_note_index==i)[0][0]
            current = aux_note[pos_of_label_name]
        if i<aux_note_index[0]:
            labels.append(0)
        else:
            if current=="(N":
                labels.append(0)
            elif current =="(AFIB":
                labels.append(1)
            else:
                labels.append(2)
    return labels
