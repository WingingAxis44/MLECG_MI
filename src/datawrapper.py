import numpy as np
import sys
import data
import datamitaf
import dataptb_xl
from sklearn.model_selection import train_test_split


datasets_AFIB = ['Arrhythmia', 'AF']

datasets_MI = ['PTB_XL']

dataset_frequencies = {'Arrhythmia': 360,
                        'AF': 250, 'PTB_XL' : 500}  #ptb-xl also has freq = 100

dataset_paths = {'Arrhythmia': 'mitdb/',
                        'AF': 'mitdbaf/',
                        'PTB_XL' : 'ptb_xl/'}

patients_all = {'Arrhythmia': data.pts,
                        'AF': datamitaf.pts}

dataset_modules = {'Arrhythmia': data, 'AF': datamitaf, 'PTB_XL': dataptb_xl}

chosen_datasets = []

#This function allows the user to select what cardiovascular disease they would like to train and test on
#It also asks the user which datasets they'd like to include for the task
#Currently only one MI dataset has been included in this study, however more can be added and integrated with this pipeline
def choose_datasets():
    
    disease_choice = eval(input('Please select either AFIB (1) or MI (2) as your classification task\n'+
    'Enter 1 or 2\n'))

    
    if(disease_choice != 1 and disease_choice != 2):
        sys.exit('Please select a disease to classify on by inputing a 1 for AFIB or 2 for MI')

    print('You selected AF') if disease_choice == 1 else print('You selected MI')
    print('Select the dataset(s) you wish to train and test the model on:')
    print('Enter number followed by a comma and a space i.e 1, 2')

    if(disease_choice==1):
        dataset_choices = input(
                "1) MIT-BIH Arrhythmia\n"+
                "2) MIT-BIH Atrial Fibrillation\n")
        tmp_dataset = datasets_AFIB

    if(disease_choice==2):
        dataset_choices = input(
            #    "1) PTB\n"+
                "1) PTB_XL\n")
        tmp_dataset = datasets_MI

    dataset_choices = dataset_choices.split(", ")

    if(len(dataset_choices) == 0):
        sys.exit('Please select at least one dataset in the format specified.')
 
    for i in dataset_choices:
        chosen_datasets.append(tmp_dataset[int(i)-1])

    
    print('You selected: ' + ' '.join(chosen_datasets))

    
    return disease_choice


#This function calls the relevant data module(s) to extract X and Y data.

def make_dataset(data_path,num_sec, samples, patient_split = True):
    X_all, rhythm_all = None, None
    
    #PTB-XL is currently the only MI dataset available on the MLECG platform

    if 'PTB_XL' in chosen_datasets:

        return dataptb_xl.make_dataset(data_path+dataset_paths['PTB_XL'], num_sec,dataset_frequencies['PTB_XL'], patient_split)

    if len(chosen_datasets)==1:

        dataset = chosen_datasets[0]
      
  
        X_all, rhythm_all = dataset_modules[dataset].make_dataset(data_path+dataset_paths[dataset], num_sec, 
        dataset_frequencies[dataset], samples[dataset])


    #This allows for including multiple datasets as a part of training and testing.
    #Currently not implemented for MI detection, but has been used for AFIB detection    
    else:
        min_freq = 100000
        rhythm_all = []
        
        for dataset, frequency in dataset_frequencies.items():
            if frequency<min_freq:
                min_freq = frequency
                
        num_cols = (int) (2*num_sec * min_freq)
    
        rhythm_all = []
        X_all = np.zeros((1,num_cols))


        for dataset in chosen_datasets:

        
            X, rhythm = dataset_modules[dataset].make_dataset(data_path+dataset_paths[dataset], num_sec, 
            dataset_frequencies[dataset], samples[dataset], True, min_freq)

            rhythm_all = rhythm_all+rhythm
            X_all = np.concatenate((X_all,X),axis=0)

        #Drop first zero row 
        X_all = X_all[1:,:] 
   
    return X_all, rhythm_all 

#Patient splitting to make a training and testing set, but excluding a validation set
#Currently not in use for any task

def split_by_patient_no_validation(test_size=0.1):
    pts_train  = {}
    pts_test = {}

    seed = None   #Truly random
    
    for dataset in chosen_datasets:
      
        pts_train[dataset], pts_test[dataset] = train_test_split(patients_all[dataset], random_state=seed, test_size=test_size)    
 
    return pts_train, pts_test

#Patient splitting to make a training, validation and testing set
#Currently not used in MI pipeline because patient specific fold assignments is
#taken from MI dataset directly
#Used for AFIB inter-patient classification currently

def split_by_patient(test_size=0.1):
    pts_train  = {}
    pts_valid = {}
    pts_test = {}
    val_split = 0.1
 
    #  seed = 4   #Same split of patients everytime

    seed = None   #Truly random
    
    for dataset in chosen_datasets:
        pts_remaining = []
        pts_remaining, pts_test[dataset] = train_test_split(patients_all[dataset], random_state=seed, test_size=test_size)    
        ratio_remaining = 1 - test_size
        ratio_val_adjusted = val_split / ratio_remaining
        pts_train[dataset], pts_valid[dataset] = train_test_split(pts_remaining, random_state=seed, test_size=ratio_val_adjusted)

    return pts_train, pts_valid, pts_test


