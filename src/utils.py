
from scipy.stats import describe
from argparse import ArgumentTypeError
import os
import numpy as np
num_classes = 3

import preprocessing
import datawrapper

from generateMetrics import (print_report, summaryReport,confusionMatrix)

#Auxillary function for converting various representations of
#Yes/No values into a singular boolean value of True or False.

def str2bool(b):
    if isinstance(b, bool):
        return b
    if b.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

#This auxillary function informs the user that they will be overwritting a previous model
#if they continue.
#The function gives the user the option to provide a new model name, therefore avoiding overwritting.
def overwriteCheck(path_to_model):

    print('WARNING: A model already exists at the given path. Are you sure you want to overwrite it?')
    inp = input('Please enter (Y)es or (N)o\n')
    if(str2bool(inp)):
        print('Overwritting model.')
    else:
        inp = input('You requested not to overwrite the model. Please specify a new model:\n'+
        'Please provide just a model name, e.g: \"model_Foo\" \n'+
        'The final model and its backups will be saved in ./trained_models/\n')
        path_to_model = inp
        if( (os.path.exists(path_to_model)) or
        (os.path.exists('./trained_models/' + path_to_model)) ):
                overwriteCheck(path_to_model)
    return path_to_model

#Function for calculating and displaying statistics about the ECG data extracted
#Also determines the threshold used for converting between model prediction probabilites
#and the associated diagnostic class label

def dataStatistics(X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, disease_choice):

    _, countTrain = np.unique(rhythm_train, return_counts=True)
    _, countValid = np.unique(rhythm_valid, return_counts=True)
    _, countTest = np.unique(rhythm_test, return_counts=True)

    countNormal = countTrain[0] +countValid[0] +countTest[0]
    countDisease = countTrain[1] +countValid[1] +countTest[1]
    countTotal = sum(countTrain + countValid + countTest)

    from tabulate import tabulate

    disease = 'AFIB' if disease_choice == 1 else 'MI'

    data = [['Training', countTrain[0], countTrain[1], sum(countTrain)], ['Validation', countValid[0], countValid[1], sum(countValid)],
    ['Testing', countTest[0], countTest[1], sum(countTest)], ['TOTAL', countNormal, countDisease, countTotal]]
    print('\nSample size splits:\n')
    print (tabulate(data, headers=["Stage", "Normal", disease, "TOTAL"], tablefmt="github"))

    for lead in range(X_train.shape[2]):
        print('\nLead: ', lead)
        print('---------')
        
        trainStats = describe(X_train[:,:,lead], nan_policy='omit',axis = None)
        validStats = describe(X_valid[:,:,lead], nan_policy='omit',axis = None)
        testStats = describe(X_test[:,:,lead], nan_policy='omit',axis = None)


        data = [['Training', '({:.4f}, {:.4f})'.format(*trainStats.minmax), trainStats.mean, trainStats.variance, trainStats.kurtosis, trainStats.skewness ],
                ['Validation', '({:.4f}, {:.4f})'.format(*validStats.minmax), validStats.mean, validStats.variance, validStats.kurtosis, validStats.skewness ], 
                ['Testing', '({:.4f}, {:.4f})'.format(*testStats.minmax), testStats.mean, testStats.variance, testStats.kurtosis, testStats.skewness]]
        print('\nSummary Statistics:\n')
        print (tabulate(data, tablefmt="github", headers=["Stage","(Min, Max)", "Mean", "Variance", "Kurtosis", "Skewness"], floatfmt=".4f"))
        print('')

    threshold = min(countDisease / countTotal, 0.5)

    return threshold

def build_datasets(test_size , num_sec,  data_path, preprocessing_config, disease_choice, patient_split=True):

#Split dataset according to a training:validation ratio.
    
    #AFIB
    if (disease_choice==1):

 
        count = 1
    
        while True:
            print('Sampling Data... Attempt ' + str(count))
            pts_train, pts_valid, pts_test = datawrapper.split_by_patient(test_size=test_size)
  
            #Build training dataset
            X_train, rhythm_train = datawrapper.make_dataset(
            samples=pts_train, num_sec=num_sec,  data_path=data_path)

            #Build validation dataset
            X_valid,  rhythm_valid = datawrapper.make_dataset(
                samples=pts_valid, num_sec=num_sec,  data_path=data_path)

            #Build test dataset
            X_test, rhythm_test = datawrapper.make_dataset(
                samples=pts_test, num_sec=num_sec,  data_path=data_path)
            
            count = count + 1
            if correctSplitRatio(X_train, rhythm_train,X_valid,rhythm_valid, X_test, rhythm_test, test_size):
                print('Data sampled effectively')
               
                break
        
    #MI
    else:
        X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test = datawrapper.make_dataset(
                samples=None, num_sec=num_sec,  data_path=data_path, patient_split=patient_split)
    
    np.savez_compressed('./trained_models/saved_data_splits', disease_choice=disease_choice,
                X_train=X_train, rhythm_train=rhythm_train, X_valid=X_valid,
                rhythm_valid=rhythm_valid, X_test=X_test, rhythm_test=rhythm_test)
    print('Data splits compressed and saved to: trained_models/saved_data_splits.npz')
  
    X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, threshold = load_datasets(preprocessing_config)
    
    return  X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, threshold


def load_datasets(preprocessing_config):
    data = np.load('./trained_models/saved_data_splits.npz')

    X_train = data['X_train']
    X_valid = data['X_valid']
    X_test = data['X_test']
    rhythm_train = data['rhythm_train']
    rhythm_valid = data['rhythm_valid']
    rhythm_test = data['rhythm_test']
    disease_choice = data['disease_choice']
    data.close()
    
    X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, threshold = transformData(X_train,rhythm_train, X_valid,rhythm_valid, 
    X_test, rhythm_test, preprocessing_config, disease_choice)
    return X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, threshold

#This function checks that the data splits that were created are representative of all classes
#and in the appropriate ratios
#Note: Only needed for AFIB datasets
def correctSplitRatio(X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, test_size):

    _, countTrain = np.unique(rhythm_train, return_counts=True)
    _, countValid = np.unique(rhythm_valid, return_counts=True)
    _, countTest = np.unique(rhythm_test, return_counts=True)

   

    if((len(countTrain) < num_classes) or (len(countValid) < num_classes) or(len(countTest) < num_classes) ):
        return False
    

    #Calculates the proportion of a class in the training vs validation set
    Normal_ratio_valid = countValid[0] / countTrain[0]
    Disease_ratio_valid = countValid[1] / countTrain[1]


    #Similarly for training vs test set
    Normal_ratio_test = countTest[0] / countTrain[0]
    Disease_ratio_test = countTest[1] / countTrain[1]

   

    return ( (X_train.shape[0] == len(rhythm_train)) and (X_valid.shape[0] == len(rhythm_valid))
    and(X_test.shape[0] == len(rhythm_test))and (Disease_ratio_valid > test_size) and (Normal_ratio_valid > test_size)
    and (Disease_ratio_test > test_size) and (Normal_ratio_test > test_size))

#This function applies non-filter type preprocessing steps to the ECG data splits
#Also prints out some useful summary statistics for the user

def transformData(X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, preprocessing_config, disease_choice):

  
    #This only changes the shape to be appropriate for Keras models.
    #It is a required pre-processing step regardless of the task
    
    X_train = np.expand_dims(X_train, axis = 2) if X_train.ndim == 2 else X_train
    X_valid = np.expand_dims(X_valid, axis = 2)  if X_valid.ndim == 2 else X_valid
    X_test = np.expand_dims(X_test, axis = 2)    if X_test.ndim == 2 else X_test


    print('Note: \"Other\" labelled ECG data dropped by default.')
    print('Window Size: ', X_train.shape[1])
    print('Number of leads used: ', X_train.shape[2])
    print('\nRaw data:')
    print('---------')
    threshold = dataStatistics(X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test,disease_choice)
    isPreProprocess = any(i for i in preprocessing_config.values())

    if(isPreProprocess):
        print('Preprocessing data...')

    
    if(preprocessing_config['normalize']):

        preprocessing.normalizeSegment(X_train)
        preprocessing.normalizeSegment(X_valid)
 

    if(preprocessing_config['oversample']):
      
        X_train, rhythm_train = preprocessing.oversample(X_train, rhythm_train )
        X_valid, rhythm_valid = preprocessing.oversample(X_valid, rhythm_valid)

    if(preprocessing_config['undersample']):
      
        X_train, rhythm_train = preprocessing.undersample(X_train, rhythm_train )
        X_valid, rhythm_valid = preprocessing.undersample(X_valid, rhythm_valid)

    if(isPreProprocess):
        print('\nAfter Preprocessing:')
        print('----------------------')

        print('Preprocessing Steps taken:')
        index = 1
        for key,value in preprocessing_config.items():
            if value:
                print(str(index) + ') ',key)
                index = index + 1

        dataStatistics(X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, disease_choice)
        print('Data Transformation complete!\n')
    else:
        print('\nNo preprocessing performed. Data distribution unchanged\n')

  
    
    return X_train,  rhythm_train, X_valid, rhythm_valid, X_test, rhythm_test, threshold

#Auxillary function for calling all the metric related functions from the generateMetrics class
def outputMetrics(y_train_preds_dense, y_test_preds_dense, rhythm_train, rhythm_test, modelName,threshold):


    print_report(rhythm_train, y_train_preds_dense, threshold, modelName, 'Train')
    print_report(rhythm_test, y_test_preds_dense, threshold, modelName,'Test')
    confusionMatrix(rhythm_train, y_train_preds_dense,modelName,'Train', threshold)
    confusionMatrix(rhythm_test, y_test_preds_dense,modelName,'Test', threshold)
    summaryReport(rhythm_train, y_train_preds_dense,threshold,'Train')
    summaryReport(rhythm_test, y_test_preds_dense, threshold, 'Test')