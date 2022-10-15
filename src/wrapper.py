# MLECG Experimental Platform
# Authors: Jarred Fisher, Shai Aarons, Joshua Rosenthal
# Since:   May, 2022

# Description:
# ------------
# Modular experimental platform for the training and testing of deep neural networks
# in the task of cardiovascular disease (CVD) detection.

# Current CVDs classifiable:
# --------------------------
# Atrial Fibrillation, Myocardial Infarction

# MLECG Feature List:
# -------------------
# Signal Filtering and data preprocessing, user input, logging,
# metric generation, contingency measures, cross-validation,
# error reporting, modular architecture development and extensive argument parsing.

# Powered by TensorFlow Machine Learning Library   

import warnings
import time
warnings.filterwarnings("ignore")

import utils

import datawrapper
from model import (generateModel)
from os import environ
import os.path
from train import trainModel
import sys

from predict import prediction

from tensorflow.keras.models import load_model
import argparse


preprocessing_options = ["none", "oversample", "normalize", "undersample"]

model_options = ["simple", "BiLSTM", "BiLSTM2", "BiLSTM_pool",
"LSTM", "LSTM_deep", "RNN", "1D_CNN", "1D_HYBRID", "LSTM_deep2",
 "LSTM_deep3", "LSTM_deep3_HighDrop", "LSTM_deep3_MedDrop", "LSTM_deep3_LowDrop" , "LSTM_deep3_LayerNorm"]


#This wrapper class is the user's gateway to the experimental platform.
#It handles all the user input (through the command line) and calls relevant methods from other classes
#The wrapper is setup to run through the entire pipeline of: 
#Data split generation, model creation, training, prediction and metric output

def main():

    args = parseArguments()

    num_sec = args.num_sec
    skip_train = args.skip_training
    batch_size = args.batch_size
    verbosity = args.verbosity
    learning_rate = args.learning_rate
    epochs = args.epochs
    test_size = args.test_size
    path_to_model = args.path_to_model
    data_path = args.data_path
    model_choice = args.model_choice
    initial_epoch = args.initial_epoch
    beat_split = args.beat_split_experiment
    resume_training = args.resume
    load_saved_train_val = args.load_saved_train_val
    disable_GPU = args.disable_GPU
  
    preprocessing_choices = args.preprocessing
   
   
    preprocessing_config = dict(oversample=False,undersample=False, normalize=False)

    #Build list of preprocessing steps requested (if any at all)
    if('none' not in preprocessing_choices):
        for i in preprocessing_choices:
            preprocessing_config[i] = True


    #When the model is a super simple model, CPU is better than GPU for training
    if (disable_GPU):
        environ["CUDA_VISIBLE_DEVICES"] = "-1"  #Disables GPU 

    #If user requests resuming training, then training is implied
    if(resume_training):    
        skip_train = False
    

    #If user requests resuming training or to skip training, then loading saved datasets is implied
    if(resume_training or skip_train):   
        load_saved_train_val=True

    #By default, patient split will be performed
    #But, if the user requests beat_splitting, patient_split will consequently be set to False
    patient_split = False if beat_split else True
    
    model = None

    modelName = path_to_model
    while('/' in modelName):                #Remove any preceding directory levels
        index = modelName.find('/')
        modelName = modelName[index+1:]
    if('backup_' in modelName):
        index = modelName.find('_')
        modelName = modelName[index+1:]     #Remove the backup_ prefix


    #If loading saved data splits is requested or deduced by other arguments passed in.
    if(load_saved_train_val):
        try:
        
            X_train,rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, threshold = utils.load_datasets(preprocessing_config)
                            
        except:
        
            sys.exit('Loading saved datasets failed.\nPlease ensure you have' +
                ' training and test splits saved if you have requested to load in sample data, are skipping training or resuming training.\n' + 
                'The program expects the splits to be saved as \"./trained_models/"saved_data_splits.npz\"\n' 
                )
        
        print('Training and Validation datasets successfully loaded from:' + 
            './trained_models/saved_data_splits.npz')
   
    #If new data splits need/ have been requested to be created
    else:

        #Ask user what type of disease they are looking to classify on (This currently assumes a binary classification)

        disease_choice = datawrapper.choose_datasets()

        X_train, rhythm_train, X_valid,rhythm_valid, X_test, rhythm_test, threshold = utils.build_datasets(test_size=test_size,
            data_path=data_path, num_sec=num_sec, preprocessing_config=preprocessing_config, disease_choice=disease_choice, patient_split=patient_split)
        
    

    if(not skip_train): #i.e. train the model
        
        model_config = dict(model_choice=model_choice, batch_size=batch_size,learning_rate=learning_rate,
                            dropout=0.25 )
            
        model = None

        #If user has requested to continue training a model (or from its backup)
        if (resume_training):

                                
            backupPath = 'trained_models/backup_'+modelName

            #If there is a backup available, present user with the option to continue training from that backup
            if(os.path.exists(backupPath)):
                print('A recent backup of this model was found at <' + backupPath+'>')

                modTimesinceEpoc = os.path.getmtime(backupPath)

                # Convert seconds since epoch to readable timestamp

                lastModified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
                print("It was last modified at:", lastModified )

                print('Would you like to continue from this backup?')
            
                inp = input('Please enter (Y)es to resume from this backup or (N)o to skip and use original model path given\n')
            

                if(utils.str2bool(inp)):
                    print('Resuming training with model from backup: '+ backupPath )
                    
                    path_to_model = backupPath
                else:
                    print('You requested not to use the backup. Retraining model at ' + path_to_model)
                               
            try:

                model = load_model(path_to_model)
            except IOError:
        
                sys.exit('Resuming training failed. Please check model path is correct')
        
            print('Model successfully loaded from: ' + path_to_model)
            print('Optimizer states:\n', str(model.optimizer.get_config()))
        
            print('Initial epoch set to: ' + str(initial_epoch))
            
        #If user not resuming training (i.e. building a fresh model)
        else:

            #Check if model exists. If it does, confirm with user that they wish to overwrite it or not    
            if(os.path.exists(path_to_model)):
                path_to_model = utils.overwriteCheck(path_to_model)


            #Build and compile model
            model = generateModel(X_train, model_config)
    
        
        path_to_model = trainModel(model=model, X_train=X_train,X_valid=X_valid, y_train = rhythm_train,
                y_valid = rhythm_valid, path_to_model=path_to_model, batch_size=batch_size, learning_rate= learning_rate,
                epochs=epochs, verbose=verbosity, inital_epoch=initial_epoch)
    

    y_train_preds_dense, y_test_preds_dense = prediction(path_to_model=path_to_model,
                                                        X_train=X_train, X_test= X_test, verbose=verbosity)

    utils.outputMetrics(y_train_preds_dense, y_test_preds_dense, rhythm_train, rhythm_test, modelName,threshold)


#This function processes all the possible arguments that can be passed through the command line
def parseArguments():

    parser = argparse.ArgumentParser(description='MLECG experimental Platform')
    parser.add_argument('data_path', type=str,
                        help='Path to data containing ECG data')

    parser.add_argument('path_to_model', type=str,
                        help='Path to model which you would like to load.'
                             ' If you opt to skip training or are resuming training, this path then implies a pre-existing model.'
                             ' If you are training a new model, this path will indicate a new model.'
                             ' Model path should follow the convention of \"./trained_models/<model>\"'
                             )

    parser.add_argument('-p', '--preprocessing', type=str.lower, nargs='+',  default="none", choices=preprocessing_options,
                        help='Specify what kind of preprocessing should be applied to the data'
                        ' Multiple preprocessing steps can be chained together where appropriate.'
                        ' Example: Normalizing data and also oversampling minority class.'
                        ' Allowed values are ' + ', '.join(preprocessing_options) + 
                        ' Default: None', metavar='')

    parser.add_argument( '-e', '--epochs', type=int, default=5,
                        help='Training Epochs. Value must be > 1 and < 1024.'
                        ' Default: 5')

    parser.add_argument('-i','--initial_epoch', type=int, default=0,
                        help='Epoch to start training from. Value must be > 0'
                        ' This should only be a different number if training was interupted'
                        ' and needs to start from a checkpoint'
                        ' Default: 0')

    parser.add_argument('-r', '--resume', action='store_true',
                        help='Flag that can be passed'
                        ' to resume training of a model. The program will look for a backup of a model based' 
                        ' on the path_to_model arg provided. There also needs to be saved '
                        ' training and testing data splits.'
                        )

    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Number between 0 and 1 specifying the learning rate for the model'
                        ' Default: 0.001')

    parser.add_argument('-s', '--skip_training', action='store_true',
                        help='Flag that can be passed'
                        ' to skip building and training the model. Thereby moving straight to predictions.'
                        ' It follows then that the model provided with the path_to_model arg' 
                        ' is already compiled and trained for at least one epoch. There are also needs to be saved '
                        ' training and testing data splits.'
                        )

    parser.add_argument('-v','--verbosity', type=int, default=1, choices=[0, 1],
                        help='Verbosity. Can choose either -v 1 for verbose'
                        ' output or -v 0 for quiet output'
                        ' Default: 1')

    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size. Value must be > 1 and < 1024.'
                        ' Default: 64')
 
    parser.add_argument('-m','--model_choice', type=str, default="simple", choices=model_options,
                        help='Specify particular type of model to build'
                        ' for this session. Note that this argument is irrelevant if you opt to skip training.'
                        ' Allowed values are ' + ', '.join(model_options) +
                        ' Default: simple (i.e. fully-connected dense model)', metavar='')

    parser.add_argument('-d','--disable_GPU', action='store_true',
                        help='Flag that can be passed to disable training with GPU.'
                        ' This is useful when training toy models as the GPU introduces memory overheads that'
                        ' can lead to longer training and predicting for such simple models.'
                        )

    parser.add_argument('-ls','--load_saved_train_val', action='store_true',
                        help='Flag that can be passed '
                        ' to load previously saved training and test data splits from disc.'
                        ' Note: If resuming training or skipping to predictions, data splits will always be loaded'
                        ' from disc to maintain integrity in the evaulation of the model.')

    parser.add_argument('-ts','--test_size', type=float, default=0.1,
                    help='Number between 0 and 1 determining how much of'
                            'the data is to be used for the test set. The remaining '
                            'is used for training and validation'
                            ' Default: 0.1')
                            
    parser.add_argument('-x','--beat_split_experiment', action='store_true',
                    help='Flag that can be passed to perform the beat splitting experiment'
                        'This is for creating the intra-patient classifier version of the models'
                        'i.e. Patient specific data will be allowed to leak between Training, validation and testing sets')

    parser.add_argument('-n', '--num_sec', type=float, default=0.4,
                        help='Number of seconds to include before and after the QRS complex.'
                        ' Default: 0.4')
  
    args, err = parser.parse_known_args()
    if err:
        warnings.warn("Unknown commands:" + str(err) + ".")

    return args


if __name__ == "__main__":
    main()




