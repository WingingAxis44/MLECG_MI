import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from datetime import datetime

from numpy import exp




def trainModel(model, X_train, y_train,  path_to_model, X_valid =None, y_valid=None, batch_size=32, epochs=5, verbose=1, learning_rate=0.01, appendCSV=False, inital_epoch=0):

    #The following path modifications is so that the models are saved correctly
    #Inside the trained_models folder

 
    modelName = path_to_model
   
        

    if(modelName[-1] == '/'): #Remove a trailing backslash if it exists
      modelName = modelName[:-1]

    while('/' in modelName):   #Remove directory specifications if specified. This is so models are forced into trained_models
      index = modelName.find('/')
   
      modelName = modelName[index+1:]
   
    if('backup_' in modelName):
      index = modelName.find('_')
      modelName = modelName[index+1:] #Remove the backup_ prefix
    
  

    if(X_valid.shape[0] == 0 or len(y_valid)==0):
      model.fit(X_train, y_train, callbacks=createCallBack(appendCSV, learning_rate,modelName), batch_size=batch_size, 
      initial_epoch=inital_epoch, epochs=epochs, verbose=verbose)
    else:
      model.fit(X_train, y_train, validation_data = (X_valid, y_valid),callbacks=createCallBack(appendCSV, learning_rate,modelName), batch_size=batch_size, 
      initial_epoch=inital_epoch, epochs=epochs, verbose=verbose)
    
  
    path_to_model = './trained_models/' + modelName
    model.save(path_to_model)

    return path_to_model



def createCallBack(appendCSV, learning_rate, modelName):

    
    callbacks = [EarlyStopping(
                              monitor ='val_loss',
                              patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                              min_delta=0.0001)]

    callbacks += [TensorBoard(log_dir='./logs/' + modelName + '/', histogram_freq=1),
                  CSVLogger('./logs/' + modelName + '/training.log', append=appendCSV)]  # Change append to true if continuing training

    callbacks += [ModelCheckpoint('./trained_models/backup_'+modelName), 
    ModelCheckpoint('./trained_models/best_'+modelName,  save_best_only=True)]

    return callbacks