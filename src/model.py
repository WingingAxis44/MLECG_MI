
from tensorflow.keras.layers import (MaxPooling1D,
    Input, add, Dropout, BatchNormalization,LayerNormalization, GlobalMaxPooling1D,
    TimeDistributed, Activation, Add, SimpleRNN , Bidirectional, LSTM, Flatten, Dense, Conv1D)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential


num_classes = 1
output_activation_fn = 'sigmoid' if num_classes == 1 else 'softmax'

def simpleSequentialModel(X_train, config):

  model = Sequential()
  
  model.add(Dense(64, input_shape=(X_train.shape[1],)))
  model.add(Activation('relu'))
  model.add(Dropout(rate=0.4))

  model.add(Dense(32))
  model.add(Activation('relu'))
  model.add(Dropout(rate=config['dropout']))

  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dropout(rate=config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model, config['learning_rate'])

def simple1D_CNN(X_train, config):

  
  model = Sequential()
  model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (X_train.shape[1],X_train.shape[2])))
  model.add(Dropout(rate=config['dropout']))
  model.add(Flatten())
  model.add(Dense(num_classes,activation=output_activation_fn))

  return  compileModel(model,config['learning_rate'])

def LSTM_deep3_LowDrop(X_train, config):

  model = Sequential(name = 'LSTM_deep3_LowDrop')
  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(0.1))
  
  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.1))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.1))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.1))
  
  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])


def LSTM_deep3_MedDrop(X_train, config):

  model = Sequential(name = 'LSTM_deep3_medDrop')
  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(0.5))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.5))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])


def LSTM_deep3_LayerNorm(X_train, config):

  model = Sequential(name = 'LSTM_deep3_layerNorm')

  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(LayerNormalization()) 
  model.add(Dropout(config['dropout']))

  
  model.add(LSTM(100,return_sequences=True))
  model.add(LayerNormalization())
  model.add(Dropout(config['dropout']))

  model.add(LSTM(100,return_sequences=True))
  model.add(LayerNormalization())
  model.add(Dropout(config['dropout']))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(LayerNormalization())
  model.add(Dropout(config['dropout']))
  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])



def LSTM_deep3_HighDrop(X_train, config):

  model = Sequential(name = 'LSTM_deep3_HighDrop')

  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(0.7))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.7))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.7))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.7))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])


def LSTM_deep3(X_train, config):

  model = Sequential(name = 'LSTM_deep3')
  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(config['dropout']))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(config['dropout']))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(config['dropout']))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(config['dropout']))
  
  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def LSTM_deep2(X_train, config):

  model = Sequential(name = 'LSTM_deep2')
  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(config['dropout']))

  model.add(LSTM(100,return_sequences=False))
  model.add(Dropout(config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def LSTM_deep(X_train, config):

  model = Sequential(name = 'LSTM_deep')
  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(config['dropout']))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(config['dropout']))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def simpleRNN_deep(X_train, config):

  model = Sequential(name = 'simpleRNN_deep')
  model.add(SimpleRNN(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(config['dropout']))

  model.add(SimpleRNN(100,return_sequences=True))
  model.add(Dropout(config['dropout']))

  model.add(SimpleRNN(100,return_sequences=True))
  model.add(Dropout(config['dropout']))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def BiLSTM_pool(X_train, config):

  model = Sequential(name = 'BiLSTM_pool')

  model.add(Bidirectional(LSTM(128,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2]))))
  model.add(GlobalMaxPooling1D()) 
  model.add(Dropout(config['dropout']))

  model.add(Dense(50, activation='relu'))
  model.add(Dropout(config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def BiLSTM(X_train, config):

  model = Sequential(name = 'BiLSTM')
  model.add(Bidirectional(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2]))))
  model.add(Dropout(config['dropout']))
  model.add(Bidirectional(LSTM(100,return_sequences=True)))
  model.add(Dropout(config['dropout']))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(config['dropout']))
  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def BiLSTM2(X_train, config):

  model = Sequential(name = 'BiLSTM2')
  model.add(Bidirectional(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2]))))
  model.add(Dropout(config['dropout']))
  model.add(Bidirectional(LSTM(100,return_sequences=False)))
  model.add(Dropout(config['dropout']))
  
  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])

def hybrid1D(X_train=None, X_valid=None, metrics = ['accuracy'],  lr = 0.001):

  model = Sequential()
  model.add(Conv1D(filters = 60, kernel_size = 5, activation = 'relu', input_shape = (X_train.shape[1],1)))
  model.add(Conv1D(filters = 80, kernel_size = 3, activation = 'relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(rate=0.05))
  model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(rate=0.15))
  model.add(Bidirectional(LSTM(units=64, activation='tanh')))
  model.add(Flatten())
  model.add(Dense(1, activation=output_activation_fn))

  return compileModel(model,lr)

def hybrid1Dv2(X_train=None, X_valid=None, metrics = ['accuracy'],  lr = 0.001):

  model = Sequential()
  model.add(Conv1D(filters = 3, kernel_size = 20, strides = 1, use_bias=False, activation = 'relu', input_shape = (X_train.shape[1],1)))
  model.add(MaxPooling1D(pool_size=2, strides=2))
  model.add(Conv1D(filters = 6, kernel_size = 10, strides= 1, use_bias=False,  activation = 'relu'))
  model.add(MaxPooling1D(pool_size=2, strides= 2))
  model.add(Conv1D(filters = 6, kernel_size = 5, strides= 1, use_bias=False,  activation = 'relu'))
  model.add(MaxPooling1D(pool_size=2, strides= 2))

  model.add(Bidirectional(LSTM(units=10, activation='tanh')))
  model.add(Dropout(rate=0.2))
  model.add(Dense(20, activation='relu'))
  model.add(Dropout(rate=0.2))
  model.add(Dense(10, activation='relu'))
  model.add(Dropout(rate=0.2))
  model.add(Dense(num_classes, activation=output_activation_fn))

  return compileModel(model,lr)


def compileModel(model, lr):

 
  opt = Adam(lr)
  loss = 'binary_crossentropy' if num_classes==1 else 'sparse_categorial_crossentropy'

  model.compile(
  loss = loss,
  optimizer=opt, metrics=['accuracy'])
  

  return model

def generateModel(X_train, config):

  model = None
  
  if(config['model_choice']=="simple"):
          
    model = simpleSequentialModel(X_train, config)

  if(config['model_choice']=="BiLSTM"):
  
    model = BiLSTM(X_train, config)

  if(config['model_choice']=="BiLSTM2"):
  
    model = BiLSTM2(X_train, config)

  if(config['model_choice']=="BiLSTM_pool"):

    model = BiLSTM_pool(X_train, config)

  if(config['model_choice']=="1D_CNN"):
  
    model = simple1D_CNN(X_train, config)

  if(config['model_choice']=="LSTM_deep"):
  
    model = LSTM_deep(X_train, config)
  

  if(config['model_choice']=="LSTM_deep2"):
  
    model = LSTM_deep2(X_train, config)

  if(config['model_choice']=="LSTM_deep3"):
  
    model = LSTM_deep3(X_train, config)

  if(config['model_choice']=="LSTM_deep3_HighDrop"):

    model = LSTM_deep3_HighDrop(X_train, config)

  if(config['model_choice']=="LSTM_deep3_MedDrop"):

    model = LSTM_deep3_MedDrop(X_train, config)


  if(config['model_choice']=="LSTM_deep3_LowDrop"):

    model = LSTM_deep3_LowDrop(X_train, config)

  
  if(config['model_choice']=="LSTM_deep3_LayerNorm"):

    model = LSTM_deep3_LayerNorm(X_train, config)

  if(config['model_choice']=="1D_HYBRID"):

    model = hybrid1D(X_train, config)

  if(config['model_choice']=="RNN"):
  
    model = simpleRNN_deep(X_train, config)
  
  
  return model