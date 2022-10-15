import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

def prediction(path_to_model, X_train, X_test, verbose=1):
      
    model = None
    try:
        model = load_model(path_to_model, compile=False)
       
    except IOError:         
        sys.exit('Loading saved model failed. Please check model path is correct')
   
    model.summary()
 
    y_train_preds_dense = model.predict(X_train,verbose=verbose)
    y_test_preds_dense = model.predict(X_test,verbose=verbose)  

    output_path = "./output/"
    # Generate dataframe
    np.savetxt(output_path+"_dnnTrain.csv", y_train_preds_dense, fmt='%.4g', header='Training predictions\nProb(x) is positive class (i.e. 1)')

    # Generate dataframe
    np.savetxt(output_path+"_dnnTest.csv", y_test_preds_dense, fmt='%.4g',header='Test predictions\nProb(x) is positive class (i.e. 1)')

    print("Output predictions saved\n")

    return y_train_preds_dense, y_test_preds_dense