from sklearn.metrics import (classification_report, f1_score , confusion_matrix,  roc_auc_score, accuracy_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from datetime import datetime

labels = ['Normal', 'MI']

def summaryReport(y_actual, y_pred, thresh, stage):

    print(stage + " stage")
    
    print(classification_report(y_actual, (y_pred>thresh), target_names=labels))

#Print metrics for model performance
#Currently setup only for Binary classfication
def print_report(y_actual, y_pred, thresh, modelName, stage):

    print('Metrics for model: ' + modelName)
    print(stage + " stage")
    print('Threshold:', thresh)
    
    auc = roc_auc_score(y_actual, y_pred)

    accuracy = accuracy_score(y_actual, (y_pred>thresh))

    recall = recall_score(y_actual, (y_pred>thresh))
    f1 = f1_score(y_actual, (y_pred>thresh))
    precision = precision_score(y_actual, (y_pred>thresh))
    
    specificity = recall_score(y_actual, (y_pred > thresh), pos_label=0)
    
    print('AUC:', auc)
    print('accuracy:' ,accuracy)
    print('recall:' , recall)
    print('specificity:' , specificity)
    print('precision:' , precision)
    print('F1:' , f1)
    print('')

    return  auc, accuracy, recall, specificity,  precision ,f1

#This function is needed when using softmax loss function (i.e. multiclass)
#Not used in this study because binary classification was performed

def classPrediction(arr):

    class_predictions = []
    for x in arr:
        class_predictions.append(np.argmax(x))

    return class_predictions

#Plot and save confusion matrix for model predictions
def confusionMatrix(y_actual, y_pred, modelName, stage, thresh):

    plt.figure('conf_matrix')

    sb.heatmap(confusion_matrix(y_actual, (y_pred>thresh), normalize='true'),
    annot = True, xticklabels = labels, yticklabels = labels, cmap = 'summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(stage + ' : ' + modelName)

    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    plt.savefig('./output/confusion_matrix' + stage + '_' + modelName + '_' + date + '.jpg')
    plt.show()

    