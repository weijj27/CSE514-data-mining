import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler
import time

def import_data(fileName, filePath):
    # import data
    filePath = filePath + fileName
    #df=pd.read_csv(filePath).to_numpy()
    df=pd.read_csv(filePath)

    # Separating out the features
    #X = df[:, :-1]# Separating out the target
    #y = df[:,-1]
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y


def svm_classification(X_train, X_test, y_train, y_test):
    # create model
    svm = SVC()

    # train data
    svm.fit(X_train, y_train)

    # predict
    test_preds = svm.predict(X_test) 

    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return 


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def svm_tune(X_train, X_test, y_train, y_test):
    kernel = ['linear', 'poly', 'rbf']
    max_iter = [50, 60, 70, 80, 90]

    # convert to dictionary
    hyperparameters = dict(kernel=kernel, max_iter=max_iter)

    # create new decision object
    svm = SVC()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(svm, hyperparameters, cv=5)

    # Fit the model
    best_model = clf.fit(X_train,y_train)

    # display all result
    result = pd.DataFrame(best_model.cv_results_)[['mean_test_score', 'params']]
    print(result)

    # Print The value of best Hyperparameters
    best_kernel = best_model.best_estimator_.get_params()['kernel']
    best_max_iter = best_model.best_estimator_.get_params()['max_iter']
    best_score = best_model.best_score_
    print('Best kernel:', best_kernel)
    print('Best max_iter:', best_max_iter)
    print('Best score:', best_score)

    # graph result
    graph_result(result, best_kernel, best_max_iter, best_score)

    return best_kernel, best_max_iter



def svm_retrain(X_train, X_test, y_train, y_test, best_kernel, best_max_iter):
    # create model
    svm = SVC(kernel=best_kernel, max_iter=best_max_iter)

    # train data
    svm.fit(X_train, y_train)

    # predict
    test_preds = svm.predict(X_test) 

    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return 



def createLable(result):
    xLable = []
    for index in range(len(result['params'])):
        xLable.append(str(result['params'][index]))
    return xLable


def graph_result(result, best_kernel, best_max_iter, best_score):
    # plot title
    plt.figure(figsize=(13,7))
    str1 = str2 = str3 = ""
    str1 = "Best kernel:" + str(best_kernel)
    str2 = "Best max_iter:" + str(best_max_iter)
    str3 = "Best score:" + str(best_score)
    s = str1 + "\n" + str2 + "\n" + str3
    plt.title(s)
    plt.xlabel('Hyperparameters')
    plt.ylabel('Scores')
    num = len(result['mean_test_score'])
    x1 = list(range(0,num))
    y1 = result['mean_test_score']
    plt.plot(x1, y1, color='blue', marker='o', linestyle='solid')

    labels = createLable(result)
    plt.xticks(x1, labels, rotation = 85)

    y11 = [ round(y,3) for y in y1 ]
    for a, b in zip(x1, y11):
        plt.text(a, b, b,  fontsize = 7)

    #y_major_locator = MultipleLocator(0.005)
    #ax = plt.gca()
    #ax.yaxis.set_major_locator(y_major_locator)

    # show a legend on the plot
    plt.legend()
    plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=None)
    plt.tight_layout()

    plt.show()
    return



def SVD_Redcution(fileName, filePath):
    filePath = filePath + fileName
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    from sklearn.decomposition import TruncatedSVD 
    X = StandardScaler().fit_transform(X)
    svd = TruncatedSVD(n_components=4, random_state=42)
    X_4 = svd.fit_transform(X)
    #print(X_4)
    return X_4, y




if __name__ == '__main__':
    # import data
    fileName = 'dataset_M_V.csv'
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/dataset/'
    
    X, y = import_data(fileName, filePath)

    # SVD reduction
    X, y = SVD_Redcution(fileName, filePath)

    # set 10% as validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # decision tree
    start = time.time()
    svm_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("SVM before tune:", end - start)

    # tune decision tree
    best_kernel, best_max_iter = svm_tune(X_train, X_test, y_train, y_test)

    # tune result
    start = time.time()
    svm_retrain(X_train, X_test, y_train, y_test, best_kernel, best_max_iter)
    end = time.time()
    print("SVM after tune:", end - start)
    print(len(X))
    print(len(y))
    # https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/


