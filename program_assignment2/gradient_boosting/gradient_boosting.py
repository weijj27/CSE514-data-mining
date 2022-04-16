from pickletools import read_string1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier  
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


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
def GBDT_classification(X_train, X_test, y_train, y_test):
    # create model
    model =  GradientBoostingClassifier()
    model.fit(X_train, y_train)
    # predict
    test_preds = model.predict(X_test) 
    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))
    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))
    return


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
def GBDT_tune(X_train, X_test, y_train, y_test):
    max_depth = [9, 8, 7, 6, 5]
    n_estimators = [270, 280, 290, 300, 310]

    hyperparameters = dict(max_depth=max_depth, n_estimators=n_estimators)
    # create new decision object
    model =  GradientBoostingClassifier()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(model, hyperparameters, cv=5)

    # Fit the model
    best_model = clf.fit(X_train,y_train)

    # display all result
    result = pd.DataFrame(best_model.cv_results_)[['mean_test_score', 'params']]
    print(result)

    # Print The value of best Hyperparameters
    best_max_depth = best_model.best_estimator_.get_params()['max_depth']
    best_n_estimators = best_model.best_estimator_.get_params()['n_estimators']
    best_score = best_model.best_score_
    print('Best max_depth:', best_max_depth)
    print('Best n_estimators:', best_n_estimators)
    print('Best score:', best_score)

    # graph result
    graph_result(result, best_max_depth, best_n_estimators, best_score)

    return best_max_depth, best_n_estimators



def createLable(result):
    xLable = []
    for index in range(len(result['params'])):
        xLable.append(str(result['params'][index]))
    return xLable


def graph_result(result, best_max_depth, best_n_estimators, best_score):
    # plot title
    plt.figure(figsize=(13,7))
    str1 = str2 = str3 = ""
    str1 = "Best max_depth:" + str(best_max_depth)
    str2 = "Best n_estimators:" + str(best_n_estimators)
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


def UMAP_Reduction(fileName, filePath):
    filePath = filePath + fileName
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    import umap
    umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=4)
    X_4 = umap_data.fit_transform(X)
    #print(X_4)
    return X_4, y


def GBDT_classification_retrain(X_train, X_test, y_train, y_test, max_depth, n_estimators):
    # create model
    model =  GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    # predict
    test_preds = model.predict(X_test) 
    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))
    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))
    return


if __name__ == '__main__':
    # import data
    fileName = 'dataset_M_V.csv'
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/dataset/'
    
    X, y = import_data(fileName, filePath)

    # UMAP reduction
    X, y = UMAP_Reduction(fileName, filePath)

    # set 10% as validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    start = time.time()
    GBDT_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("GBDT before tune:", end - start)

    best_max_depth, best_n_estimators = GBDT_tune(X_train, X_test, y_train, y_test)

    start = time.time()
    GBDT_classification_retrain(X_train, X_test, y_train, y_test, best_max_depth, best_n_estimators)
    end = time.time()
    print("GBDT after tune:", end - start)

    #https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
