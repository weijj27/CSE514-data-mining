import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.pyplot import MultipleLocator

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


def knn_classification(X_train, X_test, y_train, y_test):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)

    train_preds = knn_model.predict(X_train)    
    mse = mean_squared_error(y_train, train_preds)
    print(sqrt(mse)) 

    #Predict test data set.
    test_preds = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    print(sqrt(mse))

    #Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    #Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))
    return


def knn_tune_classification(X_train, X_test, y_train, y_test,best_neighbour, best_algorithm):
    knn_model = KNeighborsClassifier(n_neighbors=best_neighbour, algorithm=best_algorithm)
    knn_model.fit(X_train, y_train)

    train_preds = knn_model.predict(X_train)    
    mse = mean_squared_error(y_train, train_preds)
    print(sqrt(mse)) 

    #Predict test data set.
    test_preds = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    print(sqrt(mse))

    #Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    #Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))
    return


 # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
def knn_tune(X_train, X_test, y_train, y_test):
    # hyperparameters that we want to tune.
    n_neighbors = [3, 5, 7, 9, 11]
    algorithm = ['brute', 'ball_tree', 'kd_tree']

    # convert to dictionary
    hyperparameters = dict(n_neighbors=n_neighbors, algorithm=algorithm)

    # create new KNN object
    knn_model = KNeighborsClassifier()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(knn_model, hyperparameters, cv=5)

    # Fit the model
    best_model = clf.fit(X_train,y_train)

    # display all result
    result = pd.DataFrame(best_model.cv_results_)[['mean_test_score', 'params']]
    print(result)

    # Print The value of best Hyperparameters
    best_algorithm = best_model.best_estimator_.get_params()['algorithm']
    best_neighbour = best_model.best_estimator_.get_params()['n_neighbors']
    best_score = best_model.best_score_
    print('Best algorithm:', best_algorithm)
    print('Best n_neighbors:', best_neighbour)

    # graph result
    graph_result(result, best_algorithm, best_neighbour, best_score)

    '''
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print(roc_auc_score(y_test, y_pred))
    print()
    '''
    return best_algorithm, best_neighbour


def createLable(result):
    xLable = []
    for index in range(len(result['params'])):
        xLable.append(str(result['params'][index]))
    return xLable


def graph_result(result, best_algorithm, best_neighbour, best_score):
    # plot title
    plt.figure(figsize=(13,7))
    str1 = str2 = str3 = ""
    str1 = "Best algorithm:" + str(best_algorithm)
    str2 = "Best n_neighbors:" + str(best_neighbour)
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

    y_major_locator = MultipleLocator(0.005)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)

    # show a legend on the plot
    plt.legend()
    plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=None)
    plt.tight_layout()

    plt.show()
    return



if __name__ == '__main__':
    # import data
    #fileName = 'dataset_M_V.csv'
    #filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/dataset/'
    
    fileName = 'dataset_M_V_rf.csv'
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/knn/knn_reduction/'
    X, y = import_data(fileName, filePath)

    # set 10% as validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # knn classify
    start = time.time()
    knn_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("knn classify time before tune:", end - start)

    # tune knn
    best_algorithm, best_neighbour = knn_tune(X_train, X_test, y_train, y_test)

    # tune result
    start = time.time()
    knn_tune_classification(X_train, X_test, y_train, y_test, best_neighbour, best_algorithm)
    end = time.time()
    print("knn classify time after tune:", end - start)

    print(len(X))
    print(len(y))
    # https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f
    # https://realpython.com/knn-python/
    # https://sijanb.com.np/posts/how-to-tune-hyperparameter-in-k-nearest-neighbors-classifier/



