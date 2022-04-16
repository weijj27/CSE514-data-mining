import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
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


def ann_classification(X_train, X_test, y_train, y_test):
    # create model
    ann = MLPClassifier()
    # train data
    ann.fit(X_train, y_train)
    # predict
    test_preds = ann.predict(X_test) 
    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))
    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return


# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
def ann_tune(X_train, X_test, y_train, y_test):
    #activation = ['lbfgs', 'sgd', 'adam']
    activation = ['identity', 'logistic', 'tanh', 'relu']
    max_iter = [1000, 1100, 1200, 1300, 1400]
    #max_iter = ['constant', 'invscaling', 'adaptive']

    # convert to dictionary
    hyperparameters = dict(activation=activation, max_iter=max_iter)
    # create new decision object
    ann = MLPClassifier()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(ann, hyperparameters, cv=5)

    # Fit the model
    best_model = clf.fit(X_train,y_train)
    # display all result
    result = pd.DataFrame(best_model.cv_results_)[['mean_test_score', 'params']]
    print(result)

    # Print The value of best Hyperparameters
    best_activation = best_model.best_estimator_.get_params()['activation']
    best_max_iter = best_model.best_estimator_.get_params()['max_iter']
    best_score = best_model.best_score_
    print('Best activation:', best_activation)
    print('Best max_iter:', best_max_iter)
    print('Best score:', best_score)

    # graph result
    graph_result(result, best_activation, best_max_iter, best_score)

    return best_activation, best_max_iter


def ann_retrain(X_train, X_test, y_train, y_test, best_activation, best_max_iter):
    # create model
    ann = MLPClassifier(activation=best_activation, max_iter=best_max_iter)
    # train data
    ann.fit(X_train, y_train)
    # predict
    test_preds = ann.predict(X_test) 
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


def graph_result(result, best_activation, best_max_iter, best_score):
    # plot title
    plt.figure(figsize=(13,7))
    str1 = str2 = str3 = ""
    str1 = "Best activation:" + str(best_activation)
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
        plt.text(a, b, b,  fontsize = 6)

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

    X = StandardScaler().fit_transform(X)
    import umap
    umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=4)
    X_4 = umap_data.fit_transform(X)
    #print(X_4)
    return X_4, y



if __name__ == '__main__':
    # import data
    fileName = 'dataset_M_V.csv'
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/dataset/'
    
    X, y = import_data(fileName, filePath) 

    # UMAP reduction
    X, y = UMAP_Reduction(fileName, filePath) 

    # set 10% as validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # decision tree
    start = time.time()
    ann_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("ANN before tune:", end - start)

    # tune decision tree
    best_activation, best_max_iter = ann_tune(X_train, X_test, y_train, y_test)

    # tune result
    start = time.time()
    ann_retrain(X_train, X_test, y_train, y_test, best_activation, best_max_iter)
    end = time.time()
    print("ANN after tune:", end - start)

    print(len(X))
    print(len(y))