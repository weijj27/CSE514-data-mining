import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib.pyplot import MultipleLocator
from sklearn.decomposition import PCA
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


def random_forest_classification(X_train, X_test, y_train, y_test):
    # create model
    random_forest = RandomForestClassifier()

    # train data
    random_forest.fit(X_train, y_train)

    # predict
    test_preds = random_forest.predict(X_test) 

    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return 




def random_forest_tune_classification(X_train, X_test, y_train, y_test, n_estimators, max_depth):
    # create model
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # train data
    random_forest.fit(X_train, y_train)

    # predict
    test_preds = random_forest.predict(X_test) 

    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return 

# https://scikit-learn.org/0.16/modules/generated/sklearn.ensemble.RandomForestRegressor.html
def random_forest_tune(X_train, X_test, y_train, y_test):
    n_estimators = [10, 11, 12, 13, 14]
    max_depth = [100, 110, 120, 130, 140]

    # convert to dictionary
    hyperparameters = dict( max_depth=max_depth, n_estimators=n_estimators)

    # create new decision object
    random_forest = RandomForestClassifier()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(random_forest, hyperparameters, cv=5)

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
    print('Best  n_estimators:', best_n_estimators)
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

    y_major_locator = MultipleLocator(0.005)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)

    # show a legend on the plot
    plt.legend()
    plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=None)
    plt.tight_layout()

    plt.show()
    return


def PCA_Reduction(fileName, filePath):
    filePath = filePath+fileName
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    pca = PCA(n_components=4)
    pca.fit(X)
    X_4 = pca.transform(X)
    return X_4, y



if __name__ == '__main__':
    # import data
    fileName = 'dataset_M_V.csv'
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/dataset/'
    
    X, y = import_data(fileName, filePath)

    # PCA reduction
    X, y = PCA_Reduction(fileName, filePath)

    # set 10% as validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # random forest
    start = time.time()
    random_forest_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("random forest before tune:", end - start)

    # tune decision tree
    best_n_estimators, best_max_depth = random_forest_tune(X_train, X_test, y_train, y_test)

    # tune result
    start = time.time()
    random_forest_tune_classification(X_train, X_test, y_train, y_test, best_n_estimators, best_max_depth)
    end = time.time()
    print("random forest after tune:", end - start)

    print(len(X))
    print(len(y))
    # https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
    # https://medium.com/swlh/random-forest-from-model-building-to-hyperparameter-tuning-in-python-5d0c07a428eb
    # https://www.relataly.com/hyperparameter-tuning-with-grid-search/2261/#h-step-5-hyperparameter-tuning-a-classification-model-using-the-grid-search-technique

