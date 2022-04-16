import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
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


def decision_tree_classification(X_train, X_test, y_train, y_test):
    # create model
    decision_tree = DecisionTreeClassifier()

    # train data
    decision_tree.fit(X_train, y_train)

    # predict
    test_preds = decision_tree.predict(X_test) 

    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return 


def decision_tree_tune_classification(X_train, X_test, y_train, y_test,best_max_depth, best_min_samples_leaf):
    # create model
    decision_tree = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf= best_min_samples_leaf)

    # train data
    decision_tree.fit(X_train, y_train)

    # predict
    test_preds = decision_tree.predict(X_test) 

    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))

    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return 




# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
def decision_tree_tune(X_train, X_test, y_train, y_test):
    criterion = ['gini', 'entropy']
    max_depth = [11, 14, 16, 17, 8]
    min_samples_leaf = [4, 5, 6, 7, 8]

    # convert to dictionary
    hyperparameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    # create new decision object
    decision_tree = DecisionTreeClassifier()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # https://www.programcreek.com/python/example/91151/sklearn.model_selection.GridSearchCV
    clf = GridSearchCV(decision_tree, hyperparameters, cv=5)

    # Fit the model
    best_model = clf.fit(X_train,y_train)

    # display all result
    result = pd.DataFrame(best_model.cv_results_)[['mean_test_score', 'params']]
    print(result)


    # Print The value of best Hyperparameters
    best_max_depth = best_model.best_estimator_.get_params()['max_depth']
    best_min_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']
    best_score = best_model.best_score_
    print('Best max_depth:', best_max_depth)
    print('Best min_samples_leaf:', best_min_samples_leaf)
    print('Best score:', best_score)

    # graph result
    graph_result(result, best_max_depth, best_min_samples_leaf, best_score)

    return best_max_depth, best_min_samples_leaf

def createLable(result):
    xLable = []
    for index in range(len(result['params'])):
        xLable.append(str(result['params'][index]))
    return xLable


def graph_result(result, best_max_depth, best_min_samples_leaf, best_score):
    # plot title
    plt.figure(figsize=(13,7))
    str1 = str2 = str3 = ""
    str1 = "Best max_depth:" + str(best_max_depth)
    str2 = "Best min_samples_leaf:" + str(best_min_samples_leaf)
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
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/decision_tree/decision_tree_reduction/'
    X, y = import_data(fileName, filePath)

    # set 10% as validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # decision tree
    start = time.time()
    decision_tree_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("decision tree before tune:", end - start)

    # tune decision tree
    best_max_depth, best_min_samples_leaf = decision_tree_tune(X_train, X_test, y_train, y_test)

    # tune result
    start = time.time()
    decision_tree_tune_classification(X_train, X_test, y_train, y_test, best_max_depth, best_min_samples_leaf)
    end = time.time()
    print("decision tree after tune:", end - start)

    print(len(X))
    print(len(y))
    # https://www.section.io/engineering-education/hyperparmeter-tuning/
    # https://ai.plainenglish.io/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda

