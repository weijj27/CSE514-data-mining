import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
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

# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
def nb_classification(X_train, X_test, y_train, y_test):
    # create model
    bernoulli_nb =  BernoulliNB()
    bernoulli_nb.fit(X_train, y_train)
    # predict
    test_preds = bernoulli_nb.predict(X_test) 
    # Checking performance our model with classification report.
    print(classification_report(y_test, test_preds))
    # Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, test_preds))

    return

def nb_tune(X_train, X_test, y_train, y_test):
    alpha = [0.001, 0.01, 0.15, 0.45, 5.0]
    binarize = [2, 4, 5, 7, 8]
    # convert to dictionary
    hyperparameters = dict(alpha=alpha, binarize=binarize)
    # create new decision object
    bernoulli_nb =  BernoulliNB()

    # use GridSearch
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(bernoulli_nb, hyperparameters, cv=5)

    # Fit the model
    best_model = clf.fit(X_train,y_train)

    # display all result
    result = pd.DataFrame(best_model.cv_results_)[['mean_test_score', 'params']]
    print(result)

    # Print The value of best Hyperparameters
    best_alpha = best_model.best_estimator_.get_params()['alpha']
    best_binarize = best_model.best_estimator_.get_params()['binarize']
    best_score = best_model.best_score_
    print('Best alpha:', best_alpha)
    print('Best binarize:', best_binarize)
    print('Best score:', best_score)

    # graph result
    graph_result(result, best_alpha, best_binarize, best_score)

    return best_alpha, best_binarize


def nb_classification_retrain(X_train, X_test, y_train, y_test, alpha, binarize):
    # create model
    bernoulli_nb =  BernoulliNB(alpha = alpha, binarize=binarize)
    bernoulli_nb.fit(X_train, y_train)
    # predict
    test_preds = bernoulli_nb.predict(X_test) 
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


def graph_result(result, best_alpha, best_binarize, best_score):
    # plot title
    plt.figure(figsize=(13,7))
    str1 = str2 = str3 = ""
    str1 = "Best alpha:" + str(best_alpha)
    str2 = "Best binarize:" + str(best_binarize)
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



def PCA_Reduction(fileName, filePath):
    filePath = filePath + fileName
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    #X = StandardScaler().fit_transform(X)
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

    start = time.time()
    nb_classification(X_train, X_test, y_train, y_test)
    end = time.time()
    print("Naive Bayes before tune: ", end - start)

    best_alpha, best_binarize = nb_tune(X_train, X_test, y_train, y_test)

    start = time.time()
    nb_classification_retrain(X_train, X_test, y_train, y_test, best_alpha, best_binarize)
    end = time.time()
    print("Naive Bayes after tune:", end - start)

    # https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-naive-bayes
    # https://osswangxining.github.io/sklearn-classifier/