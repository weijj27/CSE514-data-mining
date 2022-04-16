from asyncore import file_dispatcher
from turtle import back
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA


def backward(filePath ,isForward):
    data = pd.read_csv(filePath)

    X = data.drop(['Class'], axis=1)
    y = data['Class']

    from mlxtend.feature_selection import SequentialFeatureSelector as sfs
    from sklearn.linear_model import LinearRegression

    lreg = LinearRegression()
    sfs1 = sfs(lreg, k_features=4, forward=isForward, verbose=1, scoring='neg_mean_squared_error')

    sfs1 = sfs1.fit(X, y)

    feat_names = list(sfs1.k_feature_names_)
    print(feat_names)
    return


def randomForest(filePath):
    data = pd.read_csv(filePath)

    X = data.drop(['Class'], axis=1)
    y = data['Class']

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(X,y)
    features = data.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-4:]
    print([features[i] for i in indices])
    return


def decisionTree(filePath):
    data = pd.read_csv(filePath)

    X = data.drop(['Class'], axis=1)
    y = data['Class']

    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=1, max_depth=10)
    model.fit(X,y)
    features = data.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-4:]
    print([features[i] for i in indices])
    return


def LassoRedcution(filePath):
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
    ])

    search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

    search.fit(X_train,y_train)

    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    print(importance)
    return


def PCA_Reduction(filePath):
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    pca = PCA(n_components=4)
    pca.fit(X)
    X_4 = pca.transform(X)
    return X_4, y


def SVD_Redcution(filePath):
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    from sklearn.decomposition import TruncatedSVD 
    svd = TruncatedSVD(n_components=4, random_state=42)
    X_4 = svd.fit_transform(X)
    #print(X_4)
    return X_4, y


def UMAP_Reduction(filePath):
    data = pd.read_csv(filePath)
    features = data.columns
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    import umap
    umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=4)
    X_4 = umap_data.fit_transform(X)
    #print(X_4)
    return X_4, y


if __name__ == '__main__':
    fileName = 'dataset_M_Y.csv'
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment2/dataset/' + fileName

    # backward reduction
    #backward(filePath, False)

    # forward reduction
    #backward(filePath, True)

    # random forest
    #randomForest(filePath)

    # decision Tree 
    #decisionTree(filePath)

    # Lasso
    # https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
    LassoRedcution(filePath)

    # PCA
    #PCA_Reduction(filePath)

    # SVD
    #SVD_Redcution(filePath)

    # UMAP
    # UMAP_Reduction(filePath) 

    # https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/
    # https://medium.com/mlearning-ai/short-python-code-for-backward-elimination-with-detailed-explanation-52894a9a7880
    # https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/