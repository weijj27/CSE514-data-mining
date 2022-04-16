from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def train_test_split(X, y, trainCount):
    X_train = X[:trainCount]
    y_train = y[:trainCount]
    X_test  = X[trainCount:]
    y_test  = y[trainCount:] 
    return X_train, X_test, y_train, y_test


def normalize(X):
    # https://zhuanlan.zhihu.com/p/49427706
    #Feature normalizing the columns (subtract mean, divide by standard deviation)
    stored_feature_means, stored_feature_stds = [], []
    Xnorm = X.copy()
    
    for icol in range(Xnorm.shape[1]):    
        # we do not normalize last colum b 
        stored_feature_means.append(np.mean(Xnorm[:,icol]))
        stored_feature_stds.append(np.std(Xnorm[:,icol]))
        #Skip the last column
        if icol == Xnorm.shape[1] - 1: continue
        #Faster to not recompute the mean and std again, just used stored values
        Xnorm[:,icol] = (Xnorm[:,icol] - stored_feature_means[icol])/stored_feature_stds[icol]
    return Xnorm, stored_feature_means, stored_feature_stds


def normalize_test(X, stored_feature_means, stored_feature_stds):
    #Feature normalizing the columns (subtract mean, divide by standard deviation)
    #stored_feature_means, stored_feature_stds = [], []
    Xnorm = X.copy()
    
    for icol in range(Xnorm.shape[1]):    
        #Skip the last column
        if icol == Xnorm.shape[1] - 1: continue
        if stored_feature_stds[icol] == 0: continue
        #Faster to not recompute the mean and std again, just used stored values
        Xnorm[:,icol] = (Xnorm[:,icol] - stored_feature_means[icol])/stored_feature_stds[icol]
    return Xnorm

def import_data(filePath):
    # importing data
    dataset = pd.read_excel(filePath)
    # get data
    dataset = dataset.iloc[:,0:9].values
    return dataset


def init_para(paraNum):
    # initial para matrix into 0
    return np.zeros(paraNum).reshape(-1,1)


def init_X(X):
    # add one column at the end of train input, initial 1
    row, col = X.shape
    X = np.concatenate((X, np.ones((row, 1))), axis=1)
    return X


def generate_poly_X(X_train_data, features): 
    X_poly_train = X_train_data
    row = X_train_data.shape[0]
    for i in range (features):
        for j in range(i, features):
            tmp = (X_train_data[:,i] * X_train_data[:,j]).reshape(-1,1)
            X_poly_train = np.concatenate((X_poly_train, tmp), axis=1)
    X_poly_train = np.concatenate((X_poly_train, np.ones((row, 1))), axis=1)
    return X_poly_train


def matrix_derivative(X, W, y):
    #https://vxy10.github.io/2016/06/25/lin-reg-matrix/
    return (W.T @ X.T @ X  - y.T @ X)

def response_var(mse, y):
    return 1 - mse / np.var(y)


def loss_function(y_train, paraMat, trainMat, size):
    diff = y_train - trainMat @ paraMat
    return np.sum(diff * diff) / size


def adjust_learning_rate(learnRate, mse, prevLoss):
    if (mse < prevLoss):
        return learnRate * 1.2
    else:
        return learnRate * 0.75


def plot(X_train, X, y, paraMat, index):
    plt.scatter(X_train, y, color = 'red', s=10)  #data for testing
    y_predict = X @ paraMat
    plt.plot(X_train, y_predict, color = 'blue') #regresion line
    #plt.title('Concrete compressive strength (Mpa) vs. Cement (kg/m^3)')
    plt.xlabel('Predictor')
    plt.ylabel('Response')
    plt.grid(True)
    #plt.savefig(f'images/{index}.jpg')
    plt.show()
    
    plt.clf()
    return 

def plot_feature_hist(X):
    plt.grid(True)
    dummy = plt.hist(X[:,0],label = 'feature 1')
    dummy = plt.hist(X[:,1],label = 'feature 2')
    dummy = plt.hist(X[:,2],label = 'feature 3')
    dummy = plt.hist(X[:,3],label = 'feature 4')
    dummy = plt.hist(X[:,4],label = 'feature 5')
    dummy = plt.hist(X[:,5],label = 'feature 6')
    dummy = plt.hist(X[:,6],label = 'feature 7')
    #dummy = plt.hist(X[:,7],label = 'feature 8')
    #plt.title('Feature Normalization Accomplished')
    plt.title('Feature With Normalization Accomplished')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    dummy = plt.legend()
    plt.savefig('images/with_norm.jpg')
    plt.show()
    return

def descendGradient(trainMat, paraMat, y_train, learnRate, maxStep):

    size = len(y_train)
    step = 0
    prevLoss = float('inf') 
    diff = float('inf') 

    while diff > 1e-18 and step < maxStep :
        # generate derivatives
        dervMat = matrix_derivative(trainMat, paraMat, y_train)
        # update parameters
        paraMat -= learnRate * dervMat.T / (size * 2)
        # calculate loss function 
        mse = loss_function(y_train, paraMat, trainMat, size)  
        '''
        if step % 1000 == 0:
            print(mse)
        '''
        diff = abs(mse - prevLoss)
        # adjust learning rate
        learnRate = adjust_learning_rate(learnRate, mse, prevLoss)

        step += 1
        prevLoss = mse
    print(step)
    return paraMat


def linear_regression_train(X_train_data, y_train_data, learnRate, maxStep, featureNum, index):    
    stored_feature_means, stored_feature_stds = [], []

    # data for each feature
    if (featureNum == 1):
        # uni-variate
        X_train = X_train_data[:,index].reshape(-1,1) 
    else:
        # multi-variate
        X_train = X_train_data[:,0:featureNum]

    #plot_feature_hist(X_train)
    #X_train, stored_feature_means, stored_feature_stds = normalize(X_train)
    
    y_train = y_train_data.reshape(-1,1)
    trainMat = init_X(X_train)

    # init parameter matrix
    paraNum = featureNum + 1 # add Constant b 
    paraMat = init_para(paraNum) 

    # regression
    paraMat = descendGradient(trainMat, paraMat, y_train, learnRate, maxStep) 

    # plot
    #if (featureNum == 1):
    #    plot(X_train, trainMat, y_train, paraMat,index) 

    calMetric(y_train, paraMat, trainMat)
    return paraMat, stored_feature_means, stored_feature_stds



def linear_regression_test(X_test_data, y_test_data, paraMat, featureNum, index, stored_feature_means, stored_feature_stds):
    # data for each feature
    if (featureNum == 1):
        # uni-variate
        X_test = X_test_data[:,index].reshape(-1,1) 
    else:
        # multi-variate
        X_test = X_test_data[:,0:featureNum]
    
    #X_test = normalize_test(X_test, stored_feature_means, stored_feature_stds)
    y_test = y_test_data.reshape(-1,1)
    testMat = init_X(X_test)
    calMetric(y_test, paraMat, testMat)
    return


def calMetric(y_observe, paraMat, X_observe):
    mse = loss_function(y_observe, paraMat, X_observe, len(y_observe))
    print("MSE: {mse}".format(mse = mse))
    y_predictor = X_observe @ paraMat
    diff = y_predictor - y_observe
    r_var = response_var(mse, y_observe)
    print("R_square: {rsp_var}".format(rsp_var = r_var))
    return


def univariate_regression(X_train_data, X_test_data, y_train_data, y_test_data, features, learnRate, maxStep):
    featureNum = 1   
    for index in range(features):
        print("================= Univariate Training ======================")
        paraMat, stored_feature_means, stored_feature_stds = linear_regression_train(X_train_data, y_train_data, learnRate, maxStep, featureNum, index)
        
        print("================= Univariate Testing =======================")
        linear_regression_test(X_test_data, y_test_data, paraMat, featureNum, index, stored_feature_means, stored_feature_stds)
    return


def multivariate_regression(X_train_data, X_test_data, y_train_data, y_test_data, features, learnRate, maxStep):
    featureNum = features # 8
    print("================= Multivariate Training ========================")
    paraMat, stored_feature_means, stored_feature_stds = linear_regression_train(X_train_data, y_train_data, learnRate, maxStep, featureNum, featureNum)

    '''
        paraMat = np.array([[ 9.80314185],
       [ 6.4218981 ],
       [ 3.88082546],
       [-5.34179487],
       [ 1.8037599 ],
       [-0.3075571 ],
       [-0.90168268],
       [ 0.11432099],
       [30.86172693]])
    '''

    print("================= Multivariate Testing =========================")
    linear_regression_test(X_test_data, y_test_data, paraMat, featureNum, featureNum, stored_feature_means, stored_feature_stds)
    return


def polynomial_regression(X_train_data, X_test_data, y_train_data, y_test_data, features, learnRate, maxStep):
    poly = 2
    featureNum = 45 #cal_features(poly, features)
    
    print("================= Polynomial Training ==========================")
    X_poly_train = generate_poly_X(X_train_data, features) 
    paraMat, stored_feature_means, stored_feature_stds = linear_regression_train(X_poly_train, y_train_data, learnRate, maxStep, featureNum, featureNum)
    
    print("================= Polynomial Testing ===========================")
    X_poly_test = generate_poly_X(X_test_data, features) 
    linear_regression_test(X_poly_test, y_test_data, paraMat, featureNum, featureNum, stored_feature_means, stored_feature_stds)
    return


if __name__ == '__main__':

    # import data
    filePath = '/Users/jjwei/Documents/WUSTL/class/data_mining/HW/program_assignment1/dataset/Concrete_Data.xls'
    dataset = import_data(filePath)

    # feature
    features = 8
    X = dataset[:,0:features]
    y = dataset[:,-1]

    # Split data into training data and test data
    trainCount = 900
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, trainCount)

    # Regression
    
    maxStep = 100000
    #learnRate = 0.00001 # with norm
    learnRate = 0.001 # without norm
    univariate_regression(X_train_data, X_test_data, y_train_data, y_test_data, features, learnRate, maxStep)

    #learnRate = 0.0001 # with norm
    learnRate = 0.0000000001 # without norm 
    multivariate_regression(X_train_data, X_test_data, y_train_data, y_test_data, features, learnRate, maxStep)

    polynomial_regression(X_train_data, X_test_data, y_train_data, y_test_data, features, learnRate, maxStep)



# Brief Description
# This project is written in Python. The input data is the CSV file which contains all the data we need. 
# The output will be the images we may need to be used to analyze and some metrics that we calculated. 
# By setting the correct path of the input CSV file, the program could be run just by a simple click of mouse.


