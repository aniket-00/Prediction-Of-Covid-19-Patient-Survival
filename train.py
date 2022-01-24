import numpy as np
import csv


def import_package():
    X=np.genfromtxt("train_X_pr.csv",delimiter=",",skip_header=1,dtype=np.float64)
    Y=np.genfromtxt("train_Y_pr.csv",delimiter=",",dtype=np.float64)

    return X,Y

def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def replace_with_mean(X):
    col_mean = np.nanmean(X, axis=0)
    a = np.where(np.isnan(X))
    X[a] = np.take(col_mean, a[1])
    
    return X

def feature_scaling(X):
    # mean normalisation

    col_indices = [2, 5]
    for i in col_indices:
        col = X[:, i]
        mean = np.mean(col)
        min = np.min(col)
        max = np.max(col)
        X[:, i] = (col-mean)/(max-min)

    
    return X


def correlation_matrix(X):
    num_var = len(X[0])
    m = len(X)
    cor_mat = np.ones((num_var, num_var))
    for i in range(num_var):
        for j in range(i, num_var):
            mean_i = np.mean(X[:, i])
            mean_j = np.mean(X[:, j])
            std_i = np.std(X[:, i])
            std_j = np.std(X[:, j])
            nume = np.sum((X[:, i]-mean_i)*(X[:, j]-mean_j))
            deno = m*(std_i)*std_j
            cor = nume/deno
            cor_mat[i][j] = cor
            cor_mat[j][i] = cor
    
    return cor_mat


def compute_cost(X, Y, W, b, Lambda):
    m = len(X)
    h = np.dot(X, W)+b
    A = sigmoid(h)
    cost = 1/m*(-np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A)))+Lambda*np.sum(np.square(W))/2)
    return cost

def gradient_descent(X,Y,W,b,Lambda):
    m=len(X)
    A=sigmoid(np.dot(X,W)+b)
    dz=A-Y
    dw = (1/m)*np.dot(X.T,dz)
    db = (np.sum(dz))/m

    return dw,db



def model(X,Y):
    Y = Y.reshape(len(Y), 1)
    w = np.zeros((X.shape[1], 1))
    b=0.1
    Lambda=5
    alpha = 0.1
    iterations = 1000
    for i in range(iterations):
        dw, db = gradient_descent(X, Y,w,b,Lambda)
        cost = compute_cost(X, Y, w, b, Lambda)
        w=w-alpha*dw
        b=b-alpha*db

    return w


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w', newline="") as weights_file:
       wr = csv.writer(weights_file)
       wr.writerows(weights)
       weights_file.close()


if __name__=="__main__":
    X, Y = import_package()
    X=replace_with_mean(X)
    X=feature_scaling(X)
    weights = model(X, Y)
    save_model(weights, 'WEIGHTS_FILE.csv')