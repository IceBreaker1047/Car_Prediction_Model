import numpy as np
from word2number import w2n
import copy
import matplotlib.pyplot as plt
import pandas as pd

file_path = "A:\AIML\Cleaned_CarPrice_Assignment.csv"
df = pd.read_csv(file_path)

y_train = df["price"]
y_train = y_train.to_numpy()
df = df.drop(columns=["price","symboling","doornumber","cylindernumber","stroke","compressionratio","peakrpm"])

df_enconded = pd.get_dummies(df, columns=["fueltype","aspiration","carbody","drivewheel","enginetype","fuelsystem"],drop_first=True)

X_train = df_enconded.astype(float).to_numpy()

def z_score_normalization(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X-mu)/sigma
    return X_norm,mu,sigma

X_norm,mu,sigma = z_score_normalization(X_train)
print(X_norm)

def compute_cost(X,y,w,b):
    cost = 0.0
    m = X.shape[0]
    for i in range (m):
        f_wb_i = np.dot(w,X[i]) + b
        err = (f_wb_i - y[i])**2
        cost += err
    cost /= 2*m
    return cost

def compute_gradient(X,y,w,b):
    m = X.shape[0]
    n = X.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range (m):
        f_wb_i = np.dot(w,X[i]) + b
        err = (f_wb_i - y[i])
        dj_db += err
        for j in range (n):
            dj_dw[j] += err*X[i,j]
    dj_dw /= m 
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range (num_iters):
        dj_dw , dj_db  = gradient_function(X,y,w,b)
        tmp_w = w-alpha*dj_dw
        tmp_b = b-alpha*dj_db
        w = tmp_w
        b = tmp_b
        if i<100000: 
            J_history.append(cost_function(X,y,w,b))
        if i%1000 == 0:
            print(f"Iteration: {i:4d} Cost: {J_history[-1]:8.8f} dj_dw: {dj_dw} dj_db: {dj_db}")
    return J_history,w,b

n = X_norm.shape[1]
initial_w = np.zeros(n)
initial_b = 0
alpha = 0.2
iterations = 10000

J_hist,w_final,b_final = gradient_descent(X_norm,y_train,initial_w,initial_b,compute_cost,compute_gradient,alpha,iterations)

print(f"The value of w found is : {w_final} and the value of b foudn is {b_final}")

x = X_norm[2]
predict = np.dot(w_final,x)+b_final

print(predict)