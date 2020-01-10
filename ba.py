#from BA_PB import *
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import pandas as pd
import numpy as np
import pickle
import uuid
from matplotlib import pyplot as plt

def BA(X,Y):
    xi=[]
    yi=[]
    di=[]
    mean=[]
    for i in range(len(X)):
        xi.append(X[i][0])
        yi.append(Y[i][0])
        di.append(X[i][0]-Y[i][0])
        mean.append((X[i][0]+Y[i][0])/2)

    
    d=np.mean(di)
    
    sd=np.std(di)
    LOAl=d-1.96*sd
    LOAu=d+1.96*sd
    print(sd)
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.scatter(mean, di,color='0.1',marker='.')
    ax.axhline(d,           color='black', linestyle='-')
    ax.axhline(LOAl, color='black', linestyle='--')
    ax.axhline(LOAu, color='black', linestyle='--')
    ax.grid(True)
    ax.xlabel('(X+Y)/2')
    ax.ylabel('X-Y')
    
    print('Srednja vrijednost razlika: '+str(d))
    print('Gornji limit: '+str(LOAu))
    print('Donji limit: '+str(LOAl))
    

def baPlotter(X,Y):
    #xVals = []
    #yVals = []
    di = []
    pi = []

    for i in range(len(X)):
        di.append(X[i]-Y[i])
        pi.append((X[i]+Y[i])/2)

    d=np.mean(di)
    sd = np.std(di)

    LOAl = d-1.96*sd
    LOAu = d+1.96*sd

    plt.figure()
    plt.grid()
    plt.scatter(pi,di, edgecolors='0.1', facecolors='none', marker='o')
    plt.axhline(d, color='r')
    plt.axhline(LOAl, label="Lower limit="+str(round(LOAl,3)), color='r', linestyle='--')
    plt.axhline(LOAu, label="Upper limit="+str(round(LOAu,2)), color='r', linestyle='-.')
    plt.legend()
    plt.title("Bland-Altman plot, $R^2$=" +
              str(round(metrics.r2_score(X,Y),2)))
    plt.xlabel("$(X+Y)/2$")
    plt.ylabel("$X-Y$")
    plt.show()

data = np.genfromtxt('../data/yacht_hydrodynamics.csv', delimiter=',')

input_data = data[:,:-1]
#print(input_data)
output_data = data[:, -1]
#print(output_data)
input_train, input_test, output_train, output_test = train_test_split(
    input_data, output_data)

model = pickle.load(
    open('0.9871453832268098-modelc413e75d-bee1-4d9c-ac8c-bfe8f155f97c.pickle', 'rb'))

output_predicted = model.predict(input_test)

print(metrics.r2_score(output_test, output_predicted))

baPlotter(output_test, output_predicted)
