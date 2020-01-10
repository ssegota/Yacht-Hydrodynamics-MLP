from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle
import uuid

hidden_layer_sizes_ = [(4,4,4,4),(4,4),(5,4,4,5),(5,4,5),(10,10,10,10,10), (5,), (7,7,7,7), (7,7), (10,20,20,10),(6,), (12,12,12), (100,)]
activation_ = ['relu','identity','logistic','tanh']
solver_ = ['adam']
learning_rate_ = ['constant','adaptive','invscaling']
learning_rate_init_ = [0.1,0.01,0.5, 0.00001]
alpha_ = [0.01,0.1,0.001, 0.0001]
max_iter_=[10000]

params = [[h_,a_,s_,lr_,lri_,al_,mi_] for h_ in hidden_layer_sizes_
                                      for a_ in activation_
                                      for s_ in solver_
                                      for lr_ in learning_rate_
                                      for lri_ in learning_rate_init_
                                      for al_ in alpha_
                                      for mi_ in max_iter_]

#for [hidden_layer_sizes_]

print("Number of parameter combs  in grid search = ", len(params))
data = np.genfromtxt('../data/yacht_hydrodynamics.csv', delimiter=',')

input_data = data[:,:-1]
print(input_data)
output_data = data[:, -1]
print(output_data)

output_data.sort()
x = np.arange(0, len(output_data), 1)

from matplotlib import pyplot as plt
plt.grid()
plt.hist(output_data, label="sorted output data")
plt.title("Histogram of the output variable values of the dataset")
plt.xlabel("residuary resistance per unit weight of displacement")
plt.show()
input_train, input_test, output_train, output_test = train_test_split(
    input_data, output_data)
#print(output_data)

for p in params:
    h=p[0]
    a = p[1]
    s = p[2]
    lr = p[3]
    lri = p[4]
    al = p[5]
    mi = p[6]
    for i in range(10):
        model = MLPRegressor(hidden_layer_sizes=h,
                            activation=a,
                            solver=s,
                            learning_rate=lr,
                            learning_rate_init=lri,
                            alpha=al,
                            max_iter=mi
                            )

        model.fit(input_train, output_train)

        output_predicted = model.predict(input_test)
        r2 = metrics.r2_score(output_test, output_predicted)
        print("r2 = % 0.2f " % r2, end="\r")
        if r2>0.97:
            uuid_=uuid.uuid4()
            file = open(str(r2)+"-params-"+str(uuid_)+".txt", 'w')
            file.write("hidden_layer_sizes="+str(h)+"\n")
            file.write("activation="+str(a)+"\n")
            file.write("solver="+str(s)+"\n")
            file.write("learning_rate="+str(lr)+"\n")
            file.write("learning_rate_init="+str(lri)+"\n")
            file.write("alpha="+str(al)+"\n")
            file.write("max_iter="+str(mi)+"\n")
            file.write("\n\nr2="+str(r2))
            file.close()

            pickle.dump(model, open(str(r2)+"-model" +
                                    str(uuid_) +".pickle", 'wb'))



