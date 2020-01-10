from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

"""
data = [22,11,5,15,22,16,11,5,9]
ingredients = ['(10, 10, 10, 10, 10)', '(10, 20, 20, 10)',
              '(12, 12, 12)', '(4, 4)', '(4, 4, 4, 4)', '(5, 4, 4, 5)', '(5, 4, 5)', '(7, 7)', '(7, 7, 7, 7)']
"""

#data = [2,97,17]
#ingredients = ['logistic','relu','tanh']


#data=[45,46,25]
#ingredients = ['adaptive', 'constant', 'invscaling']


#data = [116]
#ingredients=['0.01']


data=[17,30,32,37]
ingredients = ['0.0001', '0.001', '0.01', '0.1']

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("Distribution of L2 regularization parameters in models with $R^2>0.97$")

plt.show()
