import pandas as pd
from math import sqrt, pi, exp
import numpy as np

class GaussianNB:
    
    def __init__(self):
        
        pass
    
    def separate_by_class(self, X, Y):
        self.classes, self.class_count = np.unique(Y, return_counts=True)
        self.columns = list(X.columns)
        self.instance_count = len(Y.index) 
        data_given_class = X.groupby(Y)
        return data_given_class
    
    def fit(self, X, Y):
        data_given_class = self.separate_by_class(X, Y)
        classes = self.classes
        self.means = {}
        self.std = {}
        self.class_prior = {}
        
        for i in range (len(classes)):
            self.means[classes[i]] = data_given_class.get_group(classes[i]).apply(lambda x: sum(x)/float(len(x)))
            self.std[classes[i]] = data_given_class.get_group(classes[i]).apply(lambda x: sqrt(sum((x - sum(x)/float(len(x)))**2)/ 
                                                                                     float(len(x)-1)))
            self.class_prior[classes[i]] = self.class_count[i]/float(self.instance_count)
        return self.means, self.std, self.class_prior
    
    def gaussian(self, X, mean, stdev):
        exponent = float(exp(-((X - mean) ** 2 / (2 * stdev ** 2))))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
        
#         cls = dataset.groupby(dataset.iloc[:,-1])
    
    def predict(self, X):
        X.reset_index(inplace = True)
        y = []
        classes = self.classes
        mean = self.means
        std = self.std
#         probabilities = dict()
        for row in range(len(X.index)):
            max_prob = 0
            pred_class = 0
            for i in range(len(classes)):
                cls = classes[i]
                p = self.class_prior[cls] 
                for j in self.columns:
                    p = p * self.gaussian(X.loc[row,j], mean[cls][j], std[cls][j])
                if p > max_prob:
                    max_prob = p
                    pred_class = cls
#                 probabilities[cls] = p
#                 if probabilities[cls] > max_prob:
#                     max_prob = probabilities[cls]
#                     pred_class = cls
            y.append(pred_class) 
        return y