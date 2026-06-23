import numpy as np
import random
class KNN(object):
    def __init__(self,n_neighbour,x_train,y_train):
        self.x_train=x_train
        self.n_neighbour=n_neighbour
        self.y_train=y_train
    def predict(self,x_test):
        X_=np.zeros((len(x_test),len(self.x_train),2))
        for index_x_test,col_x_test in enumerate(x_test):

            for index_x_train,col_x_train in enumerate(self.x_train):
                x_total=0

                for i in range(len(col_x_train)):
                    x_total=x_total+(col_x_train[i]-col_x_test[i])**2
                X_[index_x_test, index_x_train, 0]=x_total
                X_[index_x_test, index_x_train, 1]=self.y_train[index_x_train]
        sorted_X = np.array([X_[i, np.argsort(X_[i, :, 1])[::-1], :]
                       for i in range(X_.shape[0])])
        y_pre=[]
        for i in range(len(x_test)):
            top_n_col2 =  sorted_X[i, :self.n_neighbour, 1]
            unique, counts = np.unique(top_n_col2, return_counts=True)
            max_count = np.max(counts)
            modes = unique[counts == max_count]
            y_pre.append(random.choice(modes))
        y_pre=np.array(y_pre,dtype=int)
        return y_pre

