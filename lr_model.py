import numpy as np
class LR(object):
    def __init__(self,x_train,y_train):
        self.x_train=np.column_stack([np.ones(len(x_train)), x_train])
        self.y_train=y_train
    def predict(self,x_test):
        x_test = np.column_stack([np.ones(len(x_test)), x_test])

        w = np.linalg.inv(self.x_train.T @ self.x_train) @ (self.x_train.T @ self.y_train)
        y_pre=x_test@w
        return y_pre