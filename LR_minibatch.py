import numpy as np
class MinibatchGD(object):
    def __init__(self,x_train,y_train,a,threshold,max_epochs,batch_size=20):
        self.batch_size=batch_size
        self.x_train = np.column_stack([np.ones(len(x_train)), x_train])
        self.y_train = y_train
        self.a = a
        self.threshold = threshold
        self.w = None
        self.max_epochs = max_epochs
    def fit(self):
        w = np.random.uniform(-0.01, 0.01, self.x_train.shape[1])
        w_prev = w.copy()
        epoch = 0
        while epoch < self.max_epochs:
            epoch += 1
            indices = np.random.choice(self.x_train.shape[0], size=self.batch_size, replace=False)
            gd = ((self.x_train[indices] @ w - np.array(self.y_train)[indices]).T @ self.x_train[indices]) / self.batch_size
            w = w - self.a * gd
            w_diff = np.linalg.norm(w - w_prev)
            if w_diff < self.threshold:
                break
            w_prev = w.copy()
        self.w = w
        return self

    def predict(self, x_test):
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        y_pre = x_test @ self.w
        return y_pre