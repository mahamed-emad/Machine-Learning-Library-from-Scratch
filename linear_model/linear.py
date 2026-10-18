import numpy as np

class Linear:
    def __init__(self, lr=0.001, batches=None, itr=1e4, seed=42):
        self.lr = lr
        self.seed = seed
        self.itr = int(itr)
        self.batches = batches
    
    def fit(self): pass

    def predict(self, x):
        x = np.hstack((np.full((x.shape[0], 1), 1), x))
        y_hat = np.dot(x, self.weights)
        return y_hat
    
    def score(self, x, y):
        y_hat = self.predict(x)
        error = np.sum((y_hat - y)**2)
        mean = np.sum((np.mean(y) - y)**2)
        r2 = 1 - (error/(mean + 1e-9))
        return r2