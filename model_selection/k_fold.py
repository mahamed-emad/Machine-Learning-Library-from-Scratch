import numpy as np

class Kfold():
    def __init__(self, k=5, base_model=None, **kwargs):
        self.k = k
        self.kwargs = kwargs
        self.base_model = base_model

    def _statis_model(self, lst):
        mean = np.mean(lst)
        variance = np.var(lst)
        return mean, variance

    def _score(self, pred, y):
        error = np.sum((pred - y) ** 2)
        mean = np.sum((np.mean(y) - y) ** 2)
        return 1 - (error / (mean + 1e-9))

    def fit(self, x, y, **kwargs):
        r2 = []
        self.models = []
        num_test = x.shape[0] // self.k
        self.valid = np.zeros((x.shape[0], 1))

        idx = np.random.permutation(range(x.shape[0]))
        x, y = x[idx], y[idx]
        for i in range(self.k):
            start, end = num_test * i, num_test * (i + 1) if i != self.k - 1 else x.shape[0]
            idx = np.full(shape=x.shape[0], fill_value=True)
            idx[start:end] = False
            x_train, y_train = x[idx, :], y[idx, :]
            x_test, y_test = x[start: end, :], y[start: end, :]
            model = self.base_model(**self.kwargs)
            model.fit(x=x_train, y=y_train, **kwargs)
            self.models.append(model)
            self.valid[start:end, :] = model.predict(x_test).reshape(-1, 1)
            r2.append(model.score(x=x_test, y=y_test))
        return self._score(self.valid, y), self._statis_model(r2)

    def predict(self, x):
        lst = np.zeros((x.shape[0], 1))
        for i in range(self.k):
            lst += self.models[i].predict(x)
        lst /= self.k
        return lst

    def score(self, x, y):
        pred = self.predict(x)
        return self._score(pred, y)