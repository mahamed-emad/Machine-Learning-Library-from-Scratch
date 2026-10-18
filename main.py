import numpy as np
import pandas as pd
from linear_model import *
from optimizers import *
from regularizers import *
from model_selection import *

data = pd.read_csv("data.csv")
data = data.to_numpy(dtype=np.float32)
print(data.shape)
print(type(data))
x, y = data[:, :-1], data[:, -1:]
_x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
data = np.hstack((x, y))
model = Linear_Regression()
model.fit(x, y, optimizer=Adam(), regulizer=Lasso())
# print(model.predict(x))
print(model.score(x, y))
model = Kfold(base_model=Linear_Regression)
print(model.fit(x=x, y=y, optimizer=Adam(), regularizer=Lasso()))
model.fit(degree=1, x=x, y=y)
print(model.score(x, y))
print(model.predict(x))

x = np.full(shape=(5, 5), fill_value=np.arange(25).reshape(5, 5))
print(x)
print((x - np.mean(x, axis=0)))
print((x - np.mean(x, axis=1)))