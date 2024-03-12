import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (3, 2)

X = np.c_[0.5, 1].T
y = [0.5, 1]
X_test = np.c_[0, 2].T

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, y)

plt.plot(X, y, "o")
plt.plot(X_test, regr.predict(X_test))

np.random.seed(0)

for _ in range(6):
    noisy_X = X + np.random.normal(loc=0, scale=0.1, size=X.shape)
    plt.plot(noisy_X, y, "o")
    regr.fit(noisy_X, y)
    plt.plot(X_test, regr.predict(X_test))

regr = linear_model.Ridge(alpha=0.1)

np.random.seed(0)

for _ in range(6):
    noisy_X = X + np.random.normal(loc=0, scale=0.1, size=X.shape)
    plt.plot(noisy_X, y, "o")
    regr.fit(noisy_X, y)
    plt.plot(X_test, regr.predict(X_test))

plt.show()
