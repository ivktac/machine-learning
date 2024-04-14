from sklearn.datasets import load_diabetes

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

import numpy as np
from matplotlib import pyplot as plt

data = load_diabetes()

X, y = data.data, data.target  # type: ignore
print(X.shape)

for Model in [Ridge, Lasso]:
    model = Model()
    print("%s: %s" % (Model.__name__, cross_val_score(model, X, y).mean()))


alphas = np.logspace(-3, -1, 30)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), X, y, cv=3).mean() for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc="lower left")
plt.xlabel("alpha")
plt.ylabel("cross validation score")
plt.tight_layout()
plt.show()
