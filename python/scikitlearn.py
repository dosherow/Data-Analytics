# sckikit-learn steps
# 1. choose a class of model by importing appropriate estimator class from scikit-learn
# 2. choose model parameters by instantiating this class with desired values
# 3. arrange data into a features matrix and target vector following discussion from before
# 4. fit the model to your data by calling the fit() method of the model instance
# 5. apply model to new data: for supervised learning, predict labels for unknown data
# using predict() method and for unsupervised learning, transform or infer properties of the data
# using transform() or predict() method.

import matplotlib.pyplot as plt
import numpy as np

# 1. import linear regression class
# 2. a class of model is not the same as an instance of a model
from sklearn.linear_model import LinearRegression

x = 10 * np.random.rand(50)
y = 2 * x - 1 + np.random.randn(50)

# 3. arrange the data
model = LinearRegression(fit_intercept=True)
X = x[:, np.newaxis]


# 4. fit model to the data
model.fit(X,y)
print(model.coef_)
print(model.intercept_)

# 5. predict labels for unknown data
xfit = np.linspace(-1, 11, num=50)
print(xfit.shape)

Xfit = xfit[:, np.newaxis]
print(Xfit.shape)

yfit = model.predict(Xfit)

plt.scatter(x,y)
plt.plot(xfit, yfit)
plt.show()
