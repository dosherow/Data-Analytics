# predicting whether the 50th portfolio return will be positive or non-positive.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import 1st cleansed CSV data file into python
df_y = pd.read_csv('10port.csv', sep=',', header=0).as_matrix()

#number of rows
nr_y = df_y.shape[0]
# print out nr_y
print(nr_y)
# number of columns
nc_y = df_y.shape[1]
# print out nc_y
print(nc_y)

# print out 1st row
# the 1st component is an empty cell and the rest 100 components are the names of
# the 100 portfolios
print(df_y[0, :])
# print out 1st column
# the 1st component is an empty cell and the rest 1259 components are the dates of
# the 1259 daily returns
print(df_y[:, 0])

# now define data matrix Y (1259 by 100) for prediction
# need to convert a data frame to a numeric matrix
Y = df_y[1:, 1:]
Y = Y.astype(np.float)
# the sample size (number of observations or data points)
n = Y.shape[0]
# the dimensionality (number of response or Y variables)
q = Y.shape[1]

# import 2nd cleansed CSV data file into python
df_x = pd.read_csv('5factors.csv', sep=',', header=0).as_matrix()

# number of rows
nr_x = df_x.shape[0]
# print out nr_x
print(nr_x)
# number of columns
nc_x = df_x.shape[1]
# print out nc_x
print(nc_x)

# print out 1st row
# the 1st component is an empty cell and the rest 6 components are the names of
# the 6 X variables
print(df_x[0, :])
# print out 1st column
# the 1st component is an empty cell and the rest 1259 components are the dates of
# the 1259 days of daily factors
print(df_x[:, 0])

# now define data matrix X (1259 by 6) for prediction
# need to convert a data frame to a numeric matrix
X_all = df_x[1:, 1:]
X_all = X_all.astype(np.float)

# the sample size (number of observations or data points)
n = X_all.shape[0]
# the dimensionality (number of predictor or X variables)
p_all = X_all.shape[1]

# preprocess the data
# consider the 50th portfolio return as the Y variable
y_id = 49
y = Y[:, y_id]
# consider the excess return of portfolio by subtracting risk-free
# interest rate (the RF column of X data matrix)
y = y - X_all[:, 5]
# consider only the 5 factors as the predictors (X variables)
X = X_all[:, 0:5]
# the sample size (number of observations or data points)
n = X.shape[0]
# the dimensionality (number of predictor or X variables)
p = X.shape[1]

# set up the data for classification
n = X.shape[0]
# add the constant column of 1's to data matrix X for the intercept term
X_aug = np.append(np.ones((n, 1)), X, axis=1)
# new dimensionality (p + 1)
p_aug = X_aug.shape[1]
# y is a column vector
# convert portfolio returns into 0/1's for classification
# 0 means return <= 0
# 1 means return > 0
y_cn = np.where(y > 0, 1, 0)

# we now implement the 2-layer MLP we designed on the slides from Week 10
# The hidden layer has m ReLU neurons and the output layer contains a
# sigmoid neuron for classification

# define the ReLU activation function
def relu(x):
    acti = np.maximum(0, x)
    return acti

# define the hidden layer of m ReLU neurons
def relu_layer(x, W):
    # x = (1, x1, ..., xp)^T is a (p+1) by 1 vector of random input (data point) and
    # W = [w_1, ..., w_m] is a (p+1) by m weight matrix with w_i = (w_{0i}, w_{1i}, ...,
    # w_{pi})^T for the hidden layer
    # aggregated signal through linear combination
    hidden = W.T.dot(x)
    # pass through ReLU activation
    acti = relu(hidden)
    return acti

# define the output layer of a sigmoid neuron
def sigmoid_layer(z, w_vec):
    # the input z = (1, z1, ..., zm)^T is the output from the hidden layer of m ReLU
    # neurons and w_vec = (w0, w1, ..., wm)^T is a (m+1) by 1 weight vector for the
    # sigmoid neuron
    # aggregated signal through linear combination
    hidden = z.T.dot(w_vec)
    # pass through sigmoid activation
    acti = np.exp(hidden) / (1 + np.exp(hidden))
    return acti

# define loss function ell.nn given by the 2-layer MLP
def ell_nn(X, y, W, w_vec):
    # sample size (number of rows)
    n = X.shape[0]
    # number of ReLU neurons
    m = W.shape[1]
    Z = np.zeros((n, m + 1))
    for i in range(0, n):
        # ith row of data matrix X as an input
        x = X[i,].reshape((X.shape[1], 1))
        # output of ReLU layer
        Z[i,] = np.append(np.array([[1]]), relu_layer(x, W), axis=0).reshape(Z.shape[1])
    # the logistic loss from the sigmoid layer (top layer)
    loss = (-y.T.dot(Z).dot(w_vec) + np.ones((1, n)).dot(np.log(1 + np.exp(Z.dot(w_vec))))) / n
    return loss

# To code up the SGD algorithm with mini-batch (more specifically, the backpropagation
# algorithm with mini-batch combined with gradient descent), we need to calculate the
# gradient for each module (neuron) of the neural network
# Define the gradient of the sigmoid layer (the top layer)
def sigmoid_layer_grad(z, y, w_vec):
    mu = np.exp(z.T.dot(w_vec)) / (1 + np.exp(z.T.dot(w_vec)))
    # gradient for sigmoid module (neuron)
    grad = (mu - y) * z
    return grad

# Define the gradient of the hidden layer of m ReLU neurons
def relu_layer_grad(x, W):
    # number of rows of weight matrix W
    n_row = W.shape[0]
    # number of ReLU neurons
    m = W.shape[1]
    # gradient matrix for W
    grad = np.zeros((n_row, m))

    for i in range(0, m):
        w_i = W[:, i].reshape(W.shape[0], 1)
        # gradient for ReLU module (neuron)
        grad[:, i] = max(0, np.asscalar(np.sign(x.T.dot(w_i)))) * x.reshape(grad.shape[0])
    return grad

# calculate the gradient of loss with respect to network weights using backpropagation
# algorithm (also known as automatic differentiation using the chain rule)
def ell_nn_grad(X, y_vec, W, w_vec):
    #Sample size(number of rows)
    n = X.shape[0]
    # number of columns of data matrix X which is n by (p+1)
    p_col = X.shape[1]
    # number of rows of weight matrix W
    n_row = W.shape[0]
    # number of ReLU neurons
    m = W.shape[1]
    # gradient matrix for W
    grad_W = np.zeros((n_row, m))
    # sum of gradients over training examples
    grad_W_sum = np.zeros((n_row, m))
    # gradient vector for w_vec
    grad_w_vec = np.zeros((m + 1, 1))
    # sum of gradients over training examples
    grad_w_vec_sum = np.zeros((m + 1, 1))

    for i in range(0, n):
        x = X[i,].reshape(X.shape[1], 1)
        y = y_vec[i]
        # define the output z = (1, z1, ..., zm)^T from the hidden layer of m ReLU neurons
        z = np.append(np.array([[1]]), relu_layer(x, W))
        z = z.reshape(z.size, 1)
        grad_w_vec = sigmoid_layer_grad(z, y, w_vec)
        grad_w_vec_sum = grad_w_vec_sum + grad_w_vec

        grad_W = relu_layer_grad(x, W)
        # apply the chain rule (walking through all the paths in the neural network starting
        # from the top (the loss layer) all the way down to that neuron (in a certain layer)
        # here * means entrywise product for matrices (as opposed to matrix product)
        grad_W = grad_W * (np.ones((p_col, 1)).dot(grad_w_vec[1:m + 1].reshape(1, m)))
        grad_W_sum = grad_W_sum + grad_W

    # average the gradients over the mini-batch
    grad_W = grad_W_sum / n
    grad_w_vec = grad_w_vec_sum / n
    # this function returns an object (a dictionary of items)
    obj = {"grad_W":grad_W,"grad_w_vec":grad_w_vec}
    return obj

# define the optimizer Adam (adaptive moment estimation) which is a very popular
# extension of the basic SGD
def opt_adam(m_W, m_w_vec, v_W, v_w_vec, grad_W, grad_w_vec, eta=0.02, gamma1=0.9, gamma2=0.999):
    # running averages of first two moments of recent gradients for the weights
    # m for first moment and v for second moment
    # eta is learning rate (step size)
    # gamma1 and gamma2 (between 0 and 1) are forgetting factors for gradients
    # and their second moments, respectively
    m_W = gamma1 * m_W + (1 - gamma1) * grad_W
    m_w_vec = gamma1 * m_w_vec + (1 - gamma1) * grad_w_vec
    v_W = gamma2 * v_W + (1 - gamma2) * grad_W * grad_W
    v_w_vec = gamma2 * v_w_vec + (1 - gamma2) * grad_w_vec * grad_w_vec

    # epsi is a small positive number
    epsi = 1e-8
    update_W = -eta * (m_W / (1.0 - gamma1)) / (np.sqrt(v_W / (1.0 - gamma2)) + epsi)
    update_w_vec = -eta * (m_w_vec / (1.0 - gamma1)) / (np.sqrt(v_w_vec / (1.0 - gamma2)) + epsi)

    # this function returns an object (a dictionary of items)
    obj = {"update_W":update_W,"update_w_vec":update_w_vec,"m_W":m_W,"m_w_vec":m_w_vec,"v_W":v_W,"v_w_vec":v_w_vec}
    return obj

# optimize the 2-layer MLP using the SGD algorithm with mini-batch and Adam
# set default mini-batch size (mbsize) to 100
# learning rate (step size) eta should not be too large
def ell_nn_opt_mb_adam(X, y, m, mbsize=100, eta=0.02, gamma1=0.9, gamma2=0.999, maxsteps=1000, tol=1e-4):
    # m is the number of ReLU neurons in the hidden layer
    # sample size (number of rows)
    n = X.shape[0]
    # dimensionality (with intercept added)
    p = X.shape[1]
    # random initialization of weights (each entry generated independently
    # from N(0, 1) distribution)
    W_ini = np.random.randn(p, m)
    w_vec_ini = np.random.randn(m + 1, 1)
    # optimal weights
    W_opt = W_ini
    w_vec_opt = w_vec_ini
    W_opt_old = W_opt
    w_vec_opt_old = w_vec_opt
    # record the paths of optimal weights and loss during the course of iterations
    W_path = W_opt.reshape(W_opt.shape[0], W_opt.shape[1], 1)
    w_vec_path = w_vec_opt
    ell_path = ell_nn(X, y, W_opt, w_vec_opt)
    # iteration count
    iter = 0
    # for early stopping (1e8 or a sufficently large number)
    update = 1e8
    # initialize m_W, m_w_vec, v_W, v_w_vec, grad_W, grad_w_vec for optimizer Adam
    m_W = np.zeros((p, m))
    m_w_vec = np.zeros((m + 1, 1))
    v_W = np.zeros((p, m))
    v_w_vec = np.zeros((m + 1, 1))
    grad_W = np.zeros((p, m))
    grad_w_vec = np.zeros((m + 1, 1))
    update_W = np.zeros((p, m))
    update_w_vec = np.zeros((m + 1, 1))

    while iter < maxsteps and update > tol:
        # take a random set of mbsize data points from 1 to n
        mb_ind = np.random.choice(n, mbsize, replace=False)
        X_mb = X[mb_ind,].reshape(mb_ind.size, p)
        y_mb = y[mb_ind,].reshape(mb_ind.size, 1)
        # stochastic gradient descent (SGD) with Adam
        # first calculate the gradients
        obj_ell_nn_grad = ell_nn_grad(X_mb, y_mb, W_opt, w_vec_opt)
        grad_W = obj_ell_nn_grad["grad_W"]
        grad_w_vec = obj_ell_nn_grad["grad_w_vec"]
        # then feed into Adam
        obj_opt_adam = opt_adam(m_W, m_w_vec, v_W, v_w_vec, grad_W, grad_w_vec, eta, gamma1, gamma2)

        update_W = obj_opt_adam["update_W"]
        update_w_vec = obj_opt_adam["update_w_vec"]
        m_W = obj_opt_adam["m_W"]
        m_w_vec = obj_opt_adam["m_w_vec"]
        v_W = obj_opt_adam["v_W"]
        v_w_vec = obj_opt_adam["v_w_vec"]

        # update the network weights
        W_opt = W_opt + update_W
        w_vec_opt = w_vec_opt + update_w_vec

        # use the relative change to check convergence for early stopping
        temp1 = np.linalg.norm(W_opt - W_opt_old) + np.linalg.norm(w_vec_opt - w_vec_opt_old)
        temp2 = max(1e-4, np.linalg.norm(W_opt_old) + np.linalg.norm(w_vec_opt_old))
        update = 1.0 * temp1 / temp2
        W_opt_old = W_opt
        w_vec_opt_old = w_vec_opt

        # add new weights and loss to the solution path after each iteration
        # append for combining multi-dimensional arrays
        W_path = np.append(W_path, W_opt.reshape(W_opt.shape[0], W_opt.shape[1], 1), axis=2)
        w_vec_path = np.append(w_vec_path, w_vec_opt, axis=1)
        ell_path = np.append(ell_path, ell_nn(X, y, W_opt, w_vec_opt))

        # keep track of loss during the course of iterations
        # print(np.append(iter, ell_nn(X, y, W_opt, w_vec_opt)))

        # increase iteration count by one
        iter = iter + 1

    # this function returns an object (a dictionary of items)
    obj = {"W_opt":W_opt,"w_vec_opt":w_vec_opt,"W_path":W_path,"w_vec_path":w_vec_path,"ell_path":ell_path,
           "iter":iter,"update":update}
    return obj


# test sample size (for the last year 1/1/2017 - 12/31/2017)
n_test = 251
# training sample size (for the 1st 4 years 1/1/2013 - 12/31/2016)
n_train = n - n_test
# traning sample
X_aug_train = X_aug[0:n_train, ]
y_train_cn = y_cn[0:n_train, ].reshape(n_train, 1)
# test sample
X_aug_test = X_aug[n_train:n, ].reshape(n_test, X_aug.shape[1])
y_test_cn = y_cn[n_train:n, ].reshape(n_test, 1)

# now do classification using the 2-layer MLP
# set the number of ReLU neurons in the hidden layer
m = 3
obj_nn = ell_nn_opt_mb_adam(X_aug_train, y_train_cn, m)
# number of iterations used
# print(obj_nn["iter"])
# # relative change from the last iteration
# print(obj_nn["update"])
#
# # The values of optimal network weights
# print(obj_nn["W_opt"])
# print(obj_nn["w_vec_opt"])

# plot the paths of loss and optimal weights during the course of iterations
# set the plotting area into a 2 by 4 panel
plt.figure(1)
plt.subplot(241)
plt.plot(obj_nn["ell_path"])
plt.subplot(242)
plt.plot(obj_nn["W_path"][0, 0, :])
plt.subplot(243)
plt.plot(obj_nn["W_path"][1, 0, :])
plt.subplot(244)
plt.plot(obj_nn["W_path"][2, 0, :])
plt.subplot(245)
plt.plot(obj_nn["w_vec_path"][0, :])
plt.subplot(246)
plt.plot(obj_nn["w_vec_path"][1, :])
plt.subplot(247)
plt.plot(obj_nn["w_vec_path"][2, :])
plt.show()

# define the neural network (deep learning) prediction rule
def pred_nn(x, W, w_vec):
    # x = (1, x1, ..., xp)^T is a (p+1) by 1 vector of random input (data point),
    # W = [w_1, ..., w_m] is a (p+1) by m weight matrix with w_i = (w_{0i}, w_{1i}, ...,
    # w_{pi})^T for the hidden layer, and w.vec = (w0, w1, ..., wm)^T is a (m+1) by 1
    # weight vector for the sigmoid neuron
    # define the output z = (1, z1, ..., zm)^T from the hidden layer of m ReLU neurons
    z = np.append(1, relu_layer(x, W))
    z = z.reshape(z.size, 1)
    # define the output from the sigmoid layer
    # mu can be interpreted as estimated probability of y label = 1
    mu = sigmoid_layer(z, w_vec)
    # predicted y label for classification
    y_pred = (mu > 0.5)
    return y_pred

# define prediction error for classification as classification error rate
def pe_nn(X, y, W, w_vec):
    #sample size
    n = y.size
    # predicted y labels
    yhat = np.zeros(n)
    for i in range(0, n):
        # ith row of data matrix X as an input
        x = X[i,].reshape(X.shape[1], 1)
        # predicted y label
        yhat[i] = pred_nn(x, W, w_vec)

    # classification error rate
    pe = np.sum(y.reshape(n) != yhat) * 1.0 / n
    return pe


# calculate the prediction error using the test sample
obj_pe_nn = pe_nn(X_aug_test, y_test_cn, obj_nn["W_opt"], obj_nn["w_vec_opt"])
# print out prediction error (classification error rate)
print(obj_pe_nn)

