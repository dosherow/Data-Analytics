# Drew Osherow Hw4 
# we want to predict the 50th portfolio return whether 
# it will be positive or non-positive using fama-5
# we will implement SGD algo with mini-batch 
# using logistic regression instead of linear regression 

# import the 1st cleaned CSV data file into R
# returns a data frame with a list of variables 
df.y = read.csv("100_Portfolios_10x10_Daily.CSV")

# number of rows
nr.y = nrow(df.y)
# print out nr.y
nr.y
# number of columns
nc.y = ncol(df.y)
# print out nc.y
nc.y

# print out 1st row
# the 1st component is an empty cell and the rest 100 components are the names of 
# the 100 portfolios
df.y[1,]
# print out 1st column
# the 1st component is an empty cell and the rest 1259 components are the dates of 
# the 1259 daily returns
df.y[,1]

# now define data matrix Y (1259 by 100) for prediction
# need to convert a data frame to a numeric matrix
Y = apply(as.matrix.noquote(df.y[2:nr.y, 2:nc.y]), 2, as.numeric)
# the sample size (number of observations or data points)
n = nrow(Y)
# the dimensionality (number of response or Y variables)
q = ncol(Y)

# import the 2nd cleaned CSV data file into R
# returns a data frame with a list of variables 
df.x = read.csv("F-F_Research_Data_5_Factors_2x3_daily 2.CSV")

# number of rows
nr.x = nrow(df.x)
# print out nr.x
nr.x
# number of columns
nc.x = ncol(df.x)
# print out nc.x
nc.x

# print out 1st row
# the 1st component is an empty cell and the rest 6 components are the names of 
# the 6 X variables
df.x[1,]
# print out 1st column
# the 1st component is an empty cell and the rest 1259 components are the dates of 
# the 1259 days of daily factors
df.x[,1]

# now define data matrix X (1259 by 6) for prediction
# need to convert a data frame to a numeric matrix
X.all = apply(as.matrix.noquote(df.x[2:nr.x, 2:nc.x]), 2, as.numeric)
# the sample size (number of observations or data points)
n = nrow(X.all)
# the dimensionality (number of predictor or X variables)
p.all = ncol(X.all)

# preprocess the data
# consider the 50th portfolio return as the Y variable
y.id = 50
y = Y[,y.id]
# consider the excess return of portfolio by subtracting risk-free 
# interest rate (the RF column of X data matrix)
y = y - X.all[,6]
# consider only the 5 factors as the predictors (X variables)
X = X.all[,1:5]
# the sample size (number of observations or data points)
n = nrow(X)
# the dimensionality (number of predictor or X variables)
p = ncol(X)

# let us start with plotting the time series of hte portfolio vs those of 5 factors
# set the plotting area into a 2 by 3 panel
par(mfrow = c(2, 3))
plot(X[,1], y)
plot(X[,2], y)
plot(X[,3], y)
plot(X[,4], y)
plot(X[,5], y)

# calculate the mean and standard deviation for each variable
mean(y)
sd(y)
mean(X[,1])
sd(X[,1])
mean(X[,2])
sd(X[,2])
mean(X[,3])
sd(X[,3])
mean(X[,4])
sd(X[,4])
mean(X[,5])
sd(X[,5])

# calculate the correlation between y and each x variable
cor(y, X[,1])
cor(y, X[,2])
cor(y, X[,3])
cor(y, X[,4])
cor(y, X[,5])

# apply SGD algo with mini-batch idea for classification 
# define logistic function 
mu.logi <- function(X, beta){
  # estimated probability of y label = 1
  mu = exp(X%*%beta)/(1 + exp(X%*%beta))
  return(mu)
}

# define loss function ell(beta)
ell.logi <- function(X, y, beta){
  # sample size (num of rows)
  n = nrow(X)
  loss = (-t(y)%*%X%*%beta + rep(1, n)%*%log(1 + exp(X%*%beta)))/n
  return(loss)
}

# define the gradient of loss function ell(Beta)
ell.logi.grad <- function(X, y, beta){
  # sample size (num of rows)
  n = nrow(X)
  grad = (t(X)%*%(mu.logi(X, beta) - y))/n
  return(grad)
}

# find optimal value of beta (with lowest loss) using gradient descent & mini-batch
# set default mini-batch size to 100
# stepsize shouldn't be too large 
ell.logi.opt.mb <- function(X, y, beta.ini, mbsize = 100, stepsize = 0.05,
                            maxsteps = 2000, tol = 1e-4){
  # sample size ( num of rows)
  n = nrow(X)
  # dimensionality 
  p = ncol(X)
  #beta.opt holds current value of optimal beta 
  beta.opt = beta.ini
  beta.opt.old = beta.opt
  # record paths of beta.opt and ell(X, y, beta.opt) during course of itereations
  beta.path = beta.ini
  ell.path = ell.logi(X, y, beta.ini)
  # iteration count
  iter = 0
  # for early stopping 
  update = 1e8
  
  while((iter < maxsteps) && (update > tol)){
    # take a random set of mbsize data points from 1 to n
    mb.ind = sample(1:n, mbsize)
    X.mb = as.matrix(X[mb.ind, ], length(mb.ind), p)
    y.mb = as.matrix(y[mb.ind, ], length(mb.ind), 1)
    
    # gradient dsscent small step size
    # use mini-batch to compute gradient of loss function
    beta.opt = beta.opt - stepsize*(ell.logi.grad(X.mb, y.mb, beta.opt))
    # use relative change to check convergence for early stopping
    update = norm(beta.opt - beta.opt.old, type="F")/max(1e-4, norm(beta.opt.old, type="F"))
    beta.opt.old = beta.opt
    # add new values beta.opt and ell(X, y, beta.opt) to solution path
    # attach new column to matrix beta.path after each iteration
    beta.path = cbind(beta.path, beta.opt)
    ell.path = c(ell.path, ell.logi(X, y, beta.opt))
    # increase iteration count by one
    iter = iter + 1
  }
  
  # this function returns an object (list of items)
  obj = list(beta.opt = beta.opt, beta.path = beta.path, ell.path = ell.path, iter = iter,
             update = update)
  return(obj)
}

# compute value of optimal beta
n = nrow(X)
# add constant column of 1's to data matrix x for intercept term
X.aug = cbind(rep(1, n), X)
# new dimensionality (p + 1)
p.aug = ncol(X.aug)
# set y as column vector
y = as.matrix(y, n, 1)
# convert portfolio returns into 0/1's for classification
# 0 means return <= 0
# 1 means return > 0
y.cn = (y > 0)

# test sample size (last year 1/1/2017 - 12-31-2017)
n.test = 251
# training sample size (1st 4 years 1/1/2013 - 12/31/2016)
n.train = n - n.test
# traning sample 
X.aug.train = X.aug[1:n.train, ]
y.train.cn = as.matrix(y.cn[1:n.train, ], n.train, 1)
# test sample 
X.aug.test = X.aug[(n.train+1):n, ]
y.test.cn = as.matrix(y.cn[(n.train+1):n, ], n.test, 1)

# set initial value to zero vector
beta.ini = as.matrix(rep(0, p.aug), p.aug, 1)
obj.logi = ell.logi.opt.mb(X.aug.train, y.train.cn, beta.ini)
# num of iterations used
obj.logi$iter
# relative change from last iteration
obj.logi$update

# value of optimal beta
obj.logi$beta.opt

# plot path of ell(X, y, beta.opt) and beta.opt during course of iterations
# set plotting area into 2 by 4 panel
par(mfrow = c(2, 4))
plot(obj.logi$ell.path)
plot(obj.logi$beta.path[1, ])
plot(obj.logi$beta.path[2, ])
plot(obj.logi$beta.path[3, ])
plot(obj.logi$beta.path[4, ])
plot(obj.logi$beta.path[5, ])
plot(obj.logi$beta.path[6, ])

# define prediction error pe(beta) as classification error rate 
pe.logi <- function(X, y, beta){
  # sampe size (num of rows)
  n = nrow(X)
  pe = sum(y != (mu.logi(X, beta) > 0.5))/n
  return(pe)
}

# calculate prediction error using test sample
obj.pe.logi = pe.logi(X.aug.test, y.test.cn, obj.logi$beta.opt)
# print out prediction error 
obj.pe.logi