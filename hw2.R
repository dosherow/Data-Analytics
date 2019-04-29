# generate data set (x_i, y_i) with i from 1 to n = 1000
# each x_i is randomly generated from N(4, 3^2) distribution and y_i = 2*x_i
n = 1000
# set x as a column vector
x = as.matrix(rnorm(n, 4, 3), n, 1)
y = 2*x

# define loss function ell(beta) = MSE(beta)
ell <- function(x, y, beta){
  n = length(x)
  loss = t(y - x%*%beta)%*%(y - x%*%beta)/n
  return(loss)
}

# Define the gradient of loss function ell(beta) = MSE(beta)
ell.grad <- function(x, y, beta){
  grad = 2*(t(x)%*%(x%*%beta - y))/n
  return(grad)
}

# find the optimal value of beta (with the lowest loss) using the idea of 
# gradient descent (skiing downhill) 
# setpsize should not be too large
ell.opt <- function(x, y, beta.ini = 0, setpsize = 0.005, maxsteps = 200, tol = 1e-4){
  # beta.opt holds the current value of the optimal beta
  beta.opt = beta.ini
  beta.opt.old = beta.opt
  # record the paths of beta.opt and ell(x, y, beta.opt) during the course 
  # of iterations
  beta.path = beta.ini
  ell.path = ell(x, y, beta.ini)
  # iteration count
  iter = 0
  # for early stopping (1e8 or a sufficently large number)
  update = 1e8
  
  while((iter < maxsteps) && (update > tol)){
    # gradient descent with a small step size (in the direction opposite to gradient)
    beta.opt = beta.opt - setpsize*(ell.grad(x, y, beta.opt))
    # use the relative change to check convergence for early stopping
    update = abs(beta.opt - beta.opt.old)/max(1e-4, abs(beta.opt.old))
    beta.opt.old = beta.opt
    # add new values beta.opt and ell(x, y, beta.opt) to the solution path
    beta.path = c(beta.path, beta.opt)
    ell.path = c(ell.path, ell(x, y, beta.opt))
    # increase iteration count by one
    iter = iter + 1
  }
  
  # this function returns an object (a list of items)
  obj = list(beta.opt = beta.opt, beta.path = beta.path, ell.path = ell.path)
  return(obj)
}

# Compute the value of the optimal beta
obj = ell.opt(x, y, 0)

# The value of the optimal beta
obj$beta.opt

# plot the paths of ell(x, y, beta.opt) and beta.opt during the course of iterations
# set the plotting area into a 1 by 2 panel
par(mfrow = c(1, 2))   
plot(obj$ell.path)
plot(obj$beta.path)

# Compute the value of the optimal beta
# initial value now set as a random number generated from N(0, 1) distribution
beta.ini = rnorm(1)
beta.ini
obj = ell.opt(x, y, beta.ini)

# The value of the optimal beta
obj$beta.opt

# plot the paths of ell(x, y, beta.opt) and beta.opt during the course of iterations
# set the plotting area into a 1 by 2 panel
par(mfrow = c(1, 2))   
plot(obj$ell.path)
plot(obj$beta.path)

# find the optimal value of beta (with the lowest loss) using the idea of 
# gradient descent (skiing downhill) with mini-batch
# set default mini-batch size (mbsize) to 50
# setpsize should not be too large
ell.opt.mb <- function(x, y, beta.ini = 0, mbsize = 50, setpsize = 0.05, 
                       maxsteps = 200, tol = 1e-4){
  # beta.opt holds the current value of the optimal beta
  beta.opt = beta.ini
  beta.opt.old = beta.opt
  # record the paths of beta.opt and ell(x, y, beta.opt) during the course 
  # of iterations
  beta.path = beta.ini
  ell.path = ell(x, y, beta.ini)
  # iteration count
  iter = 0
  # for early stopping (1e8 or a sufficently large number)
  update = 1e8
  
  while((iter < maxsteps) && (update > tol)){
    # take a random set of mbsize data points from 1 to n
    mb.ind = sample(1:n, mbsize)
    x.mb = as.matrix(x[mb.ind, ], length(mb.ind), 1)
    y.mb = as.matrix(y[mb.ind, ], length(mb.ind), 1)
    
    # gradient descent with a small step size (in the direction opposite to gradient)
    # use the mini-batch to compute the gradient of loss function
    beta.opt = beta.opt - setpsize*(ell.grad(x.mb, y.mb, beta.opt))
    # use the relative change to check convergence for early stopping
    update = abs(beta.opt - beta.opt.old)/max(1e-4, abs(beta.opt.old))
    beta.opt.old = beta.opt
    # add new values beta.opt and ell(x, y, beta.opt) to the solution path
    beta.path = c(beta.path, beta.opt)
    ell.path = c(ell.path, ell(x, y, beta.opt))
    # increase iteration count by one
    iter = iter + 1
  }
  
  # this function returns an object (a list of items)
  obj = list(beta.opt = beta.opt, beta.path = beta.path, ell.path = ell.path)
  return(obj)
}

# Compute the value of the optimal beta
obj = ell.opt.mb(x, y, 0)

# The value of the optimal beta
obj$beta.opt

# plot the paths of ell(x, y, beta.opt) and beta.opt during the course of iterations
# set the plotting area into a 1 by 2 panel
par(mfrow = c(1, 2))   
plot(obj$ell.path)
plot(obj$beta.path)

# Compute the value of the optimal beta
# initial value now set as a random number generated from N(0, 1) distribution
beta.ini = rnorm(1)
beta.ini
obj = ell.opt.mb(x, y, beta.ini)

# The value of the optimal beta
obj$beta.opt

# plot the paths of ell(x, y, beta.opt) and beta.opt during the course of iterations
# set the plotting area into a 1 by 2 panel
par(mfrow = c(1, 2))   
plot(obj$ell.path)
plot(obj$beta.path)


