# Define cost function f(x) = 1 + (x - 5)^2
f <- function(x) {
  cost = 1 + (x - 5)^2
  return(cost)
}

# compute the value of cost at x = 0
f(0)

# define a sequence of x values (a grid from -10 to 10 with spacing 0.5)
x = seq(-10, 10, 0.5)

#plot cost function f(x) = 1 + (x - 5)^2
#important to note that f(x) defined can also take a vector as an input

plot(x, f(x))

# define the gradient of cost function f(x) = 1 + (x - 5)^2
f.grad <- function(x) {
  grad = 2*(x-5)
  return(grad)
}

# find the optimal value of x (with the lowest cost) using the idea of
# gradient descent (skiing downhill)
f.opt <- function(x.ini = 0, setpsize = 0.05, maxsteps = 200, tol = 1e-4){
  # x.opt holds the current value of the optimal x
  x.opt = x.ini
  x.opt.old = x.opt
  # record the paths of x.opt and f(x.opt) during the course of iterations
  x.path = x.ini
  y.path = f(x.ini)
  # iteration count
  iter = 0
  # for early stopping (1e8 or a sufficiently large number)
  update = 1e8
  
  while((iter < maxsteps) && (update > tol)){
    # gradient descent with a small step size (in the direction opposite to gradient)
    x.opt = x.opt - setpsize*(f.grad(x.opt))
    #use the relative change to check convergence for early stopping
    update = abs(x.opt - x.opt.old)/max(1e-4, abs(x.opt.old))
    x.opt.old = x.opt
    # add new values x.opt and f(x.opt) to the solution path
    x.path = c(x.path, x.opt)
    y.path = c(y.path, f(x.opt))
    # increase iteration count by one
    iter = iter + 1
  }
  
  # this function returns an object (a list of items)
  obj = list(x.opt = x.opt, x.path = x.path, y.path = y.path)
  return(obj)
}

# compute the value of optimal x
obj = f.opt(0)

# the value of the optimal x
obj$x.opt

# plot the paths of f(x.opt) and x.opt during the course of iterations 
# set the plotting area into a 1 by 2 panel 
par(mfrow = c(1, 2))
plot(obj$y.path)
plot(obj$x.path)
