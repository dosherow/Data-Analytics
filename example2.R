# Example 2 for Week 6

# import the cleaned CSV data file into R
# returns a data frame with a list of variables 
df = read.csv("C:/Users/Jinchi Lv/Desktop/100_Portfolios_10x10_Daily.csv")

# number of rows
nr = nrow(df)
# print out nr
nr
# number of columns
nc = ncol(df)
# print out nc
nc

# print out 1st row
# the 1st component is an empty cell and the rest 100 components are the names of 
# the 100 portfolios
df[1,]
# print out 1st column
# the 1st component is an empty cell and the rest 1259 components are the dates of 
# the 1259 daily returns
df[,1]

# now define data matrix X (1259 by 100) for structural exploration
# need to convert a data frame to a numeric matrix
X = apply(as.matrix.noquote(df[2:nr, 2:nc]), 2, as.numeric)
# the sample size (number of observations or data points)
n = nrow(X)
# the dimensionality (number of variables)
p = ncol(X)

# Let us start with plotting the time series of the portfolios
# set the plotting area into a 2 by 2 panel
par(mfrow = c(2, 2))   
plot(X[,1])
plot(X[,2])
plot(X[,3])
plot(X[,4])

# calculate the mean and standard deviation for each portfolio
mean(X[,1])
sd(X[,1])
mean(X[,2])
sd(X[,2])
mean(X[,3])
sd(X[,3])
mean(X[,4])
sd(X[,4])

# Principal components analysis (PCA)
obj.pca = prcomp(X, scale = TRUE)
# scores (square roots of eigenvalues)
scores = obj.pca$sdev
# loadings (matrix whose columns contain eigenvectors)
loadings = obj.pca$rotation

# plot the scores and the top 3 principal components
# set the plotting area into a 2 by 2 panel
par(mfrow = c(2, 2))   
plot(scores)
plot(X%*%loadings[,1])
plot(X%*%loadings[,2])
plot(X%*%loadings[,3])

# Singular value decomposition (SVD) 
obj.svd = svd(X)
# D vector of singular values
D = obj.svd$d
# print out vector D
D
# print out vector D recaled by factor 1/sqrt(n) (a common choice)
D/sqrt(n)
# U matrix of left singular vectors (columns)
U = obj.svd$u
# V matrix of right singular vectors (columns)
V = obj.svd$v

# plot the rescaled singular values and the 1st SVD layer
# set the plotting area into a 1 by 3 panel
par(mfrow = c(1, 3))   
plot(D/sqrt(n))
plot(U[,1])
plot(V[,1])

# k-means clustering (applied to rows of data matrix)
# we consider the p by n data matrix X^T for clustering the p = 100 portfolios
# set the number of clusters
k = 3
obj.kmc = kmeans(t(X), k)

# cluster labels for each data point (row)
obj.kmc$cluster
# size of each cluster
obj.kmc$size
# fraction of variation due to between-cluster
obj.kmc$betweenss/obj.kmc$totss
# fraction of variation due to within-cluster
obj.kmc$tot.withinss/obj.kmc$totss

# hierarchical clustering (applied to rows of data matrix)
# we consider the p by n data matrix X^T for clustering the p = 100 portfolios
# distance matrix computation using the specified distance measure to compute 
# the distances between the rows of a data matrix
d.dist = dist(t(X))
obj.hc = hclust(d.dist)

# set the plotting area into a 1 by 1 panel
par(mfrow = c(1, 1))
# plot hierarchical clustering results with dendrogram
plot(obj.hc)
# set the number of clusters
k = 3
# cut off the tree at desired number of clusters using cutree
obj.hc.cut = cutree(obj.hc, k)
obj.hc.cut
# size of each cluster
table(obj.hc.cut)

# spectral clustering (applied to rows of data matrix)
# we consider the p by n data matrix X^T for clustering the p = 100 portfolios
# load R package
library(kernlab)
# click on Packages tab and then Install icon to install a new R package
# set the number of clusters
k = 3
obj.sc = specc(t(X), k)

# print out spectral clustering results (cluster labels, ...)
obj.sc
# size of each cluster
size(obj.sc)

# k-nearest neighbors (KNN) clustering (applied to rows of data matrix)
# we consider the p by n data matrix X^T for clustering the p = 100 portfolios
# load R package
library(kknn)
# click on Packages tab and then Install icon to install a new R package
# spectral clustering based on k-nearest neighbor (KNN) graph
# set the number of clusters
k = 3
obj.knnc = specClust(t(X), k)

# cluster labels for each data point (row)
obj.knnc$cluster
# size of each cluster
obj.knnc$size
# fraction of variation due to between-cluster
obj.knnc$betweenss/obj.knnc$totss
# fraction of variation due to within-cluster
obj.knnc$tot.withinss/obj.knnc$totss

# estimate the Gaussian graphical network using graphical Lasso (glasso) method
# we consider the n by p data matrix X for the p = 100 portfolios
# load R package
library(glasso)
# click on Packages tab and then Install icon to install a new R package
# calculate covariance matrix for p = 100 portfolios
cov.mat = var(X)
# nonnegative regularization parameter for graphical Lasso
rho = 0.1
obj.glasso = glasso(cov.mat, rho)

# estimated inverse covariance matrix (also known as precision matrix)
# print out part of this matrix and we see that some of the entries are 
# 0 (for no links between the pairs of variables)
obj.glasso$wi[1:10,1:10]

# create graph from the estimated sparse precision matrix
# load R package
library(igraph)
# click on Packages tab and then Install icon to install a new R package
adj.mat = obj.glasso$wi
# remove self-loops by seeting the diagonal entries to zero
diag(adj.mat) = rep(0, p)
# convert the modified sparse precision matrix into an adjacency matrix (0/1's)
adj.mat = (adj.mat != 0)
obj.gn = graph_from_adjacency_matrix(abs(adj.mat), mode = "undirected")
plot.igraph(obj.gn)
