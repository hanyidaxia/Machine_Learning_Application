#loading house price data
housedata <- read.csv("/Users/Min/Downloads/machine-learning-ex1/ex1/ex1data2.txt", header = FALSE)
#summary
summary(housedata)
#add a column of 1, x_0
housedata$x0 = 1
#use "area" only for regression, price = const + b * area
#construct X matrix
X <- as.matrix(housedata[,c(4,1)])
#X transpose
XT <- t(X)
# Y vector
Y <- as.matrix(housedata[,3]/1000.)
# coefficeints b =(XT X)^-1 XT Y
b <- solve(XT%*%X)%*%(XT%*%Y)
b
#check the predicted house price
#plot the true values
plot(housedata[,1], housedata[,3]/1000, xlab = "Living Area (square feet)", ylab="House Price ($1000)")
#plot the predicted line
abline(b[1], b[2], col = "red")
title("Linear Regression of House Price with Living Area")
##############
#include number of bedrooms
#construct X matrix
X1 <- as.matrix(housedata[,c(4,1,2)])
#X transpose
X1T <- t(X1)
# coefficeints b =(XT X)^-1 XT Y
b1 <- solve(X1T%*%X1)%*%(X1T%*%Y)
b1
#check the predicted values
ypredict <- X1%*%b1
#############
#scatter plot of price vs area
plot(housedata[,1], ypredict,
     main = "House Price vs. Living Area", 
     xlab = "Living Area (square feet)",
     ylab = "House Price ($1000)", col=2, pch=15, 
     xlim=c(500,5000), ylim=c(100,710))
points(housedata[,1],Y, col=2, pch=1)
#plot the regression line
abline(a=b[1], b=b[2], col=4, xlim=c(500,5000), ylim=c(100,710))
###############
#Gradient Descent Algorithm:
Jcost <- function(X, y, b){
  m <- length(y)
  return((t(X%*%b-y))%*%(X%*%b-y)/(2*m))
}

X <- as.matrix(housedata[c(4,1,2)])
X[,2] <-(X[,2] - mean(X[,2]))/sd(X[,2])
X[,3] <-(X[,3] - mean(X[,3]))/sd(X[,3])
Y <- as.matrix(workdata[,3])/1000

niter <- 400
J_history[1:niter] = 0

alpha <- 0.1
b0 <- as.matrix(c(0,0,0))
brun <- b0
m = length(Y)

for (iter in 1:niter){
  brun = b0 + alpha*t(t(Y-X%*%b0)%*%X)/m
  J_history[iter] <- Jcost(X,Y,brun)
#  print(J_history[iter])
  b0 = brun
}

b0
plot(J_history[1:50], col = "blue", xlab = "No. of Iteration", ylab = "Jcost",
     xlim=c(1,50), ylim = c(1,60000), main = "Least square cost function vs. iteration step")

#compare with normal equations

b1 <- solve(t(X)%*%X)%*%(t(X)%*%Y)
b1
