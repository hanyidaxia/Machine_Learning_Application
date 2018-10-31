#### 10/10/2018
#### linear regression function
exdata <- read.csv("ex1data2.csv",header = F)
str(exdata)
exdata$V0 <- 1
str(exdata)
X <- as.matrix(exdata[,c(4,1)])
XT <- t(X)
Y <- as.matrix(exdata[,3])
b <- solve(XT%*%X)%*%(XT%*%Y)
############# matrix combine use %*%
b
###### super clear that we got it
plot(exdata[,3]~exdata[,1])
abline(b[1],b[2],col = "gold")
X1 <- as.matrix(exdata[,c(4,1,2)])
X1T <- t(X1)
b1 <- solve(X1T%*%X1)%*%(X1T%*%Y)
b1
ypred <- X1%*%b1##############the predction
plot(ypred~exdata[,1], xlim = c(500,5000), ylim = c(100000,710000))
points(exdata[,1],Y, col = 2, pch = 1)
#abline(b[])
jcost <- function(x,y,b){
  m <- length(y)
}
jcost <- function(x,y,b){
    m <- length(y)
    return(t(X%*%b-y)%*%(X%*%b-y))/(2*m)
}


X <- as.matrix(exdata[,c(4,1,2)])
X[,2] <- (X[,2] - mean(X[,2]))/sd(X[,2])
X[,3] <- (X[,3] - mean(X[,3]))/sd(X[,3])
niter <- 400
J_history <- 1:400
J_history[1:niter]<- 0
alpha <- 0.1
b0 <- as.matrix(c(0,0,0))
# for (iter in 1:niter) {
#   brun <- b0 + alpha*t(t(Y-X%*%b0)%*%X)/m
#   J_history[iter] <- jcost(X,Y,brun)
#   b0 <- brun
# }
m <- length(Y)
for (iter in 1:niter) {
  brun <- b0 + alpha*t(t(Y-X%*%b0)%*%X)/m
  J_history[iter] <- jcost(X,Y,brun)
  b0 <- brun
}
b0



solve(t(X)%*%X)%*%(t(X)%*%Y)



plot(J_history[1:50],col = "darkred", xlab = "#of interation", ylab = "Jcost", xlim = c(1:50),
     ylim = c(1:60000))


jcost



plot(J_history[1:50],col = "darkred", xlab = "#of interation", ylab = "Jcost")
