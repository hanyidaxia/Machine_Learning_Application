#####9/26/2018
eigen(matrix(c(5,-2,-2,2),nrow = 2))
cereals <- read.csv("cereals(1).csv")
eigen(matrix(c(379.63,-188.68,-188.68,197.32),nrow = 2))
help(eigen)
str(cereals)
pcs <- princomp(data.frame(cereals$calories, cereals$rating))
help("princomp")
summary(pcs)
pcs$loadings
pcs$scores[1:5,]
cereals[1:5,c("calories","rating")]
pcs2 <- prcomp(na.omit(cereals[,-c(1:3)]))
summary(pcs2)
pcs2$sdev
pcs2$rotation[,1:5]
pcs2$x[1:5,1:5]
plot(x = c(1:13), y = pcs2$sdev^2, type = "l", col= "gold", xlab = "princple components", 
     ylab = "variances", main = "variance of priciple components")
psc3 <- prcomp(na.omit(cereals[,-c(1:3)]),scale. = T)
summary(psc3)
psc3$sdev
psc3$rotation[,1:5]
psc3$x[1:5,1:6]
par(c(1,1))
plot(x = c(1:13), y = psc3$sdev^2, type = "l", col= "gold", xlab = "princple components", 
     ylab = "variances", main = "variance of priciple components",xlim = NULL)
