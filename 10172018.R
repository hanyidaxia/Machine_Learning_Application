##### 10/17/2018
car.df <- read.csv("ToyotaCorolla(1).csv")
car.df<- car.df[1:1000,]
str(car.df)
selected.car <- c(3,4,7,8,9,10,12,13,14,17,18)
set.seed(1)
train.index <- sample(c(1:1000), 600)
train.df <- car.df[train.index, selected.car]
valid.df <- car.df[-train.index,selected.car]
car.lm <- lm(Price~., data = train.df)
options(scipen = 999)
summary(car.lm)
install.packages("forecast")
library("forecast")
car.lm.pred <- predict(car.lm, valid.df)
summary(car.lm.pred)
options(scipen = 999, digits = 0)
some.resisules <- valid.df$Price[1:20] - car.lm.pred[1:20]### calculate residules
data.frame("predicted" = car.lm.pred[1:20],"Actual" = valid.df$Price[1:20],"residule" = some.resisules)
accuracy(car.lm.pred,valid.df$Price)
hist(valid.df$Price - car.lm.pred, breaks = 25, xlab = "Residules", main = "")
## test the correlation
cor(valid.df$Price,car.lm.pred)
install.packages("gains")
library('gains')
gain <- gains(valid.df$Price[!is.na(car.lm.pred)], car.lm.pred[! is.na(car.lm.pred)])
options(scipen = 999)
price <- valid.df$Price[!is.na(valid.df$Price)]
par(pty = "s")
###### draw some pictures
plot(c(0,gain$cume.pct.of.total* sum(price))~ c(0,gain$cume.obs),
     xlab = "#cases", ylab = "cumulative price", main = "lift chart", type = "l"
     , col = "red")
lines(c(0,sum(price))~c(0,dim(valid.df)[1]), col = "gold", lty = 2)
barplot(gain$cume.mean.resp/mean(price), names.arg = gain$depth, xlab = "percentile",
        ylab = "mean response", main = "decile-wise lift chart")
str(train.df)
summary(car.lm)


install.packages("leaps")
library('leaps')
Fuel_Type <- as.data.frame(model.matrix(~0 + Fuel_Type,data = car.df))
car.df <- cbind(car.df[,-8], Fuel_Type[,])
car.df[1:8]
car.df <- car.df[, -41]
selected.car <- c(3,4,7,8,11,12,13,14,17,18,39,40)
train.df <- car.df[train.index,selected.car]
valid.df <- car.df[-train.index,selected.car]
head(train.df)
search <- regsubsets(Price~., data = train.df, nbest = 1, nvmax = dim(train.df)[2], method = "exhausive")

plot(cumsum(valid.df$Price[order(car.lm.pred,decreasing = T)]),xlab = "# of cases", ylab = "cumulative price",
            main = "lift chart", type = "l", col = "blue")







amatrak <- read.csv("AmtrakPassengers.csv")
ridership.ts <- ts(amatrak$Ridership, start = c(1991,1),end = c(2004,3),frequency = 12)
plot(ridership.ts, xlab = "Time", ylab = "Ridership")
par(pty = "r")
ridership.lm <- tslm(ridership.ts ~ trend + I(trend^2))
lines(ridership.lm$fitted.values, lwd = 2, col = "lightblue" )
ridership.lm.season <- tslm(ridership.ts~ trend + I(trend^2) + season)
lines(ridership.lm.season$fitted.values, lwd = 2, col = "red" )






hwin <- ets(ridership.ts, model = "MAA")
hwin$fitted
lines(hwin$fitted)
plot(ridership.ts, xlab = "Time", ylab = "Ridership")
lines(hwin$fitted, col = "orange", lty = 2)
hwin <- ets(ridership.ts, model = "MMM")
lines(hwin$fitted, col = "darkblue", lty = 2)
hwin <- ets(ridership.ts, model = "AAA")
plot(ridership.ts, xlab = "Time", ylab = "Ridership")
lines(hwin$fitted, col = "green", lty = 2)
hwin$mse
plot(hwin$residuals)
hist(hwin$residuals, breaks = 15)
cor(ridership.ts, hwin$fitted)
