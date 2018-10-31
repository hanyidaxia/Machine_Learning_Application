#Insurance expense
insurance <- read.csv("C:/Users/Luckyrtee/Desktop/IE7275/Data for lectures/insurance.csv")
#
set.seed(50)
#
sample <- sample.int(n = nrow(insurance), size =800, replace = F)
#training set
insurance_train <- insurance[sample,]
#validation set
insurance_validation <- insurance[-sample,]
#Linear regression
ins_model <- lm(expenses ~ ., data=insurance_train)
#
ins_model
#
summary(ins_model)
#
ins_predict <- predict(ins_model, insurance_validation)
#

# lift chart 
library(gains)
gain <- gains(insurance_validation$expenses[!is.na(ins_predict)], ins_predict[!is.na(ins_predict)])
options(scipen=999)
price <- insurance_validation$expenses[!is.na(insurance_validation$expenses)]

par(pty="s")
plot(c(0,gain$cume.pct.of.total*sum(price)/1000000)~c(0,gain$cume.obs), 
     xlab = "# cases", ylab = "Cumulative Expenses (Million)", main = "Lift Chart", type = "l", col = "blue")
#baseline
lines(c(0,sum(price)/1000000)~c(0,dim(insurance_validation)[1]), col = "gray", lty = 2)
#do it yourself
plot(cumsum(insurance_validation$expenses[order(ins_predict, decreasing=TRUE)]/1000000), 
     xlab = "# cases", ylab = "Cumulative Expenses (Million)", main = "Lift Chart", type = "l", col = "blue")
lines(c(0,sum(price)/1000000)~c(0,dim(insurance_validation)[1]), col = "gray", lty = 2)

#Decie-wise lift chart
barplot(gain$mean.resp/mean(price), names.arg = gain$depth, 
        xlab="Percentile", ylab = "Mean response", main = "Decile-wise lift chart", col=10)

ins_predict <- as.data.frame(ins_predict)
ins_predict$Actual <- insurance_validation[,7]
ins_predict$Error <- ins_predict[,1] - ins_predict[,2]
ins_predict
hist(ins_predict[,3])
boxplot(ins_predict[,3])

###########################