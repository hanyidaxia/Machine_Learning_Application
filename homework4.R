####10/27/2018####
#### homework 3####
rm()
#### task 3
setwd("C:/Users/hanyi_gm/Desktop/第五学期/Data Mining/homework4")
library('xlsx')
data1 <- read.xlsx(file = "Concrete Slump Test Data.xlsx",sheetIndex = 1)
View(data1)
drops <- c("NA.", "NA..1","NA..2")
data1 <- data1[,!names(data1) %in% drops]
##data1[1:(length(data1)-3)]
# data1[-c(12,13,14), ]
library('car')
################# Question 1 ############################
my_cols <- c("#00AFBB", "#E7B800", "#FC4E07")
pairs(data1[,c(2,4,6,8)], pch = 19,  cex = 0.5,
      col = my_cols[data1$Slump],
      lower.panel=NULL)
scatterplotMatrix(data1, spread = F, lty.smooth = 2, main = "scatter plot matrix")
## b. #######
slumplm <- lm(Slump.Flow~ Fly.Ash,data = data1 )
summary(slumplm)
data1$Slump.Flow
fitted(slumplm)
residuals(slumplm)
plot(data1$Slump.Flow, data1$Fly.Ash, xlab = "Fly.Ash", ylab = "Slump.Flow")
abline(slumplm)

slumplm1 <- lm(Slump.Flow~ Cement,data = data1 )
summary(slumplm1)
data1$Slump.Flow
fitted(slumplm1)
residuals(slumplm1)
plot(data1$Slump.Flow, data1$Cement, xlab = "Cement", ylab = "Slump.Flow")
abline(slumplm1)

slumplm2 <- lm(Slump.Flow~ SP,data = data1 )
summary(slumplm2)
data1$Slump.Flow
fitted(slumplm2)
residuals(slumplm2)
plot(data1$Slump.Flow, data1$SP, xlab = "SP", ylab = "Slump.Flow")
abline(slumplm2)

slumplm3 <- lm(Slump.Flow~ Fine.Aggregate,data = data1 )
summary(slumplm3)
data1$Slump.Flow
fitted(slumplm3)
residuals(slumplm3)
plot(data1$Slump.Flow, data1$Fine.Aggregate, xlab = "Fine.Aggregate", ylab = "Slump.Flow")
abline(slumplm3)

slumplm4 <- lm(Slump~ Cement,data = data1 )
summary(slumplm4)
data1$Slump.Flow
fitted(slumplm4)
residuals(slumplm4)
plot(data1$Slump, data1$Cement, xlab = "Cement", ylab = "Slump")
abline(slumplm4)

slumplm5 <- lm(Slump~ SP,data = data1 )
summary(slumplm5)
data1$Slump.Flow
fitted(slumplm5)
residuals(slumplm5)
plot(data1$Slump, data1$SP, xlab = "SP", ylab = "Slump")
abline(slumplm5)

slumplm6 <- lm(Slump~ Fly.Ash,data = data1 )
summary(slumplm6)
data1$Slump.Flow
fitted(slumplm6)
residuals(slumplm6)
plot(data1$Slump, data1$Fly.Ash, xlab = "Fly.Ash", ylab = "Slump")
abline(slumplm6)

slumplm7 <- lm(X28.day.Compressive.Strength ~ Cement,data = data1)
plot(data1$X28.day.Compressive.Strength, data1$Cement,xlab = "Cement", ylab = "X28")
abline(slumplm7)

slumplm8 <- lm(Slump.Flow ~ Cement*Slag*Fly.Ash*Water*SP*Coarse.Aggregate*Fine.Aggregate,data = data1 )
summary(slumplm8)

  lm(formula = Slump.Flow ~ Cement*Slag*Fly.Ash*Water*SP*Coarse.Aggregate*Fine.Aggregate,data = data1)

  slumplm9 <- lm(Slump ~ Cement*Slag*Fly.Ash*Water*SP*Coarse.Aggregate*Fine.Aggregate,data = data1 )
  summary(slumplm9)
  slumplm10 <- lm(Slump.Flow ~ Cement+Slag+Fly.Ash+Water+SP+Coarse.Aggregate+Fine.Aggregate,data = data1 )
  summary(slumplm10)
  
  
  options(repos='http://cran.rstudio.com/')
  install.packages("gvlma")
  library('gvlma')
############ Question 3 #########
  
  #### typical method ####
  slumplm10 <- lm(Slump.Flow ~ Cement+Slag+Fly.Ash+Water+SP+Coarse.Aggregate+Fine.Aggregate,data = data1 )
  summary(slumplm10)
  par(mfrow = c(1,2))
  plot(slumplm10)
  ####  enhanced method ####
  ## 1. normality
  par(mfrow = c(1,1))
  qqPlot(slumplm10, labels = row.names(data1), simulate = T, id.method = "identify", main = "QQ-plot")
### rookie mistake, P is a upper letter##
#### 2. studentized residuals
  newplot <- function(slumplm10, nbreaks = 20){
    z <- rstudent(slumplm10)
    hist(z,breaks = nbreaks, freq = F,
         xlab = "studentized residual",
         main = "Distribution of Errors")
    rug(jitter(z),col = "red")
    curve(dnorm(x,mean = mean(z),sd = sd(z)),
          add = T,col = "blue",lwd = 2)
    lines(density(z)$x, density(z)$y,col = "darkblue",lwd = 2,lty = 2)
    legend("topright",
           legend = c("normal curve", "kernel density curve"),
           lty = 1:2, col = c("yellow","orange"),cex = 0.7)
  }
newplot(slumplm10)  

#### 3. #### linearity
crPlots(slumplm10)

#### 4. #### homoscedasticity
ncvTest(slumplm10)
spreadLevelPlot(slumplm10)

#### 5. Global validation of linear model assumption
gmodel <- gvlma(slumplm10)
summary(gmodel)

#### 6. multicollinearity
vif(slumplm10)
sqrt(vif(slumplm10)) > 2


######################################Identify unusual observations and take corrective measures#######
outlierTest(slumplm10)
outplot <- function(slumplm10){
  p <- length(coefficients(slumplm10))
  n <- length(fitted(slumplm10))
  plot(hatvalues(slumplm10), main = "Index plot of hat values")
  abline(h = c(2,3)*p/n, col = "red", lty = 2)
  identify(1:n, hatvalues(slumplm10),names(hatvalues(slumplm10)))
  
}
outplot(slumplm10)

## influtial observations
cutoff <- 4/(nrow(data1) - length(slumplm10$coefficients)- 2)
plot(slumplm10, which = 4, cook.levels = cutoff)
abline(h = cutoff, lty = 2, col = "lightblue")
avPlots(slumplm10,ask = F,onepage = T, id.method = "identify")
influencePlot(slumplm10, id.method = "identify", main = "Influence Plot",
                   sub = "Circle size is proportional to cook's distance")


####### corrective measures ###########
####1. delete samples 
data1 <- data1[-69,]
####2. transforming variables
summary(powerTransform(data1$Slump.Flow))
####3. other approches
slumplm11 <- lm(Slump.Flow ~ Cement+ Slag * Fly.Ash + Water*SP + Coarse.Aggregate*Fine.Aggregate,
                data = data1)
summary(slumplm11)
par(mfrow =c(1,2))
plot(slumplm11)

slumplm12 <- lm(Slump.Flow ~ Cement + Slag*Fly.Ash + Water + SP^2 + Coarse.Aggregate*Fine.Aggregate
                , data = data1)
summary(slumplm12)
plot(slumplm12)

######################################## Select the best regression model################

anova(slumplm11,slumplm10)
AIC(slumplm10,slumplm11)

##################Fine tune the selection of predictor variables#######################
######1. step wise 
library('MASS')
slumplm11 <- lm(Slump.Flow ~ Cement+ Slag * Fly.Ash + Water*SP + Coarse.Aggregate*Fine.Aggregate,
                data = data1)
stepAIC(slumplm11,direction = "backward")

######2. all ser regression
library('leaps')
lep <- regsubsets(Slump.Flow ~ Cement+ Slag * Fly.Ash + Water*SP + Coarse.Aggregate*Fine.Aggregate,
                  data = data1, nbest = 7)
par(mfrow = c(1,1))
plot(lep, scale = "adjr2")

subsets(lep,statistic = "cp",
        main = "Cp plot for All subsets Regression")
abline(1,1,lty = 2, col = "lightblue")


### choose a good model from stepwise regression

slumplm13 <- lm(Slump.Flow ~ Slag + Fly.Ash + Water + Slag:Fly.Ash, 
                data = data1)
par(mfrow = c(1,2))
plot(slumplm13)






###############################interpret the result ###############################
data2 <- read.xlsx("Forest Fires Data.xlsx",sheetIndex = 1)
View(data2)

#########################################1.#####################################################
scatterplotMatrix(data2, spread = F, lty.smooth = 2, main = "scatter plot matrix")

#################2.###################################################################
slumplm20 <- lm(Area~ X+Y+Month*Day+FFMC*DMC + DC +ISI + Temp + RH + Wind + Rain,
                data = data2)
plot(slumplm20)

slumplm21 <- lm(Area~ X*Y + Month*Day + FFMC*DMC + DC*ISI+ Temp + RH + Wind + Rain,
                data = data2)
par(mfrow = c(1,1))
plot(slumplm21)

### use a normalized method
library('dplyr')
func2 <- function(x){
  return((x - min(x))/(max(x)- min(x)))
}
data2 <- mutate(data2,y = log(Area + 1))
data3 <- func2(data2[, -c(3,4)])
View(data2)
View(data3)
slumplm22 <- lm(y ~ X*Y + FFMC*DMC + ISI +Temp + RH+ Wind + Rain,
                data = data3)
plot(slumplm22)

slumplm23 <- lm(y ~ X*Y + FFMC*DMC + ISI + RH*Wind*Rain,data = data3)
plot(slumplm23)
#######################3.######################################################
#### typical method ####
summary(slumplm20)
par(mfrow = c(1,2))
plot(slumplm20)

summary(slumplm21)
par(mfrow = c(1,2))
plot(slumplm21)

summary(slumplm22)
par(mfrow = c(1,2))
plot(slumplm20)
####  enhanced method ####
## 1. normality
par(mfrow = c(1,1))
qqPlot(slumplm22, labels = row.names(data2), simulate = T, id.method = "identify", main = "QQ-plot")
### rookie mistake, P is a upper letter##
#### 2. studentized residuals
newplot <- function(slumplm22, nbreaks = 20){
  z <- rstudent(slumplm22)
  hist(z,breaks = nbreaks, freq = F,
       xlab = "studentized residual",
       main = "Distribution of Errors")
  rug(jitter(z),col = "red")
  curve(dnorm(x,mean = mean(z),sd = sd(z)),
        add = T,col = "blue",lwd = 2)
  lines(density(z)$x, density(z)$y,col = "darkblue",lwd = 2,lty = 2)
  legend("topright",
         legend = c("normal curve", "kernel density curve"),
         lty = 1:2, col = c("yellow","orange"),cex = 0.7)
}
newplot(slumplm22)  

#### 3. #### linearity
crPlots(slumplm22)

#### 4. #### homoscedasticity
ncvTest(slumplm22)
spreadLevelPlot(slumplm22)

#### 5. Global validation of linear model assumption
gmodel <- gvlma(slumplm22)
summary(gmodel)

#### 6. multicollinearity
vif(slumplm22)
sqrt(vif(slumplm22)) > 2

######################################Identify unusual observations and take corrective measures#######
outlierTest(slumplm23)
outplot <- function(slumplm23){
  p <- length(coefficients(slumplm23))
  n <- length(fitted(slumplm23))
  plot(hatvalues(slumplm23), main = "Index plot of hat values")
  abline(h = c(2,3)*p/n, col = "red", lty = 2)
  identify(1:n, hatvalues(slumplm23),names(hatvalues(slumplm23)))
  
}
outplot(slumplm23)


## influtial observations
cutoff <- 4/(nrow(data3) - length(slumplm23$coefficients)- 2)
plot(slumplm23, which = 4, cook.levels = cutoff)
abline(h = cutoff, lty = 2, col = "lightblue")
avPlots(slumplm23,ask = F,onepage = T, id.method = "identify")
par(mfrow = c(1,1))
influencePlot(slumplm23, id.method = "identify", main = "Influence Plot",
              sub = "Circle size is proportional to cook's distance")



####### corrective measures ###########
####1. delete samples 
# data1 <- data1[-69,]
####2. transforming variables
summary(powerTransform(data2$Area))
####3. other approches

summary(slumplm23)
par(mfrow = c(1,2))
plot(slumplm23)
plot(slumplm22)
######################################## Select the best regression model################

anova(slumplm23,slumplm22)
AIC(slumplm22,slumplm23)

##################Fine tune the selection of predictor variables#######################
######1. step wise 
library('MASS')

stepAIC(slumplm23,direction = "backward")

######2. all ser regression
library('leaps')
lep <- regsubsets(y ~ X*Y + FFMC*DMC + ISI + RH*Wind*Rain,data = data3, nbest = 10)
par(mfrow = c(1,1))
plot(lep, scale = "adjr2")

subsets(lep,statistic = "cp",
        main = "Cp plot for All subsets Regression")
abline(1,1,lty = 2, col = "lightblue")


### choose a good model from stepwise regression

slumplm24 <- lm(y ~ X + DMC + RH + Wind + Rain + RH:Rain, data = data3)
par(mfrow = c(1,2))
plot(slumplm24)
summary(slumplm24)


###############################interpret the result ###############################
