

#### Assignment 1, Group 6, JingXuan Yang & Yi Han ####
#### 9/21/2018 ####
#######################################################################################################
## Question 1##
setwd("C:/Users/hanyi_gm/Desktop/第五学期/Data Mining/homework1")
data1 <-  read.csv("forestfires.csv")
install.packages("Rcpp",  dependencies = TRUE, repos = "http://cran.us.r-project.org")
install.packages("ggpubr")
install.packages("magrittr")
install.packages("psych")
install.packages("plotly")
library(igraph)
library(yaml)
library(gmodels)
library(ggplot2)
library(devtools)
library(magrittr)
library(ggpubr)
library(RColorBrewer)
library(psych)
library(MASS)
library(dplyr)
###if want check the data directly apply this sentence: View(data1)


#### a. ####
# plot(x = data1$area, y = data1$temp, main = "Area & Temp",xlab = "Area", ylab = "Temp")
# ## not easy to find something from this plot
# ggplot(data = data1) + geom_point(aes(x = area, y= temp), color = "navy", alpha = 0.69)
# barplot(data1$temp,names.arg = "Area & Temp", xlab = "area", ylab = "temp", color  = "red")
# ggplot(data = data1) + geom_bar(aes(x = area, y = temp), color = "steelblue", stat = "identity", fill = "steelblue")
# ggplot(data = data1) + geom_histogram(aes(x = area, y = temp),color = "pink", 
#  stat = "identity")
# ggplot(data = data1) + geom_boxplot(aes(x= area,y = temp ), color = "yellow", outlier.color = "red",
#                                    outlier.shape = 7)
#### par arrangement can't combine ggplot
##par(mfcol =c(2,2))
par(mfcol = c(2,2))
plot(data1$temp ~ data1$area,main = "Area & Temp",ylab = "Area", ylim = c(0,200), xlab = "Temp")
plot(x = data1$month, y = data1$area, main = "Area & Month",ylab = "Area", xlab = "Month")
plot(x = data1$DC, y = data1$area, main = "Area & DC",ylab = "Area", xlab = "TDC")
plot(x = data1$RH, y = data1$area, main = "Area & RH",ylab = "Area", xlab = "RH")
#rm(A,B,C,D)


#### b. ####
p <- ggplot(data1) + 
  geom_histogram(aes(x = wind), binwidth = 1,color = "pink",fill = "darkblue")
p + scale_x_discrete(name = "wind (km/h)")


#### c. ####
set.seed(314.15926)
c <- scale(data1$wind, center = TRUE, scale = FALSE)
cg <- lm(formula = c ~ ., data = data1)
summary(cg)


#### d. ####
ggplot(data1,aes(x = wind)) + 
geom_histogram(aes(x = wind, y = ..density..),binwidth = 0.7,color = "pink",fill = "darkblue") +
geom_density(color = "red", size = 1)


#### e. ####
# as.factor(data1$month)
# ggplot(data1, aes(x = as.factor(month), colour = cut)) +
#   geom_bar(aes(x= month , y = wind), color = "darkblue", fill = "grey",stat = "density")+
#   geom_density()
# ggplot(data1, aes(x = wind, y =as.factor(month)))
ggplot(data1, aes(x = wind, colour = as.factor(month))) +
  geom_density(position="identity", alpha=0.6, fill = NA,size = 1) +
  scale_x_continuous(name = "Wind in each month",
                     breaks = seq(0, 2, 6),
                     limits=c(0, 10)) +
  scale_y_continuous(name = "Density") +
  ggtitle("wind density plot of each month") 


#### f. ####
###### 1. #####
pairs.panels(data1 %>% select(RH, DMC,temp, DC), 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE, # show correlation ellipses
             colour = c("green", "red", "lightblue","black")
)
###### 2. ##### how to intepret this plot
# each plot in the maticx shows the relationship between two variables
# each plot on the diagnol shows the density curve of the single variable
# upon the diagnol shows the directly correlation coeeficent of two variables


#### g. ####
########option 1 : interact two variable and put all three variable in a single plot
#qplot(interaction(ISI,DC), wind, data = data1, geom = "boxplot")+
  #aes(fill = wind)
#######not easy to intepret
#######option 2 : pair three different boxplot in a single panel
#par(mfcol= c(2,2))
# ggplot(data1, aes(x = ISI)) + geom_boxplot(aes(x = ISI, y = DC), 
#                              fill = "lightblue", outlier.color = "red", outlier.size = 3)
# ggplot(data1)+
#   geom_boxplot(aes(x = ISI, y = DC), fill = "lightblue", 
#                outlier.shape = 4,outlier.colour = "darkred",outlier.size = 7 )
par(mfcol = c(1,3))
boxplot(x= data1$ISI, outline = TRUE, 
        names = "ISI",col = "darkblue", xlab = "ISI",main = "ISI")
boxplot(x= data1$wind, outline = TRUE, 
        names = "wind ",col = "gold", xlab = "wind",main = "wind")
boxplot(x= data1$DC, outline = TRUE,
        names = "DC",col = "orange",xlab = "DC",main = "DC")


#### h. ####
############regular DMC
ggplot(data1,aes(x = DMC))+
  geom_histogram(aes(x= DMC, y = ..density..), fill = "purple", binwidth = 5)+
  geom_density(color = "black",size = 1)
########### log DMC
ggplot(data1,aes(x = log(DMC)))+
  geom_histogram(aes(x= log(DMC), y = ..density..), fill = "purple", binwidth = 0.05)+
  geom_density(color = "black",size = 1)
###the default log is ln, so it's have a bottom number bigger than 3 it's foutth power is close to 100
##basiclly the result didn't changed



######################################################################################################
#### Question 2 ####
data2 <- read.csv("M01_quasi_twitter.csv")
View(data2)

#### a. ####
typeof(data2$friends_count)
par(mfcol= c(1,2))
plot(x= log(data2$friends_count), main = "friend_count")
hist(x= log(data2$friends_count),breaks = 200,density = TRUE,
     border = "blue",main = "Dist of Friend_count",xlab = "friend_count")
ggplot(data2,aes(x= log(friends_count)))+
  geom_histogram(aes(x= log(friends_count), y = ..density..), fill = "darkblue", binwidth = 0.05)+
  geom_density(color = "red",size = 1)


#### b. ####
summary(data2$friends_count)


#### c. ####


##### d. ####
library(plotly)
colors <- c('#55f416','#193584','#ea0007')
# p2 <- plot_ly(data2, x = ~created_at_year, y = ~education, z = ~age,  size = 5, colors = colors,
#              marker = list(symbol = 'circle', sizemode = 'diameter'), sizes = c(5, 150),
#              text = ~paste('created_at_year:', created_at_year, 'education:', education, 'age:', age
#                            )) %>%
#   layout(title = '3D scatter plot',
#          scene = list(xaxis = list(title = 'created_at_year',
#                                    gridcolor = 'rgb(255, 255, 255)'，
#                                    type = 'log',
#                                    zerolinewidth = 1,
#                                    ticklen = 5,
#                                    gridwidth = 2),
#                       yaxis = list(title = 'education',
#                                    gridcolor = 'rgb(255, 255, 255)',
#                                    zerolinewidth = 1,
#                                    ticklen = 5,
#                                    gridwith = 2),
#                       zaxis = list(title = 'age',
#                                    gridcolor = 'rgb(255, 255, 255)',
#                                    type = 'log',
#                                    zerolinewidth = 1,
#                                    ticklen = 5,
#                                    gridwith = 2)),
#          paper_bgcolor = 'rgb(243, 243, 243)',
#          plot_bgcolor = 'rgb(243, 243, 243)')
# p2
p2 <- plot_ly(data2, x = ~created_at_year, y = ~education, z = ~age, 
              color = ~as.factor(created_at_year)
              ) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'created_at_year'),
                      yaxis = list(title = 'education'),
                      zaxis = list(title = 'age')))
p2
install.packages("scatterplot3d")
library(scatterplot3d) 
attach(mtcars) 
par(mfcol = c(1,1))
s3d <-scatterplot3d(data2$created_at_year,data2$education,data2$age, pch=16, highlight.3d=TRUE,
                    type="h", main="3D Scatterplot")
fit <- lm(mpg ~ wt+disp) 
s3d$plane3d(fit)

#### e. ####
#### 1 ####
a <- c(650,1000,900,300,14900)
b <- c ("UK" ,"Canada", "India", "Australia" , "US")
dt <- data.frame(a, b)
View(dt)
pie(a, labels = b, main = "Accounts hold num in 5 Country")
pct <- round(a/sum(a)*100)
b <- paste(b, pct)
b <- paste(b, "%", sep = "")
p3 <- pie(a, labels = b, col = rainbow(length(pct)),main = "Accounts hold num in 5 Country")
#### 2 ####
install.packages("plotrix")
library(plotrix)
p4 <- pie3D(a, labels = b, explode = 0.8, main = "3D pie")
layout(matrix(c(1,2), byrow = TRUE, nrow = 1))
 pie(a, labels = b, col = rainbow(length(pct)),main = "Accounts hold num in 5 Country")
 pie3D(a, labels = b, explode = 0.8, main = "3D pie")

 
#### f. #### 
 ggplot(data2, aes(x = created_at_year)) + 
   geom_histogram(aes(x = created_at_year, y = ..density.. ), fill = "#ced81e", binwidth = 0.3 )+ 
   geom_density(size = 1, color = "gold")
#### intepreting
 
 
######################################################################################################
#### Question 3. ####
#### a. ####
data3 <- read.csv("raw_data.csv")
View(data3) 
names(data3)[1] <- paste("sustainability")
names(data3)[2] <- paste("carbon_footprint")
names(data3)[3] <- paste("weight")
names(data3)[4] <- paste("required_power")
Ndata <- data.frame(scale(data3))
View(Ndata)
colMeans(Ndata)
summary(Ndata)
mean(data3["sustainability"])
typeof(data3["sustainability"])
#### b. ####
#par(mfcol= c(2,2))
boxplot(x= data3$sustainability)
boxplot(x = data3$carbon_footprint)
boxplot(x = data3$weight)
boxplot(x = data3$required_power)
par(mfcol = c(1,1))
boxplot(data3, main = "orginal 4 variables boxplot",varwidth = 7)


#### c. ####
boxplot(Ndata, main= "New 4 variables boxplot", varwidth = 7)


#### d. ####
##in b
    #1.all means of 4 variables larger than 0
    #2.carbon footprint have the obviously lower value
    #3.meanwhile the carbon footprint has the most ourlier
    #4.the required power has the largest range
##in c
####1.all means are close to 0
####2.the 4 varibles have the same range

#### e. ####
ggplot(Ndata) + 
  geom_point(aes(x = sustainability, y = carbon_footprint), color = "lightblue")
plot(x = Ndata$sustainability, y =Ndata$carbon_footprint)
cor.test(x= Ndata$sustainability,y = Ndata$carbon_footprint,method = "spearman")
ggscatter(Ndata, x= "sustainability", y = "carbon_footprint",
          cof.int = TRUE, add = "reg.line",
          cor.coef = TRUE
          )
ggscatter(data3, x= "sustainability", y = "carbon_footprint",
          cof.int = TRUE, add = "reg.line",
          cor.coef = TRUE,color = "green"
)
par(c(1,1))
