temp <- c(98.1, 95.2, 101.1)
gender <- c("male","female","male")
blood <- factor(c("o","a","ab"), levels = c("o","a","b","ab"))
sym <- factor(c("sever","baga","baba"),levels = c("baba","baga","sever"),ordered = TRUE)
sym[1]
flu <-c(TRUE, FALSE,TRUE)
subname <- c("han","da","xia")
subject1 <- list(fullname = subname[1],temperature = temp[1], flu_status = flu[1],
                 blood = blood[1],symptom = sym[1] )
subject1
subject1[2]
subject1[c("temperature","flu_status")]
subject1[2:3]
data <- data.frame(subname,temp,flu,blood,sym,gender,stringsAsFactors = FALSE)
View(data)
data$sym
m <-matrix(c(1,2,3,4),nrow = 2)
View(m)
##R basic lines training
han <- c(1,2,3)
M <- array(c('1','2'),dim = c(6,8,9))
a <- array(c('1','2'),dim = c(6,8,9))
print(a)
a  <- c(1:4)
b <- factor(a)
is.factor(b)
c = gl(2,3,labels = c("1","2","3","4"))
print(c)
height <-c("9.8","7","78","65")
weight <-c("465","3535","53535.456","0")
age <-c("2534","3578","55.65468","354313.3543")
gender <-factor(c("god","human","demon","evil"),levels = c("demon","human","evil","god"),ordered = TRUE)
love_me <-c("yes","no","yes","yes")
exit <-c(TRUE,FALSE,TRUE,TRUE)
h1 <- list(L = height[1],o = weight[5],v = gender[2],e = exit[4])
print(h1)
again <- data.frame(height,weight,age,gender,love_me,exit, ... = NA,stringsAsFactors = TRUE)
View(again)


