######10/4/2018
####### Assignment 2 
############ Group 6 JingXuan Yang, Yi Han
#######################################Question 1 ##############################################
library('foreign')
data1 <-  data.frame(read.delim("US Judge Ratings.dat"))
View(data1)
data1f <- read.ftable("US Judge Ratings.dat")
View(data1f)
data1 <- USJudgeRatings <-
  data.frame(CONT = c(5.7, 6.8, 7.2, 6.8, 7.3, 6.2, 10.6, 7, 7.3, 8.2,
                      7, 6.5, 6.7, 7, 6.5, 7.3, 8, 7.7, 8.3, 9.6, 7.1, 7.6, 6.6, 6.2, 7.5, 7.8,
                      7.1, 7.5, 7.5, 7.1, 6.6, 8.4, 6.9, 7.3, 7.7, 8.5, 6.9, 6.5, 8.3, 8.3, 9,
                      7.1, 8.6), INTG = c(7.9, 8.9, 8.1, 8.8, 6.4, 8.8, 9, 5.9, 8.9, 7.9, 8,
                                          8, 8.6, 7.5, 8.1, 8, 7.6, 7.7, 8.2, 6.9, 8.2, 7.3, 7.4, 8.3, 8.7, 8.9,
                                          8.5, 9, 8.1, 9.2, 7.4, 8, 8.5, 8.9, 6.2, 8.3, 8.3, 8.2, 7.3, 8.2, 7, 8.4,
                                          7.4), DMNR = c(7.7, 8.8, 7.8, 8.5, 4.3, 8.7, 8.9, 4.9, 8.9, 6.7, 7.6, 7.6,
                                                         8.2, 6.4, 8, 7.4, 6.6, 6.7, 7.4, 5.7, 7.7, 6.9, 6.2, 8.1, 8.5, 8.7, 8.3,
                                                         8.9, 7.7, 9, 6.9, 7.9, 7.8, 8.8, 5.1, 8.1, 8, 7.7, 7, 7.8, 5.9, 8.4, 7),
             DILG = c(7.3, 8.5, 7.8, 8.8, 6.5, 8.5, 8.7, 5.1, 8.7, 8.1, 7.4, 7.2, 6.8,
                      6.8, 8, 7.7, 7.2, 7.5, 7.8, 6.6, 7.1, 6.8, 6.2, 7.7, 8.6, 8.9, 8, 8.7,
                      8.2, 9, 8.4, 7.9, 8.5, 8.7, 5.6, 8.3, 8.1, 7.8, 6.8, 8.3, 7, 7.7, 7.5),
             CFMG = c(7.1, 7.8, 7.5, 8.3, 6, 7.9, 8.5, 5.4, 8.6, 7.9, 7.3, 7,
                      6.9, 6.5, 7.9, 7.3, 6.5, 7.4, 7.7, 6.9, 6.6, 6.7, 5.4, 7.4, 8.5,
                      8.7, 7.9, 8.4, 8, 8.4, 8, 7.8, 8.1, 8.4, 5.6, 8.4, 7.9, 7.6, 7,
                      8.4, 7, 7.5, 7.5), DECI = c(7.4, 8.1, 7.6, 8.5, 6.2, 8, 8.5, 5.9,
                                                  8.5, 8, 7.5, 7.1, 6.6, 7, 8, 7.3, 6.5, 7.5, 7.7, 6.6, 6.6, 6.8,
                                                  5.7, 7.3, 8.4, 8.8, 7.9, 8.5, 8.1, 8.6, 7.9, 7.8, 8.2, 8.5, 5.9,
                                                  8.2, 7.9, 7.7, 7.1, 8.3, 7.2, 7.7, 7.7), PREP = c(7.1, 8, 7.5, 8.7,
                                                                                                    5.7, 8.1, 8.5, 4.8, 8.4, 7.9, 7.1, 6.9, 7.1, 6.6, 7.9, 7.3, 6.8,
                                                                                                    7.1, 7.7, 6.2, 6.7, 6.4, 5.8, 7.3, 8.5, 8.9, 7.8, 8.4, 8.2, 9.1,
                                                                                                    8.2, 7.6, 8.4, 8.5, 5.6, 8.2, 7.9, 7.7, 6.7, 7.7, 6.9, 7.8, 7.4),
             FAMI = c(7.1, 8, 7.5, 8.7, 5.7, 8, 8.5, 5.1, 8.4, 8.1, 7.2, 7, 7.3,
                      6.8, 7.8, 7.2, 6.7, 7.3, 7.8, 6, 6.7, 6.3, 5.9, 7.3, 8.5, 9, 7.8,
                      8.3, 8.4, 9.1, 8.4, 7.4, 8.5, 8.5, 5.6, 8.1, 7.7, 7.7, 6.7, 7.6,
                      6.9, 8.2, 7.2), ORAL = c(7.1, 7.8, 7.3, 8.4, 5.1, 8, 8.6, 4.7, 8.4,
                                               7.7, 7.1, 7, 7.2, 6.3, 7.8, 7.1, 6.4, 7.1, 7.5, 5.8, 6.8, 6.3, 5.2,
                                               7.2, 8.4, 8.8, 7.8, 8.3, 8, 8.9, 7.7, 7.4, 8.1, 8.4, 5.3, 7.9, 7.6,
                                               7.5, 6.7, 7.5, 6.5, 8, 6.9), WRIT = c(7, 7.9, 7.4, 8.5, 5.3, 8,
                                                                                     8.4, 4.9, 8.5, 7.8, 7.2, 7.1, 7.2, 6.6, 7.8, 7.2, 6.5, 7.3, 7.6,
                                                                                     5.8, 6.8, 6.3, 5.8, 7.3, 8.4, 8.9, 7.7, 8.3, 8.1, 9, 7.9, 7.4, 8.3,
                                                                                     8.4, 5.5, 8, 7.7, 7.6, 6.7, 7.7, 6.6, 8.1, 7), PHYS = c(8.3, 8.5,
                                                                                                                                             7.9, 8.8, 5.5, 8.6, 9.1, 6.8, 8.8, 8.5, 8.4, 6.9, 8.1, 6.2, 8.4,
                                                                                                                                             8, 6.9, 8.1, 8, 7.2, 7.5, 7.4, 4.7, 7.8, 8.7, 9, 8.3, 8.8, 8.4,
                                                                                                                                             8.9, 8.4, 8.1, 8.7, 8.8, 6.3, 8, 8.1, 8.5, 8, 8.1, 7.6, 8.3, 7.8),
             RTEN = c(7.8, 8.7, 7.8, 8.7, 4.8, 8.6, 9, 5, 8.8, 7.9, 7.7, 7.2,
                      7.7, 6.5, 8, 7.6, 6.7, 7.4, 8, 6, 7.3, 6.6, 5.2, 7.6, 8.7, 9, 8.2,
                      8.7, 8.1, 9.2, 7.5, 7.9, 8.3, 8.8, 5.3, 8.2, 8, 7.7, 7, 7.9, 6.6,
                      8.1, 7.1), row.names = c("AARONSON,L.H.", "ALEXANDER,J.M.",
                                               "ARMENTANO,A.J.", "BERDON,R.I.", "BRACKEN,J.J.", "BURNS,E.B.",
                                               "CALLAHAN,R.J.", "COHEN,S.S.", "DALY,J.J.", "DANNEHY,J.F.",
                                               "DEAN,H.H.", "DEVITA,H.J.", "DRISCOLL,P.J.", "GRILLO,A.E.",
                                               "HADDEN,W.L.JR.", "HAMILL,E.C.", "HEALEY.A.H.", "HULL,T.C.",
                                               "LEVINE,I.", "LEVISTER,R.L.", "MARTIN,L.F.", "MCGRATH,J.F.",
                                               "MIGNONE,A.F.", "MISSAL,H.M.", "MULVEY,H.M.", "NARUK,H.J.",
                                               "O\'BRIEN,F.J.", "O\'SULLIVAN,T.J.", "PASKEY,L.", "RUBINOW,J.E.",
                                               "SADEN.G.A.", "SATANIELLO,A.G.", "SHEA,D.M.", "SHEA,J.F.JR.",
                                               "SIDOR,W.J.", "SPEZIALE,J.A.", "SPONZO,M.J.", "STAPLETON,J.F.",
                                               "TESTO,R.J.", "TIERNEY,W.L.JR.", "WALL,R.A.", "WRIGHT,D.B.",
                                               "ZARRILLI,K.J."))
View(data1)

install.packages("FactoMineR", dependencies = T)
library('FactoMineR')

#### 1.
library(psych)
# fa.parallel(data1)
# pca1 <- PCA(data1)
# egine1 <- get_eigenvalue(pca1)
# egine1
# fviz_pca_var(pca1,col.var = "lightblue")
# 
# pc2 <- principal(USJudgeRatings[,-1],nfactors = 1, rotate  = "varimax")
# pca1
# pc2
# USJudgeRatings

fa.parallel(USJudgeRatings[,-1],n.iter = 100, main = "scree plot with parallel analysis",fa = "pc")
##so 1 components are enough 

#### 2.
pca1 <- principal(USJudgeRatings[,-1],nfactors = 1,rotate = "none")
pca1

#### 3. 
pca_rotate <- principal(USJudgeRatings[,-1], nfactors = 1, rotate="varimax")
print(pca_rotate)

#### 4.
install.packages("factoextra")
library(factoextra)
data1_rotate$scores
pca_rotate <- psych::principal(USJudgeRatings[,-1], nfactors = 1, rotate="varimax", scores=T)
print(pca_rotate$scores)

#### 5.
factor.plot(pca_rotate,labels = rownames(pca_rotate$loadings))

#### 6.


############################################ Question 2 ################################################
data2 <- read_excel("Glass Identification Data.xlsx")
install.packages("xlsx")
library(xlsx)
library(readxl)
data2 <- data2[,-1]
data2 <- data2[,-10]
View(data2)

#### 1.
fa.parallel(data2[,-1], fa  = "pc", main = "scree of the parallel analysis")
# Harman23.cor$cov
# fa.parallel(Harman23.cor$cov[,-1],fa = "pc")

#### 2.
pca2 <-principal(data2[,-1],nfactors = 4, rotate = "none", covar = F,scores = T)
pca2
#### 3.
principal(r = data2[,-1], nfactors = 4, covar = F, rotate = "varimax")
pca2 <- principal(data2[,-1], rotate = "varimax",covar = F, nfactors = 4)
pca2
pca2_rotation <- principal(data2[,-1],nfactors = 4,rotate = "variate")
               ##2
pca2_scores <- pca2$scores
print(pca2_scores)
                ## 3
pca9 <- psych::principal(data2[,-1], nfactors = 4, rotate="varimax", scores=T)
print(pca9$scores)

#### 4.
factor.plot(pca2, labels = rownames(data2[,-1]))

#### 5.


################################### Question 3. #########################################
"Harman23.cor" <-
  structure(list(cov = structure(c(1, 0.846, 0.805, 0.859, 0.473, 
                                   0.398, 0.301, 0.382, 0.846, 1, 0.881, 0.826, 0.376, 0.326, 0.277, 
                                   0.415, 0.805, 0.881, 1, 0.801, 0.38, 0.319, 0.237, 0.345, 0.859, 
                                   0.826, 0.801, 1, 0.436, 0.329, 0.327, 0.365, 0.473, 0.376, 0.38, 
                                   0.436, 1, 0.762, 0.73, 0.629, 0.398, 0.326, 0.319, 0.329, 0.762, 
                                   1, 0.583, 0.577, 0.301, 0.277, 0.237, 0.327, 0.73, 0.583, 1, 
                                   0.539, 0.382, 0.415, 0.345, 0.365, 0.629, 0.577, 0.539, 1), .Dim = c(8, 
                                                                                                        8), .Dimnames = list(c("height", "arm.span", "forearm", "lower.leg", 
                                                                                                                               "weight", "bitro.diameter", "chest.girth", "chest.width"), c("height", 
                                                                                                                                                                                            "arm.span", "forearm", "lower.leg", "weight", "bitro.diameter", 
                                                                                                                                                                                            "chest.girth", "chest.width"))), center = c(0, 0, 0, 0, 0, 0, 
                                                                                                                                                                                                                                        0, 0), n.obs = 305), .Names = c("cov", "center", "n.obs"))

data3 <- data.frame(Harman23.cor)
View(data3)
drops <- c("center", "n.obs")
data3 <- data3[, !colnames(data3) %in% drops ]
View(data3)
str(Harman23.cor)
data3 <- Harman23.cor$cov
install.packages("GPArotation")
library("GPArotation")
#### 1.
fa.parallel(Harman23.cor$cov,fa = "pc",n.obs = 305,main = "scree of parallel analysis")

#### 2.
fa(Harman23.cor$cov, fm = "pa",nfactors = 2,rotate = "none")

#### 3.
pca3 <- principal(Harman23.cor$cov, rotate = "varimax",nfactors = 2)
pca3
pca3_fa <- fa(Harman23.cor$cov,rotate = "varimax", nfactors = 2)
pca3_fa


#### 4.
fac_score <- fa(Harman23.cor$cov,rotate = "varimax", nfactors = 2, scores = T)
head(fac_score$weights)

#### 5.
factor.plot(pca3_fa, labels = rownames(fac_score$weights))

#### 6.
fa.diagram(pca3_fa)



################################# Question 4. #############################################
str(Harman74.cor)
#### 1.
fa.parallel(Harman74.cor$cov,fa = "pc",n.obs = 305,main = "scree of parallel analysis")

#### 2.
fa(Harman74.cor$cov, fm = "pa" ,nfactors = 4,rotate = "none")

#### 3.
pca4_fa <- fa(Harman74.cor$cov,rotate = "varimax", nfactors = 4)
pca4_fa

#### 4.
fac2_score <- fa(Harman74.cor$cov,rotate = "varimax", nfactors = 4, scores = T)
head(fac2_score$weights)

#### 5.
factor.plot(pca4_fa, labels = rownames(fac2_score$weights))

#### 6.
fa.diagram(pca4_fa)


############################### Question 5. ################################################

data4 <- read_xlsx("Vertebral Column Data.xlsx")
View(data4)
data4 <- data4[,-7]

#### 1.
fa.parallel(data4, main = "scree pf parallel analysis",fa = "pc")

#### 2.
m4 <- as.matrix(data4)
dist1 <- dist(data4)
cmdscale(dist1,k = 2,eig = T)

#### 3.
score <- cmdscale(dist1,k= 2,eig = F)
factor.plot(score)

#### 4.
