##### KNN Modeling Infant Mortality in Michoacan


install.packages(c("caret","class","dplyr","e1071","FNN","gmodels","psych"))

install.packages("psych")

library(caret)
library(class)
library(dplyr)
library(e1071)
library(FNN) 
library(gmodels) 
library(psych)




## Loading Data into R

data<-read.csv("E:/Spring 2018/Job/Sample/mich_dat_mod.csv")


##changing name of outcome
colnames(data)[1] <- "mortalidad"
variable.names(data)

# put outcome in its own object
mort_outcome <- data %>% select(mortalidad)

# remove original from the data set
data <- data %>% select(-mortalidad)



## integer values
str(data)


##Scaling integer values
data[,c("pib_percapita_dolares","des_social")]<-scale(data[,c("pib_percapita_dolares","des_social")])

head(data)



#### No categorical values





set.seed(1234) # set the seed to make the partition reproducible

# 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

train_ind <- sample(seq_len(nrow(data)), size = smp_size)

# creating test and training sets that contain all of the predictors
reg_pred_train <- data[train_ind, ]
reg_pred_test <- data[-train_ind, ]



#### Same for outcome variable

mort_outcome_train <- mort_outcome[train_ind, ]
mort_outcome_test <- mort_outcome[-train_ind, ]


### KNN regression
reg_results <- knn.reg(reg_pred_train, reg_pred_test, mort_outcome_train, k = 9)
print(reg_results)

### Plotting residuals

plot(mort_outcome_test, reg_results$pred, xlab="y", ylab=expression(hat(y)))


#mean square prediction error
mean((mort_outcome_test - reg_results$pred) ^ 2)
#1.839974


#mean absolute error
mean(abs(mort_outcome_test - reg_results$pred))
#1.071264




########################################################################## PLS & PCR

install.packages(c("AppliedPredictiveModeling","elasticnet","pls", "RColorBrewer","reshape2"))



install.packages("elasticnet")
install.packages("pls")


library(AppliedPredictiveModeling)
library(elasticnet)
library(pls)
library(RColorBrewer)
library(reshape2)


data_p<-read.csv("E:/Spring 2018/Job/Sample/mich_dat_mod.csv")




##Components & Variance


pcaObj <- prcomp(data_p, center = TRUE, scale = TRUE)

pctVar <- pcaObj$sdev^2/sum(pcaObj$sdev^2)*100
head(pctVar)


library(caret)

##changing name of outcome

colnames(data_p)[1] <- "mortalidad"
variable.names(data_p)


################################# Linear Model

set.seed(1029)


ctrl <- trainControl(method = "repeatedcv", repeats = 5)

set.seed(529)
mortLm <- train(x = as.data.frame(reg_pred_train), y = mort_outcome_train, method = "lm", trControl = ctrl)
mortLm



##################################### PLS & PCA


set.seed(529)
mortPCR <- train(x = reg_pred_train, y = mort_outcome_train,
                   method = "pcr",
                   trControl = ctrl, tuneLength = 25)
set.seed(529)
mortPLS <- train(x = reg_pred_train, y = mort_outcome_train,
                   method = "pls",
                   trControl = ctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 25)
## For Figure
comps <- rbind(mortPLS$results, mortPCR$results)

comps$Model <- rep(c("PLS", "PCR"), each = 10)



#### Beautiful Plot
bookTheme()
xyplot(RMSE ~ ncomp, data = comps, 
       groups = Model, type = c("g", "o"), 
       auto.key = list(columns = 2), 
       xlab = "#Components")


########################### Elastic Net
set.seed(529)
mortENet <- train(x = reg_pred_train, y = mort_outcome_train,
                    method = "enet",
                    trControl = ctrl,
                    preProcess = c("center", "scale"),
                    tuneGrid = expand.grid(lambda = c(0, .001, .01, .1, 1),
                                             fraction = seq(0.05, 1, length = 20)))



bookTheme()
plot(mortENet, plotType = "level")


############################## Easier KNN

set.seed(921)
knnModel <- train(x = reg_pred_train,
                    y = mort_outcome_train,
                    method = "knn",
                    preProc = c("center", "scale"),
                    tuneLength = 10)

knnModel


knnPred <- predict(knnModel, newdata = reg_pred_test)
## The function 'postResample' can be used to get the test set
## perforamnce values
postResample(pred = knnPred, obs = mort_outcome_test)


############################ MARS


install.packages("earth")
install.packages("kernlab")
install.packages("latticeExtra")
install.packages("mlbench")
install.packages("nnet")
install.packages("plotmo")

library(earth)
library(kernlab)
library(latticeExtra)
library(mlbench)
library(nnet)
library(plotmo)








marsGrid <- expand.grid(degree = 1:2, nprune = seq(2,14,by=2))

set.seed(921)
marsModel <- train(x = reg_pred_train,
                     y = mort_outcome_train,
                     method = "earth",
                     preProc = c("center", "scale"),
                     tuneGrid = marsGrid)

marsPred <- predict(marsModel, newdata = reg_pred_test)
postResample(pred = marsPred, obs = mort_outcome_test)
plot(marsModel)


## Better MARS

set.seed(529)
mortMARS <- train(x = reg_pred_train, y = mort_outcome_train, 
                  method = "earth", 
                  trControl = ctrl, 
                  tuneLength = 25)



varImp(mortMARS)
varImp(mortBMARS)




##Bagged Mars

set.seed(529)
mortBMARS <- train(x = reg_pred_train, y = mort_outcome_train,
                     method = "bagEarth",
                     trControl = ctrl,
                     tuneLength = 25,
                     B = 20)


plotDat <- rbind(mortMARS$results, mortBMARS$results)
plotDat$Model <- rep(c("Basic", "Bagged"), each = nrow(mortMARS$results))


bookTheme()
xyplot(RMSE ~ nprune, 
       data = plotDat,
       type = c("g", "o"),
       groups = Model,
       auto.key = list(columns = 2))



########### SVR Model

set.seed(921)
svmRModel <- train(x = reg_pred_train,
                     y = mort_outcome_train,
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneLength = 8)

svmRPred <- predict(svmRModel, newdata = reg_pred_test)
postResample(pred = svmRPred, obs = mort_outcome_test)
plot(svmRModel, scales = list(x = list(log = 2)))





##qvsm


polyGrid <- expand.grid(degree = 1:2,
                          C = 2^seq(8, 15, length = 8),
                          scale = c(.5, .1, 0.01))
polyGrid <- polyGrid[!(polyGrid$scale == .5 & polyGrid$degree == 2),]

set.seed(529)
mortQSVM <- train(x = reg_pred_train, y = mort_outcome_train,
                    method = "svmPoly",
                    preProcess = c("center", "scale"),
                    trControl = ctrl,
                    tuneGrid = polyGrid)








###################################### Neural Net


set.seed(529)
meatNet <- train(x = reg_pred_train, y = mort_outcome_train,
                   method = "nnet",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   linout = TRUE,
                   trace = FALSE,
                   tuneGrid = expand.grid(size = 1:9,
                                            decay = c(0, .001, .01, .1)))

set.seed(529)
meatPCANet <- train(x = reg_pred_train, y = mort_outcome_train,
                      method = "nnet",
                      trControl = ctrl,
                      preProc = c("center", "scale", "pca"),
                      linout = TRUE,
                      trace = FALSE,
                      tuneGrid = expand.grid(size = 1:9,
                                               decay = c(0, .001, .01, .1)))

plotNNet <- rbind(meatNet$results, meatPCANet$results)
plotNNet$Model <- rep(c("Raw", "PCA"), each = nrow(meatNet$results))


bookTheme()
print(xyplot(RMSE ~ size|Model, 
             data = plotNNet,
             groups = decay,
             type = c("g", "o"),
             auto.key = list(columns = 4, 
                             lines = TRUE, 
                             cex = .6,
                             title = "Weight Decay"),
             scales = list(y = list(relation = "free"))))

############## Resampling Profile


meatResamples <- resamples(list(PLS = mortPLS,
                                PCA = mortPCR,
                                MARS = marsModel,
                                "ENet" = mortENet,
                                  "Bagged MARS" = mortBMARS,
                                  SVMrad = svmRModel,
                                  NNet = meatNet,
                                  KNN = knnModel))

