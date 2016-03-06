########################################################################################################
#                                            SVM model                                                 # 
########################################################################################################

###load all required packages
suppressPackageStartupMessages(require('dplyr'))
suppressPackageStartupMessages(require('ggplot2'))
suppressPackageStartupMessages(require('caret'))
suppressPackageStartupMessages(require('e1071'))
suppressPackageStartupMessages(require('party'))
suppressPackageStartupMessages(require('nnet'))
suppressPackageStartupMessages(require('randomForest'))
suppressPackageStartupMessages(require('pROC'))

#set directory to folder with downloaded file
setwd("M:/")
filepath=getwd()
setwd(paste(filepath, "R_Script/Input", sep="/"))

###lets built a funcation to preprocess file and remove all NA 
model= function(x) {
  train = read.csv(x ,na.strings=c("NA", "")) 
  
  # Convert string to factor
  train$Sex = factor(train$Sex)
  train$Pclass = factor(train$Pclass)
  
  #fill na on Embarked with S
  train$Embarked[which(is.na(train$Embarked))] ='S'
  
  # lets gets mean age for each title to fill na value
  title = c("Mr\\.", "Miss\\.", "Mrs\\.", "Master\\." ,"Dr\\.", "Ms\\.")
  for (x in title){
    train$Age[grepl(x, train$Name) & is.na(train$Age)]=mean(train$Age[grepl(x, train$Name) & !is.na(train$Age)])
  }
  
  train$family = ifelse(train$Parch >0,1,0)
  #train$familyandspouch = ifelse(train$Parch >0 | train$SibSp>0,1,0)
  
  
  #return everything as numberic
  train$Sex = as.numeric(train$Sex)
  train$Pclass = as.numeric(train$Pclass)
  train$Embarked = as.numeric(train$Embarked)
  train$Fare[is.na(train$Fare)] = median(train$Fare, na.rm = T)
  return (train)
}



##lets read train csv
train=model('train.csv')
summary(train)

#normal = preProcess(train, method = c("scale"))
#train <-  predict(normal, train)


##lets splite data to training and testing
set.seed(100)
intrain<-createDataPartition(y=train$Survive,p=0.7,list=FALSE)
trainset = train[intrain,]
testset = train[-intrain,]

##let train data 

tuned = tune.svm(Survived ~ Pclass+Sex+Age+SibSp+family+Embarked +Fare, 
                 data = train, gamma = 10^(-6:-1),cost = 10^(1:2))
fit = tune.svm(Survived ~ Pclass+Sex+Age+SibSp+family+Embarked +Fare,
               data = train, kernel="radial",
               gamma = tuned$best.parameters$gamma, 
               cost = tuned$best.parameters$cost,
               tunecontrol=tune.control(cross=10))


###parameters
fit$performances
fit = fit$best.model



####let test data
pred = predict(fit, testset, type='raw',probability=TRUE)
class = ifelse(pred >= .5,1,0)
tb = table(testset$Survive,class)
confusionMatrix(tb)


##################################ROC 
pred.rocr = prediction(pred, testset$Survived)
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")
perf.tpr.rocr = performance(pred.rocr, "tpr","fpr")
plot(perf.tpr.rocr, colorize=T,main=paste("AUC:",(perf.rocr@y.values)))

############################let get test predication for new data########################

##let read test.csv and clean data
test = model("test.csv")
summary(test)

# We can see that test data still has NA in ages that as there is Ms.(Ms. is same as Mss.) 
# in testset which we never had in train set, let put means of Mss. in this data too
test$Age[grepl("Ms\\.", test$Name) & is.na(test$Age)]=mean(test$Age[grepl("Miss\\.", test$Name) & !is.na(test$Age)])


##lets predict 
pred = predict(fit, test, type='raw',probability=TRUE)
class = as.data.frame(ifelse(pred >= .5,1,0))


##let make data frame of pred
passangerid = as.data.frame(test[,1])
class = cbind(passangerid, class)
colnames(class) = c("PassengerId", "Survived")
write.csv(class, "svm.csv", row.names=F)


rm(list=ls())

##########################################Author Bikram Dahal#########################################