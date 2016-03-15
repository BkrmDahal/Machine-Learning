########################################################################################################
#                              deeplearning Using h2o--out_of_bag                                      # 
########################################################################################################

###load all required packages
suppressPackageStartupMessages(require('dplyr'))
suppressPackageStartupMessages(require('ggplot2'))
suppressPackageStartupMessages(require('caret'))
suppressPackageStartupMessages(require('e1071'))
suppressPackageStartupMessages(require('h2o'))




###read files
train = read.csv('https://raw.githubusercontent.com/BkrmDahal/All_file_backup_for_analysis/master/train_titanic.csv' ,na.strings=c("NA", "")) 
test = read.csv("https://raw.githubusercontent.com/BkrmDahal/All_file_backup_for_analysis/master/test_titanic.csv",na.strings=c("NA", ""))

##let see the summary
summary(train)


##Feature engineering

#1. fill na on Embarked with S
train$Embarked[which(is.na(train$Embarked))] ='S'
test$Embarked[which(is.na(test$Embarked))] ='S'


#2. lets gets mean age for each title to fill na value
title = c("Mr\\.", "Miss\\.", "Mrs\\.", "Master\\." ,"Dr\\.", "Ms\\.", "Miss\\.")
for (x in title){
  train$Age[grepl(x, train$Name) & is.na(train$Age)]=mean(train$Age[grepl(x, train$Name) & !is.na(train$Age)])
}
for (x in title){
  test$Age[grepl(x, test$Name) & is.na(test$Age)]=mean(test$Age[grepl(x, test$Name) & !is.na(test$Age)])
}

# We can see that test data still has NA in ages that as there is Ms.(Ms. is same as Mss.) 
# in testset which we never had in train set, let put means of Mss. in this data too
test$Age[grepl("Ms\\.", test$Name) & is.na(test$Age)]=mean(test$Age[grepl("Miss\\.", test$Name) & !is.na(test$Age)])

#3. Lets add family member
train$family = ifelse(train$Parch >0,1,0)
test$family = ifelse(test$Parch >0,1,0)

#4. Lets add fare
train$Fare[is.na(train$Fare)] = median(train$Fare, na.rm = T)
test$Fare[is.na(test$Fare)] = median(test$Fare, na.rm = T)

##return everything as numberic
train$Sex = as.numeric(train$Sex)
train$Embarked = as.numeric(train$Embarked)
test$Sex = as.numeric(test$Sex)
test$Embarked = as.numeric(test$Embarked)


##removed any coloume that will not be used in anlysis
temp = c("PassengerId",  "Name", "Parch", "Ticket","Cabin", "Age" )
for( i in temp){
  train[, i] <- NULL
}
temp = c(  "Name", "Parch", "Ticket","Cabin", "Age" )
for( i in temp){
  test[, i] <- NULL
}

###let runmodel
##lets splite data to training and testing
set.seed(100)
intrain<-createDataPartition(y=train$Survive,p=0.8,list=FALSE)
trainset = train[intrain,]
testset = train[-intrain,]

###lets initalize h2o server

localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

##lets make h2o 
h2o_trainset <- as.h2o( trainset, "trainset")
h2o_testset <- as.h2o( testset, "testset")
h2o_test <- as.h2o(test, "test")

##lets fit
fit <- h2o.deeplearning(x = 2:7,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = h2o_trainset, # data in H2O format
                   activation = "TanhWithDropout", # or 'Tanh'
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                   hidden = c(50,50,50), # three layers of 50 nodes
                   epochs = 100)

## Using the model for predictions
h2o_yhat_test <- h2o.predict(fit, h2o_testset)

## Converting H2O format into data frame
df_yhat_test <- as.list(h2o_yhat_test)

###confusionMatrix
class = ifelse(df_yhat_test >= .5,1,0)
tb = table(testset$Survive,class)
confusionMatrix(tb)

##################################ROC 
pred.rocr = prediction(df_yhat_test, testset$Survived)
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")
perf.tpr.rocr = performance(pred.rocr, "tpr","fpr")
plot(perf.tpr.rocr, colorize=T,main=paste("AUC:",(perf.rocr@y.values)))


###lets predict for real test data
##lets predict 
h2o_yhat_test <- h2o.predict(fit, h2o_test)

## Converting H2O format into data frame
df_yhat_test <- as.list(h2o_yhat_test)
class = as.data.frame(ifelse(df_yhat_test >= .5,1,0))


##let make data frame of pred
passangerid = as.data.frame(test[,1])
class = cbind(passangerid, class)
colnames(class) = c("PassengerId", "Survived")
write.csv(class, "svm.csv", row.names=F)




##########################################Author Bikram Dahal#########################################
