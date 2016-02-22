
### Machine learning using RMS Titanic
##get data from http://www.kaggle.com/c/titanic-gettingStarted/data

###load all required packages
suppressPackageStartupMessages(require('dplyr'))
suppressPackageStartupMessages(require('ggplot2'))
suppressPackageStartupMessages(require('caret'))
suppressPackageStartupMessages(require('party'))

##set plot option
options(repr.plot.width=6, repr.plot.height=3)


#set directory to folder with downloaded file
setwd("E:/")
filepath=getwd()
setwd(paste(filepath, "R_Script/Input", sep="/"))


# read the files
train = read.csv("train.csv", na.strings=c("NA", "")) 
#let check structure of data
str(train)

#lets see  first 10 rows
head(train)

#let convert sex, survive and pclass into numerical factor
train$Sex = factor(train$Sex)
train$Survived = factor(train$Survived)
train$Pclass = factor(train$Pclass)

# If NA is present in any row than ML will ignore these rows while
#building model which is not good as we want max no of data point
#lets see summary to find NA
summary(train)

# you can see there are NA in Age, cabin and embarked
# lets remove NA from Embarked
# As there was only 3 NA value, let assign NA to most counted port
train$Embarked[which(is.na(train$Embarked))] ='S'
table(train$Embarked, useNA = "always")
train$Pclass = factor(train$Pclass)
str(train)

#As ages have 177 missing value, lets fill in value
#As we see name have intial like Mr., Mrs., Master, We can use this info to add 
#to add ages, we will find mean ages os each intial and assign these value to NA
train$Name = as.character(train$Name)
table_names = table(unlist(strsplit(train$Name, "\\s+")))
sort(table_names[grep('\\.', names(table_names))], decreasing = T)

# lets get initial of missing value
table_na = train[which(is.na(train$Age)),]
table_names = table(unlist(strsplit(table_na$Name, "\\s+")))
sort(table_names[grep('\\.', names(table_names))], decreasing = T)

# lets gets mean age for each intial
sort(table_names[grep('\\.', names(table_names))], decreasing = T)
title = c("Mr\\.", "Miss\\.", "Mrs\\.", "Master\\." ,"Dr\\.")

#means
sapply(title, function(x){
       mean(train$Age[grepl(x, train$Name) & !is.na(train$Age)])
})

# assign NA  value to means
for (x in title){
    train$Age[grepl(x, train$Name) & is.na(train$Age)]=mean(train$Age[grepl(x, train$Name) & !is.na(train$Age)])
}
summary(train$Age)

#let make some plot
(ggplot(train, aes(Survived, fill=Sex)) 
     + geom_bar(aes(color = Sex) ) 
     + xlab("") 
     + ylab("No of Passanger") 
     +  scale_x_discrete(breaks=c("0", "1"),labels=c("Perished","Survived")))


(ggplot(train, aes(Age, fill=Survived)) 
     + geom_histogram( binwidth = 2 ) 
     + xlab("Age") 
     + ylab("No of Passanger")
     + scale_fill_discrete(breaks=c("0", "1"),labels=c("Perished","Survived")))
     

#let make some plot
(ggplot(train, aes(Pclass, fill=Survived)) 
     + geom_bar( ) 
     + xlab("Class") 
     + ylab("No of Passanger") 
     +  scale_x_discrete(breaks=c("1", "2", "3"),labels=c("First","Second", "third"))
     +scale_fill_discrete(breaks=c("0", "1"),labels=c("Perished","Survived")))

##lets covert eevrything to numeric
train$Sex = as.numeric(train$Sex)
train$Embarked = as.numeric(train$Embarked)
 train$Pclass= as.numeric(train$Pclass)
##lets splite data to training and testing
intrain<-createDataPartition(y=train$Survive,p=0.7,list=FALSE)
traingset = train[intrain,]
testset = train[-intrain,]

##Logistcs Regression 
reg = glm(Survived~Age+Pclass+Sex+SibSp+Embarked, data = traingset, family=binomial)
pred = predict(reg, testset, type='response')
class = ifelse(pred > .5,1,0)
tb = table(testset$Survive,class)


#confusion matrix
confusionMatrix(tb)



