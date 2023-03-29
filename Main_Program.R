#Main File to import dataset, train, test, validate and plot learning curves

#Install pre-requisite packages if not present
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("caret")
#install.packages("dplyr")

library(caret)
library(readr)
library(dplyr)

#Read csv data
dfHawks <- readr::read_csv("Hawks.csv")
#head(dfHawks)
#View(dfHawks)

any(is.na(dfHawks))
# returns true so need to remove null values

#removing null values
hawksOmit <- na.omit(dfHawks)
#head(hawksOmit)

#data pre-processing, sorting the data. 

#removing irrelevant values
hawksOmit %>% select(-c(BandNumber, Sex, Age, Tail)) -> hawksOmit
#head(hawksOmit)

#converting character to factor as chr cannot be converted to numeric value
hawksOmit$Species <- as.factor(hawksOmit$Species)
#head(hawksOmit)

#converting fct to ints (numeric value)
hawksOmit[sapply(hawksOmit, is.factor)] <- data.matrix(hawksOmit[sapply(hawksOmit, is.factor)])
#head(hawksOmit)

#rearrange dataset to match train/test modeling from template files
hawksOmit <- hawksOmit[, c(2,4,5,3,1)]
#head(hawksOmit)

#Randomly shuffle the dataset rows (repeatedly shuffled for 5 times)
rows_count <- nrow(hawksOmit)
for(k in 1:5){
  hawksOmit<-hawksOmit[sample(rows_count),]
}
#confirm row_count
nrow(hawksOmit)
head(hawksOmit)

source("Perceptron.r")
source("Evaluation_Cross_Validation.r")
source("Evaluation_Validation.r")
source("Evaluation_Curves.r")
source("MLP_hawks.r")


#Hold out 1/3 rd validation data set
validation_instances <- sample(nrow(hawksOmit)/3)
hawksOmit_validation<-hawksOmit[validation_instances,] #1/3 rd validation set
hawksOmit_train <- hawksOmit[-validation_instances,] #2/3 rd training set

nrow(hawksOmit_train)
nrow(hawksOmit_validation)

#Build Perceptron Model
p_model <- Perceptron(0.001)

#Set number of epochs (iterations)
num_of_epochs <- 100 #Ideally, run with 1000 number of epochs but 1000 takes considerable amount (>10 min) to train

#plot Learning Curve - Accuracy vs Training Sample size
plot_learning_curve(p_model, hawksOmit_train, hawksOmit_validation, number_of_iterations = num_of_epochs)

#plot Learning Curve - Accuracy vs Number of Epochs (Iterations)
plot_learning_curve_epochs(p_model, hawksOmit_train, hawksOmit_validation)

#plot Learning Curve - Accuracy vs Learning Rate values
plot_learning_curve_learning_Rates(hawksOmit_train, hawksOmit_validation, num_of_epochs = num_of_epochs)

#Train - Test - Cross Validate accross 10 folds
Cross_Validate(p_model, hawksOmit_train, num_of_iterations = num_of_epochs, num_of_folds = 10)
#Cross_Validate(ml_model, dataset, num_of_iterations, num_of_folds)

#Validate results with held out validation dataset

Validate(p_model, hawksOmit_train, hawksOmit_validation, number_of_iterations = 10)


