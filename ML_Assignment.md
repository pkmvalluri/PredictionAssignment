Practical Machine Learning - Prediction Assignment Markdown document
================

Prediction Assignment
---------------------

The goal of this project is to predict the manner in which they did the exercise and to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

This is the "classe" variable in the training set.Also model to predict 20 different test cases.

Include caret and load files
----------------------------

First, I loaded the data both from the provided training and test data provided by COURSERA. Some values contained a "\#DIV/0!" that I replaced with an NA value: The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.0.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.0.3

``` r
setwd("C:\\Krishna\\CourseRa\\Practicle Machine Learning\\Week4\\PredictionAssignment")

trainingFile <- "pml_training.csv"
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

download.file(url, destfile = trainingFile )

training <- read.csv(trainingFile , na.strings = c("NA","#DIV/0!",""))

testingFile <- "pml_testing.csv"

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, destfile = testingFile )

testing <- read.csv(testingFile, na.strings = c("NA","#DIV/0!",""))
```

(Clean and) Remove invalid predictors
-------------------------------------

Remove columns with Near Zero Values and Remove columns with NA or is empty

``` r
subTrain <- training[, names(training)[!(nzv(training, saveMetrics = T)[, 4])]]

subTrain <- subTrain[, names(subTrain)[sapply(subTrain, function (x) ! (any(is.na(x) | x == "")))]]
```

Remove V1 which seems to be a serial number, and
------------------------------------------------

cvtd\_timestamp that is unlikely to influence the prediction

``` r
subTrain <- subTrain[,-1]
subTrain <- subTrain[, c(1:3, 5:58)]
```

Separate the data to be used for Cross Validation
-------------------------------------------------

Divide the training data into a training set and a validation set

``` r
inTrain <- createDataPartition(subTrain$classe, p = 0.6, list = FALSE)
subTraining <- subTrain[inTrain,]
subValidation <- subTrain[-inTrain,]
```

Create the prediction model using random forest
===============================================

Check if model file exists from previous run and as aadvised by the forums running on mutiple cores takes a long time so loading the model from previous runs

``` r
model <- "modelFit.RData"
if (!file.exists(model)) {

    # If not, set up the parallel clusters.  
    require(parallel)
    require(doParallel)
    cl <- makeCluster(detectCores() - 1)
    registerDoParallel(cl)
    
    fit <- train(subTraining$classe ~ ., method = "rf", data = subTraining)
    save(fit, file = "modelFit.RData")
    
    stopCluster(cl)
} else {
    # model exists from previous run, load it and use it.  
    load(file = "modelFit.RData", verbose = TRUE)
}
```

    ## Loading objects:
    ##   fit

Measure the Accuracy and Sample Error of the prediction model
=============================================================

Measure its accuracy using confusion matrix

``` r
predTrain <- predict(fit, subTraining)
```

    ## Loading required package: randomForest

    ## Warning: package 'randomForest' was built under R version 3.0.3

    ## randomForest 4.6-10

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
confusionMatrix(predTrain, subTraining$classe)
```

    ## Loading required namespace: e1071

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3348    1    0    0    0
    ##          B    0 2278    1    0    0
    ##          C    0    0 2053    1    0
    ##          D    0    0    0 1929    1
    ##          E    0    0    0    0 2164
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9997          
    ##                  95% CI : (0.9991, 0.9999)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9996          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9996   0.9995   0.9995   0.9995
    ## Specificity            0.9999   0.9999   0.9999   0.9999   1.0000
    ## Pos Pred Value         0.9997   0.9996   0.9995   0.9995   1.0000
    ## Neg Pred Value         1.0000   0.9999   0.9999   0.9999   0.9999
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1934   0.1743   0.1638   0.1838
    ## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      0.9999   0.9997   0.9997   0.9997   0.9998

Using the validation subset and create a prediction. Then measure its accuracy. From the training subset, the accuracy is very high, at above 99%. The sample error is 0.0008.
===============================================================================================================================================================================

``` r
predValidation <- predict(fit, subValidation)
confusionMatrix(predValidation, subValidation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2232    0    0    0    0
    ##          B    0 1517    0    0    0
    ##          C    0    1 1368    1    0
    ##          D    0    0    0 1284    2
    ##          E    0    0    0    1 1440
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9994          
    ##                  95% CI : (0.9985, 0.9998)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9992          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9993   1.0000   0.9984   0.9986
    ## Specificity            1.0000   1.0000   0.9997   0.9997   0.9998
    ## Pos Pred Value         1.0000   1.0000   0.9985   0.9984   0.9993
    ## Neg Pred Value         1.0000   0.9998   1.0000   0.9997   0.9997
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1933   0.1744   0.1637   0.1835
    ## Detection Prevalence   0.2845   0.1933   0.1746   0.1639   0.1837
    ## Balanced Accuracy      1.0000   0.9997   0.9998   0.9991   0.9992

The accuracy of this model is high and no need to build another prediction model for better accuracy to stack multiple prediction models. takes too long a time to run another training process. Given the level of accuracy I am fine with this prediction model.
==================================================================================================================================================================================================================================================================

From the model, the following are the list of important predictors in the model.

``` r
varImp(fit)
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 60)
    ## 
    ##                      Overall
    ## raw_timestamp_part_1 100.000
    ## num_window            53.635
    ## roll_belt             48.434
    ## pitch_forearm         28.439
    ## magnet_dumbbell_z     23.024
    ## yaw_belt              20.020
    ## magnet_dumbbell_y     17.830
    ## pitch_belt            16.885
    ## roll_forearm          13.296
    ## accel_dumbbell_y       8.140
    ## roll_dumbbell          7.799
    ## magnet_dumbbell_x      7.462
    ## accel_belt_z           6.566
    ## accel_forearm_x        6.489
    ## magnet_belt_z          6.283
    ## total_accel_dumbbell   6.153
    ## magnet_belt_y          6.079
    ## accel_dumbbell_z       5.720
    ## yaw_dumbbell           3.761
    ## magnet_arm_x           3.405

Based on the validation accuracy at over 99% and Cross-Validation out-of-sample error rate of 0.03%, with CI between 99.87% to 99.97%, the prediction model should be applied to the final testing set, and predict the classe in the 20 test cases.

``` r
fit$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 31
    ## 
    ##         OOB estimate of  error rate: 0.11%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3348    0    0    0    0 0.0000000000
    ## B    3 2275    1    0    0 0.0017551558
    ## C    0    5 2049    0    0 0.0024342746
    ## D    0    0    1 1927    2 0.0015544041
    ## E    0    0    0    1 2164 0.0004618938

Apply the prediction model to the testing data. The predicted classification are (and were 100% accurate):
==========================================================================================================

``` r
predTesting <- predict(fit, testing)

predTesting
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

generate the files for submission with the given R code from the assignment.
============================================================================

``` r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predTesting)
```
