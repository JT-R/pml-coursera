Practical Machine Learning course project
========================================================
Jakub Tomaszewski
--------------------------------------------------------
**2014-06-21**

In this document I present the approach I followed to accomplish the final course project. The main goal was to create, validate and choose possibly the best predictive model solving the task of class separation in the given dataset.

## Initial data preparation

After creating the project in RStudio and downloading training and test dataset, I used the following code to read these datasets:

```
pml <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
```

*caret* package was used to fit a collection of machine learning algorithms and to assess their goodness.

*Hmisc::describe* function allowed me to get sense of the data and to find out that some attributes have a lot of missing values.

As it appears after using the *caret::nearZeroVar* function, variability of these attributes was too small to be useful in explaining the differences between classes. These attributes were excluded from the group of the possible predictors in the models created later. Furthermore, I removed the attributes containing missing values in the test set.

```
library(caret)
library(Hmisc)
which_empty <- names(which(apply(test,2,function(x) any(is.na(x)))))
column_selection <- intersect(colnames(pml)[which(nearZeroVar(pml, saveMetrics=TRUE)$nzv==FALSE)],colnames(test))
column_selection <- setdiff(column_selection,c("X","user_name",which_empty))
pml <- pml[,union(column_selection,"classe")]
test <- test[,column_selection]
```

The next step was to create a partition of the *pml-training* dataset into 3 disjoint parts:
* training set (50% of obs.) to train the models,
* validation set (25% of obs.) to tune the parameters of fitted models,
* testing set (25% of obs.) to assess the goodness of fit of these models.

```
set.seed(87)
n <- nrow(pml)
training_indices <- sample(1:n,0.5*n)
validation_indices <- sample(setdiff(1:n,training_indices),0.25*n)
testing_indices <- setdiff(1:n,union(training_indices,validation_indices))
training <- pml[training_indices,]
validation <- pml[validation_indices,]
testing <- pml[testing_indices,]
```

## Modelling

Having the data ready for modelling, I created 3 models with default tuning parameters:
* CART tree
* Linear Discriminant Analysis classifier
* Random forest

```
CV_control <- trainControl(
                method = "repeatedcv",
                number = 10,
                repeats = 5)

model1 <- train(classe~.,method="rpart",data=training,trControl=CV_control)
model2 <- train(classe~.,method="lda",data=training,trControl=CV_control)
model3 <- train(classe~.,method="rf",data=training,trControl=CV_control)
```

When the estimates and partition rules were calculated, predictions were made and evaluated on the independent validation set:

```
pred1 <- predict(model1,newdata=validation)
pred2 <- predict(model2,newdata=validation)
pred3 <- predict(model3,newdata=validation)
confusionMatrix(pred1,validation[complete.cases(validation),"classe"])
confusionMatrix(pred2,validation[complete.cases(validation),"classe"])
confusionMatrix(pred3,validation[complete.cases(validation),"classe"])
```

It was surprising that even without tuning parameters and further ensembling of the models, cross-validation accuracy of predictions made by random forest model was almost $99\%$! 

In case of LDA and CART models, accuracy was significantly worse, respectively: $84.83\%$ and $53.52\%$. Depending on the seed chosen before partitioning the initial dataset, out of sample error for LDA and random forest varies between $0\%-1\%$. Default parameters of these models were not changed for obtaining the further improvement of their accuracy (because there was no need to improve the classifier performance), so in this case validation dataset played a role of the testing dataset. Finally, random forest model appeared to be the most accurate of all evaluated models.

Dear Reader!
--------------------------------------------------------
Thank you for your effort reading this report. I will be grateful to receive your advices on improvement of my modelling approach! I hope you had as much fun as I did participating in this course. :)

