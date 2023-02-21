#Installation of necessary libraries

install.packages("Amelia")
install.packages("dplyr")
install.packages("gbm")
install.packages("randomForest")
install.packages("ggplot2")
install.packages("datarium")
install.packages("reshape2")
install.packages("ggpubr")
install.packages("corrplot")
install.packages("pROC")
install.packages("tree")
install.packages("ROCR")


bc <- read.csv('bco.csv', header = FALSE, sep = ",", na.strings = c(""))
print(head(bc))

colnames(bc) <- c("ID","clump_thickness","uniformity_of_cell_size",
                  "uniformity_of_cell_shape","marginal_adhesion",
                  "single_epithelial_cell_size","bare_nuclei","bland_chromatin",
                  "normal_nucleoli","mitoses","Class")

head(bc)
str(bc)
summary(bc)

bc$bare_nuclei <- as.integer(bc$bare_nuclei)
str(bc)
summary(bc)

library("Amelia")
missmap(bc, main = 'Missing Map', col = c('yellow','black'), legend = FALSE)
bc <- na.omit(bc)
missmap(bc, main = 'Missing Map', col = c('yellow','black'), legend = FALSE)

str(bc)
summary(bc)

###Plots for understanding the data

library(ggplot2)
library(ggpubr)

fig1 <- ggplot(bc, aes(clump_thickness)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig2 <- ggplot(bc, aes(uniformity_of_cell_size)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig3 <- ggplot(bc, aes(uniformity_of_cell_shape)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig4 <- ggplot(bc, aes(marginal_adhesion)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig5 <- ggplot(bc, aes(single_epithelial_cell_size)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig6 <- ggplot(bc, aes(bare_nuclei)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig7 <- ggplot(bc, aes(bland_chromatin)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig8 <- ggplot(bc, aes(normal_nucleoli)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig9 <- ggplot(bc, aes(mitoses)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig10 <- ggplot(bc, aes(Class)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')

ggarrange(fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9, fig10, ncol = 3, nrow = 4, 
          align=("v"))

###Divide test and training set

set.seed(12345)
train = runif(nrow(bc))<0.6
addmargins(table(train,bc$Class))

###Examine correlation matrix

library("corrplot")

corrplot(cor(bc[2:11]), method = "circle")

###RANDOM FOREST
library(randomForest)
set.seed(12345)
fit = randomForest(x=bc[train, 2:10], y = factor(1-bc$Class[train]), 
                   xtest=bc[!train, 2:10], ntree=100, keep.forest = T)
fit

set.seed(1) # Bagging
bag.bc <- randomForest(Class ~ . -ID, data = bc[train,], mtry = 10, 
                             importance = TRUE)
yhat.bag <- predict(bag.bc, newdata = bc[!train,])
mean((yhat.bag - bc[!train,]$Class)^2)

set.seed(1) # Random Forest 
rf.bc <- randomForest(Class ~ . -ID, data = bc[train,], mtry = 5, 
                            importance = TRUE)
yhat.rf <- predict(rf.bc, newdata = bc[!train,])
mean((yhat.rf - bc[!train,]$Class)^2)


###Area under the curve
library(pROC)
library(ROCR)

auc(1-bc$Class[!train],fit$test$votes[,2])
plot.roc(bc$Class[!train], fit$test$votes[,2], print.auc = T )

###Predicted Probabilities of the test set
hist(fit$test$votes[,2], main="Predicted Probabilities for test set")

###Variable importance plot
varImpPlot(fit)

###Partial Dependence on uniformity_of_cell_size 
pdp1 <- partialPlot(fit, bc[train,], "uniformity_of_cell_size")

###Partial Dependence on uniformity_of_cell_shape
pdp2 <- partialPlot(fit, bc[train,], "uniformity_of_cell_shape")

###Partial Dependence on bare_nuclei
pdp3 <- partialPlot(fit, bc[train,], "bare_nuclei")

###GBM, additive (interaction.depth =1)

library(gbm)
set.seed(1)
boost.bc <- gbm( Class ~. -ID, data = bc[train,], distribution = "gaussian", 
                 n.trees = 10000, interaction.depth = 1)
boost.bc
summary(boost.bc)

yhat.boost <- predict(boost.bc, newdata = bc[!train,], n.trees = 10000)
mean((yhat.boost - bc[!train,]$Class)^2)

#AUC values for boosted tree

phat = predict(boost.bc, bc[!train,], n.trees = 10000, type="response")
auc(bc$Class[!train], phat)

###GBM with interaction depth = 2
set.seed(1)
boost.bc2 <- gbm( Class ~. -ID, data = bc[train,], distribution = "gaussian", 
                 n.trees = 10000, interaction.depth = 2)
boost.bc2
summary(boost.bc2)

yhat.boost2 <- predict(boost.bc2, newdata = bc[!train,], n.trees = 10000)
mean((yhat.boost2 - bc[!train,]$Class)^2)

#AUC values for boosted tree

phat = predict(boost.bc2, bc[!train,], n.trees = 10000, type="response")
auc(bc$Class[!train], phat)

#Fitting a simple tree
library(tree)
fit.tree = tree(factor(Class) ~ . -ID, data = bc[train,])
fit.tree

plot(fit.tree, type = "uniform")
text(fit.tree)
summary(fit.tree)

Y.test = bc[!train,]$Class ### Calculating test-MSE
pred.bc = predict(fit.tree, bc[!train,])
MSE.test.tree = mean((Y.test - pred.bc)^2)

#Apply k-fold cross-validation for pruning

set.seed(3)
### Performing cross-validation
cv.bc = cv.tree(fit.tree, FUN = prune.misclass)
cv.bc

### Plotting the results
plot(cv.bc$size, cv.bc$dev, type = "b")

### optimal terminal node size (3)
nodes.opt = cv.bc$size[which.min(cv.bc$dev)]
### pruning tree
prune.tree.bc = prune.misclass(fit.tree, best = nodes.opt)
### Plotting pruned tree with 3 terminal nodes
prune.tree.bc

plot(prune.tree.bc)
text(prune.tree.bc, pretty = 0)
summary(prune.tree.bc)

### Calculating test-MSE of pruned tree
pred.pruned.bc = predict(prune.tree.bc, bc[!train,])
MSE.pruned.bc = mean((Y.test - pred.pruned.bc)^2)

### Estimating fitted values for pruned and unpruned tree
Y.train = bc[train,]$Class
y.pred.pru.train = predict(prune.tree.bc, bc[train,], type = "class")
y.pred.unpru.train = predict(fit.tree, bc[train,], type = "class")
### Creating confusion matrices
conf.mat.pru.train = table(y.pred.pru.train, Y.train)
conf.mat.unpru.train = table(y.pred.unpru.train, Y.train)
### Calculating training error rates
train.err.pru = 1 - sum(diag(conf.mat.pru.train)) / sum(conf.mat.pru.train)
train.err.unpru = 1 - sum(diag(conf.mat.unpru.train)) / sum(conf.mat.unpru.train)


### Estimating fitted values for pruned and unpruned tree
y.pred.pru.test = predict(prune.tree.bc, bc[!train,], type = "class")
y.pred.unpru.test = predict(fit.tree, bc[!train,], type = "class")
### Creating confusion matrices
conf.mat.pru.test = table(y.pred.pru.test, Y.test)
conf.mat.unpru.test = table(y.pred.unpru.test, Y.test)
### Calculating test error rates
test.err.pru = 1 - sum(diag(conf.mat.pru.test)) / sum(conf.mat.pru.test)
test.err.unpru = 1 - sum(diag(conf.mat.unpru.test)) / sum(conf.mat.unpru.test)

###Multiple Linear Regression Comparison

#Multiple Linear Regression on the training set
reg.train <- lm(Class ~ . -ID , data = bc[train,])
summary(reg.train)

### Calculating test-MSE for multiple linear regression
pred.reg = predict(reg.train, bc[!train,])
MSE.reg = mean((Y.test - pred.reg)^2)




