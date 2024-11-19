###################################################
## Fitting trees to the data for best model estimation.
## model creation - Gayathree, Maru, Jenny, Ronak.
## first we train a single tree and check its RMSE_train
## Second we train a random forest and check its RMSE_train
## Third we train a Boosted tree and check its RMSE_train
## Lastly we check the variable importance of each individual variable used to train.
###################################################

library(tree)
library(MASS)
library(readr)

################################################################################
## Fit a big tree to the NBA data using rpart
## Vary across many minsplit values.
## For each minsplit, vary across CP values and get best fit.
## Compare across each best fit for minimum validation/oos error.
## train a final tree using best parameters
## Get the best train, val and test data RMSE value.
################################################################################


library(tree)
library(rpart)
library(MASS)
library(randomForest)
#--------------------------------------------------
#reading the data
nba <- read_csv("nba_filtered_cleaned_for_model_prep.csv")
nba$position  <- factor(nba$position )

#create train,val,test sets
set.seed(99)
n=nrow(nba)
n1=floor(n/2)
n2=floor(n/4)
n3=n-n1-n2
ii = sample(1:n,n)
nbatrain=nba[ii[1:n1],]
nbaval = nba[ii[n1+1:n2],]
nbatest = nba[ii[n1+n2+1:n3],]
################################################################################


#--------------------------------------------------
#define range of model parameters
set.seed(1)
p <- ncol(nbatrain) - 1
mtryv <- seq(from = 4, to = 22, by = 4)
ntreev <- seq(from = 50, to = 1000, by = 100)
maxnodesv <- seq(from = 1, to = 100, by = 10)
parmrf <- expand.grid(mtryv, ntreev,maxnodesv)
colnames(parmrf) <- c("mtry", "ntree","maxnodesv")
nset <- nrow(parmrf)
olrf <- rep(0, nset)
ilrf <- rep(0, nset)
rffitv <- vector("list", nset)
for (i in 1:nset) {
  cat("doing rf ", i, " out of ", nset, "\n")
  temprf <- randomForest(points_per_game  ~ ., data = nbatrain, mtry = parmrf[i, 1], ntree = parmrf[i, 2], maxnodes = parmrf[i, 3])
  ifit <- predict(temprf)
  ofit <- predict(temprf, newdata = nbaval)
  olrf[i] <- sum((nbaval$points_per_game - ofit)^2)
  ilrf[i] <- sum((nbatrain$points_per_game - ifit)^2)
  rffitv[[i]] <- temprf
}
ilrf <- round(sqrt(ilrf / nrow(nbatrain)), 3)
olrf <- round(sqrt(olrf / nrow(nbaval)), 3)
#----------------------------------------

#print(cbind(parmrf, olrf, ilrf))

#----------------------------------------
# get best rf model
iirf <- which.min(olrf)
therf <- rffitv[[iirf]]
# Extract parameters of the best model
best_mtry <- parmrf$mtry[iirf]
best_ntree <- parmrf$ntree[iirf]
best_maxnodes <- parmrf$maxnodes[iirf]
cat("The value of parameters (mtry, ntree, maxnodes) for best model is:", best_mtry, ", ", best_ntree, ", ", best_maxnodes, "\n")


#get the final error with best model
final_errorval <- sqrt(sum((nbaval$points_per_game - predict(therf, nbaval))^2)/nrow(nbaval))
final_errortrain <- sqrt(sum((nbatrain$points_per_game - predict(therf, nbatrain))^2)/nrow(nbatrain))
final_errortest <- sqrt(sum((nbatest$points_per_game - predict(therf, nbatest))^2)/nrow(nbatest))

cat("The value of random forest train error is:", final_errortrain, "\n")
cat("The value of random forest val train error is:", final_errorval, "\n")
cat("The value of random forest test train error is:", final_errortest, "\n")

#--------------------------------------------------
# plot variable importance

par(cex = 0.5)  # Adjust the font size as needed
varImpPlot(therf)

#--------------------------------------------------

rm(list = ls())


