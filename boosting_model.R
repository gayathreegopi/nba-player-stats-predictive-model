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
library(gbm)
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
idv <- seq(from = 4, to = 20, by = 4)
ntv <- seq(from = 10, to = 700, by = 50)
lamv <- seq(from = 0.001, to = 0.05, by = 0.005)
parmb <- expand.grid(idv, ntv, lamv)
colnames(parmb) <- c("tdepth", "ntree", "lam")
print(parmb)
nset <- nrow(parmb)
olb <- rep(0, nset)
ilb <- rep(0, nset)
bfitv <- vector("list", nset)
for (i in 1:nset) {
  cat("doing boost ", i, " out of ", nset, "\n")
  tempboost <- gbm(points_per_game  ~ .,
                   data = nbatrain, distribution = "gaussian",
                   interaction.depth = parmb[i, 1], n.trees = parmb[i, 2], shrinkage = parmb[i, 3]
  )
  ifit <- predict(tempboost, n.trees = parmb[i, 2])
  ofit <- predict(tempboost, newdata = nbaval, n.trees = parmb[i, 2])
  olb[i] <- sum((nbaval$points_per_game - ofit)^2)
  ilb[i] <- sum((nbatrain$points_per_game - ifit)^2)
  bfitv[[i]] <- tempboost
}
ilb <- round(sqrt(ilb / nrow(nbatrain)), 3)
olb <- round(sqrt(olb / nrow(nbaval)), 3)
#--------------------------------------------------
# print losses

print(cbind(parmb, olb, ilb))

#--------------------------------------------------

#--------------------------------------------------
# plot variable importance
p <- ncol(nbatrain) - 1 # want number of variables for later
vsum <- summary(finb) # this will have the variable importance info
row.names(vsum) <- NULL # drop varable names from rows.



# write variable importance table
cat("\\begin{verbatim}\n")
print(vsum)
cat("\\end{verbatim}\n")

# write val preds
iib <- which.min(olb)
theb <- bfitv[[iib]]
thebpred <- predict(theb, newdata = nbaval, n.trees = parmb[iib, 2])

rm(list = ls())


