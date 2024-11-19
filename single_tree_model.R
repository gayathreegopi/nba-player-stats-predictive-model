###################################################
## Fitting trees to the data for best model estimation.
## first we train a single tree and check its RMSE_train
## Second we train a random forest and check its RMSE_train
## Third we train a Boosted tree and check its RMSE_train
## Lastly we check the variable importance of each individual variable used to train.
###################################################

library(tree)
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



#--------------------------------------------------
#creating a vector to define range of minsplit values
minsplit_values <- c(2:10)

best_parameters_from_models <- list()

for (min_split in 2:10) {
  # get big tree
  big.tree <- rpart(points_per_game  ~ .,
                    method = "anova", data = nbatrain,
                    control = rpart.control(minsplit = min_split, cp = .00005)
  )
  nbig <- length(unique(big.tree$where))
  cat("size of big tree: ", nbig, "\n")
  
  # fit on train, predict on val for vector of cp.
  cpvec <- big.tree$cptable[, "CP"] # cp values to try
  ntree <- length(cpvec) # number of cv values = number of trees fit.
  iltree <- rep(0, ntree) # in-sample loss
  oltree <- rep(0, ntree) # out-of-sample loss
  sztree <- rep(0, ntree) # size of each tree
  for (i in 1:ntree) {
    if ((i %% 10) == 0) cat("tree i: ", i, "\n")
    temptree <- prune(big.tree, cp = cpvec[i])
    sztree[i] <- length(unique(temptree$where))
    iltree[i] <- sum((nbatrain$points_per_game - predict(temptree))^2)
    ofit <- predict(temptree, nbaval)
    oltree[i] <- sum((nbaval$points_per_game - ofit)^2)
  }
  oltree <- sqrt(oltree / nrow(nbaval))
  iltree <- sqrt(iltree / nrow(nbatrain))
  
  # plot losses
  rgl <- range(c(iltree, oltree))
  plot(range(sztree), rgl, type = "n", xlab = "tree size", ylab = "loss")
  points(sztree, iltree, pch = 15, col = "red")
  points(sztree, oltree, pch = 16, col = "blue")
  legend("topright", legend = c("in-sample", "out-of-sample"), lwd = 3, col = c("red", "blue"))
  title(main = paste("minsplit =", min_split))
  
  
  # write val preds to our list
  iitree <- which.min(oltree)
  thetree <- prune(big.tree, cp = cpvec[iitree])
  thetreepred <- predict(thetree, nbaval)
  thetreepred_val <- predict(thetree, nbaval)
  error_val <- sqrt(sum((nbaval$points_per_game - thetreepred_val)^2)/nrow(nbaval))
  thetreepred_train <- predict(thetree, nbatrain)
  error_train <- sqrt(sum((nbatrain$points_per_game - thetreepred_train)^2)/nrow(nbatrain))
  best_parameters_from_models[[min_split]] <- list(bestcp = cpvec[iitree], 
                       best_sizeoftree = big.tree$cptable[iitree, "nsplit"], 
                       error_val = error_val, 
                       error_train = error_train,
                       min_split = min_split)
} 
#convert final list to dataframe to observe
best_parameters_from_models_df <- data.frame(matrix(unlist(best_parameters_from_models), nrow=length(best_parameters_from_models), byrow=TRUE))
colnames(best_parameters_from_models_df) <- c("bestcp", "best_sizeoftree", "error_val", "error_train","min_split")

#train the best model
best_model_single_tree_parameters <- best_parameters_from_models_df[which.min(best_parameters_from_models_df$error_val), ]
big.tree <- rpart(points_per_game  ~ .,
                  method = "anova", data = nbatrain,
                  control = rpart.control(minsplit = best_model_single_tree_parameters[,"min_split"], cp = best_model_single_tree_parameters[,"bestcp"]
                                          )
)
finaltree <- prune(big.tree, cp = best_model_single_tree_parameters[,"bestcp"])
#plot best tree
par(mfrow = c(1, 1))
plotcp(big.tree)
#plot best actual tree
plot(finaltree, uniform = TRUE, branch = 01, margin = 0.01)  # Increase branch and margin
text(finaltree, digits = 2, cex = 0.5, use.n = TRUE, fancy = TRUE, bg = "lightblue")  # Reduce font size (cex)


#get the final error with best model
final_errorval <- sqrt(sum((nbaval$points_per_game - predict(finaltree, nbaval))^2)/nrow(nbaval))
final_errortrain <- sqrt(sum((nbatrain$points_per_game - predict(finaltree, nbatrain))^2)/nrow(nbatrain))
final_errortest <- sqrt(sum((nbatest$points_per_game - predict(finaltree, nbatest))^2)/nrow(nbatest))

cat("The value of single tree train error is:", final_errortrain, "\n")
cat("The value of single val train error is:", final_errorval, "\n")
cat("The value of single test train error is:", final_errortest, "\n")

rm(list = ls())

#--------------------------------------------------
