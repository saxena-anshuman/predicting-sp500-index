# Clearing the environment
rm(list = ls())
cat('\014')
gc()

# Loading the required packages
library(pacman)
pacman::p_load(dplyr, xgboost)

# Setting seed to ensure reproducibility
set.seed(123)

# Loading the data
data = read.csv(url("https://s3.amazonaws.com/istreet-questions-us-east-1/417582/train.csv"), header = F)

# Creating a variable with random values, distributed uniformally 
data = 
  data %>% 
  mutate(rand = runif(nrow(data)))

# Splitting the dataset into train and test (70-30 split)
idx_train = sample(1:nrow(data), size = 0.7*nrow(data), replace = F)
idx_test = (1:nrow(data))[-idx_train]

X_train = as.matrix(data %>% select(-V501))[idx_train,]

Y_train = data$V501[idx_train]

median(Y_train) # 2354.395
mean(Y_train) # 2353.432
# The mean and median values are very close, indicating normal distribution of the dependent variable

# Running an xgboost model to derive a list of most important variables
# Parameter tuning
params = list(eta = 0.1, # slow learning
              max_depth = 1, # single depth for selection
              objective = "reg:linear",
              eval_metric = "rmse")

# Fit xgboost
xgb = xgboost(data = X_train,
              label = Y_train,
              nrounds = 50000, # lots of rounds to ensure good ranking on all variables
              early_stopping_rounds = 50, # stop if the model isn't improving
              print_every_n = 1000,
              params = params)

# List of most important variables
base_vars = 
  xgb.importance(model = xgb, feature_names = colnames(X_train)) %>% 
  filter(cumsum(Feature == "rand") == 0) # drop anything less important than the random variable

# Writing a function to check for multicollinearity and retain the more important variable in case of high correlation
run_correlation_check = function(mat, cor_thr = 0.7){
  cor_mat = cor(mat)
  diag(cor_mat) = 0
  
  i = 1
  test = 1
  
  while(test > 0){
    idx = which(abs(cor_mat[i,]) >= cor_thr)
    
    if(length(idx) == 0){
      i = i + 1
    }else{
      cor_mat = cor_mat[-idx, -idx]
    }
    
    test = sum(cor_mat >= cor_thr)
  }
  
  cat("\n", "Updated Max Correlation:", max(cor_mat), "\n")
  return(rownames(cor_mat))
}

# Running correlation check
cor_check = run_correlation_check(X_train[,base_vars$Feature])

# Running first iteration of linear model
train = as.data.frame(as.matrix(data)[idx_train, c(cor_check, "V501")])

mdl = lm(V501 ~ ., data = train)
summary(mdl)

# Dropping insignificant variables
varsToRemove = c("V214", "V386", "V220", "V425", "V80", "V198", "V455", "V137", "V254", "V212", "V289", "V244", "V145", "V429", "V297", "V38", "V103", "V239")

# Running second iteration of linear model
train_2 = as.data.frame(as.matrix(data %>% select(setdiff(cor_check, varsToRemove), "V501"))[idx_train, ])

mdl_2 = lm(V501 ~ ., data = train_2)
summary(mdl_2)

# Dropping insignificant variables and running third iteration 
train_3 = train_2 %>% select(-V228, -V331)

mdl_3 = lm(V501 ~ ., data = train_3)
summary(mdl_3) # adjusted R^2 value same as previous iterations, suggesting no predicted variance loss over iterations

hist(mdl_3$residuals) # the median of residuals is centered around 0, suggesting normal distribution

# Deriving AICc of all the 3 linear models
extractAIC(mdl) # -1959.113
extractAIC(mdl_2) # -1973.746
extractAIC(mdl_3) # -1974.236
# Model 3 has the lowest AIC, suggesting best fit

# Checking RMSE
sqrt(mean(mdl_3$residuals^2)) # 0.69

# Checking for heteroskedasticity 
plot(train$V501, resid(mdl_3)) # Do not observe heteroskedasticity

# Checking for autocorrelation
library(lmtest)
dwtest(mdl_3, alternative = "two.sided") # Fail to reject the null hypothesis, suggesting there is no autocorrelation

# Testing the final model on foreign data
test = as.data.frame(as.matrix(data %>% select(colnames(train_3)))[idx_test,])

test$pred = predict(mdl_3, test)

# Linear line suggests good prediction
plot(test$V501, test$pred)

# Checking RMSE on test data
sqrt(mean((test$V501-test$pred)^2)) # 0.76 - comparable values of RMSE suggest low overfitting

# Things to investigate:
# need to assess variables with negative coefficient
# since its an index, there should be a positive relationship between the index and its components (i.e. the dependent and independent variables) 
# however, it is possible there are industries within the index that have negative correlation resulting in negative coefficients

