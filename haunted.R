library(tidymodels,quietly=T)
library(vroom,quietly=T)
library(tidyverse,quietly=T)
library(discrim,quietly=T) # for naive bayes engine
library(naivebayes,quietly=T) # naive bayes
library(embed,quietly=T) #used for target encoding
library(parallel,quietly=T)
library(kernlab,quietly=T) #for svm
library(keras) # for neural net
library(bonsai) # boosted trees & bart
library(lightgbm) # boosted trees & bart
library(dbarts) # bart

source("myFunctions.R")

# Data Gathering ----------------------------------------------------------

test <- vroom("./Data/test.csv")
train <- vroom("./Data/train.csv")

## Recipes ------------------------------------------------------------------

#   Treat `id` as a predictor (somehow this makes the Naive Bayes models better)
hauntedRecipeNoID <- recipe(type ~ . , data=train) %>% 
                        step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) 
prpped <- prep(hauntedRecipeNoID)
bakd <- bake(prpped, new_data=train)


klaR_recipe <- recipe(type ~ . , data=train) %>%
	step_mutate_at(color, fn = factor)

# prep(hauntedRecipeNoID,verbose=T)

#   Ignore `id` by updating its role
hauntedRecipe <- recipe(type ~ . , data=train) %>% 
                    update_role(id,new_role="ID") %>%
                    step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) #%>% 
                    # step_kpca_rbf(all_predictors()) 

prepped<- prep(hauntedRecipe)
baked <- bake(prepped, new_data=train)


## Naive Bayes -------------------------------------------------------------
run_naive_bayes <- function(numCores, numLevels, numFolds.v){
  funcStart <- proc.time()
  #   model
  naiveModel <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
    set_mode("classification") %>% 
    set_engine("klaR")
  
  #   workflow
  naiveWF <- workflow() %>% 
   # add_recipe(hauntedRecipeNoID) %>% 	##### switch recipe to klaR_recipe 
    add_recipe(klaR_recipe) %>%
    add_model(naiveModel)
	  


  #   tuning
  naiveGrid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = numLevels)
  
  #   folds for cross validation
  naiveFolds <- vfold_cv(train, v=numFolds.v, repeats=1)
  
  cl <- makePSOCKcluster(numCores)
  doParallel::registerDoParallel(cl)
  
  #   fit model with cross validation
  naiveResultsCV <- naiveWF %>% 
    tune_grid(resamples=naiveFolds,
              grid=naiveGrid,
              metric_set("accuracy"))
  
  #   find best tune
  naiveBestTune <- naiveResultsCV %>%
    select_best("accuracy")
  
  #   finalize the Workflow & fit it
  naiveFinalWF <-
    naiveWF %>%
    finalize_workflow(naiveBestTune) %>%
    fit(data=train)
  
  #   predict and export
  outputCSV <-  predict_export(naiveFinalWF,"naiveBayes_klaR")
  stopCluster(cl)
  
  funcRunTimeSeconds <- (proc.time() - funcStart)[3]
  period <- seconds_to_period(funcRunTimeSeconds)
  
  time <- sprintf('%02d:%02d:%02d', hour(period), minute(period), round(second(period),0))

  return(list(time,outputCSV))
} # end run_naive_bayes() function

## SVM ---------------------------------------------------------------------
run_svm_radial <- function(numCores, numLevels, numFolds.v){
  funcStart <- proc.time()  
  
  #   make model with radial kernel
  svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
                      set_mode("classification") %>%
                      set_engine("kernlab")
  
  #   create workflow
  svmRadialWF <- workflow() %>% 
                    add_recipe(hauntedRecipe) %>% 
                    add_model(svmRadial)
  
  #   tuning
  svmRadialGrid <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = numLevels)

  #   folds for cross validation
  svmRadialFolds <- vfold_cv(train, v=numFolds.v, repeats=1)
  
  cl <- makePSOCKcluster(numCores)
  doParallel::registerDoParallel(cl)
  
  #   fit model with cross validation
  svmRadialResultsCV <- svmRadialWF %>% 
                          tune_grid(resamples=svmRadialFolds,
                                    grid=svmRadialGrid,
                                    metric_set("accuracy"))
  
  #   find best tune
  svmRadialBestTune <- svmRadialResultsCV %>%
    select_best("accuracy")
  
  #   finalize the Workflow & fit it
  svmRadialFinalWF <- svmRadialWF %>%
                        finalize_workflow(svmRadialBestTune) %>%
                        fit(data=train)
  
  #   predict and export
  outputCSV <-  predict_export(svmRadialFinalWF,"svmRadial")
  stopCluster(cl)
  
  ########################################
  
  funcRunTimeSeconds <- (proc.time() - funcStart)[3]
  period <- seconds_to_period(funcRunTimeSeconds)
  
  time <- sprintf('%02d:%02d:%02d', hour(period), minute(period), round(second(period),0)) 
  
  return(list(time,outputCSV))
} # end run_svm_radial() function


# Random Forest -----------------------------------------------------------
run_random_forest <- function(numCores, numLevels, numFolds.v, numTrees){
    funcStart <- proc.time()  
  
    randForestModel <- rand_forest(mtry = tune(),
                                   min_n=tune(),
                                   trees=numTrees) %>%
                                   # trees = 100) %>% #trees=numTrees) %>% 
      set_engine("ranger") %>% 
      set_mode("classification")
    
    forestWF <- workflow() %>% 
      add_recipe(hauntedRecipe) %>% 
      add_model(randForestModel)
    
    # create tuning grid
    # forest_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(baked)-1)) ),
    #                                    min_n(),
    #                                    levels = 2)
    # 
    # # split data for cross validation
    # rfolds <- vfold_cv(train, v = 2, repeats=1)
    
    forest_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(baked)-1)) ),
                                       min_n(),
                                       levels = numLevels)

    # split data for cross validation
    rfolds <- vfold_cv(train, v = numFolds.v, repeats=1)
    
    #cl <- makePSOCKcluster(4)
    cl <- makePSOCKcluster(numCores)
    doParallel::registerDoParallel(cl)
    
    # run cross validation
    treeCVResults <- forestWF %>% 
      tune_grid(resamples = rfolds,
                grid = forest_tuning_grid)
               # metrics=metric_set(accuracy)) 
    
    # select best model
    best_tuneForest <- treeCVResults %>% 
      select_best("accuracy")
    
    # finalize workflow
    finalForestWF <- 
      forestWF %>% 
      finalize_workflow(best_tuneForest) %>% 
      fit(data=train)
    
    #   predict and export
    outputCSV <-  predict_export(finalForestWF,"randomForest")
    stopCluster(cl)
    
    ##############################################
    
    funcRunTimeSeconds <- (proc.time() - funcStart)[3]
    period <- seconds_to_period(funcRunTimeSeconds)
    
    time <- sprintf('%02d:%02d:%02d', hour(period), minute(period), round(second(period),0)) 
    
    return(list(time,outputCSV))
}
    


# Neural Network ---------------------------------------------------
run_nn <- function(numCores, numLevels, numFolds.v){
  funcStart <- proc.time() 
  
  nn_recipe <- recipe(type ~ ., data=train) %>%
  #  update_role(id, new_role="id") %>%
    step_mutate_at(color, fn = factor) %>% ## Turn color to factor then dummy encode color
    step_dummy(color) %>%
    step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]
  
  nn_model <- mlp(hidden_units = tune(),
                  epochs = 100, #or 100 or 250
                  activation="relu") %>%
              set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
              set_mode("classification")
  ########### Keras: https://stackoverflow.com/questions/44611325/r-keras-package-error-python-module-tensorflow-contrib-keras-python-keras-was-n
  ###########  did all but step 4 from the answer install_github
  
  
  # nn_model <- mlp(hidden_units = tune(),
  #                 epochs = 50) %>%  #or 100 or 250
  #                 # activation="relu") %>%
  #   set_engine("nnet") %>% #verbose = 0 prints off less
  #   set_mode("classification")
  
  nn_wf <- workflow() %>%
    add_recipe(nn_recipe) %>%
    add_model(nn_model)
  
  nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 100)),
                              levels=numLevels)
  
  nn_folds <- vfold_cv(train, v = numFolds.v, repeats=1)
  
  tuned_nn <- nn_wf %>%
      tune_grid(grid=nn_tuneGrid,
                resamples=nn_folds,
                metrics=metric_set(accuracy))
  
  #   find best tune
  nn_bestTune <- tuned_nn %>%
    select_best("accuracy")
  
  #   finalize the Workflow & fit it
  final_nn_wf <- nn_wf %>%
    finalize_workflow(nn_bestTune) %>%
    fit(data=train)
  
  #   predict and export
  outputCSV <-  predict_export(final_nn_wf,"nn_Keras")
  
  
  funcRunTimeSeconds <- (proc.time() - funcStart)[3]
  period <- seconds_to_period(funcRunTimeSeconds)
  
  time <- sprintf('%02d:%02d:%02d', hour(period), minute(period), round(second(period),0)) 
  
  return(list(time,outputCSV))

}





# Boosted Trees -----------------------------------------------------------
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boostWF <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(hauntedRecipe)

#   tuning grid
boostGrid <- grid_regular(tree_depth(),
                          trees(),
                          learn_rate(),
                          levels = 20)
# folds
boostFolds <- vfold_cv(train, v=10, repeats=1)

cl <- makePSOCKcluster(4)
doParallel::registerDoParallel(cl)

#   fit model with cross validation
s <- proc.time()
boostResultsCV <- boostWF %>% 
  tune_grid(resamples=boostFolds,
            grid=boostGrid,
            metrics=metric_set(accuracy))

#   find best tune
boostBestTune <- boostResultsCV %>%
  select_best("accuracy")

#   finalize the Workflow & fit it
boostFinalWF <- boostWF %>%
  finalize_workflow(boostBestTune) %>%
  fit(data=train)

#   predict and export
outputCSV <-  predict_export(boostFinalWF,"lightGBM")
stopCluster(cl)
proc.time()-s

# BART --------------------------------------------------------------------

# 
# bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
#   set_engine("dbarts") %>% # might need to install
#   set_mode("classification")
# 
# bartWF <- workflow() %>% 
#   add_model(bart_model) %>% 
#   # add_recipe(hauntedRecipeNoID)
#   add_recipe(hauntedRecipe)
# 
# #   tuning
# bartGrid <- grid_regular(trees(),
#                               levels = 15)
# 
# #   folds for cross validation
# bartFolds <- vfold_cv(train, v=10, repeats=1)
# 
# cl <- makePSOCKcluster(4)
# doParallel::registerDoParallel(cl)
# 
# #   fit model with cross validation
# s <- proc.time()
# bartResultsCV <- bartWF %>% 
#   tune_grid(resamples=bartFolds,
#             grid=bartGrid,
#             metrics=metric_set(accuracy))
# 
# #   find best tune
# bartBestTune <- bartResultsCV %>%
#   select_best("accuracy")
# 
# #   finalize the Workflow & fit it
# bartFinalWF <- bartWF %>%
#   finalize_workflow(bartBestTune) %>%
#   fit(data=train)
# 
# #   predict and export
# outputCSV <-  predict_export(bartFinalWF,"bart")
# stopCluster(cl)
# proc.time()-s