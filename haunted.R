library(tidymodels,quietly=T)
library(vroom,quietly=T)
library(tidyverse,quietly=T)
library(discrim,quietly=T) # for naive bayes engine
library(naivebayes,quietly=T) # naive bayes
library(embed,quietly=T) #used for target encoding
library(parallel,quietly=T)
library(kernlab,quietly=T) #for svm

source("myFunctions.R")

# Data Gathering ----------------------------------------------------------

test <- vroom("./Data/test.csv")
train <- vroom("./Data/train.csv")

## Recipes ------------------------------------------------------------------

#   Treat `id` as a predictor (somehow this makes the Naive Bayes models better)
hauntedRecipeNoID <- recipe(type ~ . , data=train) %>% 
                        step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) 

# prep(hauntedRecipeNoID,verbose=T)

#   Ignore `id` by updating its role
hauntedRecipe <- recipe(type ~ . , data=train) %>% 
                    update_role(id,new_role="ID") %>%
                    step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) %>% 
                    step_kpca_rbf(all_predictors()) 

prepped<- prep(hauntedRecipe)
baked <- bake(prepped, new_data=train)

print(ncol(baked))
## Naive Bayes -------------------------------------------------------------
run_naive_bayes <- function(numCores, numLevels, numFolds.v){
  funcStart <- proc.time()
  #   model
  naiveModel <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
    set_mode("classification") %>% 
    set_engine("naivebayes")
  
  #   workflow
  naiveWF <- workflow() %>% 
    add_recipe(hauntedRecipeNoID) %>% 
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
  outputCSV <-  predict_export(naiveFinalWF,"naiveBayesGLM")
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
    
    cl <- makePSOCKcluster(4)
    # cl <- makePSOCKcluster(numCores)
    doParallel::registerDoParallel(cl)
    
    # run cross validation
    treeCVResults <- forestWF %>% 
      tune_grid(resamples = rfolds,
                grid = forest_tuning_grid,
                metrics=metric_set(accuracy)) 
    
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
    
