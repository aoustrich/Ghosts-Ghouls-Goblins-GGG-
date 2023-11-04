library(tidymodels)
library(vroom)
library(tidyverse)
library(discrim) # for naive bayes engine
library(naivebayes) # naive bayes
library(embed) #used for target encoding
library(parallel)
library(kernlab) #for svm

source("myFunctions.R")

# Data Gathering ----------------------------------------------------------

test <- vroom("test.csv")
train <- vroom("train.csv")

# # EDA ---------------------------------------------------------------------
# ggplot(data=train, aes(x=type, y=bone_length)) +
#   geom_boxplot()

# Data Imputation (Practice) ------------------------------------------------
#missing <- vroom("trainWithMissingValues.csv")

#imputeRecipe <- recipe(type ~ . , data=missing) %>% 
 # update_role(id,new_role="ID") %>% 
  #step_string2factor(all_nominal_predictors()) %>% 
  #step_impute_knn(hair_length, impute_with = c('has_soul','color'), neighbors=5 ) %>% 
  #step_impute_knn(rotting_flesh, impute_with = c('has_soul','color','hair_length'), neighbors=5 ) %>% 
 # step_impute_knn(bone_length, impute_with = c('has_soul','color','hair_length','rotting_flesh'),neighbors=5)

  
#preppedImpute <- prep(imputeRecipe)
#bakedImpute <- bake(preppedImpute, new_data = missing)


#rmse_vec(train[is.na(missing)],bakedImpute[is.na(missing)])


# # Make function to predict and export ----------------------------------------
# predict_export <- function(workflowName, newFileName){
#   preds <- workflowName %>%
#             predict(new_data = test, type="class")
#   
#   submission <- as.data.frame(cbind(test$id, as.character(preds$.pred_class)))
#   colnames(submission) <- c("id","type")
#   
#   directory = "./submissions/"
#   
#   fileCount <- function(newName) {
#     files <- list.files(path="./submissions/",pattern = paste0("^", newName, "_[0-9]+.csv"), recursive=T)
#     if(length(files) == 0){
#       return(1)
#     }
#     else{
#       maxNum <- max(as.numeric(str_extract_all(files,"[0-9]+")))
#       return(maxNum+1)
#     }
#   }
#   
#   fileNum <- fileCount(newFileName)
#   
#   outputFilePath = paste0(directory,newFileName,"_",fileNum,".csv")
#   
#   vroom_write(submission, file = outputFilePath, delim=',')
# }

## Recipes ------------------------------------------------------------------

#   Treat `id` as a predictor (somehow this makes the Naive Bayes models better)
hauntedRecipeNoID <- recipe(type ~ . , data=train) %>% 
                        step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) 

# prep(hauntedRecipeNoID,verbose=T)

#   Ignore `id` by updating its role
hauntedRecipe <- recipe(type ~ . , data=train) %>% 
                    update_role(id,new_role="ID") %>%
                    step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) 

# prep(hauntedRecipe,
#       verbose = T,
#       retain = T,
#       strings_as_factors = T)

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
  
  cvStart <- proc.time()
  #   fit model with cross validation
  naiveResultsCV <- naiveWF %>% 
    tune_grid(resamples=naiveFolds,
              grid=naiveGrid,
              metric_set("accuracy"))
  
  cvStart - proc.time()
  
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
  
  cvStart <- proc.time()
  #   fit model with cross validation
  svmRadialResultsCV <- svmRadialWF %>% 
                          tune_grid(resamples=svmRadialFolds,
                                    grid=svmRadialGrid,
                                    metric_set("accuracy"))
  
  cvStart - proc.time()
  
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



