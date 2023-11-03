library(tidymodels)
library(vroom)
library(tidyverse)
library(discrim) # for naive bayes engine
library(naivebayes) # naive bayes
library(embed) #used for target encoding
library(parallel)

# Data Gathering ----------------------------------------------------------

test <- vroom("test.csv")
train <- vroom("train.csv")

# train$color <- as.factor(train$color)
# train$type <- as.factor(train$type)

# # EDA ---------------------------------------------------------------------
# ggplot(data=train, aes(x=type, y=bone_length)) +
#   geom_boxplot()



# Data Imputation (Practice) ------------------------------------------------
missing <- vroom("trainWithMissingValues.csv")

imputeRecipe <- recipe(type ~ . , data=missing) %>% 
  update_role(id,new_role="ID") %>% 
  step_string2factor(all_nominal_predictors()) %>% 
  step_impute_knn(hair_length, impute_with = c('has_soul','color'), neighbors=5 ) %>% 
  step_impute_knn(rotting_flesh, impute_with = c('has_soul','color','hair_length'), neighbors=5 ) %>% 
  step_impute_knn(bone_length, impute_with = c('has_soul','color','hair_length','rotting_flesh'),neighbors=5)

  
preppedImpute <- prep(imputeRecipe)
bakedImpute <- bake(preppedImpute, new_data = missing)


rmse_vec(train[is.na(missing)],bakedImpute[is.na(missing)])


# Prep for Kaggle ---------------------------------------------------------

# Make function to predict and export
predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  
  preds <- workflowName %>%
    predict(new_data = test, type="class")
  
  submission <- as.data.frame(cbind(test$id, as.character(preds$.pred_class)))
  colnames(submission) <- c("id","type")
  
  directory = "./submissions/"
  path = paste0(directory,fileName)
  
  vroom_write(submission, file = path, delim=',')
  # vroom_write(submission, file = "./submissions/naiveBayes.csv", delim=',')
  
}

# Recipe ------------------------------------------------------------------
hauntedRecipe <- recipe(type ~ . , data=train) %>% 
  # update_role(id,new_role="ID") %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) 
  # step_dummy(all_nominal_predictors()) %>% 
  # step_normalize(all_numeric_predictors())

preppedHauntedRecipe <- prep(hauntedRecipe)
# bakedHaunted <- bake(preppedHauntedRecipe, new_data=train)

## Naive Bayes -------------------------------------------------------------
#   model
naiveModel <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

#   workflow
naiveWF <- workflow() %>% 
  add_recipe(hauntedRecipe) %>% 
  add_model(naiveModel)

#   tuning
naiveGrid <- grid_regular(Laplace(),
                          smoothness(),
                          levels = 25)

#   folds for cross validation
naiveFolds <- vfold_cv(train, v=15, repeats=1)

cl <- makePSOCKcluster(4)
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
predict_export(naiveFinalWF,"naiveBayesGLM5.csv")


stopCluster(cl)


