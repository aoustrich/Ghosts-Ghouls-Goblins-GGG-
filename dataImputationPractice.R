library(tidyverse)
library(tidymodels)
library(vroom)


# Read in Data ------------------------------------------------------------

test <- vroom("./Data/test.csv")
train <- vroom("./Data/train.csv")
missing <- vroom("./Data/trainWithMissingValues.csv")

# Impute Values -----------------------------------------------------------

imputeRecipe <- recipe(type ~ . , data=missing) %>%
                  update_role(id,new_role="ID") %>%
                  step_string2factor(all_nominal_predictors()) %>%
                  step_impute_knn(hair_length, 
                                  impute_with = c('has_soul','color'), 
                                  neighbors=5 ) %>%
                  step_impute_knn(rotting_flesh,
                                  impute_with = c('has_soul','color','hair_length'), 
                                  neighbors=5 ) %>%
                  step_impute_knn(bone_length,
                                  impute_with = c('has_soul','color','hair_length',
                                                  'rotting_flesh'),
                                  neighbors=5)


preppedImpute <- prep(imputeRecipe)
bakedImpute <- bake(preppedImpute, new_data = missing)

rmse_vec(train[is.na(missing)],bakedImpute[is.na(missing)])