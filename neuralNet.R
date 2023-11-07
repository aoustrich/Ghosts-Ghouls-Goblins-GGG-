source("haunted.R")
source("myFunctions.R")

args <- commandArgs(trailingOnly = TRUE)

# cores <- parseArgValues(args,"-C")
cores <- 0
levels <- parseArgValues(args,"-L")
folds <- parseArgValues(args,"-V")

results <- run_nn(cores, levels, folds)

store_print(results,cores,levels,folds,trees=0)