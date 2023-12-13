source("haunted.R")
source("myFunctions.R")

args <- commandArgs(trailingOnly = TRUE)

cores <- parseArgValues(args,"-C")
levels <- parseArgValues(args,"-L")
folds <- parseArgValues(args,"-V")
trees <- parseArgValues(args,"-T")

results <- run_naive_bayes(cores, levels, folds)

store_print(results,cores,levels,folds)
