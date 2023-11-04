source("haunted.R")
source("myFunctions.R")

args <- commandArgs(trailingOnly = TRUE)

cores <- parseArgValues(args,"-C")
levels <- parseArgValues(args,"-L")
folds <- parseArgValues(args,"-V")

run_svm_radial(cores, levels, folds)