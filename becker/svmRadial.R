source("haunted.R")
source("myFunctions.R")

args <- commandArgs(trailingOnly = TRUE)

cores <- parseArgValues(args,"-C")
levels <- parseArgValues(args,"-L")
folds <- parseArgValues(args,"-V")

results <- run_svm_radial(cores, levels, folds)

store_print(results,cores,levels,folds)
 





# runTime <- results[1]
# outputFileName <- results[2]
# 
# output <- data.frame(outputFileName,runTime,cores,levels,folds,NA)
# 
# write.table(output, "./submissions/allResults.csv",
#             na = "$",
#             quote = FALSE,
#             row.names = FALSE,
#             col.names = FALSE,
#             append = TRUE,
#             sep = ",")
# 
# paste0(runTime, " to run on " , cores," Cores, ", levels, " Levels, ", folds, " Folds")