# Make function to predict and export ----------------------------------------
predict_export <- function(workflowName, newFileName){
  preds <- workflowName %>%
    predict(new_data = test, type="class")
  
  submission <- as.data.frame(cbind(test$id, as.character(preds$.pred_class)))
  colnames(submission) <- c("id","type")
  
  directory = "./submissions/"
  
  fileCount <- function(newName) {
    files <- list.files(path="./submissions/",pattern = paste0("^", newName, "_[0-9]+.csv"), recursive=T)
    if(length(files) == 0){
      return(1)
    }
    else{
      maxNum <- max(as.numeric(str_extract_all(files,"[0-9]+")))
      return(maxNum+1)
    }
  }
  
  fileNum <- fileCount(newFileName)
  
  outputFilePath = paste0(directory,newFileName,"_",fileNum,".csv")
  
  vroom_write(submission, file = outputFilePath, delim=',')
}


# Get CLI Arguments -------------------------------------------------------

parseArgValues <- function(argsList,argument){
  # Parse the arguments
  for (i in 1:length(argsList)) {
    if (argsList[i] == argument) {
      arg_value <- as.numeric(argsList[i + 1])
    }
  }
  return(arg_value)
}
  
  
  
  
  
  
  
  
  
  
  
