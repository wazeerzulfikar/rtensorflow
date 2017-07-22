library(rtensorflow)

check_mnist <- function(path) {
  initializeSessionVariables()
  loadSavedModel(path, c("train", "serve"))
  
  data <- read.csv(file="../mnist_data/train.csv", header=TRUE, sep=',')
  print ("Data read successful")
  
  y_train <- data[,"label"]
  
  drops <- c("label")
  X_train <- data[ , !(names(data) %in% drops)]
  print(class(X_train))
  feedInput("x",X_train[c(1:100),])
  feedInput("y",y_train[c(1:100)])
  print(y_train[c(1:100)])
  feedInput("keep_prob",c(0.75))
  output_list <- runSession("output")
  for (i in 1:nrow(output_list[["output"]])) {
    print(which.max(output_list[["output"]][i,])-1)
  }
  deleteSessionVariables()
  return(output_list[["output"]])
}

check_mnist("./saved-models/mnist-model")