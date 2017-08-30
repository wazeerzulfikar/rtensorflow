library(rtensorflow)

check_mnist <- function(model_path, csv_path) {
  initializeSessionVariables()
  loadSavedModel(model_path, c("train", "serve"))
  
  training_iters <- 2000
  batch_size <- 128
  display_step <- 10
  
  # Read MNIST data CSV file

  data <- read.csv(file=csv_path, header=TRUE, sep=',')
  print ("Data read successful")
  
  # Extract label column
  y_train <- data[,"label"]

  # One hot Encoder for the labels
  col <- 10
  row <- length(y_train)
  onehot <- array(data=rep(0, col * row),dim=c(row, col))
  onehot[cbind(1:row, y_train + 1)] <- 1
  y_train <- onehot

  # Drop label for getting X training data

  drops <- c("label")
  X_train <- data[ , !(names(data) %in% drops)]
  X_train <- X_train/255

  step <- 0
  for (i in 1:training_iters) {
      samples <- sample(1:nrow(X_train), batch_size, replace=FALSE)
      feedInput("x",X_train[samples,])
      feedInput("y",y_train[samples,])
      feedInput("keep_prob",c(0.75))
      runSession(c("train"))
    if (step%%display_step==0) {
      feedInput("x",X_train[samples,])
      feedInput("y",y_train[samples,])
      feedInput("keep_prob",c(1.))
      display <- runSession(c("cost","accuracy"))
    
      cat("Iter ",i, ",  ")
      cat("Cost=", display[["cost"]])
      cat(",  Training Accuracy=", display[["accuracy"]],"\n")
    }
      step <- step+1
  }
  
  print ("Optimization Finished!")
  
  deleteSessionVariables()
  
}

