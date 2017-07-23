library(rtensorflow)

check_mnist <- function(model_path, csv_path) {
  initializeSessionVariables()
  loadSavedModel(model_path, c("train", "serve"))
  
  # Read MNIST data CSV file
  data <- read.csv(file=csv_path, header=TRUE, sep=',')
  print ("Data read successful")
  
  # Extract label column
  y_train <- data[,"label"]

  # One hot encode the labels
  col <- 10
  row <- length(y_train)
  onehot <- array(data=rep(0,col*row),dim=c(row,col))
  i <- 1
  for (j in y_train) {
    onehot[i,j+1] <- 1
    i=i+1
  }
  y_train <- onehot
  
  # Drop label for getting X training data
  drops <- c("label")
  X_train <- data[ , !(names(data) %in% drops)]

  X_train <- X_train/255
  step <- 0
  for (i in seq(1, 10000, by = 128)) {
     
	# Feed data to respective placeholders

	feedInput("x",X_train[c(i:i+128),])
	feedInput("y",y_train[c(i:i+128),])
	feedInput("keep_prob",c(0.75))
	runSession(c("train"))

    if (step%%10==0) {
      feedInput("x",X_train[c(i:i+128),])
      feedInput("y",y_train[c(i:i+128),])
      feedInput("keep_prob",c(1.))
      display <- runSession(c("cost","accuracy"))
    
      cat("Iter ",i, ",  ")
      cat("Cost=", display[["cost"]])
      cat(",  Accuracy=", display[["accuracy"]],"\n")
    }
      step <- step+1
  }

  deleteSessionVariables()
}

