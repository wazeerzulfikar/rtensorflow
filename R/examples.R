#' @useDynLib rtensorflow
#' @importFrom Rcpp sourceCpp
#' @importFrom utils read.csv
#' @exportPattern "^[[:alpha:]]+"
#' 
#' @title Import and Run Graph
#' @description Function to import and run graph using C API
#' @param path Path to the graph, "/tests/models/feed_forward_graph.pb"
#' @param feed Integer vector of size 3, feed for the network
#' @return Output Value of Network 

import_run_graph <- function(path, feed){
  
  initializeSessionVariables()
  loadGraphFromFile(path)
  feedInput("input", feed)
  output_list <- runSession("output")
  
  deleteSessionVariables()
  return(output_list[["output"]])
}

#' @title Build and Run Graph
#' @description Function to build and run graph using C API
#' @param feed Integer vector of size 3, feed for the network
#' @param dtype Datatype to be used, one of {"int32","double"}
#' @return Output Value of Network 

build_run_graph <- function(feed, dtype="float") {
  
  initializeSessionVariables()
  
  input <- Placeholder(dtype, shape = c(1,3), name="input")

  w1 <- Constant(rep(1,12), dtype=dtype, shape=c(3,4))
  b1 <- Constant(rep(1,4), dtype=dtype, shape=c(4))
  w2 <- Constant(rep(1,4), dtype=dtype, shape=c(4,1))
  b2 <- Constant(rep(1,1), dtype=dtype, shape=c(1))
  
  hidden_matmul <- MatMul(input, w1)
  hidden_layer <- Add(hidden_matmul, b1)
  
  output_matmul <- MatMul(hidden_layer, w2)
  output_layer <- Add(output_matmul, b2, name="out")

  feedInput(input, feed)
  output <- runSession("out")

  resetGraph()

  return(output[[output_layer]])
}

#' @title Build an Add graph
#' @description Graph to add two vectors and output the complement
#' @return Output Value of Network 
add_graph <- function() {
  
  initializeSessionVariables()
  
  a <- Placeholder("double", shape=c(-1,4))
  b <- Placeholder("double", shape=c(1))

  neg <- Neg(Add(a,b))

  feed <- data.frame(a=c(-0.2, 0.2),b=c(0.42,-0.42),c=c(0.13,-0.13),d=c(-0.54,0.54))

  feedInput(a, feed)
  feedInput(b, 0.3)
  
  output <- runSession(neg)

  deleteSessionVariables()
  return (output[[neg]])
}

#' @title Load Saved Model
#' @description Load an MNIST graph and Train it. But first run mnist.py to create a saved model before running this function.
#' @param model_path Path to saved model
#' @param csv_path Path to csv file with training data
#' @return Output Value of Network 

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
  y_train <- data[, "label"]
  
  # One hot Encoder for the labels
  col <- 10
  row <- length(y_train)
  onehot <- array(data=rep(0, col * row),dim=c(row, col))
  onehot[cbind(1:row, y_train + 1)] <- 1
  y_train <- onehot
  
  # Drop label for getting X training data
  
  drops <- "label"
  X_train <- data[, !(names(data) %in% drops)]
  X_train <- X_train / 255
  
  step <- 0
  for (i in 1:training_iters) {
    samples <- sample(1:nrow(X_train), batch_size, replace=FALSE)
    feedInput("x", X_train[samples,])
    feedInput("y", y_train[samples,])
    feedInput("keep_prob", 0.75)
    runSession("train")
    if (step %% display_step == 0) {
      feedInput("x", X_train[samples,])
      feedInput("y", y_train[samples,])
      feedInput("keep_prob", 1.)
      display <- runSession(c("cost", "accuracy"))
      
      cat(sprintf(
        "Iter %d, Cost=%f, Training Accuracy=%f\n",
        i, display[["cost"]], display[["accuracy"]]))
    }
    step <- step + 1
  }
  
  print ("Optimization Finished!")
  
  deleteSessionVariables()
  
}
