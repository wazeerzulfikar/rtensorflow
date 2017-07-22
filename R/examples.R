#' @useDynLib rtensorflow
#' @importFrom Rcpp sourceCpp
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

  w1 <- Constant(rep(1,12), dtype = dtype, shape = c(3,4))
  b1 <- Constant(rep(1,4),  dtype = dtype, shape = c(4))
  w2 <- Constant(rep(1,4),  dtype = dtype, shape = c(4,1))
  b2 <- Constant(rep(1,1),  dtype = dtype, shape = c(1))
  
  hidden_matmul <- MatMul(input, w1)
  hidden_layer <- Add(hidden_matmul, b1)
  output_matmul <- MatMul(hidden_layer, w2)
  output_layer <- Add(output_matmul, b2, name="out")
  
  feedInput(input, feed)
  output <- runSession("out")
  
  deleteSessionVariables()
  
  return(output[[output_layer]])
}

#' @title Build an Add graph
#' @description Function to build and run graph using C API
#' @return Output Value of Network 
add_graph <- function() {
  
  initializeSessionVariables()
  
  a <- Placeholder("double", shape=c(-1,4))
  b <- Placeholder("double", shape=c(1))
  c <- Constant(c(3,4,3,6), dtype="double")
  
  neg <- Neg(Add(a,b))
  
  out <- Sigmoid(neg)
  
  feed <- data.frame(a=c(-0.2, 0.2),b=c(0.42,-0.42),c=c(0.13,-0.13),d=c(-0.54,0.54))

  feedInput(a,feed)
  feedInput(b,c(0.3))
  
  output <- runSession(c(out,neg))
  deleteSessionVariables()
  return (output[[out]])
}

check_load_saved_model <- function(path){
  initializeSessionVariables()
  loadSavedModel(path, c("train", "serve"))
  # Training the regressor
  for (i in 1:10) {
    feedInput("x",rep(i,1))
    feedInput("y", rep(i,1))
    output <- runSession(c("train", "loss"))
    cat("Loss: ",output[["loss"]],"\n")
  }
  
  # Testing the regressor
  feedInput("x", rep(4,1))
  feedInput("y", rep(4,1))
  output <- runSession("y_hat")
  
  deleteSessionVariables()
  return (output[["y_hat"]])
}
