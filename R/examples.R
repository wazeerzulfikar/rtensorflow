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
  output <- runSession("output")
  
  deleteSessionVariables()
  return(output)
  
}

#' @title Build and Run Graph
#' @description Function to build and run graph using C API
#' @param feed Integer vector of size 3, feed for the network
#' @param dtype Datatype to be used, one of {"int32","double"}
#' @return Output Value of Network 

build_run_graph <- function(feed, dtype="double") {
  
  initializeSessionVariables()
  
  input <- Placeholder(dtype, shape = c(1,3), name="input")

  w1 <- Constant(rep(1,12), dtype = dtype, shape = c(3,4))
  b1 <- Constant(rep(1,4),  dtype = dtype, shape = c(4))
  w2 <- Constant(rep(1,4),  dtype = dtype, shape = c(4,1))
  b2 <- Constant(rep(1,1),  dtype = dtype, shape = c(1))
  
  hidden_matmul <- MatMul(input, w1)
  hidden_layer <- Add(hidden_matmul, b1)
  output_matmul <- MatMul(hidden_layer, w2)
  output_layer <- Add(output_matmul, b2)
  
  feedInput(input, feed)
  output <- runSession(output_layer)
  
  deleteSessionVariables()
  
  return(output)
}

#' @title Build an Add graph
#' @description Function to build and run graph using C API
#' @return Output Value of Network 
add_graph <- function() {
  
  initializeSessionVariables()
  
  a <- Placeholder(c(4), dtype="double")
  b <- Placeholder(c(1), dtype="double")
  c <- Constant(c(3,4,3,6), dtype="double")
  
  add <- Add(a,b)
  
  neg <- Neg(add)
  
  out <- Sigmoid(add)
  
  feedInput(a,c(-0.2,0.42,0.13,-0.54))
  feedInput(b,c(0.3))
  
  output <- runSession(out)
  
  deleteSessionVariables()
  return (output)
}
