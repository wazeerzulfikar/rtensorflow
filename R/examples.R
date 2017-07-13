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
  feedInput("input", feed,dim=c(1,3))
  output <- runSession("output")
  
  deleteSessionVariables()
  return(output)
  
}

#' @title Build and Run Graph
#' @description Function to build and run graph using C API
#' @param feed Integer vector of size 3, feed for the network
#' @param dtype Datatype to be used, one of {"int32","double"}
#' @return Output Value of Network 

build_run_graph <- function(feed, dtype="int32") {
  
  initializeSessionVariables()
  
  input <- Placeholder(dtype=dtype, name="input")

  w1 <- Constant(rep(1,12), dim = c(3,4), dtype = dtype)
  b1 <- Constant(rep(1,4), dim = c(4), dtype = dtype)
  w2 <- Constant(rep(1,4), dim = c(4,1), dtype = dtype)
  b2 <- Constant(rep(1,1), dim = c(1), dtype = dtype)
  
  hidden_matmul <- MatMul(input, w1)
  hidden <- Add(hidden_matmul, b1)
  output_matmul <- MatMul(hidden, w2)
  output <- Add(output_matmul, b2)
  
  feedInput(input, feed, dim = c(1,3))
  output <- runSession(output)
  
  deleteSessionVariables()
  
  return(output)
}

#' @title Build an Add graph
#' @description Function to build and run graph using C API
#' @return Output Value of Network 
add_graph <- function() {
  
  initializeSessionVariables()
  
  a <- Constant(c(1,2,3,4), dtype="int32")
  b <- Constant(c(1), dim=c(1), dtype="int32")
  
  add <- Add(a,b)
  
  neg <- Neg(add)
  
  output <- runSession(output)
  
  deleteSessionVariables()
  return (output)
}
