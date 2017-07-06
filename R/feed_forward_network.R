#' @useDynLib rtensorflow
#' @importFrom Rcpp sourceCpp
#' @exportPattern "^[[:alpha:]]+"
#' 
#' @title Import and Run Graph
#' @description Function to import and run graph using C API
#' @param path Path to the graph, "/tests/models/feed_forward_graph.pb"
#' @param feed Integer vector of size 3, feed for the network
#' @return Output Value of Network 

import_run_graph <- function(path,feed){
  instantiateSessionVariables()
  loadGraphFromFile(path)
  feedInput("input", feed, "int32")
  setOutput("output")
  runSession()
  output <- printIntOutputs()
  return(output)
}

#' @title Build and Run Graph
#' @description Function to build and run graph using C API
#' @param feed Integer vector of size 3, feed for the network
#' @param dtype Datatype to be used, one of {"int32","double"}
#' @return Output Value of Network 

build_run_graph <- function(feed, dtype="int32"){
  instantiateSessionVariables()
  
  input <- Placeholder(dtype)

  w1 <- Constant(rep(1,12),c(3,4),dtype)
  b1 <- Constant(rep(1,4),c(4),dtype)
  w2 <- Constant(rep(1,4),c(4,1),dtype)
  b2 <- Constant(rep(1,1),c(1),dtype)
  
  hidden_matmul <- MatMul(input,w1)
  hidden <- Add(hidden_matmul, b1)
  output_matmul <- MatMul(hidden,w2)
  output <- Add(output_matmul, b2)
  
  feedInput(input, feed, dtype)
  setOutput(output)
  runSession()
  
  if (identical(dtype,"int32")){
    output <- printIntOutputs()
  }
  else{
    output <- printDoubleOutputs()
  }
  return(output)
}
