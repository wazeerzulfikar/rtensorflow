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
  feedInput("input", feed)
  setOutput("output")
  runSession()
  output <- printOutput()
  return(output)
}

#' @title Build and Run Graph
#' @description Function to build and run graph using C API
#' @param feed Integer vector of size 3, feed for the network
#' @return Output Value of Network 

build_run_graph <- function(feed){
  instantiateSessionVariables()
  
  input <- Placeholder("input")
  
  w1 <- Constant(c(3,4),"w1")
  b1 <- Constant(c(4),"b1")
  w2 <- Constant(c(4,1),"w2")
  b2 <- Constant(c(1),"b2")
  
  hidden_matmul <- MatMul(input,w1,"hidden_matmul")
  hidden <-  Add(hidden_matmul, b1, "hidden")
  output_matmul <- MatMul(hidden,w2,"output_matmul")
  output <-  Add(output_matmul, b2, "output")
  
  feedInput("input", feed)
  setOutput("output")
  runSession()
  output <- printOutput()
  
  return(output)
}