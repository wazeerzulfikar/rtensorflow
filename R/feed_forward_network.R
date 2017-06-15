#' @useDynLib rtensorflow
#' @importFrom Rcpp sourceCpp
#' @exportPattern "^[[:alpha:]]+"
#' 
#' @title Import and Run Graph
#' @description Function to import and run graph using C API
#' @param path Path to the graph, "/tests/models/feed_forward_graph.pb"
#' @param imput Integer vector of size 3, feed for the network
#' @return Output Value of Network 

import_run_graph <- function(path,input) {
  output <- c_import_run_ff_graph(path,input)
  return(output)
}
