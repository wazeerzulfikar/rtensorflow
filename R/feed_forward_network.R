#' @useDynLib rtensorflow
#' @importFrom Rcpp sourceCpp
#' @exportPattern "^[[:alpha:]]+"
#' @param path Path to the graph, "/tests/models/feed_forward_graph.pb"
#' @return Output Value of Network 

import_graph <- function(path) {
  output <- c_import_ff_graph(path)
  return(output)
}
