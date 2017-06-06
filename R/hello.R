#' @useDynLib rtensorflow
#' @importFrom Rcpp sourceCpp
#' @exportPattern "^[[:alpha:]]+"

hello <- function() {
  print("Hello, world!")
  return("Hello World")
}
