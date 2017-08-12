#' @title Generate Unique Name
#'
#' @description Generates a unique name for the node in the graph
#' 
#' @param extra_length Number of characters to be randomly generated for identifier
#' @param op_name Name of op to be used as base of unique generated name
#' 
#' @return Unique generated name for node
generateUniqueName <- function(extra_length=5, op_name="") {

  randomString <- paste(sample(c(0:9, letters, LETTERS),
                               extra_length, replace=TRUE),
                        collapse="")
  randomString <- paste(op_name,randomString, sep="")

  return(randomString)
}

#' @title Feed Input
#'
#' @description Easy-to-use wrapper for setFeedInput
#' 
#' @param input_node Node to which tensor must be fed to graph
#' @param feed Vector to be fed as input to the graph
#' 
#' @return Integer status
feedInput <- function(input_node, feed) {
  feed_vector <- c()
  
  if(!is.null(dim(feed))) {
    for (i in 1:nrow(feed)) {
      feed_vector <- c(feed_vector, as.numeric(feed[i,]))
    }
  } else {
    feed_vector <- feed
  }

  return (setFeedInput(input_node, feed_vector))
}

#' @title Run Interactive Session
#'
#' @description Runs the current Tensorflow session
#' 
#' @param op_names Node to be set as output of graph
#' 
#' @return Multidimensional output matrix
runSession <- function(op_names) {
  output <- runInternalSession(op_names)
  output_list <- list()
  for (op in op_names) {
    if (identical(output[[op]], "No Output")) {
      output_list[[op]] <- 0
    } else {
      if (length(output[[op]][["dim"]]) == 0) {
        output[[op]][["dim"]] <- length(output[[op]][["val"]])
      }
      output_array <- aperm(array(data = output[[op]][["val"]], dim = rev(output[[op]][["dim"]])))
      output_list[[op]] <- output_array
    }
  }
  return (output_list)
}

# Wrappers to create mathematical ops for graph

#' @title Placeholder Op
#'
#' @description Sets a placeholder op for a value that will be fed into the computation.
#' 
#' @param dtype Shape of Tensor
#' @param shape Datatype of Tensor
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Placeholder <- function(dtype, shape=NULL, name=NULL) {

  if(is.null(name)){
    name <- generateUniqueName(op_name = "Placeholder")
  }
  
  return (getPlaceholder(shape, dtype, "Placeholder", name))
}

#' @title Variable Op
#'
#' @description Sets a variable op
#' 
#' @param val Vector for value of Constant node
#' @param dtype Shape of Tensor
#' @param shape Datatype of Tensor
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Variable <- function(val, dtype, shape=length(val), name=NULL) {
  
  if(is.null(name)){
    name <- generateUniqueName(op_name = "Variable")
  }
  
  op <- getPlaceholder(shape, dtype, "VariableV2", name)
  
  return (op)
}


#' @title Constant Op
#'
#' @description  Easy-to-use wrapper for getConstant. Returns a constant tensor.
#' 
#' @param val Vector for value of Constant node
#' @param dtype Datatype of val
#' @param shape Vector indicating dimensions of val
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Constant <- function(val, dtype="float32", shape=length(val), name=NULL) {
  
   if (is.null(name)){
    name <- generateUniqueName(op_name = "Const")
  }
  
  return (getSourceOp(val, shape, dtype,"Const", name))
}

#' @title Add Op
#'
#' @description Returns x + y element-wise.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node, If not specified, a random name is generated
#' 
#' @return Unique name of node
Add <- function(l_op, r_op, name=NULL) {
  
  if (is.null(name)){
    name <- generateUniqueName(op_name = "Add")
  }
  
  return (getBinaryOp(l_op,r_op,"Add",name))
}

#' @title MatMul Op
#'
#' @description Multiply the matrix x by the matrix y.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node, If not specified, a random name is generated
#' 
#' @return Unique name of node
MatMul <- function(l_op, r_op, name=NULL) {

  if (is.null(name)){
    name <- generateUniqueName(op_name = "MatMul")
  }
  
  return (getBinaryOp(l_op,r_op,"MatMul",name))
}

#' @title Pow Op
#'
#' @description Computes the power of one value to another.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Pow <- function(l_op, r_op, name=NULL) {
  
  if (is.null(name)){
    name <- generateUniqueName(op_name = "Pow")
  }
  
  return (getBinaryOp(l_op,r_op,"Pow",name))
}

#' @title Neg Op
#'
#' @description Computes numerical negative value element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Neg <- function(inp, name=NULL) {
  
  if (is.null(name)){
    name <- generateUniqueName(op_name = "Neg")
  }
  
  return (getUnaryOp(inp, "Neg", name))
}

#' @title Tanh Op
#'
#' @description Computes hyperbolic tangent of `x` element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Tanh <- function(inp, name=NULL) {
  
  if (is.null(name)){
    name <- generateUniqueName(op_name = "Tanh")
  }
  
  return (getUnaryOp(inp, "Tanh", name))
}

#' @title Sigmoid Op
#'
#' @description Computes sigmoid of `x` element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Sigmoid <- function(inp, name=NULL) {
  
  if (is.null(name)) {
    name <- generateUniqueName(op_name = "Sigmoid")
  }
  
  return (getUnaryOp(inp, "Sigmoid", name))
}

#' @title Relu Op
#'
#' @description Computes rectified linear: `max(features, 0)`.
#' 
#' @param inp Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Relu <- function(inp, name=NULL) {
  
  if(is.null(name)) {
    name <- generateUniqueName(op_name = "Relu")
  }
  
  return (getUnaryOp(inp, "Relu", name))
}

#' @title Cos Op
#'
#' @description Computes cos of x element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Cos <- function(inp, name=NULL) {
  
  if (is.null(name)) {
    name <- generateUniqueName(op_name = "Cos")
  }
  
  return (getUnaryOp(inp, "Cos", name))
}

#' @title Softmax Op
#'
#' @description Computes softmax activations.
#' 
#' @param inp Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Softmax <- function(inp, name=NULL) {
  
  if(is.null(name)){
    name <- generateUniqueName(op_name = "Softmax")
  }
  
  return (getUnaryOp(inp, "Softmax", name))
}

#' @title Equal Op
#'
#' @description Returns the truth value of (x == y) element-wise.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node. If not specified, a random name is generated
#' 
#' @return Unique name of node
Equal <- function(l_op, r_op, name=NULL) {
  
  if(is.null(name)){
    name <- generateUniqueName(op_name = "Equal")
  }
  
  return (getBinaryOp(l_op, r_op, "Equal", name))
}

