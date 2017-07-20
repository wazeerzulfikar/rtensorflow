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
  return (setFeedInput(input_node, feed))
}

#' @title Run Session
#'
#' @description Runs the current Tensorflow session
#' 
#' @param op_name Node to be set as output of graph
#' 
#' @return Multidimensional output matrix
runSession <- function(op_name, train = FALSE) {
  output <- runInternalSession(op_name, train);
  if (train) {
    return (0)
  }

  output_array <- array(data = output$val, dim = output$dim)
  return (output_array)
}

# Wrappers to create mathematical ops for graph

#' @title Placeholder Wrapper
#'
#' @description  Easy-to-use wrapper for getPlaceholder. Sets a placeholder op for a value that will be fed into the computation.
#' 
#' @param dtype Shape of Tensor
#' @param shape Datatype of Tensor
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Placeholder <- function(dtype, shape = NULL, name = "Placeholder") {
  
  if(identical(name,"Placeholder")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getPlaceholder(shape, dtype, name))
}

#' @title Constant Wrapper
#'
#' @description  Easy-to-use wrapper for getConstant. Returns a constant tensor.
#' 
#' @param val Vector for value of Constant node
#' @param dtype Datatype of val
#' @param shape Vector indicating dimensions of val
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Constant <- function(val, dtype = "int32", shape = c(length(val)), name = "Const") {
  
  if(identical(name,"Const")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getConstant(val, shape, dtype, name))
}

#' @title Add Op
#'
#' @description Returns x + y element-wise.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Add <- function(l_op, r_op, name="Add") {
  
  if(identical(name,"Add")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op,r_op,"Add",name))
}

#' @title MatMul Op
#'
#' @description Multiply the matrix x by the matrix y.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
MatMul <- function(l_op, r_op, name="MatMul") {
  
  if(identical(name,"MatMul")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op,r_op,"MatMul",name))
}

#' @title Pow Op
#'
#' @description Computes the power of one value to another.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Pow <- function(l_op, r_op, name="Pow") {
  
  if(identical(name,"Pow")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op,r_op,"Pow",name))
}

#' @title Neg Op
#'
#' @description Computes numerical negative value element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Neg <- function(inp, name="Neg") {
  
  if (identical(name, "Neg")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp, "Neg", name))
}

#' @title Tanh Op
#'
#' @description Computes hyperbolic tangent of `x` element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Tanh <- function(inp, name="Tanh") {
  
  if (identical(name, "Tanh")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp, "Tanh", name))
}

#' @title Sigmoid Op
#'
#' @description Computes sigmoid of `x` element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Sigmoid <- function(inp, name="Sigmoid") {
  
  if (identical(name, "Sigmoid")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp, "Sigmoid", name))
}

#' @title Relu Op
#'
#' @description Computes rectified linear: `max(features, 0)`.
#' 
#' @param inp Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Relu <- function(inp, name="Relu") {
  
  if (identical(name, "Relu")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp, "Relu", name))
}

#' @title Cos Op
#'
#' @description Computes cos of x element-wise.
#' 
#' @param inp Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Cos <- function(inp, name="Cos") {
  
  if (identical(name, "Cos")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp, "Cos", name))
}

#' @title Softmax Op
#'
#' @description Computes softmax activations.
#' 
#' @param inp Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Softmax <- function(inp, name="Softmax") {
  
  if (identical(name, "Softmax")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp, "Softmax", name))
}

#' @title Equal Op
#'
#' @description Returns the truth value of (x == y) element-wise.
#' 
#' @param l_op Input node
#' @param r_op Input node
#' @param name Optional custom name for node
#' 
#' @return Unique name of node
Equal <- function(l_op, r_op, name="Equal") {
  
  if (identical(name,"Equal")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op, r_op, "Equal", name))
}

