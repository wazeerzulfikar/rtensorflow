# Wrappers to create mathematical ops for graph

Constant <- function(val,dim,dtype){
  if (missing(dim)){
    dim <- c(length(val))
  }
  return (getConstant(val,dim,dtype))
}

Add <- function(l_op,r_op){
  return (getBinaryOp(l_op,r_op,"Add"))
}

MatMul <- function(l_op,r_op){
  return (getBinaryOp(l_op,r_op,"MatMul"))
}

Pow <- function(l_op,r_op){
  return (getBinaryOp(l_op,r_op,"Pow"))
}

Neg <- function(inp){
  return (getUnaryOp(inp,"Neg"))
}

fetchOutput <- function(dtype="int32"){
  
  output <- getOutput(dtype);
  output_array <- array(data = output$val, dim = output$dim)
  
  return (output_array)
} 