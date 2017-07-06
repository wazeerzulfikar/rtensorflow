# Wrappers to create mathematical ops for graph

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