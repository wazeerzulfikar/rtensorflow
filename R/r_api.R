# Generate random unique name for ops

generateUniqueName <- function(extra_length=5, op_name="") {

  randomString <- paste(sample(c(0:9, letters, LETTERS),
                               extra_length, replace=TRUE),
                        collapse="")
  randomString <- paste(op_name,randomString, sep="")

  return(randomString)
}

# Helper to set feed dim in while setting Input

feedInput <- function(input, feed, dim = c(length(feed)), dtype="int32") {
  return (setFeedInput(input, feed, dim, dtype))
}

# Wrappers to create mathematical ops for graph

Placeholder <- function(name="Placeholder", dtype="int32") {
  
  if(identical(name,"Placeholder")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getPlaceholder(dtype,name))
}

Constant <- function(val,dim = c(length(val)),dtype="int32",name="Const"){
  
  if(identical(name,"Const")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getConstant(val,dim,dtype,name))
}

Add <- function(l_op, r_op, name="Add"){
  
  if(identical(name,"Add")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op,r_op,"Add",name))
}

MatMul <- function(l_op, r_op, name="MatMul"){
  
  if(identical(name,"MatMul")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op,r_op,"MatMul",name))
}

Pow <- function(l_op, r_op, name="Pow"){
  
  if(identical(name,"Pow")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getBinaryOp(l_op,r_op,"Pow",name))
}

Neg <- function(inp, name="Neg"){
  
  if(identical(name,"Add")){
    name <- generateUniqueName(op_name = name)
  }
  
  return (getUnaryOp(inp,"Neg", name))
}

fetchOutput <- function(dtype="int32"){
  
  output <- getOutput(dtype);
  output_array <- array(data = output$val, dim = output$dim)
  
  return (output_array)
} 