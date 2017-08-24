test_that("Check Build Run Graph", {
  
  status <- initializeSessionVariables()
  
  expect_equal(status, 0)
  
  dtype <- "int32"
  
  input <- Placeholder(dtype, shape = c(1,3), name="input")
  
  expect_that(input, is_a("character") )
  expect_equal(input, "input")
  
  w1 <- Constant(rep(1,12), dtype=dtype, shape=c(3,4))
  b1 <- Constant(rep(1,4), dtype=dtype, shape=c(4))
  w2 <- Constant(rep(1,4), dtype=dtype, shape=c(4,1))
  b2 <- Constant(rep(1,1), dtype=dtype, shape=c(1))
  
  expect_that(w1, is_a("character") )
  expect_equal(substr(w1,0,5), "Const")
  
  hidden_matmul <- MatMul(input, w1)
  
  expect_that(hidden_matmul, is_a("character") )
  expect_equal(substr(hidden_matmul,0,6), "MatMul")
  
  hidden_layer <- Add(hidden_matmul, b1, name = "hidden")
  
  properties <- getOpProperties(hidden_layer)
  
  actual_properties <- list()
  actual_properties["op_type"] <- "Add"
  actual_properties["num_inputs"] <- 2
  actual_properties["num_outputs"] <- 1
  
  expect_that(properties, is_a("list") )
  expect_equal(properties, actual_properties)
  
  expect_that(hidden_layer, is_a("character") )
  expect_equal(hidden_layer, "hidden")
  
  output_matmul <- MatMul(hidden_layer, w2)
  output_layer <- Add(output_matmul, b2, name="out")
  
  expect_that(output_layer, is_a("character") )
  expect_equal(output_layer, "out")
  
  status <- feedInput(input, c(1,2,3))
  
  expect_equal(status, 0)
  
  output <- runSession("out")
  
  error <- printError()
  
  expect_equal(error, "No Error")
  
  result <- output[[output_layer]]
  
  expect_that(result, is_a("matrix") )
  expect_equal(result, array(data=c(29),dim=c(1,1)) )
  
  resetGraph()
  
  nodes <- getNodeList()
  expect_equal(nodes, list())
  
  deleteSessionVariables()
  
})
