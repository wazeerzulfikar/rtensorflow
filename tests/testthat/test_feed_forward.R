test_that("Check Import Run Graph", {
  
  path <- "../models/feed_forward_graph.pb"
  
  result <- import_run_graph(path,c(1,2,3))
  
  expect_that( result, is_a("matrix") )
  expect_equal( result, array(data=c(29),dim=c(1,1)) )
  
  result <- import_run_graph(path,c(652,211,42))
  
  expect_that( result[0], is_a("integer") )
  expect_equal( result, array(data=c(3625),dim=c(1,1)) )

  file <- loadGraphFromFile("./wrong_path")
  
  expect_that( file, is_a("numeric") )
  expect_equal( file, -1 )
  
})

test_that("Check Build Run Graph", {
  
  result <- build_run_graph(c(1,2,3),dtype="int32")
  
  expect_that( result, is_a("matrix") )
  expect_equal( result, array(data=c(29),dim=c(1,1)) )
  
  result <- build_run_graph(c(45.6,3.1,4),dtype="double")
  
  expect_that( result, is_a("matrix") )
  expect_equal( result, array(data=c(215.8),dim=c(1,1)) )
  
  result <- build_run_graph(c(652,211,42),dtype="int32")
  
  expect_that( result[0], is_a("integer") )
  expect_equal( result, array(data=c(3625),dim=c(1,1)) )
  
})
