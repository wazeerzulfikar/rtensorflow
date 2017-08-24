test_that("Check Import Run Graph", {
  
  #  library(reticulate)
  # py_run_file("../regress.py")
  
  path <- "../saved-graphs/feed_forward_graph.pb"
  
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