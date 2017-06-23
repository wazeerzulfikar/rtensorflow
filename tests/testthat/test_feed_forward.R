test_that("Check Import Run Graph", {
  
  path <- "../models/feed_forward_graph.pb"
  
  result <- import_run_graph(path,c(1,2,3))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 29 )
  
  result <- import_run_graph(path,c(652,211,42))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 3625 )

  file <- loadGraphFromFile("./wrong_path")
  
  expect_that( file, is_a("integer") )
  expect_equal( file, -1 )
  
  file <- loadGraphFromFile(path)
  input_status <- feedInput("input",c(1,2,3,4))
  
  expect_that( input_status, is_a("integer") )
  expect_equal( input_status, -1 )
})

test_that("Check Build Run Graph", {
  
  result <- build_run_graph(c(1,2,3))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 29 )
  
  result <- build_run_graph(c(7,12,3))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 93 )
  
  result <- build_run_graph(c(652,211,42))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 3625 )
  
})
