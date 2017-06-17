test_that("Check Import Run Graph", {
  
  result <- import_run_graph("../models/feed_forward_graph.pb",c(1,2,3))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 29 )
})

test_that("Check Build Run Graph", {
  
  result <- build_run_graph(c(1,2,3))
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 29 )
})
