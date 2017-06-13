test_that("Check Neural Net", {
  
  result <- import_graph("../models/feed_forward_graph.pb")
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 29 )
})
