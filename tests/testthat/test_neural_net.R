test_that("Check Neural Net", {
  
  result <- neural_net(c(1,2,3),4)
  
  expect_that( result, is_a("integer") )
  expect_equal( result, 40 )
})
