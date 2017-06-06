test_that("Check Hello", {
  
  result <- hello()
  
  expect_that( result, is_a("character") )
  expect_equal( result, "Hello World" )
})

test_that("Check Factorial", {
  result <- factorial(4)
  
  expect_that( result, is_a("integer"))
  expect_equal( result, 24)
})