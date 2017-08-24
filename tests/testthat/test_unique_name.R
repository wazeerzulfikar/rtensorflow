test_that("Unique Name Generator", {
      
  base <- "Placeholder"
  unique_name <- generateUniqueName(extra_length = 5, op_name = base)
  
  expect_that(unique_name, is_a("character") )
  expect_equal(substr(unique_name, 0, 11), "Placeholder")
  expect_equal(nchar(unique_name), 16)
  
})