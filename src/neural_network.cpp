#include <Rcpp.h>
using namespace Rcpp;
#include <stdlib.h>


// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//' @title Feed-Forward Network
//' @param input Integer Vector for input to network, of any length.
//' @param n_hidden Integer for number of hidden neurons.
//' @description Simple feed forward network with X input neurons, 4 hidden neurons and 1 ouptut neuron.
//' @return int Output of Network
//' @export
// [[Rcpp::export]]

int neural_net(NumericVector input, int n_hidden){
  
  // Simple feed forward network with X input neurons,
  // n_hidden hidden neurons and 1 output neuron. All weights and biases are equal to 1.
  
  int weights_1_2[n_hidden][input.length()];
  int biases_1_2[n_hidden][input.length()];
  int output_2[n_hidden];
  
  int weights_2_3[1][n_hidden];
  int biases_2_3[1][n_hidden];
  int output_3[1];
  
  int i,j,output;
  
  // Initialising all variables with 1
  for(i=0;i<n_hidden;i++){
    for(j=0;j<input.length();j++){
      weights_1_2[i][j] = 1; 
      biases_1_2[i][j] = 1; 
    }
  }
  
  for(i=0;i<4;i++){
    weights_2_3[0][i] = 1; 
    biases_2_3[0][i] = 1; 
  }
  
  // Forward pass
  for(i=0;i<4;i++){
    output = 0;
    for(j=0;j<3;j++){
      output += weights_1_2[i][j]*input[j] + biases_1_2[i][j];
    }
    output_2[i] = output;
  }
  
  output = 0;
  for(i=0;i<4;i++){
    output +=  weights_2_3[0][i]*output_2[i] + biases_2_3[0][i];
  }
  output_3[0]=output;
  
  return output_3[0];
  
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
neural_net(42)
*/
