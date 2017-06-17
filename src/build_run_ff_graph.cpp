#include <Rcpp.h>
using namespace Rcpp;
#include <stdio.h>
#include <iostream>
using namespace std;
#include <tensorflow/c/c_api.h>
#include <stdlib.h>   
#include <vector>  
#include <string>

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
// 

void tensors_deallocator(void* data, size_t length,void* arg) { 
  delete [] static_cast<int*> (data);
}


TF_Operation* Placeholder(TF_Graph* graph, TF_Status* status, const char* name="input" ){
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc,"dtype",TF_INT32);
  return TF_FinishOperation(desc, status);
}

TF_Operation* Constant(TF_Tensor* tensor, TF_Graph* graph, TF_Status* status, const char* name="const"){
  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrTensor(desc, "value", tensor, status);
  if(TF_GetCode(status)!=TF_OK){
    return nullptr;
  }
  TF_SetAttrType(desc,"dtype",TF_TensorType(tensor));
  return TF_FinishOperation(desc, status);
}

TF_Operation* Add(TF_Operation* l,TF_Operation* r, TF_Graph* graph, TF_Status* status, const char* name="add"){
  TF_OperationDescription* desc = TF_NewOperation(graph, "Add", name);
  TF_AddInput(desc, {l,0});
  TF_AddInput(desc, {r,0});
  return TF_FinishOperation(desc, status);
}

TF_Operation* MatMul(TF_Operation* l, TF_Operation* r, TF_Graph* graph, TF_Status* status, const char* name="matmul"){
  TF_OperationDescription* desc = TF_NewOperation(graph,"MatMul", name);
  TF_AddInput(desc, {l,0});
  TF_AddInput(desc, {r,0});
  return TF_FinishOperation(desc, status);
}

static TF_Tensor* ones(int row, int col){
  //Function for returning a Tensor of required dimension, filled with 1's
  int* arr = new int[row*col];
  for(int i=0;i<row*col;i++){
    arr[i] = 1;
  }
  const int64_t dim[2] = {row,col};
  const int numBytes = sizeof(int);
  return TF_NewTensor(
    TF_INT32, dim, 2, arr, numBytes*row*col,
    &tensors_deallocator,
    nullptr);
  
}

// [[Rcpp::export]]

int c_build_run_ff_graph(IntegerVector inp) {
  // Builds and runs a simple feedforward network. Input layer with 3 neurons, hidden layer 
  // with 4 and a single output neuron. 
  
  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  TF_SessionOptions * options = TF_NewSessionOptions();
  
  TF_Session * session = TF_NewSession(graph, options, status);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error instantiating variables");
    return 1;
  }
  
  printf("Sucessfully instantiated session variables\n");
  
  TF_Operation* input = Placeholder(graph, status);
  
  TF_Operation* w1 = Constant(ones(3,4),graph,status, "w1");
  TF_Operation* b1 = Constant(ones(4,1),graph,status, "b1");
  TF_Operation* w2 = Constant(ones(4,1),graph,status, "w2");
  TF_Operation* b2 = Constant(ones(1,1),graph,status, "b2");
  
  TF_Operation* hidden_matmul = MatMul(input,w1,graph,status,"hidden_matmul");
  TF_Operation* hidden = Add(hidden_matmul, b1, graph, status, "hidden");
  TF_Operation* output_matmul = MatMul(hidden,w2,graph,status,"output_matmul");
  TF_Operation* output = Add(output_matmul, b2, graph, status, "output");
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error instantiating operations\n");
    cout<<TF_Message(status)<<endl;
    return 1;
  }
  
  printf("Sucessfully added ops to graph\n");
  
  std::vector<TF_Operation*> targets;
  
  std::vector<TF_Output> inputs_;
  std::vector<TF_Tensor*> input_values_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> output_values_;
  
  TF_Output o;
  o.index = 0;
  o.oper = output;
  outputs_.push_back(o);
  
  TF_Output i;
  i.index = 0;
  i.oper = input;
  inputs_.push_back(i);
  
  //Fill int array with input values
  int* c_inp = new int[inp.size()];
  int iter;
  for(iter=0;iter<inp.size();iter++){
    c_inp[iter] = inp[iter];
  }
  
  int64_t dim[2] = {1,3};
  bool deallocator_called = false;
  TF_Tensor* feed = TF_NewTensor(
    TF_INT32, dim, 2, c_inp, sizeof(c_inp),
    &tensors_deallocator,
    &deallocator_called);
    
  input_values_.push_back(feed);
    
  //Run
    
  output_values_.resize(outputs_.size(), nullptr);
    
    
  TF_Operation* const* targets_ptr =
      targets.empty() ? nullptr : &targets[0];
    
    
  const TF_Output* inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
  TF_Tensor* const* input_values_ptr =
      input_values_.empty() ? nullptr : &input_values_[0];
    
  const TF_Output* outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
  TF_Tensor** output_values_ptr =
      output_values_.empty() ? nullptr : &output_values_[0];
    
  printf("Running the Session..\n");
  
  TF_SessionRun( session, nullptr,
                inputs_ptr, input_values_ptr, inputs_.size(),  // Inputs
                outputs_ptr, output_values_ptr, outputs_.size(),  // Outputs
                targets_ptr, targets.size(),  // Operations
                nullptr, status );
    
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session");
    return 1;
  }
    
  TF_Tensor* out = output_values_[0];
  void* output_contents = TF_TensorData(out);
    
  cout <<"Output size: "<< output_values_.size() <<endl;
    
  printf("Output Value: %i\n", *((int*) output_contents));
  
  TF_DeleteTensor(feed);  
  TF_CloseSession( session, status );
  TF_DeleteSession( session, status );
    
  TF_DeleteStatus(status);
    
  TF_DeleteGraph(graph);
    
  return *((int*) output_contents);
}


