#include <Rcpp.h>
using namespace Rcpp;
#include "utils.h"

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
    &tensor_deallocator,
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
  
  setOutputs({output});
 
  if(inp.size()!=3){
    cout<<"Wrong size of Input"<<endl;
    return -1;
  }
  
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
    &tensor_deallocator,
    &deallocator_called);
    
  setInputs({{input,feed}});

  //Run
    
  setPointers();
  
  printf("Running the Session..\n");
    
  runSession(session,status);
    
  int out = getOutputs();
  
  printf("Output Value: %i\n", out);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session");
    return 1;
  }
    
  TF_CloseSession( session, status );
  TF_DeleteSession( session, status );
    
  TF_DeleteStatus(status);
    
  TF_DeleteGraph(graph);
    
  return out;
}


