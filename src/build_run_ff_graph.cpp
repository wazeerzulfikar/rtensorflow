#include <Rcpp.h>
using namespace Rcpp;
#include "utils.h"

// [[Rcpp::export]]

int c_build_run_ff_graph(NumericVector inp) {
  // Builds and runs a simple feedforward network. Input layer with 3 neurons, hidden layer 
  // with 4 and a single output neuron. 
  
  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  TF_SessionOptions* options = TF_NewSessionOptions();
  
  TF_Session* session = TF_NewSession(graph, options, status);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error instantiating variables");
    return 1;
  }
  
  printf("Sucessfully instantiated session variables\n");
  
  TF_Operation* input = Placeholder(graph, status);
  
  TF_Operation* w1 = Constant(ones({3,4}),graph,status, "w1");
  TF_Operation* b1 = Constant(ones({4,}),graph,status, "b1");
  TF_Operation* w2 = Constant(ones({4,1}),graph,status, "w2");
  TF_Operation* b2 = Constant(ones({1,}),graph,status, "b2");
  
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
  
  TF_Tensor* feed = parseInputs(inp,{1,3},"int32");
    
  setInputs({{input,feed}});

  //Run
    
  setPointers();
  
  printf("Running the Session..\n");
    
  runSession(session,status);
    
  int out = getIntOutputs();
  
  printf("Output Value: %i\n", out);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session");
    return 1;
  }
    
  TF_DeleteTensor(feed);
  
  TF_CloseSession( session, status );
  TF_DeleteSession( session, status );
    
  TF_DeleteStatus(status);
    
  TF_DeleteGraph(graph);
    
  return out;
}


