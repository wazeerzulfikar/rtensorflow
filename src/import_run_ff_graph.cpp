#include <Rcpp.h>
using namespace Rcpp;
#include "utils.h"

// [[Rcpp::export]]

int c_import_run_ff_graph(std::string path, IntegerVector inp) {
  
  TF_Buffer* graph_def = read_file(path); 
  if (graph_def == nullptr){
    return -1;
  }
  TF_Graph* graph = TF_NewGraph();
  
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions * options = TF_NewSessionOptions();
  
  TF_Session * session = TF_NewSession( graph, options, status );
  
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  
  TF_GraphImportGraphDef(graph,graph_def, opts, status);
  
  TF_DeleteImportGraphDefOptions(opts);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error importing graph");
    return -1;
  }
  
  printf("Sucessfully imported graph\n");
  
  TF_Operation * input = TF_GraphOperationByName(graph,"input");
  TF_Operation * output = TF_GraphOperationByName(graph,"output");
  
  setOutputs({output});
  
  if(inp.size()!=3){
    cout<<"Wrong size of Input"<<endl;
    return -1;
  }
  
  int* c_inp = new int[inp.size()];
  int iter;
  for(iter=0;iter<inp.size();iter++){
    c_inp[iter] = inp[iter];
  }
  
  const int64_t dim[2] = {1,3};
  TF_Tensor* feed = TF_NewTensor(
    TF_INT32, dim, 2, c_inp, sizeof(c_inp),
    &tensor_deallocator,
    nullptr);
  
  setInputs({{input,feed}});

  //Run

  printf("Running the Session.. \n");
  
  setPointers();
  
  runSession(session,status);
  
  int out = getOutputs();
  
  printf("Output Value: %i\n", out);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session");
    cout << TF_Message(status);
    return 1;
  }
  
  
  TF_CloseSession( session, status );
  TF_DeleteSession( session, status );
  
  TF_DeleteStatus(status);
  TF_DeleteBuffer(graph_def);
  
  TF_DeleteGraph(graph);
  
  return out;
}


