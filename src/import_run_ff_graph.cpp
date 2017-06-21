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

void deallocator(void* data, size_t length) {                                             
  free(data);                                                                       
}     

void tensor_deallocator(void* data, size_t length,void* arg) {                                             
  delete[] static_cast<int*>(data);
}

TF_Buffer* read_file(std::string path){
  
  const char * fpath = path.c_str();
  FILE *f = fopen(fpath, "rb");
  
  if (f==NULL){
    printf("\n File not Found.\n");
    return nullptr;
  }
  
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);                                                                  
  fseek(f, 0, SEEK_SET);                                           
  
  void* data = malloc(fsize);                                                             
  fread(data, fsize, 1, f);
  fclose(f);
  
  TF_Buffer* buf = TF_NewBuffer();                                                        
  buf->data = data;
  buf->length = fsize;                                                                    
  buf->data_deallocator = deallocator;
  
  return buf;
}

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
  
  printf("Running the Session.. \n");
  
  TF_SessionRun( session, nullptr,
                 inputs_ptr, input_values_ptr, inputs_.size(),  // Inputs
                 outputs_ptr, output_values_ptr, outputs_.size(),  // Outputs
                 targets_ptr, targets.size(),  // Operations
                 nullptr, status );
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session");
    cout << TF_Message(status);
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
  TF_DeleteBuffer(graph_def);
  
  TF_DeleteGraph(graph);
  
  return *((int*) output_contents);
}


