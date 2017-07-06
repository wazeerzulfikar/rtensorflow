#include <Rcpp.h>
using namespace Rcpp;
#include "utils.h"

TF_Status* status;
TF_Graph* graph;
TF_SessionOptions* options;
TF_Session* session;
std::map <string,TF_Operation*> op_list;

// [[Rcpp::export]]
int instantiateSessionVariables(){
  status = TF_NewStatus();
  graph = TF_NewGraph();
  options = TF_NewSessionOptions();
  
  session = TF_NewSession(graph, options, status);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error instantiating variables");
    return -1;
  }
  
  printf("Sucessfully instantiated session variables\n");
  return 0;
}

// [[Rcpp::export]]
int loadGraphFromFile(std::string path){
  TF_Buffer* graph_def = read_file(path); 
  if (graph_def == nullptr){
    printf("File not found");
    return -1;
  }
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_def);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error importing graph");
    return -1;
  }
  
  printf("Sucessfully imported graph\n");
  return 0;
}

// [[Rcpp::export]]
int feedInput(std::string op_name, NumericVector inp, std::string type) {
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* input = TF_GraphOperationByName(graph,op_name_ptr);
  if(inp.size()!=3){
    cout<<"Wrong size of Input"<<endl;
    return -1;
  }
  
  TF_Tensor* feed = parseInputs(inp,{1,3},type);
  
  setInputs({{input,feed}});
  
  return 0;
}

// [[Rcpp::export]]
int setOutput(std::string op_name){
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* output = TF_GraphOperationByName(graph,op_name_ptr);
  
  setOutputs({output});
  return 0;
}

// [[Rcpp::export]]
int runSession(){
  printf("Running the Session.. \n");
  setPointers();
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session\n");
    cout << TF_Message(status) << endl;
    return -1;
  }
  
  runSession(session,status);
  
  if (TF_GetCode(status)!=TF_OK){
    printf("Error running session\n");
    cout << TF_Message(status) << endl;
    return -1;
  }
  
  TF_CloseSession( session, status );
  
  return 0;
}

// [[Rcpp::export]]
int printIntOutputs(){
  return getIntOutputs();
}

// [[Rcpp::export]]
double printDoubleOutputs(){
  return getDoubleOutputs();
}

// [[Rcpp::export]]
int deleteSessionVariables() {

  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
  TF_DeleteGraph(graph);
  
  return 0;
}

//Graph Building Functions

// [[Rcpp::export]]
std::string Placeholder(std::string dtype){
  pair<char*,TF_Operation*> op;
  op = Placeholder(graph, status, dtype);
  op_list.emplace(op.first,op.second);
  return op.first;
}

// [[Rcpp::export]]
std::string getConstant(NumericVector val, std::vector<int64_t> dim, std::string dtype){
  TF_Tensor* val_t = parseInputs(val,dim,dtype);
  pair<char*,TF_Operation*> op;
  op = Constant(val_t,graph,status);
  op_list.emplace(op.first,op.second);
  return op.first;
}

// [[Rcpp::export]]
std::string getUnaryOp(std::string inp, std::string op_name){
  pair<char*,TF_Operation*> op;
  TF_Operation* i = op_list.at(inp);
  op = Unary_Op(i, graph, status, op_name);
  op_list.emplace(op.first,op.second);
  return op.first;
}

// [[Rcpp::export]]
std::string getBinaryOp(std::string l_op, std::string r_op, std::string op_name){
  pair<char*,TF_Operation*> op;
  TF_Operation* l = op_list.at(l_op);
  TF_Operation* r = op_list.at(r_op);
  op = Binary_Op(l, r, graph, status, op_name);
  op_list.emplace(op.first,op.second);
  return op.first;
}

//Debug Helpers
// [[Rcpp::export]]
void printOpList(){
  for (auto const& x : op_list)
  {
    std::cout << x.first<< ':' << x.second<< std::endl ;
  }
}

// [[Rcpp::export]]
void locateError(){
  if(TF_GetCode(status)!=TF_OK){
    cout<<"Here is the error"<<endl;
    cout<<TF_Message(status)<<endl;
  }
}
