#include <Rcpp.h>
using namespace Rcpp;
#include "utils.h"

TF_Status* status;
TF_Graph* graph;
TF_SessionOptions* options;
TF_Session* session;
std::map <string, TF_Operation*> op_list;


//' @title Initialize Session Variables
//' 
//' @description Initializes all global variables and allocate space for each
//' 
//' @return Integer status 
//' 
//' @examples
//' initializeSessionVariables()
//' 
// [[Rcpp::export]]
int initializeSessionVariables() {
  status = TF_NewStatus();
  graph = TF_NewGraph();
  options = TF_NewSessionOptions();
  
  session = TF_NewSession(graph, options, status);
  op_list.clear();
  
  if (TF_GetCode(status)!=TF_OK) {
    cout << "Error instantiating variables" << endl;
    return -1;
  }
  
  cout << "Sucessfully instantiated session variables" << endl;
  return 0;
}

//' @title Load Graph from File
//' 
//' @description Assigns graph variable with one indicated by path
//' 
//' @param path Path to the graph
//' 
//' @return Integer status
//' 
//' @examples
//' loadGraphFromFile("/tests/models/feed_forward_graph.pb")
//' 
// [[Rcpp::export]]
int loadGraphFromFile(std::string path) {
  TF_Buffer* graph_def = read_file(path.c_str()); 
  if (graph_def == nullptr) {
    cout << "File not found" << endl;
    return -1;
  }
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_def);
  
  if (TF_GetCode(status)!=TF_OK) {
    cout << "Error importing graph" << endl;
    return -1;
  }
  
  cout << "Sucessfully imported graph\n" << endl;
  return 0;
}

//' @title Feed Input Ops
//' 
//' @description Sets the input node of graph, and feeds a tensor to it
//' 
//' @param op_name Op name (Node) to be set as input to graph
//' @param inp Vector to be fed to graph
//' @param dim Vector indicating dimensions of inp
//' @param dtype Datatype of inp 
//' 
//' @return Integer status 
//' 
// [[Rcpp::export]]
int setFeedInput(std::string op_name, NumericVector inp, std::vector<int64_t> dim) {
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* input = TF_GraphOperationByName(graph, op_name_ptr);
  TF_DataType dtype = TF_OperationOutputType({input,0});

  TF_Tensor* feed = parseInputs(inp,dim,dtype);
  
  setInputs({{input,feed}});
  
  return 0;
}

List fetchOutput(TF_DataType dtype) {
  NumericVector output_val;
  if(dtype == 3) {
    pair<int*, int64_t> out;
    out = getIntOutput();
    output_val = NumericVector(out.second);
    for (int i = 0; i < out.second; ++i){
      output_val[i] = out.first[i];
    }
  } else {
    pair<double*, int64_t> out;
    out = getDoubleOutput();
    output_val = NumericVector(out.second);
    for (int i = 0; i < out.second; ++i) {
      output_val[i] = out.first[i];
    }
  }
  List output;
  output["val"] = output_val;
  output["dim"] = getOutputDimensions();
  return output;
}

//' @title Run Session
//' 
//' @description Runs the Current Session
//' 
//' @return Integer status
//' 
//' @examples
//' runSession()
//' 
// [[Rcpp::export]]
List runInternalSession(std::string op_name) {
  cout << "Running the Session.. " << endl;
  
  TF_Operation* output = setOutputNode(op_name, graph);
  TF_DataType dtype = TF_OperationOutputType({output,0});
  
  setPointers();
  
  if (TF_GetCode(status)!=TF_OK) {
    cout << "Error in graph" << endl;
    cout << TF_Message(status) << endl;
    return -1;
  }
  
  runSession(session,status);
  
  if (TF_GetCode(status)!=TF_OK) {
    cout << "Error running session" << endl;
    cout << TF_Message(status) << endl;
    return -1;
  }
  
  return fetchOutput(dtype);
}

//' @title Close and Delete Session Variables
//' 
//' @description Closes session and frees all memory associated with it
//' 
//' @return Integer status
//' 
//' @examples
//' initializeSessionVariables()
//' deleteSessionVariables()
//' 
// [[Rcpp::export]]
int deleteSessionVariables() {

  resetInputValues();
  resetOutputValues();
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
  TF_DeleteGraph(graph);
  
  return 0;
}

//Graph Building Functions

//' @title Placeholder
//' 
//' @description Adds a placeholder operation to the graph
//' 
//' @param dtype Datatype of input
//' @param unique_name Unique name for the node
//' 
//' @return Unique node name
//' 
// [[Rcpp::export]]
std::string getPlaceholder(std::string dtype, std::string unique_name) {
  pair<string, TF_Operation*> op;
  op = Placeholder("Placeholder", unique_name, dtype, graph, status);
  op_list.emplace(op.first, op.second);
  return op.first;
}

//' @title Constant
//' 
//' @description Adds a constant operation to the graph
//' 
//' @param val Tensor to be initialized as Constant
//' @param dim Vector indicating dimensions of val
//' @param dtype Datatype of input
//' @param unique_name Unique name for the node
//' 
//' @return Unique node name
//' 
// [[Rcpp::export]]
std::string getConstant(NumericVector val, std::vector<int64_t> dim, std::string dtype, std::string unique_name) {
  TF_Tensor* val_t = parseCustomInputs(val, dim, dtype);
  pair<string, TF_Operation*> op;
  op = Constant("Const", unique_name, val_t, graph, status);
  op_list.emplace(op.first, op.second);
  return op.first;
}

//' @title Unary Op
//' 
//' @description Adds a unary operation to the graph
//' 
//' @param inp Input node
//' @param op_name Type of operation for node
//' @param unique_name Unique name for the node
//' 
//' @return Unique node name
//' 
// [[Rcpp::export]]
std::string getUnaryOp(std::string inp, std::string op_name, std::string unique_name) {
  pair<string, TF_Operation*> op;
  TF_Operation* i = op_list.at(inp);
  op = Unary_Op(op_name, unique_name, i, graph, status);
  op_list.emplace(op.first, op.second);
  return op.first;
}

//' @title Binary Op
//' 
//' @description Adds a binary operation to the graph
//' 
//' @param l_op Input node
//' @param r_op Input node
//' @param op_name Type of operation for node
//' @param unique_name Unique name for the node
//' 
//' @return Unique node name
//' 
// [[Rcpp::export]]
std::string getBinaryOp(std::string l_op, std::string r_op, std::string op_name, std::string unique_name) {
  pair<string, TF_Operation*> op;
  TF_Operation* l = op_list.at(l_op);
  TF_Operation* r = op_list.at(r_op);
  op = Binary_Op( op_name, unique_name, l, r, graph, status);
  op_list.emplace(op.first, op.second);
  return op.first;
}

//Debug Helpers

//' @title Print Node List
//' 
//' @description Debug helper, prints all nodes currently in the graph
//' 
//' @return NULL 
//' 
//' @examples
//' printNodeList()
//' 
// [[Rcpp::export]]
void printNodeList() {
  for (auto const& op : op_list) {
    cout << op.first << ':' << op.second << endl ;
  }
}

//' @title Locate Error
//' 
//' @description Debug helper, Locates the error and prints description of the error
//' 
//' @return NULL
//' 
//' @examples
//' locateError()
//' 
// [[Rcpp::export]]
void locateError() {
  if (TF_GetCode(status)!=TF_OK) {
    cout << "Here is the error :"<< endl;
    cout << TF_Message(status) << endl;
  }
}
