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
  
  resetInputValues();
  resetOutputValues();
  resetTargets();
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

//' @title Load Saved Model
//' 
//' @description Loads a saved TensorFlow model built in python
//' 
//' @param path Path to the saved model
//' @param tags Tags associated with the graph (from {"serve", "train, "gpu"})
//' 
//' @return Integer Status
//' 
// [[Rcpp::export]]
int loadSavedModel(std::string path, CharacterVector tags) {
  TF_Buffer* run_options = TF_NewBufferFromString("", 0);
  TF_Buffer* metagraph = TF_NewBuffer();
  char** tags_ptr = new char*[tags.size()];
  for (int i=0; i < tags.size(); ++i){
    tags_ptr[i]=tags[i];
  }
  
  session = TF_LoadSessionFromSavedModel(
    options, run_options, path.c_str(), tags_ptr, tags.size(), graph, metagraph, status);
  
  if (TF_GetCode(status)!=TF_OK) {
    cout << "Error: ";
    cout << TF_Message(status) << endl;
    return -1;
  }
  
  return 0;
}

//' @title Feed Input Ops
//' 
//' @description Sets the input node of graph, and feeds a tensor to it
//' 
//' @param op_name Op name (Node) to which tensor must be fed to graph
//' @param inp Vector to be fed to graph
//' 
//' @return Integer status 
//' 
// [[Rcpp::export]]
int setFeedInput(std::string op_name, List inp) {
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* input = TF_GraphOperationByName(graph, op_name_ptr);

  TF_DataType dtype = TF_OperationOutputType({input,0});

  int num_dims = TF_GraphGetTensorNumDims(graph, {input, 0}, status);
  int64_t* shape;
  
  if (num_dims==-1) {
    num_dims = 1;
    shape = new int64_t[num_dims];
    shape[0] = -1;
  } else {
    shape = new int64_t[num_dims];
    TF_GraphGetTensorShape(graph, {input,0}, shape, num_dims, status);
  }
  
  vector<int64_t> shape_vector;
  for (int i=0; i < num_dims; ++i){
    shape_vector.emplace_back(shape[i]);

  }
  TF_Tensor* feed = parseInputs(inp,shape_vector,dtype);
  setInputs({{input,feed}});
  
  return 0;
}

//' @title Run Internal Session
//' 
//' @description Runs the Current Session
//' 
//' @param op_names Node to be set as output of graph
//' 
//' @return R List containing output tensor and dimensions
//' 
// [[Rcpp::export]]
List runInternalSession(std::vector<std::string> op_names) {
//  cout << "Running the Session.. " << endl;
  List output;
  vector<pair<string, TF_Operation*>>output_operations;
  for (string op_name : op_names){
    TF_Operation* op = TF_GraphOperationByName(graph, op_name.c_str());
    const char* type = TF_OperationOpType(op);
    if (strcmp(type,"NoOp")==0) {
      output_operations.emplace_back(op_name, setTargetNode(op_name, graph));
    } else {
      output_operations.emplace_back(op_name, setOutputNode(op_name, graph));
    }
  }

    setPointers();
  
    if (TF_GetCode(status)!=TF_OK) {
      cout << "Error in graph" << endl;
      cout << TF_Message(status) << endl;
      return -1;
    }
  
    runSession(session, status);
  
    if (TF_GetCode(status)!=TF_OK) {
      cout << "Error running session" << endl;
      cout << TF_Message(status) << endl;
      return -1;
    }
  
  int output_index = 0;
  for (auto op : output_operations) {
    const char* type = TF_OperationOpType(op.second);
    if (strcmp(type,"NoOp")==0) {
      output[op.first] = "No Output";
    } else {
      TF_DataType dtype = TF_OperationOutputType({op.second,0});
      output[op.first] = fetchOutput(dtype, output_index);
      output_index+=1;
    }
  }
  
  resetInputValues();
  resetOutputValues();
  return output;
}

//' @title Reset Graph
//' 
//' @description Resets the graph by clearing all nodes created
//' 
//' @return Integer status
//' 
// [[Rcpp::export]]
int resetGraph() {
  op_list.clear();
  return 0;
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
  resetTargets();
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
//' @param shape Shape of Tensor
//' @param dtype Datatype of input
//' @param unique_name Unique name for the node
//' 
//' @return Unique node name
//' 
// [[Rcpp::export]]
std::string getPlaceholder(std::vector<int64_t> shape, std::string dtype, std::string unique_name) {
  pair<string, TF_Operation*> op;
  op = Placeholder("Placeholder", unique_name, shape, getDataType(dtype), graph, status);
  op_list.emplace(op.first, op.second);
  return op.first;
}

//' @title Source Op
//' 
//' @description Adds a source operation to the graph
//' 
//' @param val Tensor to be initialized as Constant
//' @param dim Vector indicating dimensions of val
//' @param dtype Datatype of input
//' @param op_name Type of operation for node
//' @param unique_name Unique name for the node
//' 
//' @return Unique node name
//' 
// [[Rcpp::export]]
std::string getSourceOp(List val, std::vector<int64_t> dim, std::string dtype, std::string op_name, std::string unique_name) {
  TF_Tensor* val_t = parseInputs(val, dim, getDataType(dtype));
  pair<string, TF_Operation*> op;
  op = SourceOp(op_name, unique_name, val_t, graph, status);
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

// [[Rcpp::export]]
void getOpDetails(std::string op_name) {
  TF_Operation* op = op_list.at(op_name);
  cout<<"Num inputs : "<<TF_OperationNumInputs(op)<<endl;
  cout<<"Type inputs : "<<TF_OperationInputType({op,0})<<endl;
  cout<<"Num outputs : "<<TF_OperationNumOutputs(op)<<endl;
}
