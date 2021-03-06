#ifndef RTENSORFLOW_SRC_UTILS_H_
#define RTENSORFLOW_SRC_UTILS_H_

#include <Rcpp.h>
using namespace Rcpp;
#include <utility>
#include <string>
using namespace std;
#include <stddef.h>
#include <tensorflow/c/c_api.h>

void deallocator(void* data, size_t length);

template<typename T> void tensor_deallocator (void* data, size_t length, void* arg);

TF_Buffer* read_file (const char* path);

void resetInputValues();

void resetOutputValues ();

void resetTargets();

void setInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs);

void setOutputs(std::vector<TF_Operation*> outputs);

void setTargets(std::vector<TF_Operation*> targets);

TF_DataType getDataType (string dtype);

template<typename T> TF_Tensor* getTensor(List inp, int64_t* shape, int ndims, TF_DataType dtype);

TF_Tensor* parseInputs(List inp, int64_t* shape, int ndims, TF_DataType dtype);

TF_Operation* setOutputNode(std::string op_name, TF_Graph* graph);

TF_Operation* setTargetNode(std::string op_name, TF_Graph* graph); 

void setPointers();

void runSession(TF_Session* session, TF_Status* status);

std::vector<int64_t> getOutputDimensions(int output_index);

List fetchOutput(TF_DataType dtype, int output_index);

template <typename T> std::vector<T>  getOutputs(int output_index);
  
// Operation Helpers

std::pair<string, TF_Operation*> Placeholder(string op_name, string unique_name, vector<int64_t> shape, TF_DataType dtype, TF_Graph* graph, TF_Status* status);

std::pair<string, TF_Operation*> SourceOp(string op_name, string unique_name, TF_Tensor* tensor, TF_Graph* graph, TF_Status* status);

std::pair<string, TF_Operation*> Unary_Op(string op_name, string unique_name, TF_Operation* inp, TF_Graph* graph, TF_Status* status);

std::pair<string, TF_Operation*> Binary_Op(string op_name, string unique_name, TF_Operation* l,TF_Operation* r, TF_Graph* graph, TF_Status* status);

#endif  // RTENSORFLOW_SRC_UTILS_H_