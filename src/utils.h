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

void setInputs (std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs);

void setOutputs(std::vector<TF_Operation*> outputs);

TF_DataType getDataType (string dtype);

template<typename T> TF_Tensor* getTensor(NumericVector inp, std::vector<int64_t> dimensions, TF_DataType dtype);

TF_Tensor* parseInputs(NumericVector inp, std::vector<int64_t> dimensions, TF_DataType dtype);

TF_Tensor* ones(std::vector<int64_t> dimensions);

void setPointers();

void runSession(TF_Session* session, TF_Status* status);

template <typename T> std::vector<T>  getOutputs();

std::vector<int64_t> getOutputDimensions();

TF_Operation* setOutputNode(std::string op_name, TF_Graph* graph);

List fetchOutput(TF_DataType dtype);
  
// Operation Helpers

std::pair<string, TF_Operation*> Placeholder(string op_name, string unique_name, vector<int64_t> shape, TF_DataType dtype, TF_Graph* graph, TF_Status* status);

std::pair<string, TF_Operation*> Constant(string op_name, string unique_name, TF_Tensor* tensor, TF_Graph* graph, TF_Status* status);

std::pair<string, TF_Operation*> Unary_Op(string op_name, string unique_name, TF_Operation* inp, TF_Graph* graph, TF_Status* status);

std::pair<string, TF_Operation*> Binary_Op(string op_name, string unique_name, TF_Operation* l,TF_Operation* r, TF_Graph* graph, TF_Status* status);

#endif  // RTENSORFLOW_SRC_UTILS_H_