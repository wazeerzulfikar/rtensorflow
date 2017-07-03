#include <Rcpp.h>
using namespace Rcpp;
#include <stdio.h>
#include <iostream>
using namespace std;
#include <tensorflow/c/c_api.h>
#include <stdlib.h>   
#include <vector>  
#include <string>

void deallocator(void* data, size_t length);

template<typename T> void tensor_deallocator(void* data, size_t length,void* arg);

TF_Buffer* read_file(std::string path);

void deleteInputValues();

void resetOutputValues();

void setInputs(std::vector<std::pair<TF_Operation*,TF_Tensor*>> inputs);

void setOutputs(std::vector<TF_Operation*> outputs);

template<typename T> TF_Tensor* getTensor(T* arr,std::vector<int64_t> dimensions);

TF_Tensor* parseInputs(NumericVector inp, std::vector<int64_t> dimensions, string dtype="int32");

TF_Tensor* ones(std::vector<int64_t> dimensions);

void setPointers();

void runSession(TF_Session* session, TF_Status* status);

template<typename T>T getOutputs();

int getIntOutputs();

double getDoubleOutputs();

// Operation Helpers

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* status, const char* name="input",string dtype="int32");

TF_Operation* Constant(TF_Tensor* tensor, TF_Graph* graph, TF_Status* status, const char* name="const");

TF_Operation* Add(TF_Operation* l,TF_Operation* r, TF_Graph* graph, TF_Status* status, const char* name="add");

TF_Operation* MatMul(TF_Operation* l, TF_Operation* r, TF_Graph* graph, TF_Status* status, const char* name="matmul");