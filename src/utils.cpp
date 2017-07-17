#include "utils.h"
#include <Rcpp.h>
using namespace Rcpp;
#include <stdio.h>
#include <iostream>
using namespace std;
#include <tensorflow/c/c_api.h>
#include <stdlib.h>   
#include <vector>  
#include <string>

std::vector<TF_Operation*> targets;
std::vector<TF_Output> inputs_;
std::vector<TF_Tensor*> input_values_;
std::vector<TF_Output> outputs_;
std::vector<TF_Tensor*> output_values_;

TF_Operation* const* targets_ptr;
const TF_Output* inputs_ptr;
TF_Tensor* const* input_values_ptr;
const TF_Output* outputs_ptr;
TF_Tensor** output_values_ptr;


void deallocator(void* data, size_t length) {                                             
  free(data);                                                                       
}     

template<typename T> void tensor_deallocator(void* data, size_t length,void* arg) {                                             
  delete[] static_cast<T*>(data);
}

TF_Buffer* read_file(const char* path) {
  
  FILE *f = fopen(path, "rb");
  
  if (f==NULL) return nullptr;
  
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

void resetInputValues() {
  for (int i = 0; i < input_values_.size(); ++i) {
    TF_DeleteTensor(input_values_[i]);
  }
  input_values_.clear();
  inputs_.clear();
}

void resetOutputValues() {
  for (int i = 0; i < output_values_.size(); ++i) {
    if (output_values_[i] != nullptr){
      TF_DeleteTensor(output_values_[i]);
    }
  }
  output_values_.clear();
  outputs_.clear();
}

void setInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
  for (const auto& i : inputs) {
    inputs_.emplace_back(TF_Output{i.first, 0});
    input_values_.emplace_back(i.second);
  }
}

void setOutputs(std::vector<TF_Operation*> outputs) {
  for (TF_Operation* o : outputs){
    outputs_.emplace_back(TF_Output{o,0});
  }
  output_values_.resize(outputs_.size(), nullptr);
}

TF_DataType getDataType (string dtype) {
  if (dtype=="int32") return TF_INT32;
  else if (dtype=="double") return TF_DOUBLE;
  else if(dtype=="boolean") return TF_BOOL;
  return TF_FLOAT;
}

template <typename T> TF_Tensor* getTensor(NumericVector inp, std::vector<int64_t> dimensions, TF_DataType dtype) {
  int no_dims = dimensions.size();
  int64_t length=1;
  int64_t* dim = new int64_t[dimensions.size()];
  for (int i = 0; i < dimensions.size(); ++i) {
    length *= dimensions.at(i);
    dim[i] = dimensions.at(i);
  }
  
  T* c_inp = new T[inp.size()];
  for (int iter=0; iter < inp.size(); ++iter) {
    c_inp[iter] = inp[iter];
  }

  return TF_NewTensor(
    dtype, dim, no_dims, c_inp, sizeof(T)*length,
    &tensor_deallocator<T>,
    nullptr);
}

TF_Tensor* parseInputs(NumericVector inp, std::vector<int64_t> dimensions, TF_DataType dtype) {
  if (dtype==3) {
    return getTensor<int>(inp, dimensions, dtype);
  } else {
    return getTensor<double>(inp, dimensions,dtype);
  }
}

TF_Tensor* ones(std::vector<int64_t> dimensions) {
  //Function for returning a Tensor of required dimension, filled with 1's
  int64_t length=1;
  for (int i = 0; i < dimensions.size(); ++i) {
    length *= dimensions.at(i);
  }
  NumericVector arr;
    for (int i = 0; i < length; ++i) {
    arr[i] = 1;
  }
  return getTensor<int>(arr, dimensions, TF_INT32);
}

void setPointers() {
  targets_ptr = targets.empty() ? nullptr : &targets[0];
  
  inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
  input_values_ptr = input_values_.empty() ? nullptr : &input_values_[0];
  
  outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
  output_values_ptr = output_values_.empty() ? nullptr : &output_values_[0];
}

void runSession(TF_Session* session, TF_Status* status) {
  TF_SessionRun(session, nullptr,
                 inputs_ptr, input_values_ptr, inputs_.size(),  // Inputs
                 outputs_ptr, output_values_ptr, outputs_.size(),  // Outputs
                 targets_ptr, targets.size(),  // Operations
                 nullptr, status );
}

template <typename T> pair<T*, int64_t> getOutputs() {
  TF_Tensor* out = output_values_[0];
  if (out == nullptr) {
    return {nullptr,0};
  }
  
  int64_t length = 1;
  for (int i = 0; i < TF_NumDims(out); ++i) {
    length *= TF_Dim(out,i);
  }

  void* output_contents = TF_TensorData(out);
  return {(T*) output_contents,length};
}

std::vector<int64_t> getOutputDimensions() {
  TF_Tensor* out = output_values_[0];
  std::vector<int64_t> dim(TF_NumDims(out));
  
  if (out == nullptr) return {0};
  
  for (int i = 0; i < TF_NumDims(out); ++i) {
    dim [i]= TF_Dim(out,i);
  }
  
  return dim;
}

TF_Operation* setOutputNode(std::string op_name, TF_Graph* graph) {
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* output = TF_GraphOperationByName(graph, op_name_ptr);
  setOutputs({output});
  return output;
}

List fetchOutput(TF_DataType dtype) {
  NumericVector output_val;
  if (dtype == 1) {
    pair<float*, int64_t> out;
    out = getOutputs<float>();
    output_val = NumericVector(out.second);
    for (int i = 0; i < out.second; ++i) {
      output_val[i] = out.first[i];
    }
  } else if (dtype == 2) {
      pair<double*, int64_t> out;
      out = getOutputs<double>();
      output_val = NumericVector(out.second);
      for (int i = 0; i < out.second; ++i) {
        output_val[i] = out.first[i];
    } 
  } else if (dtype == 3) {
    pair<int*, int64_t> out;
    out = getOutputs<int>();
    output_val = NumericVector(out.second);
    for (int i = 0; i < out.second; ++i){
      output_val[i] = out.first[i];
    } 
  } else if (dtype == 10) {
      pair<bool*, int64_t> out;
      out = getOutputs<bool>();
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


// Operation Helpers

pair<string, TF_Operation*> Placeholder(string op_name, string unique_name, vector<int64_t> shape, TF_DataType dtype, TF_Graph* graph, TF_Status* status) {
  
  TF_OperationDescription* desc = TF_NewOperation(graph, op_name.c_str(), unique_name.c_str());
  TF_SetAttrType(desc, "dtype", dtype);
  int64_t* dim = new int64_t[shape.size()];
  
  for (int i = 0; i < shape.size(); ++i) {
    dim[i] = shape.at(i);
  }
  
  TF_SetAttrShape(desc, "shape", dim, shape.size());
  TF_Operation* op = TF_FinishOperation(desc, status);
  
  return {unique_name,op};
}

pair<string, TF_Operation*> Constant(string op_name, string unique_name, TF_Tensor* tensor, TF_Graph* graph, TF_Status* status) {
  TF_OperationDescription* desc = TF_NewOperation(graph, op_name.c_str(), unique_name.c_str());
  TF_SetAttrTensor(desc, "value", tensor, status);
  
  if(TF_GetCode(status)!=TF_OK) return {nullptr,nullptr};
  
  TF_SetAttrType(desc,"dtype",TF_TensorType(tensor));
  TF_Operation* op = TF_FinishOperation(desc, status);
  
  return {unique_name,op};
}

pair<string, TF_Operation*> Unary_Op(string op_name, string unique_name, TF_Operation* inp, TF_Graph* graph, TF_Status* status) {
  TF_OperationDescription* desc = TF_NewOperation(graph, op_name.c_str(), unique_name.c_str());
  TF_AddInput(desc, {inp,0});
  TF_Operation* op = TF_FinishOperation(desc, status);
  
  return {unique_name,op};
}

pair<string, TF_Operation*> Binary_Op(string op_name, string unique_name, TF_Operation* l,TF_Operation* r, TF_Graph* graph, TF_Status* status) {
  TF_OperationDescription* desc = TF_NewOperation(graph, op_name.c_str(), unique_name.c_str());
  TF_AddInput(desc, {l,0});
  TF_AddInput(desc, {r,0});
  TF_Operation* op = TF_FinishOperation(desc, status);
  
  return {unique_name,op};
}

