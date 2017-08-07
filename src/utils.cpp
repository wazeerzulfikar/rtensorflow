#include "utils.h"
#include <Rcpp.h>
using namespace Rcpp;
#include <stdio.h>
#include <iostream>
#include <tensorflow/c/c_api.h>
#include <stdlib.h>   
#include <vector>  
#include <string>
#include <algorithm>

std::vector<TF_Operation*> targets_;
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
  for (auto item : input_values_) {
    TF_DeleteTensor(item);
  }
  input_values_.clear();
  inputs_.clear();
}

void resetOutputValues() {
  for (auto item : output_values_) {
    if(item != nullptr) {
      TF_DeleteTensor(item);
    }
  }
  output_values_.clear();
  outputs_.clear();
}

void resetTargets() {
  targets_.clear();
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

void setTargets(std::vector<TF_Operation*> targets) {
  for (TF_Operation* t : targets){
    targets_.emplace_back(t);
  }
}

TF_DataType getDataType (string dtype) {
  if (dtype=="int32") return TF_INT32;
  else if (dtype=="double") return TF_DOUBLE;
  else if(dtype=="boolean") return TF_BOOL;
  return TF_FLOAT;
}

template <typename T> TF_Tensor* getTensor(List inp, int64_t* shape, int ndims, TF_DataType dtype) {
  int unknown_dim = -1;
  int64_t length = 1;
  
  for (int i = 0; i < ndims; ++i) {
    if (shape[i]==-1) {
      unknown_dim = i;
    } else {
      length *= shape[i];
    }
  }
  
  if (unknown_dim > -1) {
    shape[unknown_dim] = static_cast<int>(inp.size()/length);
  }
  
  T* c_inp = new T[inp.size()];
  for (int64_t iter=0; iter < inp.size(); ++iter) {
    c_inp[iter] = inp[iter];
  }

  return TF_NewTensor(
    dtype, shape, ndims, c_inp, sizeof(T)*inp.size(),
    &tensor_deallocator<T>,
    nullptr);
}

TF_Tensor* parseInputs(List inp, int64_t* dimensions, int ndims, TF_DataType dtype) {
  if (dtype == TF_FLOAT) {
    return getTensor<float>(inp, dimensions, ndims, dtype);
  } else if (dtype == TF_DOUBLE) {
    return getTensor<double>(inp, dimensions, ndims, dtype);
  } else if (dtype == TF_INT32) {
    return getTensor<int>(inp, dimensions, ndims, dtype);
  } else if (dtype == TF_BOOL) {
    return getTensor<bool>(inp, dimensions, ndims, dtype);
  }
  return nullptr;
}

TF_Operation* setOutputNode(std::string op_name, TF_Graph* graph) {
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* output = TF_GraphOperationByName(graph, op_name_ptr);
  setOutputs({output});
  return output;
}

TF_Operation* setTargetNode(std::string op_name, TF_Graph* graph) {
  const char* op_name_ptr = op_name.c_str();
  TF_Operation* target = TF_GraphOperationByName(graph, op_name_ptr);
  setTargets({target});
  return target;
}

void setPointers() {
  targets_ptr = targets_.empty() ? nullptr : &targets_[0];
  
  inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
  input_values_ptr = input_values_.empty() ? nullptr : &input_values_[0];
  
  outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
  output_values_ptr = output_values_.empty() ? nullptr : &output_values_[0];
}

void runSession(TF_Session* session, TF_Status* status) {
  TF_SessionRun(session, nullptr,
                 inputs_ptr, input_values_ptr, inputs_.size(),  // Inputs
                 outputs_ptr, output_values_ptr, outputs_.size(),  // Outputs
                 targets_ptr, targets_.size(),  // Operations
                 nullptr, status );
}

std::vector<int64_t> getOutputDimensions(int output_index) {
  TF_Tensor* out = output_values_[output_index];
  std::vector<int64_t> dim(TF_NumDims(out));
  if (out == nullptr) return {0};

  for (int i = 0; i < TF_NumDims(out); ++i) {
    dim [i]= TF_Dim(out,i);
  }
  
  return dim;
}

List fetchOutput(TF_DataType dtype, int output_index) {
  List output;
  
  if (dtype == TF_FLOAT) {
    vector<float> output_val = getOutputs<float>(output_index);
    output["val"] = output_val;
  } else if (dtype == TF_DOUBLE) {
    vector<double> output_val = getOutputs<double>(output_index);
    output["val"] = output_val;
  } else if (dtype == TF_INT32) {
    vector<int> output_val = getOutputs<int>(output_index);
    output["val"] = output_val;
  } else if (dtype == TF_BOOL) {
    vector<bool> output_val = getOutputs<bool>(output_index);
    output["val"] = output_val;
  }
  
  output["dim"] = getOutputDimensions(output_index);
  return output;
}

template <typename T> std::vector<T> getOutputs(int output_index) {
  TF_Tensor* out = output_values_[output_index];
  
  vector<int64_t> dim = getOutputDimensions(output_index);
  int64_t length = 1;
  for (auto& d : dim) {
    length *= d;
  }
  
  void* output_contents = TF_TensorData(out);
  
  vector<T> output_vector = vector<T>(length);
  
  T* output_ptr = static_cast<T*>(output_contents);
  
  for (int i = 0; i < length; ++i) {
    output_vector[i] = output_ptr[i];
  }
  
  return output_vector;
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

pair<string, TF_Operation*> SourceOp(string op_name, string unique_name, TF_Tensor* tensor, TF_Graph* graph, TF_Status* status) {
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

