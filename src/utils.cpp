#include "utils.h"

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

void deleteInputValues(){
  for(int i=0;i<input_values_.size();i++){
    TF_DeleteTensor(input_values_[i]);
  }
  input_values_.clear();
}

void resetOutputValues(){
  for(int i=0;i<output_values_.size();i++){
    if (output_values_[i] != nullptr){
      TF_DeleteTensor(output_values_[i]);
    }
  }
  output_values_.clear();
}

void setInputs(std::vector<std::pair<TF_Operation*,TF_Tensor*>> inputs){
  deleteInputValues();
  inputs_.clear();
  for (const auto& i : inputs) {
    inputs_.emplace_back(TF_Output{i.first, 0});
    input_values_.emplace_back(i.second);
  }
}

void setOutputs(std::vector<TF_Operation*> outputs){
  resetOutputValues();
  outputs_.clear();
  for(TF_Operation* o : outputs){
    outputs_.emplace_back(TF_Output{o,0});
  }
  output_values_.resize(outputs_.size(), nullptr);
}

void setPointers(){
  targets_ptr = targets.empty() ? nullptr : &targets[0];
  
  inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
  input_values_ptr = input_values_.empty() ? nullptr : &input_values_[0];
  
  outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
  output_values_ptr = output_values_.empty() ? nullptr : &output_values_[0];
}

void runSession(TF_Session* session, TF_Status* status){
  TF_SessionRun( session, nullptr,
                 inputs_ptr, input_values_ptr, inputs_.size(),  // Inputs
                 outputs_ptr, output_values_ptr, outputs_.size(),  // Outputs
                 targets_ptr, targets.size(),  // Operations
                 nullptr, status );
};

int getOutputs(){
  TF_Tensor* out = output_values_[0];
  void* output_contents = TF_TensorData(out);
  return *((int*) output_contents);
}