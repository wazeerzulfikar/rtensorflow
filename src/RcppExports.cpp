// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// initializeSessionVariables
int initializeSessionVariables();
RcppExport SEXP _rtensorflow_initializeSessionVariables() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(initializeSessionVariables());
    return rcpp_result_gen;
END_RCPP
}
// loadGraphFromFile
int loadGraphFromFile(std::string path);
RcppExport SEXP _rtensorflow_loadGraphFromFile(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    rcpp_result_gen = Rcpp::wrap(loadGraphFromFile(path));
    return rcpp_result_gen;
END_RCPP
}
// setFeedInput
int setFeedInput(std::string op_name, List inp);
RcppExport SEXP _rtensorflow_setFeedInput(SEXP op_nameSEXP, SEXP inpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type op_name(op_nameSEXP);
    Rcpp::traits::input_parameter< List >::type inp(inpSEXP);
    rcpp_result_gen = Rcpp::wrap(setFeedInput(op_name, inp));
    return rcpp_result_gen;
END_RCPP
}
// runInternalSession
List runInternalSession(std::vector<std::string> op_names);
RcppExport SEXP _rtensorflow_runInternalSession(SEXP op_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<std::string> >::type op_names(op_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(runInternalSession(op_names));
    return rcpp_result_gen;
END_RCPP
}
// deleteSessionVariables
int deleteSessionVariables();
RcppExport SEXP _rtensorflow_deleteSessionVariables() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(deleteSessionVariables());
    return rcpp_result_gen;
END_RCPP
}
// getPlaceholder
std::string getPlaceholder(std::vector<int64_t> shape, std::string dtype, std::string unique_name);
RcppExport SEXP _rtensorflow_getPlaceholder(SEXP shapeSEXP, SEXP dtypeSEXP, SEXP unique_nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<int64_t> >::type shape(shapeSEXP);
    Rcpp::traits::input_parameter< std::string >::type dtype(dtypeSEXP);
    Rcpp::traits::input_parameter< std::string >::type unique_name(unique_nameSEXP);
    rcpp_result_gen = Rcpp::wrap(getPlaceholder(shape, dtype, unique_name));
    return rcpp_result_gen;
END_RCPP
}
// getConstant
std::string getConstant(List val, std::vector<int64_t> dim, std::string dtype, std::string unique_name);
RcppExport SEXP _rtensorflow_getConstant(SEXP valSEXP, SEXP dimSEXP, SEXP dtypeSEXP, SEXP unique_nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type val(valSEXP);
    Rcpp::traits::input_parameter< std::vector<int64_t> >::type dim(dimSEXP);
    Rcpp::traits::input_parameter< std::string >::type dtype(dtypeSEXP);
    Rcpp::traits::input_parameter< std::string >::type unique_name(unique_nameSEXP);
    rcpp_result_gen = Rcpp::wrap(getConstant(val, dim, dtype, unique_name));
    return rcpp_result_gen;
END_RCPP
}
// getUnaryOp
std::string getUnaryOp(std::string inp, std::string op_name, std::string unique_name);
RcppExport SEXP _rtensorflow_getUnaryOp(SEXP inpSEXP, SEXP op_nameSEXP, SEXP unique_nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type inp(inpSEXP);
    Rcpp::traits::input_parameter< std::string >::type op_name(op_nameSEXP);
    Rcpp::traits::input_parameter< std::string >::type unique_name(unique_nameSEXP);
    rcpp_result_gen = Rcpp::wrap(getUnaryOp(inp, op_name, unique_name));
    return rcpp_result_gen;
END_RCPP
}
// getBinaryOp
std::string getBinaryOp(std::string l_op, std::string r_op, std::string op_name, std::string unique_name);
RcppExport SEXP _rtensorflow_getBinaryOp(SEXP l_opSEXP, SEXP r_opSEXP, SEXP op_nameSEXP, SEXP unique_nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type l_op(l_opSEXP);
    Rcpp::traits::input_parameter< std::string >::type r_op(r_opSEXP);
    Rcpp::traits::input_parameter< std::string >::type op_name(op_nameSEXP);
    Rcpp::traits::input_parameter< std::string >::type unique_name(unique_nameSEXP);
    rcpp_result_gen = Rcpp::wrap(getBinaryOp(l_op, r_op, op_name, unique_name));
    return rcpp_result_gen;
END_RCPP
}
// loadSavedModel
void loadSavedModel(std::string path, CharacterVector tags);
RcppExport SEXP _rtensorflow_loadSavedModel(SEXP pathSEXP, SEXP tagsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type tags(tagsSEXP);
    loadSavedModel(path, tags);
    return R_NilValue;
END_RCPP
}
// printNodeList
void printNodeList();
RcppExport SEXP _rtensorflow_printNodeList() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    printNodeList();
    return R_NilValue;
END_RCPP
}
// locateError
void locateError();
RcppExport SEXP _rtensorflow_locateError() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    locateError();
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rtensorflow_initializeSessionVariables", (DL_FUNC) &_rtensorflow_initializeSessionVariables, 0},
    {"_rtensorflow_loadGraphFromFile", (DL_FUNC) &_rtensorflow_loadGraphFromFile, 1},
    {"_rtensorflow_setFeedInput", (DL_FUNC) &_rtensorflow_setFeedInput, 2},
    {"_rtensorflow_runInternalSession", (DL_FUNC) &_rtensorflow_runInternalSession, 1},
    {"_rtensorflow_deleteSessionVariables", (DL_FUNC) &_rtensorflow_deleteSessionVariables, 0},
    {"_rtensorflow_getPlaceholder", (DL_FUNC) &_rtensorflow_getPlaceholder, 3},
    {"_rtensorflow_getConstant", (DL_FUNC) &_rtensorflow_getConstant, 4},
    {"_rtensorflow_getUnaryOp", (DL_FUNC) &_rtensorflow_getUnaryOp, 3},
    {"_rtensorflow_getBinaryOp", (DL_FUNC) &_rtensorflow_getBinaryOp, 4},
    {"_rtensorflow_loadSavedModel", (DL_FUNC) &_rtensorflow_loadSavedModel, 2},
    {"_rtensorflow_printNodeList", (DL_FUNC) &_rtensorflow_printNodeList, 0},
    {"_rtensorflow_locateError", (DL_FUNC) &_rtensorflow_locateError, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_rtensorflow(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
