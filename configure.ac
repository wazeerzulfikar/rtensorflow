AC_PREREQ([2.69])
AC_INIT(DESCRIPTION)


# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.
AC_CHECK_LIB([tensorflow],[TF_Version],[],[AC_MSG_ERROR([Couldn't find or include Tensorflow])])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])
AC_CHECK_HEADER([tensorflow/c/c_api.h],[],[AC_MSG_ERROR([Couldn't find or include Tensorflow])])


# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_INT64_T
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC

RTENS_CPPFLAGS="-std=c++11"
RTENS_LIBS="-ltensorflow"

AC_SUBST(RTENS_CPPFLAGS)
AC_SUBST(RTENS_LIBS)

AC_OUTPUT(src/Makevars)
