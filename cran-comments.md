## Test environments
* local OS X install, R 3.4.0
* Debian 8.0 (Jessie)

## R CMD check results
There were no ERRORs or WARNINGs or NOTEs. 

## Upstream dependencies
As this package provides R language bindings for an external library, TensorFlow, this package depends on a non-R shared object, libtensorflow. 

libtensorflow.so can be obtained from [here](https://www.tensorflow.org/install/install_c)

More information on how to obtain libtensorflow.so is available in the ReadMe file