# <img alt="TensorFlow" src="https://www.tensorflow.org/images/tf_logo_transp.png" width="170" padding-right="10"/> R Binding

RTensorFlow provides idiomatic R language
bindings for [TensorFlow](http://tensorflow.org).

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow GitHub page](https://github.com/tensorflow/tensorflow)

## Getting Started

Since this package depends on the TensorFlow C API, it needs to be installed from Tensorflow website. 

### Manual Tensorflow C API Installation

The Tensorflow C API `libtensorflow.so` can directly be installed onto the host machine by following the instructions in this link

* [Install Tensorflow for C](https://www.tensorflow.org/install/install_c)

Make sure you install the library in the default directory (`/usr/local`)

**macOS Note :** There is a homebrew formula to install `libtensorflow` without any hassle.

## Usage

Upon installing `libtensorflow.so` in the default directory,

1. `install.packages("devtools")`
2. `library(devtools)`
3.  `install_github("wazeerzulfikar/rtensorflow")`
4. `library(rtensorflow)`
5. For examples on how to use the R API, look at `R/examples.R`

## Contributing
Developers and users are welcome to contribute!

## License
This project is licensed under the terms of the Apache 2.0 license.
