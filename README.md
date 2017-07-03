# <img alt="TensorFlow" src="https://www.tensorflow.org/images/tf_logo_transp.png" width="170" padding-right="10"/> R Binding

RTensorFlow provides idiomatic R language
bindings for [TensorFlow](http://tensorflow.org).

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow GitHub page](https://github.com/tensorflow/tensorflow)

## Getting Started

Since this package depends on the TensorFlow C API, it needs to be installed from Tensorflow website or compiled from source first. 

### Manual Tensorflow C API Installation

The Tensorflow C API `libtensorflow.so` can directly be installed onto the host machine by following the instructions in this link

* [Install Tensorflow for C](https://www.tensorflow.org/install/install_c)

Make sure you install the library in the default directory (`/usr/local`)

### Manual TensorFlow Compilation

See [TensorFlow from source](https://www.tensorflow.org/install/install_sources) first.
The Python/pip steps are not necessary, but building `tensorflow:libtensorflow.so` is.

In short:

1. Install [SWIG](http://www.swig.org) and [NumPy](http://www.numpy.org).  The
   version from your distro's package manager should be fine for these two.
2. [Install Bazel](http://bazel.io/docs/install.html), which you may need to do
   from source.
3. `git clone https://github.com/tensorflow/tensorflow`
4. `cd tensorflow`
5. `./configure`
6. `bazel build --compilation_mode=opt --copt=-march=native --jobs=1 tensorflow:libtensorflow.so`

   Using `--jobs=1` is recommended unless you have a lot of RAM, because
   TensorFlow's build is very memory intensive.

Copy `$TENSORFLOW_SRC/bazel-bin/tensorflow/libtensorflow.so` to `/usr/local/lib`.
If this is not possible, add `$TENSORFLOW_SRC/bazel-bin/tensorflow` to
`LD_LIBRARY_PATH`.

You may need to run `ldconfig` to reset `ld`'s cache after copying `libtensorflow.so`.

**OSX Note**: If you are running on OSX, there is a
[Homebrew PR](https://github.com/Homebrew/homebrew-core/pull/10273) in process which, once merged,
will make it easy to install `libtensorflow` wihout hassle. In the meantime, you can take a look at
[snipsco/tensorflow-build](https://github.com/snipsco/tensorflow-build) which provides a homebrew
tap that does essentially the same.

## Usage

Upon installing `libtensorflow.so` in the default directory,

1. `install.packages("devtools")`
2. `library(devtools)`
3.  `install_github(wazeerzulfikar/rtensorflow`
4. `./configure`
5. For examples on how to use the R API, look at `R/feed_forward_network.R`

## Contributing
Developers and users are welcome to contribute!

## License
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/tensorflow/rust/blob/master/LICENSE).
