This project is a kind of boilerplate to be used to get acquainted
with a process of TensorFlow Serving client development
using C++ as a programming language and CMake as a build system.

PREREQUISITES
-

To use a TensorFlow Serving client provided in this project
you should install next dependencies:

* Protocol Buffers (https://github.com/google/protobuf/tree/master/src):
  * `sudo apt-get install autoconf automake libtool curl make g++ unzip`
  * `git clone https://github.com/google/protobuf`
  * `cd protobuf`
  * `./autogen.sh`
  * `./configure && make && make check && sudo make install && sudo ldconfig`
  
* gRPC (https://github.com/grpc/grpc/tree/master/src/cpp):
  * `sudo apt-get install build-essential autoconf libtool`
  * `git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc`
  * `cd grpc`
  * `git submodule update --init`
  * `make`
  * `sudo make install`

COMPILATION AND USAGE
-

Before the compilation make sure that you have provided some host
with a running TensorFlow Serving model in the `main.cpp` file.

The compilation is quite similar to ones which use CMake system.
But make sure to provide a path to the installed Protobuf Compiler
using `Protobuf_PROTOC_EXECUTABLE` for it.
For example you can use a bunch of commands (from the root of the repo):

`mkdir build && cd build && cmake -D Protobuf_PROTOC_EXECUTABLE=/usr/local/bin/protoc .. && make`

to be sent to the server:

`./tf-serving-client-cpp`

to add vocabulary look up (data processing module) in server:
https://www.tensorflow.org/tfx/serving/custom_servable -- Create a new kind of servable
https://github.com/tensorflow/serving/issues/582 -- GitHub issues

