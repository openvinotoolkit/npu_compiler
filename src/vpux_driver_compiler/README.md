# What is Driver Compiler

This guide introduces Driver Compiler for Intel® Neural Processing Unit (NPU) devices. Driver Compiler is a set of C++ libraries providing a common API that allows the User Mode Driver to access compiler functions through vcl* interface methods. The action here is essentially compiling the IR format to the blob format.

To learn more about Driver Compiler, please see [intel_npu/README.md](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) in [OpenVINO Project].


## Components

The main components for Driver Compiler are :
* [CHANGES.txt](CHANGES.txt) contains the Driver Compiler history of changes.
* [docs](./docs/) - documents that describe building and testing the Driver Compiler.
* [loader](./src/loader/) - contains cmakefile to build and pack the elf from thirdparty used for some testing purposes.
* [vpux_compiler_l0](./src/vpux_compiler_l0/) - contains source files of Driver Compiler.
* [test](./test/) - contains test tools.


## Basic workflow

The main entrypoint for Driver Compiler is `vclCompilerCreate`. The basic work flow is as follow:
```C
...
vclCompilerCreate
...
vclCompilerGetProperties
...
/* If you want to query the supported layers of a network, please call following three lines. */
...
vclQueryNetworkCreate
...
/* vclQueryNetwork should be called twice, first time to retrieve data size, second time to get data. */
vclQueryNetwork
...
vclQueryNetworkDestroy
...
/* Fill buffer/weights with data read from command line arguments. Will set result blob size. */
...
vclExecutableCreate
...
vclExecutableGetSeriablizableBlob
...
blobSize > 0
blob = (uint8_t*)malloc(blobSize)
vclExecutableGetSeriablizableBlob
...
/* If log handle is created with vclCompilerCreate, can call vclLogHandleGetString to get last error message.*/
...
vclLogHandleGetString
...
logSize > 0
log = (char*)malloc(logSize)
vclLogHandleGetString
...
vclExecutableDestroy
vclCompilerDestroy
...
```


## How to build related targets locally

Driver Compiler provides npu_driver_compiler, compilerTest, profilingTest and loaderTest to compile network and test. To build Driver Compiler related targets locally, refer to

- (Recommended) build using CMake Presets, requiring CMake version 3.19 or higher.
    - [linux](./docs/how_to_build_driver_compiler_withCmakePresets_on_linux.md)
    - [windows](./docs/how_to_build_driver_compiler_withCmakePresets_on_windows.md)

- build with cmake options
    - [linux](./docs/how_to_build_driver_compiler_on_linux.md)
    - [windows](./docs/how_to_build_driver_compiler_on_windows.md)



## How to release Driver Compiler package

Unlike local build of Driver Compiler related targets, we need to apply some patches to [OpenVINO Project] and [NPU-Plugin Project] and also pack the elf, pdb and tbb files together to meet driver requirements.

## How to test

Please refer to [How to test](./docs/how_to_test.md).


## How to debug

Please refer to [How to debug](./docs/how_to_debug.md).

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
