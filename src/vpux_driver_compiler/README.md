# What is Driver Compiler

This guide introduces Driver Compiler (also known as Compiler in Driver or CiD) for Intel® Neural Processing Unit (NPU) devices. Driver Compiler is a set of C++ libraries providing a common API that allows the User Mode Driver to access compiler functions through vcl* interface methods. The action here is essentially compiling the IR format to the blob format.

To learn more about Driver Compiler, please see [intel_npu/README.md](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) in [OpenVINO Project].


## Components

The main components for Driver Compiler are:
* [CHANGES.txt](CHANGES.txt) contains the Driver Compiler history of changes.
* [docs](./docs/) - documents that describe building and testing the Driver Compiler.
* [loader](./src/loader/) - contains cmakefile to build and pack the elf from thirdparty used for some testing purposes.
* [vpux_compiler_l0](./src/vpux_compiler_l0/) - contains source files of Driver Compiler.
* [test](./test/) - contains test tools.

You can refer to the [API documentation](./docs/api_reference.md) for the usage workflow and detailed API descriptions.


## Requirements to build related targets locally

It is recommended to read the requirements section beforehand to ensure that you meet the prerequisites for local building.
- [Linux](./docs/requirements.md#linux-requirements)
- [Windows](./docs/requirements.md#windows-requirements)

Here are the steps to clone repositories and **configure environment variables** before building.
- [Pre-build preparation](./docs/prebuild.md#pre-build-preparation)


## How to build related targets locally

Driver Compiler provides `npu_driver_compiler`, `compilerTest`, `profilingTest` and `loaderTest` to compile network and test. To build Driver Compiler-related targets locally, please refer to

- (Recommended) Build using CMake Presets, requiring CMake version 3.19 or higher.
    - [Linux](./docs/build/build_with_cmake_presets_linux.md)
    - [Windows](./docs/build/build_with_cmake_presets_windows.md)
    - [Android](./docs/build/build_with_cmake_presets_android.md)

- Build with cmake options
    - [Linux](./docs/build/build_linux.md)
    - [Windows](./docs/build/build_windows.md)

- (Advanced) Build with LLVM Cache
    - [Linux](./docs/build/build_with_llvm_cache_linux.md)
    - [Windows](./docs/build/build_with_llvm_cache_windows.md)

## How to enable sideloading

Please refer to [how to sideload the Driver Compiler](./docs/test_and_debug/enable_sideloading.md).


## How to test

Please refer to [how to test](./docs/test_and_debug/test.md).

Please refer to [how to test with legacy methods](./docs/test_and_debug/legacy_test.md).


## How to debug

Please refer to [how to debug](./docs/test_and_debug/debug.md).

Please refer to [how to debug with legacy methods](./docs/test_and_debug/legacy_debug.md).


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
