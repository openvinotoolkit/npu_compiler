# How to build Driver Compiler on Linux

## Dependencies

Before you start to build Driver Compiler targets, please check the necessary components.
- Hardware
    - Minimum requirements: 32GB RAM
- Software
    - [CMake](https://cmake.org/download/) 3.22.1 for Ubuntu 22.04 (version 3.13 or higher)
    - GCC 11.4.0 for Ubuntu 22.04 (version 7.5 or higher)
    - Python 3.9 - 3.12
    - Git for Linux (requires installing `git lfs`)
    - Ninja (optional, used for the documentation related to the installation part)

> Notice: RAM is not mandatory either. If your RAM is less than 32GB, you can compensate by reducing the number of threads during the build or by increasing the swap memory.

## Using Cmake Options

Driver Compiler is built with OpenVINO static runtime. To build the library and related tests (npu_driver_compiler, npu_elf, compilerTest, profilingTest, loaderTest) using following commands :

1. Clone repos:

    Clone [OpenVINO Project] repo and [NPU-Plugin Project] repo to special location. **Or** just unpack OPENVINO and NPU-Plugin source code to special location.

    <details>
    <summary>Instructions</summary>

    ```sh
    # set the proxy, if required.
    # export  http_proxy=xxxx
    # export  https_proxy=xxxx

    cd /home/useraccount/workspace (Just an example, you should use your own path.)
    git clone https://github.com/openvinotoolkit/openvino.git 
    cd openvino
    git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
    git submodule update --init --recursiv


    cd /home/useraccount/workspace (Just an example, you should use your own path.)
    git clone https://github.com/openvinotoolkit/npu_compiler.git
    cd npu_compiler
    git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
    git submodule update --init --recursive

    export OPENVINO_HOME=/home/useraccount/workspace/openvino (need change to your own path)
    export NPU_PLUGIN_HOME=/home/useraccount/workspace/npu_compiler (need change to your own path)
    ```
    </details>

    > Notice: If you are building the Driver Compiler targets with the goal of composing the Linux driver, it is important to pay attention to the version or commit of the [OpenVino Project] being used. Make sure to check the current supported version of the [OpenVino Project] from `OpenVINO built from source` entry in the table under the `Common` section in the [release notes](https://github.com/intel/linux-npu-driver/releases/) or within the [Linux NPU driver code](https://github.com/intel/linux-npu-driver/blob/main/compiler/compiler_source.cmake#L20).

2. Create build folder and run build instructions:
    
    2.1 Build instructions:

    Before build with the following instructions, please make sure `OPENVINO_HOME` and `NPU_PLUGIN_HOME` enviroment variables have been set.

    <details>
    <summary>Instructions</summary>

    ```sh
    cd $OPENVINO_HOME
    mkdir build-x86_64
    cd build-x86_64

    cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=OFF \
    -D OPENVINO_EXTRA_MODULES=$NPU_PLUGIN_HOME \
    -D ENABLE_LTO=OFF \
    -D ENABLE_FASTER_BUILD=OFF \
    -D ENABLE_CPPLINT=OFF \
    -D ENABLE_TESTS=OFF \
    -D ENABLE_FUNCTIONAL_TESTS=OFF \
    -D ENABLE_SAMPLES=OFF \
    -D ENABLE_JS=OFF \
    -D ENABLE_PYTHON=OFF \
    -D ENABLE_PYTHON_PACKAGING=OFF \
    -D ENABLE_WHEEL=OFF \
    -D ENABLE_OV_ONNX_FRONTEND=OFF \
    -D ENABLE_OV_PYTORCH_FRONTEND=OFF \
    -D ENABLE_OV_PADDLE_FRONTEND=OFF \
    -D ENABLE_OV_TF_FRONTEND=OFF \
    -D ENABLE_OV_TF_LITE_FRONTEND=OFF \
    -D ENABLE_OV_JAX_FRONTEND=OFF \
    -D ENABLE_OV_IR_FRONTEND=ON \
    -D THREADING=TBB \
    -D ENABLE_TBBBIND_2_5=OFF \
    -D ENABLE_SYSTEM_TBB=OFF \
    -D ENABLE_TBB_RELEASE_ONLY=OFF \
    -D ENABLE_HETERO=OFF \
    -D ENABLE_MULTI=OFF \
    -D ENABLE_AUTO=OFF \
    -D ENABLE_AUTO_BATCH=OFF \
    -D ENABLE_TEMPLATE=OFF \
    -D ENABLE_PROXY=OFF \
    -D ENABLE_INTEL_CPU=OFF \
    -D ENABLE_INTEL_GPU=OFF \
    -D ENABLE_NPU_PLUGIN_ENGINE=OFF \
    -D ENABLE_ZEROAPI_BACKEND=OFF \
    -D ENABLE_DRIVER_COMPILER_ADAPTER=OFF \
    -D ENABLE_INTEL_NPU_INTERNAL=OFF \
    -D ENABLE_INTEL_NPU_PROTOPIPE=OFF \
    -D BUILD_COMPILER_FOR_DRIVER=ON \
    -D ENABLE_PRIVATE_TESTS=OFF \
    -D ENABLE_NPU_LSP_SERVER=OFF \
    ..

    cmake --build . --config Release --target npu_driver_compiler compilerTest profilingTest vpuxCompilerL0Test loaderTest -j8
    ```
    </details>

    > Notice: If you encounter the following error during building `c++: internal compiler error: Killed (program cc1plus)`, you should consider decreasing the number of threads during compiling or try increasing the swap file size. For instance, to decrease the thread count, you could consider using `-j4` to decrease the thread number to 4 or a smaller value. 

    2.2 Build instructions notes:

    Many build options are listed here. To clarify these options, the following explains the list of CMake parameters.

    <details>
    <summary>2.2.1 Common build option </summary>

    ```sh
        # Build type
        CMAKE_BUILD_TYPE

        # Build library type
        BUILD_SHARED_LIBS
    ```

    </details>


    <details>
    <summary>2.2.2 Build option list in OpenVino Project</summary>

    For more details on the build options, please refer to [features.cmake](https://github.com/openvinotoolkit/openvino/blob/master/cmake/features.cmake) and intel NPU's [features.cmake](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/cmake/features.cmake) in [OpenVINO Project], which provide explanations for all the available build options.

    ```sh
        # Specify external repo
        OPENVINO_EXTRA_MODULES

        # Build optimization option
        ENABLE_LTO
        ENABLE_FASTER_BUILD

        # Cpplint checks during build time
        ENABLE_CPPLINT

        # Tests and samples
        ENABLE_TESTS
        ENABLE_FUNCTIONAL_TESTS
        ENABLE_SAMPLES

        # Enable JS API
        ENABLE_JS

        # Enable Python API and generate python binary
        ENABLE_PYTHON
        ENABLE_PYTHON_PACKAGING
        ENABLE_WHEEL

        # Frontend
        ENABLE_OV_ONNX_FRONTEND
        ENABLE_OV_PYTORCH_FRONTEND
        ENABLE_OV_PADDLE_FRONTEND
        ENABLE_OV_TF_FRONTEND
        ENABLE_OV_TF_LITE_FRONTEND
        ENABLE_OV_JAX_FRONTEND
        ENABLE_OV_IR_FRONTEND

        # TBB related option
        THREADING
        ENABLE_TBBBIND_2_5
        ENABLE_SYSTEM_TBB
        ENABLE_TBB_RELEASE_ONLY

        # Plugin platform
        ENABLE_HETERO
        ENABLE_MULTI
        ENABLE_AUTO
        ENABLE_AUTO_BATCH
        ENABLE_PROXY
        ENABLE_INTEL_CPU
        ENABLE_INTEL_GPU

        # NPU plugin and its tools related option
        ENABLE_NPU_PLUGIN_ENGINE
        ENABLE_ZEROAPI_BACKEND
        ENABLE_DRIVER_COMPILER_ADAPTER
        ENABLE_INTEL_NPU_INTERNAL
        ENABLE_INTEL_NPU_PROTOPIPE
    ```
    </details>

    <details>
    <summary>2.2.3 Build option list in NPU-Plugin Project</summary>

    For more details on the build options, please refer to this [features.cmake](https://github.com/openvinotoolkit/npu_compiler/blob/develop/cmake/features.cmake) file in [NPU-Plugin Project], which provides explanations for all the available build options.

    ```sh
        # Build Driver Compiler targets
        BUILD_COMPILER_FOR_DRIVER

        # Compiler private tests
        ENABLE_PRIVATE_TESTS

        # Debug tools
        ENABLE_NPU_LSP_SERVER        
    ```
    </details>

    2.3 (Optional) Instruction notes about TBB:

    <details>
    <summary>2.3.1 Default tbb location</summary>

    The build instructions uses the `-DENABLE_SYSTEM_TBB=OFF` option, which means that the TBB library downloaded by [OpenVINO Project] will be used. The download path for this TBB library is `$OPENVINO_HOME/temp/tbb`. Within the downloaded TBB folder, `$OPENVINO_HOME/temp/tbb/lib/libtbb.so.12` and `$OPENVINO_HOME/temp/tbb/lib/libtbbmalloc.so.2` are required for the Release version. 

    </details>

    <details>
    <summary>2.3.2 Use different TBB version</summary>

    If you wish to build with system TBB, you need install TBB in your local system first and then use `-DENABLE_SYSTEM_TBB=ON` option to instead of `-DENABLE_SYSTEM_TBB=OFF` option.

    If you wish to build with a specific version of TBB, you can download it from [oneTBB Project] and unzip its release package. Then use the `-DENABLE_SYSTEM_TBB=OFF -DTBBROOT=/home/username/path/to/downloaded/tbb` option to build.

    </details>

    <details>
    <summary>2.3.3 Do not use TBB</summary>

    If you wish to build without TBB (which will result in a slower build process), you need change `-D THREADING=TBB` to `-D THREADING=SEQ`. More info about SEQ mode, please refer to this [file](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md#options-affecting-binary-size).

    </details>

3. (Optional) Prepare final Driver Compiler package for driver:

    <details>
    <summary>Instructions</summary>

    All Driver Compiler related targets have now been generated in `$OPENVINO_HOME/bin/intel/Release` folder, where the binary libnpu_driver_compiler.so can be found. The following instructions are provided to pack Driver Compiler related targets to the specified location.

    ```sh
        #install Driver compiler related targets to current path. A `cid` folder will be generated to `$OPENVINO_HOME/build-x86_64/`.
        cd $OPENVINO_HOME/build-x86_64
        cmake --install . --prefix $PWD/ --component CiD

        # or to get a related compressed file. A RELEASE-CiD.tar.gz compressed file will be generated to `$OPENVINO_HOME/build-x86_64/`.
        cpack -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME="RELEASE" -G "TGZ"
    ```
    </details>

    > Notice: It is not recommended to use `cmake --install . --prefix /usr --component CiD` to install the Driver Compiler targets on the system, as this will not only  install `libnpu_driver_compiler.so` but also many other related targets (such as `elf`, `compilerTest`) to the specified folder.

    
### See also

Follow the blow guide to build the Driver Compiler library and test targets with Ninja:
 * `Using ninja` section of [how-to-build.md](../../../guides/how-to-build.md) of [NPU-Plugin Project].

To use cmake presets to build, please see
* [how to build Driver Compiler with Cmake Presets on Linux](./how_to_build_driver_compiler_withCmakePresets_on_linux.md)

Driver compiler build is a static build, to get a static build of [NPU-Plugin Project] repo, please see
 * [how to build static](../../../guides/how-to-build-static.md).

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
[oneTBB Project]: https://github.com/oneapi-src/oneTBB
