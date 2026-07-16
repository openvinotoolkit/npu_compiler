# Build Driver Compiler with LLVM Cache Using CMake Options on Linux
## Prerequisites

It is highly recommended to read either [build with CMake Options](./build_linux.md) or [build with CMake Presets](./build_with_cmake_presets_linux.md) before proceeding to the next section.
> **Before you start:**  
> - Review [Requirements](../requirements.md#linux-requirements).
> - Follow [Repository Setup](../prebuild.md#pre-build-preparation).
> - Check [FAQ](../FAQ.md) for [common clone issues](../FAQ.md#common-clone-failure-issues) and [common build issues](../FAQ.md#common-build-and-install-issues).
> - See [TBB selection](../FAQ.md#choosing-the-right-tbb-linux--windows) for threading library options.


## Environment Setup

Use these environment variables for both quick and detailed builds:

<details>
<summary>Commands</summary>

```sh
export OPENVINO_HOME=/path/to/cloned/openvino
export NPU_PLUGIN_HOME=/path/to/cloned/applications.ai.vpu-accelerators.vpux-plugin
export CONFIG=Release                    # Or Debug/RelWithDebInfo
export TBBROOT=/tbb/path/you/want/to/use # Optional, only if using custom TBB
export LLVMPKG_CACHE_HOME=/path/to/saved/llvm_cache_${CONFIG}
export LLVM_LIB_PATH=${OPENVINO_HOME}/build_llvm_${CONFIG}/build-modules/vpux_plugin/thirdparty/llvm-project/llvm
```
</details>


## Recommended CMake Options

Use these options for both quick and detailed builds:

<details>
<summary>Commands</summary>

```sh
export CommonBuildOptions="-D ENABLE_LTO=OFF \
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
                          -D ENABLE_DIRECTML=OFF \
                          -D ENABLE_NPU_LSP_SERVER=OFF"
```
</details>

## Quick Build Commands (For Immediate Use)

This section provides the complete build commands for users who need to build the Driver Compiler using LLVM cache without diving into the detailed configuration steps.

<details>
<summary>Commands</summary>

```sh
cd ${OPENVINO_HOME}
mkdir llvm_cache_${CONFIG}
mkdir ${OPENVINO_HOME}/build_llvm_${CONFIG}
cd ${OPENVINO_HOME}/build_llvm_${CONFIG}

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=${CONFIG} \
    -D OPENVINO_EXTRA_MODULES=${NPU_PLUGIN_HOME} \
    ..

ninja build-modules/vpux_plugin/thirdparty/llvm-project/llvm/all -j$(nproc)

cmake --install ${LLVM_LIB_PATH} --config ${CONFIG} --prefix ${LLVMPKG_CACHE_HOME}

mkdir -p ${OPENVINO_HOME}/build_${CONFIG}
cd ${OPENVINO_HOME}/build_${CONFIG}

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=${CONFIG} \
    -D BUILD_SHARED_LIBS=OFF \
    -D OPENVINO_EXTRA_MODULES=${NPU_PLUGIN_HOME} \
    -D ENABLE_PREBUILT_LLVM_MLIR_LIBS=ON \
    -D MLIR_BINARY_PKG_DIR=${LLVMPKG_CACHE_HOME} \
    ${CommonBuildOptions} \
    ..

ninja openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest -j$(nproc)

# Optional, compress and pack all CiD targets
cpack -V -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME="${CONFIG}" -G "TGZ"
```
</details>

LLVM cache binaries are built and found in `${LLVMPKG_CACHE_HOME}` directory, and it can be used for future Driver Compiler build.

The Driver Compiler package is built and can be found in `${OPENVINO_HOME}/bin/intel64/${CONFIG}` directory. **You can ignore the remaining sections** of this document.


## Detailed Step-by-step Guide (For a Deeper Understanding on the Build Process)

`LLVM` is a thirdparty dependency for NPU compilers, and `LLVM` targets are built along with Driver Compiler. Since a [specific commit](https://github.com/openvinotoolkit/npu_compiler/tree/develop/thirdparty) of LLVM source code is used and is not frequently updated, therefore, it can be prebuilt and stored for future use to reduce build time and resource usage.

### Build and Cache LLVM Binaries

To enable Driver Compiler build with cached `LLVM`, the cache itself should be built separately.

1. Create a build directory and run the following commands:
    
    <details>
    <summary>Commands</summary>

    ```sh
    mkdir ${OPENVINO_HOME}/build_llvm_${CONFIG}
    cd ${OPENVINO_HOME}/build_llvm_${CONFIG}

    cmake \
        -G Ninja \
        -D CMAKE_BUILD_TYPE=${CONFIG} \
        -D OPENVINO_EXTRA_MODULES=${NPU_PLUGIN_HOME} \
        ..

    ninja build-modules/vpux_plugin/thirdparty/llvm-project/llvm/all -j${nproc}
    ```
    </details>

2. Install LLVM binaries for future build

    <details>
    <summary>Commands</summary>

    ```sh
    mkdir ${LLVMPKG_CACHE_HOME}
    export LLVM_LIB_PATH=${OPENVINO_HOME}/build_llvm_${CONFIG}/build-modules/vpux_plugin/thirdparty/llvm-project/llvm
    cmake --install ${LLVM_LIB_PATH} --config ${CONFIG} --prefix ${LLVMPKG_CACHE_HOME}
    ```
    </details>

    The LLVM binaries will be installed in the `${LLVMPKG_CACHE_HOME}` directory.

### Build Driver Compiler Using LLVM Cache

Once the LLVM cache is built and installed into a directory, it can be used for any future CiD builds, whether it is a clean or incremental build. You should see a significant reduction in build time when the cache is enabled. The following section is a clean build command with LLVM cache.

>Note: If you want to use the LLVM cache in other builds, please carefully check the LLVM cache `build_manifest.txt` to ensure that all the parameters used are consistent.

1. Create a build directory and run the following commands:

    Set `ENABLE_PREBUILT_LLVM_MLIR_LIBS` to `ON` and `MLIR_BINARY_PKG_DIR` to `${LLVMPKG_CACHE_HOME}` to enable prebuilt LLVM cache.

    <details>
    <summary>Command</summary>

    ```sh
    mkdir -p ${OPENVINO_HOME}/build_${CONFIG}
    cd ${OPENVINO_HOME}/build_${CONFIG}

    cmake \
        -G Ninja \
        -D CMAKE_BUILD_TYPE=${CONFIG} \
        -D BUILD_SHARED_LIBS=OFF \
        -D OPENVINO_EXTRA_MODULES=${NPU_PLUGIN_HOME} \
        -D ENABLE_PREBUILT_LLVM_MLIR_LIBS=ON \
        -D MLIR_BINARY_PKG_DIR=${LLVMPKG_CACHE_HOME} \
        ${CommonBuildOptions} \
        ..

    ninja openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest -j${nproc}
    ```
    </details>

2. Create final Driver Compiler package for driver (Optional):

    All Driver Compiler-related targets will be generated in `${OPENVINO_HOME}/bin/intel64/${CONFIG}` directory.
    - **Compress and pack all CiD components:**
        <details>
        <summary>Compress and pack</summary>

        ```sh
        # Compress and pack all CiD components
        cd ${OPENVINO_HOME}/build_${CONFIG}
        cpack -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME="${CONFIG}" -G "TGZ"
        ```
        </details>
    - **Install all CiD components to a specific location:**
        <details>
        <summary>Install</summary>

        ```sh
        # Install all CiD components
        cd ${OPENVINO_HOME}/build_${CONFIG}
        cmake --install . --prefix $(pwd)/ --component CiD
        ```
        </details>
