# Build Driver Compiler on Windows

This guide explains how to build the Driver Compiler (also known as Compiler in Driver or CiD) on Windows. The Driver Compiler is built as an extra module of a static OpenVINO package.
> **Before you start:**
> - Review [Requirements](../requirements.md#windows-requirements).
> - Follow [Repository Setup](../prebuild.md#pre-build-preparation).
> - Check [FAQ](../FAQ.md) for [common clone issues](../FAQ.md#common-clone-failure-issues) and [common build issues](../FAQ.md#common-build-and-install-issues).
> - See [TBB selection](../FAQ.md#choosing-the-right-tbb-linux--windows) for threading library options.


## Environment Setup

Use these environment variables for both quick and detailed builds in your **x64 Native Tools Command Prompt for VS 2022 (Administrator)**:

<details>
<summary>Commands</summary>

```bat
set OPENVINO_HOME=C:\path\to\cloned\openvino
set NPU_PLUGIN_HOME=C:\path\to\cloned\applications.ai.vpu-accelerators.vpux-plugin
@REM Or Debug/RelWithDebInfo
set CONFIG=Release
@REM Optional, only if using custom TBB, but required for enabling sideloading
set TBBROOT=C:\tbb\path\you\want\to\use
```
</details>


## Recommended CMake Options

Use these options for both quick and detailed builds:

<details>
<summary>Commands</summary>

```bat
set CommonBuildOptions=-D ENABLE_LTO=OFF ^
                       -D ENABLE_FASTER_BUILD=OFF ^
                       -D ENABLE_CPPLINT=OFF ^
                       -D ENABLE_TESTS=OFF ^
                       -D ENABLE_FUNCTIONAL_TESTS=OFF ^
                       -D ENABLE_SAMPLES=OFF ^
                       -D ENABLE_JS=OFF ^
                       -D ENABLE_PYTHON=OFF ^
                       -D ENABLE_PYTHON_PACKAGING=OFF ^
                       -D ENABLE_WHEEL=OFF ^
                       -D ENABLE_OV_ONNX_FRONTEND=OFF ^
                       -D ENABLE_OV_PYTORCH_FRONTEND=OFF ^
                       -D ENABLE_OV_PADDLE_FRONTEND=OFF ^
                       -D ENABLE_OV_TF_FRONTEND=OFF ^
                       -D ENABLE_OV_TF_LITE_FRONTEND=OFF ^
                       -D ENABLE_OV_JAX_FRONTEND=OFF ^
                       -D ENABLE_OV_IR_FRONTEND=ON ^
                       -D THREADING=TBB ^
                       -D ENABLE_TBBBIND_2_5=OFF ^
                       -D ENABLE_SYSTEM_TBB=OFF ^
                       -D ENABLE_TBB_RELEASE_ONLY=OFF ^
                       -D ENABLE_HETERO=OFF ^
                       -D ENABLE_MULTI=OFF ^
                       -D ENABLE_AUTO=OFF ^
                       -D ENABLE_AUTO_BATCH=OFF ^
                       -D ENABLE_TEMPLATE=OFF ^
                       -D ENABLE_PROXY=OFF ^
                       -D ENABLE_INTEL_CPU=OFF ^
                       -D ENABLE_INTEL_GPU=OFF ^
                       -D ENABLE_NPU_PLUGIN_ENGINE=OFF ^
                       -D ENABLE_ZEROAPI_BACKEND=OFF ^
                       -D ENABLE_DRIVER_COMPILER_ADAPTER=OFF ^
                       -D ENABLE_INTEL_NPU_INTERNAL=OFF ^
                       -D ENABLE_INTEL_NPU_PROTOPIPE=OFF ^
                       -D BUILD_COMPILER_FOR_DRIVER=ON ^
                       -D ENABLE_PRIVATE_TESTS=OFF ^
                       -D ENABLE_DIRECTML=OFF ^
                       -D ENABLE_NPU_LSP_SERVER=OFF ^
                       -D CMAKE_TOOLCHAIN_FILE=%OPENVINO_HOME%\cmake\toolchains\onecoreuap.toolchain.cmake
```
For more information about the build options, refer to [this section](../FAQ.md#additional-notes).
</details>


## Quick Build Commands (For Immediate Use)

This section provides the complete build commands for users who need to build the Driver Compiler without diving into the detailed configuration steps.

<details>
<summary>Commands</summary>

```bat
md %OPENVINO_HOME%\build_%CONFIG%
cd %OPENVINO_HOME%\build_%CONFIG%

cmake ^
    -D CMAKE_BUILD_TYPE=%CONFIG% ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D OPENVINO_EXTRA_MODULES=%NPU_PLUGIN_HOME% ^
    %CommonBuildOptions% ^
    ..

cmake --build . --config %CONFIG% --target openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest --parallel %NUMBER_OF_PROCESSORS%

@REM Optional, compress and pack all CiD targets
cpack -V -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME=%CONFIG% -G "ZIP"
```
</details>

The Driver Compiler package is built and can be found in `%OPENVINO_HOME%\bin\intel64\%CONFIG%` directory. **You can ignore the remaining sections** of this document.


## Detailed Step-by-step Guide (For a Deeper Understanding on the Build Process)
### Create Build Directory

<details>
<summary>Commands</summary>

```bat
md %OPENVINO_HOME%\build_%CONFIG%
```
</details>

### Configure CMake

If you do not wish to use Ninja for compilation, please remove the line `-G Ninja`.

<details>
<summary>Commands</summary>

```bat
cd %OPENVINO_HOME%\build_%CONFIG%

cmake ^
    -G Ninja ^
    -D CMAKE_BUILD_TYPE=%CONFIG% ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D OPENVINO_EXTRA_MODULES=%NPU_PLUGIN_HOME% ^
    %CommonBuildOptions% ^
    ..
```
</details>

### Build Targets

<details>
<summary>Commands</summary>

```bat
ninja openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest -j %NUMBER_OF_PROCESSORS%
@REM Or, if not using Ninja:
@REM cmake --build . --target openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest --parallel %NUMBER_OF_PROCESSORS%
```
</details>

### Packaging (Optional)

All Driver Compiler-related targets will be generated in `%OPENVINO_HOME%\bin\intel64\%CONFIG%` folder.
- **Compress and pack all CiD components:**
    <details>
    <summary>Compress and pack</summary>

    ```bat
    @REM Compress and pack all CiD components
    cd %OPENVINO_HOME%\build_%CONFIG%
    cpack -V -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME=%CONFIG% -G "ZIP"
    ```
    </details>
- **Install all CiD components to a specific location:**
    <details>
    <summary>Install</summary>

    ```bat
    @REM Install all CiD components
    cd %OPENVINO_HOME%\build_%CONFIG%
    cmake --install .\ --prefix .\ --component CiD --verbose
    ```
    </details>
