# Build Driver Compiler with CMake Presets on Windows

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
@REM Optional: only if using custom TBB, but required for enabling sideloading
set TBBROOT=C:\tbb\path\you\want\to\use
```
</details>


## Quick Build Commands (For Immediate Use)

This section provides the complete build commands for users who need to build the Driver Compiler without diving into the detailed configuration steps.

<details>
<summary>Commands</summary>

```bat
cd %OPENVINO_HOME%
mklink %OPENVINO_HOME%\CMakePresets.json %NPU_PLUGIN_HOME%\CMakePresets.json

cmake --preset cid-windows

cd build_%CONFIG%
cmake --build . --target openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest --parallel %NUMBER_OF_PROCESSORS%

@REM Optional, compress and pack all CiD targets
cpack -V -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME=%CONFIG% -G "ZIP"
```
</details>

The Driver Compiler package is built and can be found in `%OPENVINO_HOME%\bin\intel64\%CONFIG%` directory. **You can ignore the remaining sections** of this document.


## Detailed Step-by-step Guide (For a Deeper Understanding on the Build Process)
### Create a symbolic link to the preset file

<details>
<summary>Commands</summary>

```bat
cd %OPENVINO_HOME%
mklink %OPENVINO_HOME%\CMakePresets.json %NPU_PLUGIN_HOME%\CMakePresets.json
```
>Note: Ensure that CMakePresets.json does not already exist in the directory before linking.
</details>

### Configure CMake

If you want to add the specified TBBROOT field in the CMake preset, please refer to [this](../FAQ.md#tbb-settings-for-cmake-presets).

<details>
<summary>Commands</summary>

```bat
cd %OPENVINO_HOME%
cmake --preset cid-windows
```

The build options for the `cid-windows` preset can be found [here](../../../../CMakePresets.json#L389). For additional details on these build options, refer to [this section](../FAQ.md#additional-notes).
</details>

### Build Targets

<details>
<summary>Commands</summary>

```bat
cd build_%CONFIG%
ninja openvino_intel_npu_compiler openvino_intel_npu_compiler_loader compilerTest profilingTest vpuxCompilerL0Test loaderTest -j %NUMBER_OF_PROCESSORS%
```
</details>

### Packaging (Optional)

All Driver Compiler-related targets will be generated in `%OPENVINO_HOME%\bin\intel64\%CONFIG%` directory.
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


## Notes

1. The `cid-windows` preset must be built from within the OpenVINO directory.
2. Ninja is the default generator of the build and must be installed unless `configurePresets` preset is modified.
3. By default, the `cid-windows` preset is configured to build only Driver Compiler targets. To include additional targets, custom CMake presets that inherit from the `cid` preset should be manually created.
