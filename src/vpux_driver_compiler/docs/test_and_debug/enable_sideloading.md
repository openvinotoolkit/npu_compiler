# Enable Sideloading for Driver Compiler

This guide explains how to sideload the Driver Compiler (also known as Compiler in Driver and CiD) on Linux and Windows. Sideloading allows you to test a custom-built Driver Compiler without reinstalling the driver.

For example, sideloading can be used with `compile_tool` or `benchmark_app` from OpenVINO to compile or infer neural-network models with an external Driver Compiler.

## Table of Contents
- [General Steps](#general-steps)
- [Linux Sideloading Example](#linux-sideloading-example)
- [Windows Sideloading Example](#windows-sideloading-example)
- [Notes](#notes)

## General Steps

1. **Create a new directory** (e.g., `cid_alt_rel`, `cid_alt_deb`, or `cid_alt_rdi`) for your sideload libraries.
2. **Copy the Driver Compiler libraries**:
   - For Linux: `libopenvino_intel_npu_compiler_loader.so` and `libopenvino_intel_npu_compiler.so`
   - For Windows: `openvino_intel_npu_compiler_loader.dll` and `openvino_intel_npu_compiler.dll`
   - For Windows Debug builds, rename: 
      `openvino_intel_npu_compiler_loaderd.dll` to `openvino_intel_npu_compiler_loader.dll`
      `openvino_intel_npu_compilerd.dll` to `openvino_intel_npu_compiler.dll`
3. **Copy the required oneTBB libraries** (release or debug, matching your build type) into the same directory.
4. **Set the appropriate environment variable** to point to this directory:
   - Linux: `export LD_LIBRARY_PATH=/path/to/your/dir`
   - Windows: `set NPU_ALT_DEPENDENCY_PATH=C:\path\to\your\dir`
5. **(Windows Debug only)**: Copy `ucrtbased.dll` from `C:\Windows\System32` into your sideloading directory.

>**Note**: For sideloading oneTBB libraries on Windows, you must use the OneCore version. If you are unsure how to build the OneCore version of oneTBB, please refer to [the instructions for building the OneCore version of oneTBB](../FAQ.md#windows-onecore-tbb-build).


## Linux Sideloading Example

<details>
<summary>Example for Release</summary>

```sh
# 1. Prepare directory
mkdir cid_alt_rel && cd cid_alt_rel

# 2. Copy Driver Compiler
cp /path/to/libopenvino_intel_npu_compiler_loader.so .
cp /path/to/libopenvino_intel_npu_compiler.so .

# 3. Copy Release oneTBB libraries
# Release:
cp /path/to/libtbb.so.xx.xx libtbb.so.12
cp /path/to/libtbbmalloc.so.xx.xx libtbbmalloc.so.2

# 4. Set sideloading environment variable
export LD_LIBRARY_PATH=$(pwd)
```
>Note: Use ldd libopenvino_intel_npu_compiler.so to check which TBB variant is required.
</details>

<details>
<summary>Example for RelWithDebInfo</summary>

```sh
# 1. Prepare directory
mkdir cid_alt_rdi && cd cid_alt_rdi

# 2. Copy Driver Compiler
cp /path/to/libopenvino_intel_npu_compiler_loader.so .
cp /path/to/libopenvino_intel_npu_compiler.so .

# 3. Copy **Debug** oneTBB libraries
cp /path/to/libtbb_debug.so.xx.xx libtbb_debug.so.12
cp /path/to/libtbbmalloc_debug.so.xx.xx libtbbmalloc_debug.so.2

# 4. Set sideloading environment variable
export LD_LIBRARY_PATH=$(pwd)
```
</details>

<details>
<summary>Example for Debug</summary>

```sh
# 1. Prepare directory
mkdir cid_alt_deb && cd cid_alt_deb

# 2. Copy Driver Compiler
cp /path/to/libopenvino_intel_npu_compiler_loader.so .
cp /path/to/libopenvino_intel_npu_compiler.so .

# 3. Copy Debug oneTBB libraries
cp /path/to/libtbb_debug.so.xx.xx libtbb_debug.so.12
cp /path/to/libtbbmalloc_debug.so.xx.xx libtbbmalloc_debug.so.2

# 4. Set sideloading environment variable
export LD_LIBRARY_PATH=$(pwd)
```
</details>

## Windows Sideloading Example

<details>
<summary>Example for Release</summary>

```bat
@REM 1. Prepare directory
md cid_alt_rel
cd cid_alt_rel

@REM 2. Copy Driver Compiler
copy C:\path\to\openvino_intel_npu_compiler_loader.dll .
copy C:\path\to\openvino_intel_npu_compiler.dll .

@REM 3. Copy Release oneTBB libraries
@REM Release:
copy C:\path\to\tbb12.dll .
copy C:\path\to\tbbmalloc.dll .

@REM 4. Set sideloading environment variable
set NPU_ALT_DEPENDENCY_PATH=%cd%
```
</details>

<details>
<summary>Example for RelWithDebInfo</summary>

```bat
@REM 1. Prepare directory
md cid_alt_rdi
cd cid_alt_rdi

@REM 2. Copy Driver Compiler
copy C:\path\to\openvino_intel_npu_compiler_loader.dll .
copy C:\path\to\openvino_intel_npu_compiler.dll .

@REM 3. Copy **Debug** oneTBB libraries
copy C:\path\to\tbb12_debug.dll .
copy C:\path\to\tbbmalloc_debug.dll .

@REM 4. Set sideloading environment variable
set NPU_ALT_DEPENDENCY_PATH=%cd%
```
</details>

<details>
<summary>Example for Debug</summary>

```bat
@REM 1. Prepare directory
md cid_alt_deb
cd cid_alt_deb

@REM 2. Copy Debug Driver Compiler and rename for Debug build
copy C:\path\to\openvino_intel_npu_compiler_loaderd.dll openvino_intel_npu_compiler_loader.dll
copy C:\path\to\openvino_intel_npu_compilerd.dll openvino_intel_npu_compiler.dll

@REM 3. Copy Debug oneTBB libraries
@REM Debug:
copy C:\path\to\tbb12_debug.dll .
copy C:\path\to\tbbmalloc_debug.dll .

@REM For Debug builds, copy Debug CRT library
copy C:\Windows\System32\ucrtbased.dll .

@REM 4. Set sideloading environment variable
set NPU_ALT_DEPENDENCY_PATH=%cd%
```
>Note: For Debug builds, always rename the DLL and copy `ucrtbased.dll` if not present.
</details>


## Notes

- **TBB Source:** You can use prebuilt TBB from 
    - [oneTBB releases](https://github.com/oneapi-src/oneTBB/releases),
    - or your own build of TBB
- **Library Names:** Always rename TBB libraries as required by your platform and build type.
- **Dependency Check:** Use `ldd` (Linux) or `dumpbin /dependents` (Windows) to verify all dependencies are satisfied.
- **RelWithDebInfo:** On both platforms, this build type may require debug TBB libraries.
