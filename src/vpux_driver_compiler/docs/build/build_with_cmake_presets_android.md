# Build Driver Compiler with CMake Presets for Android

This guide explains how to build the Driver Compiler (also known as Compiler in Driver or CiD) for Android in Ubuntu. The Driver Compiler is built as an extra module of a static OpenVINO package.
> **Before you start:**
> - Review Linux [Requirements](../requirements.md#linux-requirements).
> - Follow [Repository Setup](../prebuild.md#pre-build-preparation).
> - Check [FAQ](../FAQ.md) for [common clone issues](../FAQ.md#common-clone-failure-issues) and [common build issues](../FAQ.md#common-build-and-install-issues).
> - See [TBB selection](../FAQ.md#choosing-the-right-tbb-linux--windows) for threading library options.

> [!WARNING]
> Today the manual is missing building TBB for Android


## Environment Setup

Use these environment variables for both quick and detailed builds:

```sh
export OPENVINO_HOME=/path/to/openvino
export NPU_PLUGIN_HOME=/path/to/cloned/applications.ai.vpu-accelerators.vpux-plugin
export CONFIG=Release        # Or Debug/RelWithDebInfo
```

Install Android NDK in your system:
```sh
curl -L https://dl.google.com/android/repository/android-ndk-r27c-linux.zip -o /tmp/packages/android-ndk.zip
sudo unzip /tmp/packages/android-ndk.zip -d /opt/
```

## Enable building Intel NPU in OpenVINO

Today the OpenVINO does not enable building Intel NPU by default for Android. To enable it, the OpenVINO repository must be patched in cmake/features.cmake file with following change:

```
diff --git a/cmake/features.cmake b/cmake/features.cmake
index 0c1bfa2033..877267c290 100644
--- a/cmake/features.cmake
+++ b/cmake/features.cmake
@@ -43,7 +43,7 @@ endif()

 ov_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ${ENABLE_ONEDNN_FOR_GPU_DEFAULT} "ENABLE_INTEL_GPU" OFF)

-ov_dependent_option (ENABLE_INTEL_NPU "NPU plugin for OpenVINO runtime" ON "X86_64;WIN32 OR LINUX" OFF)
+ov_dependent_option (ENABLE_INTEL_NPU "NPU plugin for OpenVINO runtime" ON "X86_64;WIN32 OR LINUX OR ANDROID" OFF)
 ov_dependent_option (ENABLE_INTEL_NPU_INTERNAL "NPU plugin internal components for OpenVINO runtime" ON "ENABLE_INTEL_NPU" OFF)

 ov_option (ENABLE_DEBUG_CAPS "enable OpenVINO debug capabilities at runtime" OFF)
--
```


## Build Commands

This section provides the complete build commands.

```sh
ln -s ${NPU_PLUGIN_HOME}/CMakePresets.json ${OPENVINO_HOME}/CMakePresets.json

# Build native tools
cmake -G Ninja -B build-native -S . --preset cid-linux --config Release
cmake --build build-native --parallel $(nproc) --target npureg-tblgen flatc

# Build Driver compiler for Android
cmake -G Ninja -B build-android -S . --preset cid-linux \
  -DTHREADING=SEQ \
  -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r27c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=x86_64 \
  -DANDROID_PLATFORM=android-34 \
  -DANDROID_STL=c++_shared \
  -DCMAKE_CXX_FLAGS_INIT='-frtti'
cmake --build build-android --parallel $(nproc) --target openvino_intel_npu_compiler openvino_intel_npu_compiler_loader

# Optional, compress and pack all CiD targets
cpack -V -D CPACK_COMPONENTS_ALL=CiD -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_PACKAGE_FILE_NAME="${CONFIG}" -G "TGZ"
```

The Driver Compiler package is built and can be found in `${OPENVINO_HOME}/bin/intel64/${CONFIG}` directory. **You can ignore the remaining sections** of this document.


## Notes

1. The `cid-linux` preset must be built from within the OpenVINO directory.
2. Ninja is the default generator of the build and must be installed unless `configurePresets` preset is modified.
3. By default, the `cid-linux` preset is configured to build only Driver Compiler targets. To include additional targets, custom CMake Presets that inherit from the `cid` preset should be manually created.
4. If Makefile is used as the generator, the additional targets are required for native tools:
```sh
# Extra targets for native tools
cmake --build build-native --parallel $(nproc) --target npureg-tblgen flatc mlir-headers mlir-generic-headers mlir-linalg-ods-yaml-gen

# Copy native tools to the Android build directory
mkdir -p build-android/build-modules/npu_compiler/thirdparty/llvm-project/llvm/NATIVE
cp -r build-native/build-modules/npu_compiler/thirdparty/llvm-project/llvm/bin build-android/build-modules/npu_compiler/thirdparty/llvm-project/llvm/NATIVE/
```
