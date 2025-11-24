# Requirements and Setup to Build Driver Compiler
## Linux Requirements

Ensure the following components are available on your system before building Driver Compiler.

- Hardware:
    - Minimum 32 GB RAM required

- Software:
    - [CMake](https://cmake.org/download) 3.22.1 for Ubuntu 22.04 (version 3.13 or higher)
    - GCC 11.4.0 for Ubuntu 22.04 (version 7.5 or higher)
    - Python 3.9 - 3.12
    - Git for Linux (required for `git lfs`)
    - Ninja
    - Additional dependency to build with CMake Presets:
        - ccache (Download latest version of ccache binaries or [build from source](https://github.com/ccache/ccache/releases))
            >Note: The build options defined in the `CMakePresets.json` require ccache, and it must be installed to use the default presets. If installation is not possible, remove or update the relevant entries in [CMakePresets.json](../../../CMakePresets.json#L7-L16) accordingly.

>Note: 32GB RAM is recommended, but not strictly required. For systems with smaller RAM, you can compensate by reducing the number of build threads or increasing swap memory.

## Windows Requirements

Ensure the following components are available on your system before building Driver Compiler. Make sure they are added to your system PATH after installation.

- Hardware:
    - Minimum 40 GB of disk space required

- Software:
    - [CMake](https://cmake.org/download) 3.13 or higher
    - Microsoft Visual Studio 2022
        >Note: Windows SDK and spectre libraries are required to build OpenVINO and NPU-Plugin. Install them from Microsoft Visual Studio Installer: Modify -> Individual components -> Search components: "Windows SDK" and "Spectre x64/x86 latest".
    - SDK (install from Microsoft Visual Studio or [here](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive)) and [WDK](https://learn.microsoft.com/en-ie/windows-hardware/drivers/other-wdk-downloads#step-2-install-the-wdk). Ensure the versions match your Windows system.
    - Python 3.9 - 3.12
    - Git for Windows (required for `git lfs`)
    - Ninja
    - Additional dependency to build with CMake Presets:
        - ccache (Download latest version of ccache binaries or [build from source](https://github.com/ccache/ccache/releases))
            >Note: The build options defined in the CMakePresets.json require ccache, and it must be installed to use the default presets. If installation is not possible, the relevant entries in [CMakePresets.json](../../../CMakePresets.json#L7-L16) should be removed and updated accordingly.
## Next Steps

Pre-build Preparation:
* [pre-build](./prebuild.md)

Linux: build with CMake Options (Recommended for the first Driver Compiler build):
* [how to build Driver Compiler on Linux](./build/build_linux.md)

Linux: build with CMake Presets:
* [how to build Driver Compiler with CMake Presets on Linux](./build/build_with_cmake_presets_linux.md)

Windows: build with CMake Options (Recommended for the first Driver Compiler build):
* [how to build Driver Compiler on Windows](./build/build_windows.md)

Windows: build with CMake Presets:
* [how to build Driver Compiler with CMake Presets on Windows](./build/build_with_cmake_presets_windows.md)
