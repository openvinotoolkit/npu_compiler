# How to build Driver Compiler with cmake presets on Linux

## Dependencies

Before you start to build Driver Compiler targets, please check the necessary components. 
- Hardware
    - Minimum requirements: 32GB RAM
- Software
    - [CMake](https://cmake.org/download/) 3.22.1 for Ubuntu 22.04 (version 3.19 or higher)
    - GCC 11.4.0 for Ubuntu 22.04 (version 7.5 or higher)
    - Python 3.9 - 3.12
    - Git for Linux (requires installing `git lfs`)
    - Ccache
    - Ninja

> Notice: RAM is not mandatory either. If your RAM is less than 32GB, you can compensate by reducing the number of threads during the build or by increasing the swap memory.

> Notice: Ccache and Ninja are required for the build options defined in the CMake Presets. Therefore, both of these tools are necessary. If you are unable to install them, please remove and update the relevant sections in [CMakePresets.json](https://github.com/openvinotoolkit/npu_compiler/blob/develop/CMakePresets.json#L7C1-L16C19).

## Using CMakePresets to build

#### Using CMakePresets to build and using NPU-Plugin as an extra module of OpenVINO

Here provides a default pre-configured CMake presets for users named: "npuCidReleasexxx", `npuCidReleaseLinux` for Linux and `npuCidReleaseWindows` for Windows. The setting is to build [NPU-Plugin Project] as an extra module of [OpenVINO Project]. In this case, `NPU_PLUGIN_HOME` environment variable must be set.

1. Clone repos:

    <details>
    <summary>Instructions</summary>

    ```sh
        # set the proxy, if required.
        # export  http_proxy=xxxx
        # export  https_proxy=xxxx

        cd /home/useraccount/workspace (Just an example, you could use your own branch/tag/commit.)
        git clone https://github.com/openvinotoolkit/openvino.git 
        cd openvino
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive

        cd /home/useraccount/workspace (Just an example, you could use your own branch/tag/commit.)
        git clone https://github.com/openvinotoolkit/npu_compiler.git
        cd npu_compiler
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive
    ```
    </details>

    > Notice: If you are building the Driver Compiler targets with the goal of composing the Linux driver, it is important to pay attention to the version or commit of the [OpenVino Project] being used. Make sure to check the current supported version of the [OpenVino Project] from `OpenVINO built from source` entry in the table under the `Common` section in the [release notes](https://github.com/intel/linux-npu-driver/releases/) or within the [Linux NPU driver code](https://github.com/intel/linux-npu-driver/blob/main/compiler/compiler_source.cmake#L20).

2. Set environment variables and create a symlink for a preset file in OpenVINO Project root:

    <details>
    <summary>Instructions</summary>
    
    ```sh
        # set the enviroment variables
        export OPENVINO_HOME=/home/useraccount/workspace/openvino (need change to your own path)
        export NPU_PLUGIN_HOME=/home/useraccount/workspace/npu_compiler (need change to your own path)

        cd $OPENVINO_HOME
        ln -s $NPU_PLUGIN_HOME/CMakePresets.json ./CMakePresets.json
    ```
    </details>

3. Build with the following commands:

    Before build with the following instructions, please make sure `OPENVINO_HOME` and `NPU_PLUGIN_HOME` enviroment variables have been set.

    <details>
    <summary>Instructions</summary>
    
    ```sh
        cd $OPENVINO_HOME
        cmake --preset npuCidReleaseLinux
        cd build-x86_64/Release/
        cmake --build ./ --target npu_driver_compiler compilerTest profilingTest vpuxCompilerL0Test loaderTest -j8
    ```
    </details>

    The defined build option for npuCidReleaseLinux Cmake Preset is listed [here](../../../CMakePresets.json#L240). For additional information about its build options, please refer to section `2.2 Build instructions notes` in [how to build Driver Compiler on linux](./how_to_build_driver_compiler_on_linux.md).

4. (Optional) Prepare final Driver Compiler package for driver:

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


5. (Optional) Instruction notes about TBB:

    <details>
    <summary>5.1 Default tbb location</summary>

    The [build instructions](../../../CMakePresets.json#L274) uses the `"ENABLE_SYSTEM_TBB": false` option, which means that the TBB library downloaded by [OpenVINO Project] will be used. The download path for this TBB library is `$OPENVINO_HOME/temp/tbb`. Within the downloaded TBB folder, `$OPENVINO_HOME/temp/tbb/lib/libtbb.so.12` and `$OPENVINO_HOME/temp/tbb/lib/libtbbmalloc.so.2` are required for the Release version. 

    </details>

    <details>
    <summary>5.2 Use different TBB version</summary>

    If you wish to build with system TBB, you need install TBB in your local system first and then use `"ENABLE_SYSTEM_TBB": true` option to instead of `"ENABLE_SYSTEM_TBB": false` option in [here](../../../CMakePresets.json#L274).

    If you wish to build with a specific version of TBB, you can download it from [oneTBB Project] and unzip its release package. Then, add the following new lines after line 251 in [CMakePresets.json](../../../CMakePresets.json#L251) file.

    ```sh
        "TBBROOT": {
            "type": "FILEPATH",
            "value": "/home/username/path/to/downloaded/tbb"
        }
    ```

    </details>

    <details>
    <summary>5.3 Do not use TBB</summary>

    If you wish to build without TBB (which will result in a slower build process), you need change `"value": "TBB"` to `"value": "SEQ"` in [here](../../../CMakePresets.json#L228). More info about SEQ mode, please refer to this [file](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md#options-affecting-binary-size).

    </details>

### Note

1. Presets step for "npuCidReleasexxx" need to be built in [OpenVINO Project] folder.
2. The presets for "npuCidReleasexxx" define the build directory build-x86_64/Release.
3. The presets are configured to use Ninja as default generator, so installing Ninja package is an extra requirement.
4. Currently Presets for "npuCidReleasexxx" will build the smallest size targrts of Driver Compiler. If the user wishes to build Driver Compiler and other targets, can driectly inherit "Cid" preset and enable the needed build option to self configuration presets.

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
[oneTBB Project]: https://github.com/oneapi-src/oneTBB
