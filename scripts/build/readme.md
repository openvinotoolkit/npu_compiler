# Build script
The script can automate most of the manual flow. Some features:
- OpenVINO checkout/update to the commit from `openvino_config.json` (or provided by `--ov_commit` parameter)
- Create OV cmake presets (include Cmake presets from npu compiler repo)
- OpenVINO project configure/build from presets
- NPU project configure/build from presets
- Structured logging with retention
- Build metrics collection and preparing artifacts for analysis
- Build profiling and preparing artifacts for analysis

> Note: These instructions are Linux-only for now. They may also work on Windows, but this has not been tested yet.

# Motivation
- To keep entry point as simple as possible
- To prevent documentation maintenance (as it almost always becomes outdated)
- To have dedicated place to share best build practices between developers

# Environment setup
- git 
    ```
    apt update && apt install -y --no-install-recommends git
    ```
- openvino build dependencies (this script installs cmake version 3.24+)
    ```
    wget https://raw.githubusercontent.com/openvinotoolkit/openvino/master/install_build_dependencies.sh \
        && chmod +x install_build_dependencies.sh \
        && ./install_build_dependencies.sh
    ```
- recommended development tools
    ```
    apt update && apt install -y --no-install-recommends \
        clang-20 \
        mold
    ```
- install python and requirements
    > Note: consider using [miniforge](https://github.com/conda-forge/miniforge) for environment management 

    ```
    python -m pip install -r scripts/build/requirements.txt
    ```
- install ClangBuildAnalyzer (if detailed build profiling is required)
    ```
    git clone https://github.com/aras-p/ClangBuildAnalyzer.git
    cd ClangBuildAnalyzer
    cmake -S . -B build
    cmake --build build
    cmake --install build
    ```

# Build
- To check full list of available parameters:
    ```
    python scripts/build/build.py --help
    ```
  or check the source code of [build.py](build.py)
- You can use the script with your custom presets (just pass the preset names to parameters)

> Note: examples below are just for reference to cover some common use cases, please consider using options required for your specific use case

## Fast developer build release
```
source scripts/build/developer_build_fast_release.sh
```
> Note: please check script for openvino repo layout

## Fast developer build debug
```
source scripts/build/developer_build_fast_debug.sh
```
## Build with precise time measurement
```
source scripts/build/developer_build_time_metric.sh
```
Artifacts formed as this script outputs should be used for [post-commit-metrics](https://cait-npu-scale.vpu-apps.ti.intel.com/post-commit-metrics/summary/latest) tracking  
