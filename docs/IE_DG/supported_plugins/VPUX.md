# VPUX Plugin

## Introducing VPUX Plugin

VPUX Plugin was developed for inference of neural networks on the supported Intel&reg; Movidius&trade; VPU devices:

  * Gen 3 Intel&reg; Movidius&trade; VPU (3700VE)
  * Gen 3 Intel&reg; Movidius&trade; VPU (3400VE)
  * Intel&reg; Movidius&trade; S 3900V VPU
  * Intel&reg; Movidius&trade; S 3800V VPU
  * Intel&reg; Vision Accelerator Design PCIe card with Intel Movidius&trade; S VPU

## Supported Platforms

OpenVINO™ toolkit is officially supported and validated on the following platforms:

| Host              | OS (64-bit)                          |
| :---              | :---                                 |
| Development       | Ubuntu* 18.04, MS Windows* 10        |

### Offline Compilation

To run inference using VPUX plugin, Inference Engine Intermediate Representation needs to be compiled for a certain VPU device. Sometimes, compilation may take a while (several minutes), so it makes sense to compile a network before execution. Compilation can be done by a tool called `compile_tool`. An example of the command line running `compile_tool`:
```
compile_tool -d VPUX.3700 -m model.xml -c vpu.config
```
Where `VPUX` is a name of the plugin to be used, `3700` defines a VPU platform to be used for compilation (Gen 3 Intel&reg; Movidius&trade; VPU (3700VE)), `model.xml` - a model to be compiled, `vpu.config` (optional) is a text file with config options.

If the platform is not specified, VPUX Plugin tries to determine it by analyzing all available system devices:
```
compile_tool -d VPUX -m model.xml
```

If system doesn't have any devices and platform for compilation is not provided, you will get an error `No devices found - platform must be explicitly specified for compilation. Example: -d VPUX.3700 instead of -d VPUX.`

The table below contains VPU devices and corresponding VPU platform:

| VPU device                                    | VPU platform |
| :-------------------------------------------  | :----------- |
| Gen 3 Intel&reg; Movidius&trade; VPU (3700VE) |   3700    |
| Gen 3 Intel&reg; Movidius&trade; VPU (3400VE) |   3400    |
| Intel&reg; Movidius&trade; S 3900V VPU        |   3900    |
| Intel&reg; Movidius&trade; S 3800V VPU        |   3800    |

To compile without loading to the device set environment variable IE_VPUX_CREATE_EXECUTOR to 0:
```
export IE_VPUX_CREATE_EXECUTOR=0
```
This is a temporary workaround that will be replaced later.

### Inference

For inference you should provide device parameter (see the table `Supported Configuration Parameters` below). Here are the examples of the command line running `benchmark_app`:
```
benchmark_app -d VPUX -m model.xml
```
Run inference on any available VPU device
```
benchmark_app -d VPUX.3900 -m model.xml
```
Run inference on any available slice of Intel&reg; Movidius&trade; S 3900V VPU
```
benchmark_app -d VPUX.3800.0 -m model.xml
```
Run inference on the first slice of Intel&reg; Movidius&trade; S 3800V VPU

## Supported Configuration Parameters

The VPUX plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                      |
| :---                  | :---             | :---             | :---                                                                             |
| `LOG_LEVEL`                  |`LOG_LEVEL_NONE`/ `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_DEBUG`/ `LOG_LEVEL_TRACE`|`LOG_LEVEL_NONE`  |Set log level for VPUX plugin |
| `PERF_COUNT`                 | `YES`/`NO`                                                                                   |`NO`              |Enable or disable performance counter|
| `DEVICE_ID`                  | empty/ `3400[.[0-3]]`/ `3700[.[0-3]]`/ `3900[.[0-3]]` / `3800[.[0-3]]`                              | empty (auto detection)                   |Device identifier `platform.slice` |
| `PERFORMANCE_HINT`           | `THROUGHPUT`/`LATENCY`                                                                       | `THROUGHPUT` (for the benchmark app) | Profile which determines the number of DPU groups (tiles) and the number of inference requests if none of them is modified manually. The default parameter values for each profile are documented in the [Performance Hint: Default Number of DPU Groups and Inference Requests](#performance-hint-default-number-of-dpu-groups-and-inference-requests) section |
| `VPUX_INFERENCE_SHAVES`      | positive integer, `0`                                                                        | `0`              |Number of shaves for model execution, if `0` is set, count of SHAVEs will be evaluated automatically|
| `VPUX_CSRAM_SIZE`            | integer                                                                                      | `-1`             |Set the size of CSRAM in bytes, if `-1` is set, compiler will evaluate size of CSRAM automatically|
| `VPU_COMPILER_CUSTOM_LAYERS` | string                                                                                       | empty            | Path to custom layer binding xml file. Custom layer has higher priority over native implementation |
| `VPU_COMPILER_FORCE_HOST_QUANTIZATION`                 | `YES`/`NO`                                                                                   |`NO`              |Enable or disable FP32/FP16 to U8 input quantization on VPUX Plugin side via CPU|

### Performance Hint: Default Number of DPU Groups and Inference Requests

The following table shows the default parameter values used when setting the `THROUGHPUT` performance hint profile:

| VPU Platform        | Number of DPU Groups | Number of Inference Requests    |
| :---                | :---                 | :---                            |
| 3700                | 1                    | 8                               |
| 3720                | 2 (all of them)      | 4                               |

The default parameter values applied when using the `LATENCY` profile:

| VPU Platform        | Number of DPU Groups | Number of Inference Requests    |
| :---                | :---                 | :---                            |
| 3700                | 4 (all of them)      | 1                               |
| 3720                | 2 (all of them)      | 1                               |

# See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)
