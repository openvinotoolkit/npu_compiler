# How to debug

## Logs

To change the compiler behavior a config file can be provided to the `compilerTest` tool. For example, to change the logging level, use:
```
LOG_LEVEL="LOG_TRACE"
```

A full config for googlenet-v1 is as follow:
``` bash
--inputs_precisions="input:fp16" --inputs_layouts="input:NCHW" --outputs_precisions="InceptionV1/Logits/Predictions/Softmax:fp16" --outputs_layouts="InceptionV1/Logits/Predictions/Softmax:NC" --config NPU_PLATFORM="3720" DEVICE_ID="NPU.3720" LOG_LEVEL="LOG_TRACE" NPU_COMPILATION_MODE="DefaultHW"  NPU_COMPILATION_MODE_PARAMS="swap-transpose-with-fq=1 force-z-major-concat=1 quant-dequant-removal=1 propagate-quant-dequant=0"

```


## Other tools

One can also use the tools from [NPU-Plugin Project] and [OpenVINO Project].

### compile_tool

`compile_tool` can compile network to blob. If you test it for Driver Compiler, you need set the config option in config file.

The general command on git bash is:
``` bash
./compile_tool -m <model_path> -d NPU.3720 -c <config_file_path>
```

Here is an example:
```bash
./compile_tool -m path/to/googlenet-v1.xml -d NPU.3720 -c /path/to/config.txt
```
where the content of config.txt is:
```bash
NPU_COMPILER_TYPE DRIVER
```


### benchmark_app

`benchmark_app` is used to estimate inference performance. If you test it for Driver Compiler, you need set the config option in config file.

The general command in git bash:
```bash
./benchmark_app -m <model_path> -load_config=<config_file_path> -d NPU.3720
```

Here is an example:
``` bash
./benchmark_app -m /path/to/mobilenet-v2.xml -load_config=/path/to/config.txt -d NPU
```
where the content of config.txt is:
```
{ "NPU": { "NPU_COMPILER_TYPE":"DRIVER", "NPU_PLATFORM":"3720", "LOG_LEVEL":"LOG_INFO" } }
```

### timetest suite

`timetest suite` is used to measure both total and partial execution time. You can install timetest suite as following the [time_tests/README.md](https://github.com/openvinotoolkit/openvino/blob/master/tests/time_tests/README.md). If you test it for Driver Compiler, you need set the config option in config file.

The general command in git bash:
```bash
python3 ./scripts/run_timetest.py ../../bin/intel64/Release/timetest_infer_api_2.exe -m <model_path> -d NPU.3720 -f <config_file_path>
```

Here is an example:
```bash
python3 scripts\run_timetest.py build\src\timetests\Release\timetest_infer.exe -m googlenet-v1.xml -d NPU.3720 -f config.txt
```
where the content of config.txt is:
```
NPU_COMPILER_TYPE DRIVER
```

>Note: For more debug method and detil, refer to **[how to debug](../../vpux_compiler/docs/guides/how_to_debug.md)** in vpux_compiler part.


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
