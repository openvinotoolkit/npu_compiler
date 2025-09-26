# Legacy Test Methods for Driver Compiler

>Note: The usage of `compilerTest` has been updated. This document describes legacy test methods for historical reference only. For up-to-date debugging, please refer to the [main test guide](./test.md).

## Legacy Usage of compilerTest

`compilerTest` demonstrates the full Driver Compiler API. You can use IR models for testing in Git Bash (Windows) or Linux shell.
General usage:
```sh
./compilerTest <modelfile>.xml <modelfile>.bin output.net
./compilerTest <modelfile>.xml <modelfile>.bin output.net config.file
```

>Note: In `compilerTest`, if you do not pass a configuration file in command line, an empty configuration will be used for the tested model. The `config.file` is used for updating the IR V10 version model's precision and layout. You can also use an empty `config.file` as input. For `config.file` details, please refer to the next section [Explanation for Configuration](#explanation-for-configuration).

### Explanation for Configuration

I/O Layouts and Precisions are no longer mandatory for all models under current VCL API. For **IRv10** models users are still allowed to pass different I/O Layouts & IO Precisions to override (For example, forcing a IRv10 FP32 model to run in FP16) the original settings. However, passing different I/O Layouts and I/O Precisions will **not override default values for all IRv11** models.

For example, a configuration for googlenet-v1 for legacy usage is as follows:
```sh
--inputs_precisions="input:fp16"
--inputs_layouts="input:NCHW"
--outputs_precisions="InceptionV1/Logits/Predictions/Softmax:fp16"
--outputs_layouts="InceptionV1/Logits/Predictions/Softmax:NC"
--config NPU_PLATFORM="4000" DEVICE_ID="NPU.4000" NPU_COMPILATION_MODE="DefaultHW" NPU_COMPILATION_MODE_PARAMS="swap-transpose-with-fq=1 force-z-major-concat=1 quant-dequant-removal=1 propagate-quant-dequant=0"
```

In the configuration, the contents are:
- `inputs_precisions`: Precision of input node to be used.
- `inputs_layouts`: Layout of input node to be used.
- `outputs_precisions`: Precision of output node to be used.
- `outputs_layouts`: Layout of output node to be used.
- `config`: sets compile configurations as defined in [`Supported Properties` part](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md#supported-properties).


To obtain a configuration file, first run the test model with benchmark_app from the [OpenVINO Project] to get the model's node names. Run `./benchmark_app -m /path/to/model.xml` in Windows Git Bash or Linux shell, for example, using googlenet-v1:
```sh
./benchmark_app -m /path/to/googlenet-v1.xml
```
The output of `[step4/11] Reading model files` and `[step6/11] Configuring input of the model` shows the input and output node information. The googlenet-v1 log output is shown below:
    ![alt text](../imgs/image_config.png)

Each parameter consists of a node name and precision, separated by a colon. If there are multiple input or output nodes, separate each node with a space.
Another method to generate the configuration file can be found [here](../api_reference.md#vclallocatedexecutablecreate2).


## See Also

See the main test and debug guide:

* [how to test](./test.md)
* [how to debug](./debug.md)


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
