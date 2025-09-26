# Driver Compiler Test Methods

This guide covers the test tools provided by the Driver Compiler for validating functionality and performance. All tests run in Git Bash (Windows) or Linux shell.

## compilerTest

`compilerTest` demonstrates the full Driver Compiler API.
General command:
```sh
./compilerTest -m xxx.xml -d NPU.XXXX
```
Commonly used command line parameters (same command line options as [compile_tool](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/tools/compile_tool/main.cpp)):

- (required) `-m model.xml`: Specifies the model path. Please ensure the model IR file is complete.

- (required) `-d NPU.XXXX`: Specifies the simulated platform.

- (optional) `-o output.net`: Specifies the output network name.

- (optional) `-c config.file`: Uses the same configuration format as `compile_tool`.
To save the serialized IR, please use the `CID_GET_SERIALIZED_MODEL` environment variable.

## profilingTest

`profilingTest` is used to output profiling information.

General command:
```sh
./profilingTest <blobfile>.blob profiling-0.bin
```

To get the blob file, use compilerTest or [compile_tool](https://github.com/openvinotoolkit/npu_compiler/tree/master/tools/compile_tool) of the [NPU-Plugin Project].

To get the profiling-0.bin and more profiling details, please see [how to use profiling.md](../../../../guides/how-to-use-profiling.md) in the [NPU-Plugin Project].


## loaderTest

`loaderTest` is used to check whether Driver Compiler header is available.

General command:
```sh
./loaderTest -v=1
./loaderTest -v=0
```

## vpuxCompilerL0Test

`vpuxCompilerL0Test` is the test suite of the Driver Compiler. Its test range is defined in [test_smoke.json](../../test/functional/scripts/test_smoke.json) and [test.json](../../test/functional/scripts/test.json).

### Setup

Set `POR_PATH` manually. `POR_PATH` is the test models' root folder.
```sh
# Copy and unpack POR model to special location
tar -xvjf path/to/por_model.tar.bz2
export POR_PATH=/path/to/por_model
```

Set `CID_TOOL` to load the configuration JSON files. You can use the configuration JSON files in the [vpux_driver_compiler/test/functional/scripts](../../test/functional/scripts) folder of this repository
```sh
# Set the configuration JSON files
export CID_TOOL=/path/to/configuration/JSON
```

<details>
<summary>Configuration JSON Format</summary>

The [configuration JSON](../../test/functional/scripts) files contain:

- `device`: Platform on which it will run.
- `enabled`: Whether this model will be executed.
- `network`: Name of the model to run (the simple_function model is defined [here](../../test/functional/vcl_tests_common.cpp#L126)).
- `path`: Relative path to the current model using the POR model or a custom model.
- `info`: Model configuration, blank by default as buildFlags are no longer mandatory for all models.
</details>

### Running Tests

Run all tests:
```sh
./vpuxCompilerL0Test
```
Or run specific tests using `gtest_filter`, e.g., to test resnet-50-pytorch:
```sh
./vpuxCompilerL0Test --gtest_filter=*resnet*50*pytorch*
```


>Note: For more debugging methods and details, refer to [how to debug](../../../vpux_compiler/docs/guides/how_to_debug.md) in vpux_compiler part.


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
