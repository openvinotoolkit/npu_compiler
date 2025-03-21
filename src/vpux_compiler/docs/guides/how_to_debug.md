# How to debug

The compiler offers a number of debug capabilities. In order to benefit from all of them, it is recommended to use a Debug build with the `DEVELOPER_BUILD` flag enabled (the `-DENABLE_DEVELOPER_BUILD=ON` CMake option).

## Logs

The project offers a logging functionality, which different levels of verbosity (from traces to fatal errors) - see the `Logger` class for more information. During compilation, these logs can be printed by making use of the `LOG_LEVEL` (default `LogLevel::None`) compiler config option, example:
`LOG_LEVEL LOG_INFO`
Alternative approach is to use `OV_NPU_LOG_LEVEL` env variable in which log level can be overridden. This is also the only way to set log level when running vpux-opt or unit tests.

If only certain logs are to be printed `IE_NPU_LOG_FILTER` environment variable can be configured. The majority of the logs have a name which can be used for filtering. For passes, the name corresponds to the pass argument name (e.g. `convert-layers-to-VPUIP`). This variable is also compatible with POSIX Regex, which allows capturing more than one pass. Examples of values for the variable:

```sh
export IE_NPU_LOG_FILTER=convert-layers-to-VPUIP
export IE_NPU_LOG_FILTER=convert-layers-to-VPU.*
export IE_NPU_LOG_FILTER="convert-layers-to-VPU|convert-layers-to-VPUIP"
```

The variable can be used with applications which follow the normal compilation flow (e.g. `compile_tool`), as well as tools such as `vpux-opt`.

Level of logs that are printed when filter is applied is controlled by `LOG_LEVEL` compiler config option

### Pass Timings

Logs are also available for displaying the duration per-pass. These logs are printed at the `INFO` level and can be enabled with:

```sh
export IE_NPU_LOG_FILTER=vpux-compiler
```

The moment a pass starts and ends will be printed, along with a timestamp. At the end of the compilation, a full report is also printed which includes:
- the duration of each pass
- the percentage of time spent in each pass, relative to the entire compilation
- the total compilation time

### Pass Instrumentation

#### Memory Usage Collector

This instrumentation checks whether the peak memory consumption of the process has grown after the execution of every pass. If it did, a log will be printed to inform the user. The peak memory usage of the process will also be printed at the start and end of the compilation.

This feature can be configured with the `enable-memory-usage-collector` compilation option. By default, this instrumentation is enabled for developer builds.

#### Function Statistics

This instrumentation prints information about the function(s) found in the IR, after each pass. It targets two types of data:

1. Under the `function-statistics-instrumentation-ops` log, it keeps track of the number of operations for each function in the IR. In case the number of operations changed after the execution of a pass, a log will be printed. Additionally, the current number of operations for all functions is printed after a ModulePass is executed.
2. Under the `function-statistics-instrumentation-const` log, it prints how many constants are found in a function and what the total size of the constants is. In case the pass is a ModulePass, this information will be printed for each function.

This feature can be configured with the `enable-function-statistics-instrumentation` compilation option. By default, this instrumentation is disabled.

## IR Printing

One of the most useful debug features of MLIR is by printing the Intermediate Representation (IR) of a model. During compilation, the printing can be done before or after passes and can be controlled using the following variables:

- `IE_NPU_IR_PRINTING_FILTER`: accepts a POSIX Regex filter which describes what passes should have their IR printed
    - examples:
        - `export IE_NPU_IR_PRINTING_FILTER=ConvertLayersToVPUIP`
        - `export IE_NPU_IR_PRINTING_FILTER="InitResources|ConvertLayers.*"`
    - by default, it will print to stdout the IRs after the specified passes

- `IE_NPU_IR_PRINTING_ORDER`: controls whether to print the IRs before or after the passes (case-insensitive)
    - `export IE_NPU_IR_PRINTING_ORDER=before` - print IRs before the selected passes
    - `export IE_NPU_IR_PRINTING_ORDER=before_after` - print IRs before and after the selected passes
    - `export IE_NPU_IR_PRINTING_ORDER=after` (default) - print IRs after the selected passes

- `IE_NPU_IR_PRINTING_FILE`: prints the IRs in the file specified in this variable, instead of stdout
    - example: `export IE_NPU_IR_PRINTING_FILE=dump.mlir`

- `IE_NPU_PRINT_FULL_IR`: controls the scope of printing
    - `export IE_NPU_PRINT_FULL_IR=0` (default) - only the affected scope will be printed (e.g. only the main Function of the IR, without the parent Module)
    - `export IE_NPU_PRINT_FULL_IR=1` - the entire IR will be printed, including the parent Module operation and dialect dictionary which holds dense resource values (this mode disables MLIR multi-threading)

- `IE_NPU_PRINT_FULL_CONSTANT`: controls whether to print large constants
    - `export IE_NPU_PRINT_FULL_CONSTANT=0` (default) - avoids printing constants whose number of elements is larger than the upper limit (by default 16)
    - `export IE_NPU_PRINT_FULL_CONSTANT=1` - prints the entire content of all constants, regardless of their size
        - this mode is useful when a printed IR is intended to be used for further execution (e.g. running another pass on it), as it cannot be parsed without the full constants.

- `IE_NPU_USE_SHARED_CONSTANTS`: controls whether to share constants with openvino. Note that
    - `export IE_NPU_USE_SHARED_CONSTANTS=0` - copy OV constants into MLIR context and represent them as DenseElementsAttr in the IR. This allows pretty-printing of constants but will result in higher memory usage and potentially change in schedule due to the identical constants being uniqued in the IR by the GreedyPatternRewriter.
    - `export IE_NPU_USE_SHARED_CONSTANTS=1` (default) - use OV constants without copying them. Internally constants are represented as DenseResourceElementsAttr.

- `IE_NPU_PRINT_HEX_CONSTANT`: controls whether to allow printing constants as hex values
    - `export IE_NPU_PRINT_HEX_CONSTANT=0` - prints the individual values of the constants in a human-readable format
    - `export IE_NPU_PRINT_HEX_CONSTANT=1` (default) - prints the values of the constants as hex values

- `IE_NPU_PRINT_DEBUG_INFO`: controls whether to print the locations of the operations into the IRs
    - `export IE_NPU_PRINT_DEBUG_INFO=0` (default) - no locations will be printed into the IR
    - `export IE_NPU_PRINT_DEBUG_INFO=1` - locations will be printed for each operation
        - this is useful in order to cross-reference an operation from the IR with the layers found in the original OpenVINO IR used for compilation; for each operation, the first string in the location should correspond with the name of the original OpenVINO layer

- `IE_NPU_PRINT_DEBUG_INFO_PRETTY_FORM`: controls whether to print the locations of the operations inline with the operations; this variable is only applicable when `IE_NPU_PRINT_DEBUG_INFO` is enabled
    - `export IE_NPU_PRINT_DEBUG_INFO_PRETTY_FORM=0` (default) - the locations will be printed after the module opration, with the operation having an alias after it instead of the direct location
    - `export IE_NPU_PRINT_DEBUG_INFO_PRETTY_FORM=1` - locations will be printed inline with each operation

## vpux-opt and vpux-translate

These tools can be used to call specific parts of the compiler (frontend, backend, passes) from the command-line.

### vpux-translate

`vpux-translate` allows calling the frontend and backend of the compiler. For example:

- importing OpenVINO IR into IE dialect IR: `vpux-translate --vpu-arch=NPU37XX --import-IE <path to xml> -o <MLIR file name>`
- exporting VPUIP dialect IR into ELF file: `vpux-translate --vpu-arch=NPU37XX --export-ELF <input MLIR file> > <output ELF file>`

The full list of supported frontends and backends can be found in [vpux-translate.cpp](../../../../tools/vpux-translate/vpux-translate.cpp).

The rest of the options can be found by calling `vpux-translate --help`.

### vpux-opt

`vpux-opt` can be used to call individual passes or pipelines over IRs. For example:

```sh
# Call the ConvertLayersToVPU pass over the input IR found in the file
vpux-opt --vpu-arch=NPU37XX --convert-layers-to-VPU <MLIR file name>

# Call the ExpandActivationChannels and Canonicalizer passes over the IR
vpux-opt --vpu-arch=NPU37XX --expand-activation-channels --canonicalizer <MLIR file name>

# Call the LowPrecision pipeline over the IR
vpux-opt --vpu-arch=NPU37XX --low-precision <MLIR file name>
```

The tool offers some features which can help in debugging the target code:

- `-debug-only=dialect-conversion` - prints a trace of all the changes attempted or done for conversion patterns (e.g. legalization checks for operations, attempted pattern matches, rewriter actions such as insertions, replacements etc.)
- `--mlir-elide-elementsattrs-if-larger=<uint>` - avoid printing constants whose number of elements is larger than the given integer
- `--mlir-print-ir-after-failure` - prints the entire IR which causes a failure to occur (e.g. when a verifier fails)
    - has multiple variations for printing before or after specific passes, all passes or all changes (`--mlir-print-ir-after=<pass-arg>`, `--mlir-print-ir-after-all`, `--mlir-print-ir-after-change`, `--mlir-print-ir-before=<pass-arg>`, `--mlir-print-ir-before-all`)
- `--mlir-print-op-generic` - prints operations using the generic printer instead of the custom one; can be useful in some situations, such as if the custom printer crashes
- `--mlir-print-debuginfo` - include locations into the printed IR
- `--mlir-print-value-users` - prints the users of a value in a comment
- `--verify-each=0` - avoid running verifiers, if their failures prevent the printing of the IR

The rest of the options can be found by calling `vpux-opt --help`.

### Compiling a model using vpux-translate & vpux-opt

The tools can be used to perform a full compilation of a model:

```sh
# Import an OpenVINO IR into IE dialect
./vpux-translate --vpu-arch=NPU37XX --import-IE <path to OV IR> -o net.mlir

# Call the DefaultHWMode pipeline over the imported IR
./vpux-opt --vpu-arch=NPU37XX --default-hw-mode --lower-VPUIP-to-ELF net.mlir -o net_out.mlir

# Export the final IR into an ELF file
./vpux-translate --vpu-arch=NPU37XX --export-ELF net_out.mlir > net.blob
```

Since each pass or pipeline can be specified individually, the user can have full control over the calling order of the passes if that is necessary. Options such as `--mlir-print-debuginfo` can also be included for both tools to also track the changes done to the original layers.

## IR Visualization using Dot Graph

The compiler contains a pass which can generate a Dot Graph representation of the IR, called `PrintDot`. It can be used with the following methods:

- Using the `IE_NPU_PRINT_DOT` environment variable (applicable for `DEVELOPER_BUILD`)
    - the variable accepts the name of the pass whose output IR will be represented as a graph (`pass`), as well as the name of the dot file that will contain it (`output`)
    - multiple passes can be covered by separating them using commas
    - examples:
        - `export IE_NPU_PRINT_DOT="output=temp.dot pass=OptimizeAsyncDeps"`
        - `export IE_NPU_PRINT_DOT="output=temp.dot pass=OptimizeAsyncDeps,output=temp2.dot pass=AddBuffersForNetResults"`

- Using `vpux-opt` with the `--print-dot` argument
  - example: `vpux-opt [target passes] --print-dot="output=temp.dot"`

- Using the compiler pipeline, by manually adding the `PrintDot` pass where desired and rebuilding the project
  - example: `pm.addPass(createPrintDot(<FileName>, <Options>));`

## Debugging barrier dependencies

At several stages of compilation the structure of the control graph may change due to adding new tasks,
reordering tasks, optimizing tasks dependencies etc. It is possible to check what exactly has changed after applying some
transformations either across several passes, within single pass or some individual routine. BarrierInfo class
offers methods to dump actual state of barrier dependencies at any stage of compilation where the concept of barriers is defined (i.e. in the VPUIP and VPURT dialects). This is typically done
as follows:

```cpp
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    barrierInfo.dumpBarriers("beforeTransformation"); // dump state before barrier optimization
    barrierInfo.optimizeBarriers(/* checkValidSlotCount */ false, _considerTaskFifoDependency);
    barrierInfo.dumpBarriers("afterTransformation"); // dump state after barrier optimization
```

The subsequent compilation will generate a set of files prefixed "beforeTransformation" and "afterTransformation" respectively, containing the corresponding states of the graph.
Extensions of the dumped files are as follows:
- taskUpdateBarriers - contains list of indexes of update barriers for each task in the graph. Each row starts with a unique task index.
- taskWaitBarriers - contains list of indexes of wait barriers for each task in the graph. Each row starts with a unique task index.
- barrierProducerMap - contains list of indexes of producers for each barrier in the graph. Each row starts with a unique barrier index.
- barrierConsumerMap - contains list of indexes of consumers for each barrier in the graph. Each row starts with a unique barrier index.
- slots - contains number of slots occupied by a task when it waits or updates a barrier. Each row starts with a unique task index.
- taskQueueTypeMap - contains list task indexes in each queue. Each row starts with queue name, followed by a number tasks in the queue (provided within parentheses), followed by a colon.

At times, it is also handy to print out the `module` or the main function, in order to make a connection between tasks/barriers indexes from the dump files and their IR representation. Tracing graph changes is not limited to methods of the BarrierInfo class, but an object of that class needs to be created in order to dump the graph dependencies with the actual state of the IR.

Quick investigation of the dependencies from the dump files can be easier than browsing the IR. A plotting tool (located in `tools/barrier-plotter`) can process the dump files and generate `dot` graphs, ready for viewing with e.g. `xdot`. It also exports the `dot` files to `PDF`.

Example usage:

```python
plot_barriers.py beforeTransformation -o beforeTransformation
```
This command will read dump files `beforeTransformation.taskUpdateBarriers`, `beforeTransformation.taskWaitBarriers`, `beforeTransformation.slots` and `beforeTransformation.taskQueueTypeMap` and will produce `beforeTransformation.dot` and `beforeTransformation.dot.pdf`. The output file prefix need not be identical with the dump files prefix and can be adjusted to indicate the dump point and/or compilation configuration.

**Notes**
- By default, declarations and constants are not included. To include them, add `print-declarations=true and print-const=true` to the pass options.
- The `xdot` application can be used to visualize the dot graph. For big networks however, the application may fail to show the graph. The size of the graph can be reduced by specifying which operations should be included, with the `start-after` & `stop-before` pass options. They should be configured with the exact names of the operation from the compiler IR.
    - example: `start-after=pool2 stop-before=conv4/WithoutBiases`
- In case the IR contains single function the name given for the output file will be appended with `_main`
    - example: for `output=temp.dot` result will be available in `temp_main.dot`
- In case the IR contains multiple functions, each function will be printed in a separate .dot file.
    - example: for `output=temp.dot` printing two functions will generate `temp_main.dot`, `temp_foo1.dot`, `temp_foo2.dot`.

## Crash Reproducer

MLIR offers a feature which allows the creation of reproducers in the event of a crash or pass failure. These reproducers will contain the input IR to the failing pass / pipeline as well as the instructions for reproducing it. Executing a reproducer using `vpux-opt` should then result in the same error. This is a very useful feature to simplify the debugging process.

An example of how to generate and run a reproducer using `compile_tool`, by making use of the `IE_NPU_CRASH_REPRODUCER_FILE` environment variable:

```sh
# Compile the model and generate a reproducer into the `reproducer.mlir`` file
IE_NPU_CRASH_REPRODUCER_FILE=reproducer.mlir compile_tool -d VPUX.3720 -ip FP16 -op FP16 -il NCHW -iml NCHW -ol NC -oml NC -m net.xml

# Execute the reproducer
vpux-opt --vpu-arch=NPU37XX reproducer.mlir
```

It is also possible to generate reproducers with `vpux-opt` directly, by using the `--mlir-pass-pipeline-crash-reproducer` argument. For example:

```sh
# Generate the reproducer
vpux-opt --vpu-arch=NPU37XX --convert-layers-to-VPU --mlir-pass-pipeline-crash-reproducer=reproducer.mlir net.mlir

# Execute the reproducer
vpux-opt --vpu-arch=NPU37XX reproducer.mlir
```

### Local reproducers

MLIR can generate two kinds of reproducers: full or local scope. The full scope reproducers will include the original input IR, coupled with the full pipeline description which includes every individual pass with its own options. The local scope will only include the input IR to the crashing pass and the pass argument name, coupled with its options if any exist.

When compiling a network, the reproducer scope can be controlled using the `IE_NPU_GEN_LOCAL_REPRODUCER` environment variable:

* `IE_NPU_GEN_LOCAL_REPRODUCER=0` - full reproducer scope
* `IE_NPU_GEN_LOCAL_REPRODUCER=1` (default) - local reproducer scope


When using `vpux-opt`, the local reproducer can also be used. For example:

```sh
vpux-opt --vpu-arch=NPU37XX --low-precision --mlir-pass-pipeline-crash-reproducer=reproducer.mlir --mlir-pass-pipeline-local-reproducer --mlir-disable-threading net.mlir
```

MLIR multi-threading has to be disabled when running the local reproducer. Internally, when using `IE_NPU_GEN_LOCAL_REPRODUCER` this is already handled, but using this option with `vpux-opt` requires it to be explicitly disabled.

To also note that when the local reproducer is used, MLIR will try to reduce the reproducer's scope to a local one. In case it cannot, it will fallback to a full scope.

More information can be found [here](https://mlir.llvm.org/docs/PassManagement/#crash-and-failure-reproduction).

## Printing MLIR objects

Many MLIR components come with their own printing interface. For example, `mlir::Operation`, `mlir::Type`, `mlir::Value` etc. have the `dump()` method which prints the data into the stderr stream. In order to print to another stream, the `print(raw_ostream &os)` method is available. Example usage:

```
mlir::Operation *op = ...;

op->dump();              // prints to stderr

op->print(llvm::outs()); // prints to stdout
```

It should also be possible to print to a file by using a `raw_fd_ostream` object.

**Note:** If printing something besides using dump, it may be worth ensuring the text is flushed using `std::endl` instead of `\n`. Otherwise, the extra text may be printed after the dumps are finished.

## Replace unsupported software operations

In case you are trying to see the compilation support for a network that has one or more unsupported software operations, there is a feature which can replace the operation with a dummy one. This way, the compilation can continue past the limitation of this missing operation, so that other potential issues can be identified.

The replacement of the unsupported software operations can be done by a compilation option. It should be placed in a configuration file, which is then passed to a tool such as `compile_tool` (e.g. using the `-c` argument). The configuration file should have the following content:

```
NPU_COMPILATION_MODE_PARAMS dummy-op-replacement=true
```

## Serializing canonical OV IR

In the frontend of the compiler, the OpenVINO model goes through a number of nGraph passes which perform some transformations over it. After these passes, the model gets parsed into the IE dialect from which the compiler starts running.

In order to dump the OpenVINO model that is obtained after the nGraph passes, use the following environment variable:

```sh
export NPU_SERIALIZE_CANONICAL_MODEL=1
```

## Strategy manager JSON

The strategy manager supports the possibility of dumping a JSON file with the decisions taken with regards to the multi-clustering & tiling strategies for each operation. Similarly, the per-operation strategies can be manually set using a JSON file.

To dump the strategy decisions, one of the following options can be used:

```
// Using a configuration file passed to the compilation tool, such as compile_tool, with the following content:
NPU_COMPILATION_MODE_PARAMS write-strategy-to-json

// Using the following environment variable, available for Debug & developer builds
export IE_NPU_WRITE_STRATEGY_JSON=1
```

The strategies will be dumped into a file called `strategy_out.json`. The name of the file can be controlled in Debug and developer builds using the `IE_NPU_WRITE_STRATEGY_JSON_LOC` environment variable.

For manually forcing strategies for the layers, the following options are available:

```
// Using a configuration file passed to the compilation tool, such as compile_tool, with the following content:
NPU_COMPILATION_MODE_PARAMS read-strategy-from-json

// Using the following environment variable, available for Debug & developer builds
export IE_NPU_READ_STRATEGY_JSON=1
```

The strategies will be read from a file called `strategy_in.json`, but the name can also be controller for Debug and developer builds using the `IE_NPU_READ_STRATEGY_JSON_LOC` environment variable.

## Compiler schedule trace

At the end of compilation, there is a way to dump the final schedule to a JSON trace file and print additional performance metrics. For more details refer to [how-to-get-schedule-trace-and-analysis.md](../../../../guides/how-to-get-schedule-trace-and-analysis.md) guide.

## Memory scheduler statistics

The memory scheduler (`feasible-allocation` pass) can generate a scheduling JSON trace file and provide additional statistics about the schedule. This option is available for Debug and developer builds and can be controlled using the following environment variable:

```
export IE_NPU_ENABLE_SCHEDULE_STATISTICS=1
```

The resulting `scheduleTrace.json` file will contain an entry for each task of the following form:

```JSON
{"name":"TASK_NAME", "cat":"NCE", "ph":"X", "ts":0, "dur":7175, "pid":0, "tid":0},
```

## Profiling inference performance

Please refer to [how-to-use-profiling.md](../../../../guides/how-to-use-profiling.md) guide.

## VSCode

There are some Visual Studio Code extensions which can help improve the workflow with MLIR.

### MLIR extension

The [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir) extension can enable IDE features when working with MLIR-specific files, such as `.mlir` ones encountered in LIT tests or `.td` ones for TableGen definitions.

#### MLIRLSP

[MLIRLSP](https://mlir.llvm.org/docs/Tools/MLIRLSP/) allows more complex IDE integration (adds diagnostics, definition and references finding, hover information and easier navigation). By default, MLIR includes a tool called `mlir-lsp-server` which is useful when working only with the builtin dialects. For more complex cases, such as ours, a dedicated tool must be created that registers the dialects and passes. In our compiler, a tool called `npu-lsp-server` exists. This tool can be configured to be used in the MLIR extension.

**Note:** some `.mlir` files may not work properly when LSP is configured in the extension. This was observed when dumping some intermediate IR that contains constants with opaque content (e.g. the content is elided), since these constant operations cannot be parsed.

### Natvis LLVM & MLIR

This can customize how variables are displayed in the debugger, can simplify debugging for complex C++ types.

LLVM offers an `llvm.natvis` file which supports a number of classes coming with LLVM. For example, `SmallVector` for which it can display the size and capacity in the variable watch. The file can be found at the `thirdparty/llvm-project/llvm/utils/LLVMVisualizers/llvm.natvis` path.

In order to use the natvis file in VSCode, add the following to the cppdbg launch.json configuration:

```
"visualizerFile": "${workspaceFolder}/thirdparty/llvm-project/llvm/utils/LLVMVisualizers/llvm.natvis",
```

MLIR also has a natvis file at the `thirdparty/llvm-project/mlir/utils/MLIRVisualizers/mlir.natvis` path, however it is very small for the moment.

## General notes

### Assertions

It is recommended to use a Debug build for development, since assertions are active. On Release builds, the assertions are skipped, so the compilation might have undefined behavior. Debugging crashes due to assertions is also easy in Debug builds since debuggers such as gdb or lldb can stop at the crash point and offer a stack trace.

### Valgrind

[Valgrind](https://valgrind.org/) is a powerful collection of tools which can help identify cases of memory mismanagement, threading issues, but also profile the code. It can be called by prefixing the target command with `valgrind`. For example:

```
valgrind ./compile_tool ...
```
