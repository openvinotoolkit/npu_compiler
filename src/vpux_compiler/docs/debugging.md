# Debugging Technics

**Note:** Some of the Debugging Technics works only in Developer Build mode.
To turn it on add `-D ENABLE_DEVELOPER_BUILD=ON` CMake option to the plugin build.

## vpux-translate and vpu-opt Tools

These two tools can be used to call specific parts of the compiler (frontend, backend, passes) from the command line.

`vpux-translate` is a CMD wrapper for compiler frontend and backend.
It allows to:

* Convert InferenceEngine XML IR to MLIR in **IE Dialect** (`vpux-translate --import-IE <path to xml> -o <MLIR file name>`).
* Convert MLIR in **VPUIP Dialect** (`vpux-translate --export-VPUIP <path to MLIR file> -o <graph blob file name>`).

`vpux-opt` is a CMD wrapper for compiler passes.
It allows to run single pass or a sequence of passes on MLIR file.
For example, `vpux-opt --lower-IE-to-IERT IE.mlir -o IERT.mlir`.

For all available passes please check `vpux-opt --help` output.

## Adding and printing Op names

Compile network with --mlir-print-debuginfo flag:
`./vpux-translate --import-IE <xml path> --mlir-print-debuginfo -o net.mlir`
`./vpux-opt --set-compile-params="vpu-arch=VPU3400_A0" --reference-mode net.mlir --mlir-print-debuginfo -o net_out.mlir`
To print names in the code use:
```cpp
if (const auto loc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
    //Option #1
    StringRef name = loc.getName().strref();
    std::cout << loc.getName().strref().data() << std::endl;
    //Option #2
    loc.getName().dump();
    std::cout << std::endl;
}
```

## Generating MLIR without big constants

`./vpux-opt --mlir-elide-elementsattrs-if-larger 8 net.mlir -o net_user.mlir`

## Generating of Dot graph

There is a pass which will generate Dot graph during pipeline execution.
1. Generation using enviroment variable in the DEVELOPER_BUILD:
  Provide standard print-dot pass options to the `IE_VPUX_PRINT_DOT`, if you want couple of entries separate them using comma.
  Example: `export IE_VPUX_PRINT_DOT="output=temp.dot pass=OptimizeAsyncDeps,output=temp2.dot pass=AddBuffersForNetResults"`
2. Generation using vpux-opt tool with option `--print-dot`:
  With this option user can specify the output file and additional generation options(see `vpu-opt -h`). Note: option `pass` is not available in this mode.
  Example: `--print-dot="output=dot1.dot"`
3. Via compiler pipeline. Add printDot pass into compiler pipline in the code and rebuild the project.
  Example: `pm.addPass(createPrintDot(<FileName>, <Options>));` to generate file with <FileName> name.

## IR dumping (Developer build)

The **VPUX NN Compiler** allows to dump Internal Representation before/after selected Passes.
The feature is enabled with `IE_VPUX_IR_PRINTING_FILTER` environment variable.
It should be set to POSIX Regex filter for pass argument name (see `vpux-opt --help`).
For example, `export IE_VPUX_IR_PRINTING_FILTER=convert-.*-to-VPUIP`.

`IE_VPUX_PRINT_FULL_IR` environment variable controls the scope of printing:

* `IE_VPUX_PRINT_FULL_IR=0` (default) - only the affected scope will be printed after each pass (for example, single Function).
* `IE_VPUX_PRINT_FULL_IR=1` - full IR from Module level will be printed. This mode disables multi-threading in compiler.

## Crash reproducer (Developer build)

The **VPUX NN Compiler** allows to dump Internal Representation after the failed Pass.
The feature is enabled with `IE_VPUX_CRASH_REPRODUCER_FILE` environment variable.
It should be set to the output file path.

`IE_VPUX_GEN_LOCAL_REPRODUCER` environment variable controls the scope of printing:

* `IE_VPUX_GEN_LOCAL_REPRODUCER=0` - full IR from Module level will be printed.
* `IE_VPUX_GEN_LOCAL_REPRODUCER=1` (default) - only the affected scope will be printed.

## Pass Logging (Developer build)

The Logging messages during passes can be enabled with `IE_VPUX_COMPILER_LOG_FILTER` environment variable.
It should be set to Posix Regex filter for pass argument name (see `vpux-opt --help`).
For example, `export IE_VPUX_COMPILER_LOG_FILTER=convert-.*-to-VPUIP`.

## Pass Timing

The **VPUX NN Compiler** can print the Pass performance information.
It is printed via Logger at `INFO` level, so it will be visible, if that level is enabled.
Also it can be enabled with `export IE_VPUX_COMPILER_LOG_FILTER=vpux-compiler`.

## VS Code extension

There is a [MLIR](https://mlir.llvm.org/docs/Tools/MLIRLSP/) extension by LLVM Extensions provided for VS Code with extra MLIR code analysis support.

To enable full feature support set `Mlir:Server_path` parameter in the extension configuration to `vpux-lsp-server` application.
The application is built as a part of the plugin build and can be found in OpenVINO binaries directory (for ex.: \<openvino-dir\>/bin/intel64/Debug/vpux-lsp-server).

**Note:**

* LSP stands for [Language Server Protocol](https://microsoft.github.io/language-server-protocol/).
* If you have [MLIR Highlighting for VSCode](https://marketplace.visualstudio.com/items?itemName=MomenAbdelkarim-WyattCalandro-LuisPrieto.mlir) extension by Momen Abdelkarim-Wyatt Calandro-Luis Prieto installed - remove it.
