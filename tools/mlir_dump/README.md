MLIR Dump tools
===

This folder contains simple (yet powerful) scripts related to MLIR dump
processing.

The list of tools:
1. [IR dump splitter](#ir-dump-splitter)


In general, these tools modify large IR dump files in a particular way to
streamline debugging and/or development process. For the current NPU compiler,
running
```bash
export IE_NPU_IR_PRINTING_FILTER=.*
export IE_NPU_IR_PRINTING_ORDER=before_after
export IE_NPU_IR_PRINTING_FILE=tmp.mlir
./compile_tool ...
```
produces a single 'tmp.mlir' file that contains MLIR dumps before and after
every pass. For large-ish models, these files tend to become **huge** (from tens
of MBs to GBs).

"Modern" IDEs (e.g. VSCode) give up at about hundreds of MBs of text content and
stop rendering it, "modern" diff tools (e.g. VSCode, meld, kdiff3) give up at
about tens of MBs of text content to compare.

In order to successfully analyze IR dumps (e.g. compare "working" and "non
working" IR, look at IR after different passes, find a smaller bug reproducing
IR, etc.), something has to be done with the full IR dump. The tools presented
here achieve just that.

## IR dump splitter

The [IR dump splitter](./split_ir_dump.py) splits the IR dump produced by NPU
compiler (or other tools) into individual IR files. For example, given an IR
file 'tmp.mlir':

```mlir
// -----// IR Dump Before InitResources (init-resources) //----- //
func.func @main(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg : tensor<1x2xf32>
}
// -----// IR Dump After InitResources (init-resources) //----- //
func.func @main(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg : tensor<1x2xf32>
}
```

the splitter is going to produce 'tmp_before_InitResources_0000.mlir':
```mlir
// -----// IR Dump Before InitResources (init-resources) //----- //
func.func @main(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg : tensor<1x2xf32>
}
```
and 'tmp_after_InitResources_0000.mlir':
```mlir
// -----// IR Dump After InitResources (init-resources) //----- //
func.func @main(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg : tensor<1x2xf32>
}
```

The new file is added for every `// -----// IR Dump` section discovered. Consult
the full documentation of the script for more details and possible options.
