# Profiling of NPU compiler

One aspect that the developer should be aware of when developing the NPU
compiler is model compilation time. In order to ease compilation time
improvements, several profiling "tools" and utilities exist. This document
describes the tools available in NPU compiler.

Typically, a profiling tool can be enabled via compile_tool config:
`NPU_COMPILATION_MODE_PARAMS compiler-profiler-tool=<tool with extra options>`.

The general idea behind all of the tools is that they rely on the [MLIR Action
Tracing framework](https://mlir.llvm.org/docs/ActionTracing).

> Note: all of the profiling tools work on top of a **developer build**. At the
> moment, this is intentional, as there's no expectation that anyone would be
> interested in profiling a non-developer build.

## Table of contents

1. [Profiling of NPU compiler](#profiling-of-npu-compiler)
2. [Table of contents](#table-of-contents)
3. [MLIR built-in solution](#mlir-built-in-profiler--tracer)
4. [Callgrind-based profiler for Linux](#callgrind-based-profiling-linux)
5. [VTune profiler](#vtune-profiler)

## MLIR built-in profiler / tracer

Out of the box, MLIR provides a very basic capability to report timings of
various actions that happen during compilation. The report itself is a JSON file
which is compatible with `chrome://tracing` and [Perfetto
UI](https://perfetto.dev/) visualizers, which makes it easy enough to analyze
the high-level bottlenecks.

In order to set up this profiler, one needs to:
* Specify the MLIR profiler tool in the config (the tool requires an output path
  where the JSON report would be placed):
```
NPU_COMPILATION_MODE_PARAMS compiler-profiler-tool=mlir-profiler=path=/work/dump.json
```
* Run `compile_tool` using such a config:
```
RelWithDebInfo/compile_tool -d NPU.5010 -c config_with_mlir_profiler.conf -m resnet-50.xml
```

Once compilation ends, there should be a `/work/dump.json` file generated that
contains the timings for all the actions. For a model compilation, one can
expect a result similar to the picture below (when visualized using
`chrome://tracing`): ![Example of a report](./images/mlir_profiler_example.png)

## Callgrind-based profiling (Linux)

On Linux-based systems, [Valgrind](https://valgrind.org/) is a ready-made "tool
box" with a set of profiling tools that can be used by developers for various
tasks. One of such tools,
[callgrind](https://valgrind.org/docs/manual/cl-manual.html), is capable of
generating quite sophisticated profiling information for further analysis. On
top of this, callgrind provides a way to configure ["intrusive"
profiling](https://valgrind.org/docs/manual/cl-manual.html#cl-manual.clientrequests),
that is, for the user program to call special APIs from callgrind in order to
specify profiling data from which regions to collect. This API is used in the
NPU compiler in order to be able to "filter out" the full profile and instead
collect only the necessary parts of the program execution.

To enable this profiler, one needs to:
* Specify "callgrind" as a compilation profiler in the config:
```
NPU_COMPILATION_MODE_PARAMS compiler-profiler-tool=callgrind=regex=run-ngraph-passes|convert-shape-to-4d
```
This command would collect profiling information for two "actions": all of
NGraph passes and `convert-shape-to-4d` pass, ignoring everything else. For most
use cases, one is expected to profile a single pass at a time, but there could
be situations where several passes are of interest simultaneously.
* Additionally, one can generate multiple dump files where each dump is
  dedicated to a specific profiled action. This could be especially useful when
  profiled action is run multiple times during compilation (for example, a pass
  runs twice and only second run is problematic):
```
NPU_COMPILATION_MODE_PARAMS compiler-profiler-tool=callgrind='regex=convert-shape-to-4d separate-dumps=1'
```
> Note: use `callgrind_annotate` to understand what specific dump is about.
```
$ callgrind_annotate callgrind.out.2860728.1

--------------------------------------------------------------------------------
Profile data file 'callgrind.out.2860728.1' (creator: callgrind-3.22.0)
--------------------------------------------------------------------------------
I1 cache:
D1 cache:
LL cache:
Timerange: Basic block 0 - 180405
Trigger: Client Request: ConvertShapeTo4D_1
...
```

* Run `compile_tool` using such a config **via valgrind**:
```
LD_PRELOAD=libopenvino_intel_npu_compiler.so valgrind --tool=callgrind --collect-atstart=no --instr-atstart=no RelWithDebInfo/compile_tool -d NPU.5010 -c config_with_callgrind_profiler.conf -m resnet-50.xml
```
> **Important**: Due to the way NPU stack works (there's a dynamically loaded
> plugin + compiler), `LD_PRELOAD` is an **essential** workaround for Valgrind
> to successfully map sampled data to the source code locations. Without this,
> Valgrind seems unable to "find debug symbols for NPU compiler" (this seems to
> have something to do with `dlopen()` / `dlclose()` mechanism).

> Note: `--collect-atstart=no` and `--instr-atstart=no` are additional knobs to
> ensure callgrind does not collect any data before compilation proceeds to the
> "point of interest".

* Visualize the produced profiling output with any tool that can read
  callgrind's output files, for example,
  [kcachegrind](https://kcachegrind.github.io/html/Home.html):

![Information on NGraph passes](./images/callgrind_profiler_visualization_1.png)

![Information on ConvertShapeTo4D](./images/callgrind_profiler_visualization_2.png)

* When multiple dump files are generated, `kcachegrind` is able to visualize
  their statistics "together" when the main program file is opened. For example,
  opening a `callgrind.out.2860728` (where two extra dumps are
  `callgrind.out.2860728.1` and `callgrind.out.2860728.2`) with profiling data
  for `convert-shape-to-4d` calls (there are two calls in the observed program)
  could look the following way:

![Information on multiple separate calls to
ConvertShapeTo4D](./images/callgrind_profiler_visualization_multidump.png)

## VTune profiler

The VTune profiler integrates with the compiler through MLIR action tracing and
Intel ITT instrumentation. It records compiler actions as VTune tasks, which
can then be inspected in the VTune GUI or exported as reports.

To enable it, pass the VTune profiler through the compiler config:
```
NPU_COMPILATION_MODE_PARAMS compiler-profiler-tool=vtune='regex=.*'
```

The `regex` value selects which MLIR actions are traced. An empty regex is also
valid and means that all actions are traced.

Typical usage is:
1. Build a developer configuration of the compiler.
2. Run the model compilation under VTune collection.
3. Open the generated result in the VTune GUI and inspect the task tree.

In VTune, the compiler actions appear in the task-oriented views, such as
Bottom-up or Top-down tree. The screenshot below shows the tasks grouped by
action name, along with task time and task count.

![VTune profiling results](./images/vtune_profiling.png)

If you want to narrow the captured actions while debugging, use a more specific
regex such as `regex=InitResources|FeasibleAllocation|ConvertShapeTo4D`.
