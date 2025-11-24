# Project structure

This document is written on the basis of discussions taken as part of the task of preparing the compiler for development in open-source. This means that (debatable) decisions on the structure of the project structure were made based on the following conditions:
- Enable/disable a specific platform by a CMake option and a corresponding define;
- Make a clear, convenient process of adding a new device;
- Ensure code reuse for different device generations.

## Compiler overview

### Dialects

![NPU compilation pipeline](images/compilation_flow.png)

Regardless of the device version, the compilation flow has the same appearance at the dialect level. These dialects represent different levels of detail. The IR is lowered from high level abstractions to more detailed representation step-by-step during compilation. The compilation pipeline consists of the "atomic“ passes. Each pass in compilation pipeline must represent one single transformation to reach one specific goal (either IR adaptation or IR optimization). More information is available from the [Compiler HLD](https://docs.intel.com/documents/MovidiusExternal/vpu27/Common/SW/HLD/external/VPUX_NN_Compiler.html) or the [presentation](https://videoportal.intel.com/media/0_dnxf87in).

It is also necessary to describe the dependence of dialects from an architectural point of view:

![Dialect dependencies](images/dialects.png)

Only a low-level dialect might depend on a high-level one, but not vice versa. Ideally, based on [DIP](https://en.wikipedia.org/wiki/Dependency_inversion_principle), they should all depend on abstraction. A simple example is shown in the diagram above: `AdjustLayoutsPass` pass is written in a general manner and depends on the `LayoutInfoOpInterface` interface, which is implemented in the VPU dialect. Thus `AdjustLayoutsPass` is protected from changes in HW details and can easily be reused. This example is somewhat simplified relative to the actual implementation. More information can be found in [MLIR Overview](https://mlir.llvm.org/docs/Interfaces) and [below](#operation-interfaces).

### Libraries

![Library dependencies](images/libraries.png)

This high-level diagram covers the main dependencies between libraries inside the compiler. It makes sense to divide libraries into two "types": common and HW-specific.
Common part consists of:
- frontend: to import NGraph to IE dialect
- core: data structures required by compiler
- utils: helpers to work with core data structures
- act_kernels: shave utilities
- conversion: passes for lowering dialects
- [dialect]_IR: dialect operations, attributes and types
- [dialect]_transforms: passes to perform transformations over IR
- [dialect]_interfaces: interfaces and base classes on which passes may depend
- other utility libraries

HW-specific part consists of implementation of interfaces, passes, operations and other device-specific details/utilities. There is one library for each device version. For convenience, the diagram shows it in the form of separate libraries, so `npu_compiler_[dialectN]37xx` means the dialect folder in the [NPU37XX](../../include/vpux/compiler/NPU37XX) directory.

## Passes

### Common passes

These are fully HW-agnostic passes. This means you will get the same result for any input IR regardless of the platform version. Such passes have to be placed in a common part. Please refer to [primer_mlir](primer_mlir.md#passes) to get more information.

### HW-specific passes

Hardware specific passes are designed to work on a particular platform. And from development perspective, the only difference is that necessary to use appropriate HW folder. For example, 37XX-specific passes for IE dialect:
- Declaration [path](../../tblgen/vpux/compiler/NPU37XX/dialect/IE/passes.td) in TableGen;
- Declaration [path](../../include/vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp) for constructor;
- Implementation [folder](../../src/NPU37XX/dialect/IE/transforms/passes).

You are allowed to reuse passes from an older HW version for a newer one if the required feature is a strict superset:

```C++
// 40XX
void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions40XX& options, Logger log) {
    // ...
    // Use pass from previous version here
    pm.addPass(IE::arch37xx::createHwSpecific1Pass(log));
    IE::buildName1Pipeline(pm, log);
    // ...
}
```

HW-specific passes must also be registered in [vpux-opt](../../../../tools/vpux-opt/vpux-opt.cpp) for validation purposes:

```C++
// ...
vpux::IE::arch37xx::registerPasses();
// ...
```

### "Mixed" passes

Mixed passes share a common core algorithm but utilise hardware specific information to make decisions.

#### Interface-based approach

Lets say we have `StrategyManager` pass in VPU dialect that can be applied for all HW generations. At the same time, the general algorithm from this pass needs information about possible strategies that are different for different devices. So we have to store strategies separately for HW components, because, for example, for the newest device it is private information.

![Interface-based approach scheme](images/interface_based.png)

Following this approach, the development of a "mixed" pass is similar to a common pass. The difference here is that we have to create a concrete instance of the corresponding type in the common part, using, for example, the factory method:

```C++
std::unique_ptr<IStrategyGetter> vpux::VPU::createMCStrategyGetter(ArchKind arch, int64_t numClusters) {
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return std::make_unique<arch37xx::StrategyGetter>();
    }
    case config::ArchKind::NPU40XX: {
        return std::make_unique<arch40xx::StrategyGetter>(numClusters);
    }
    case ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unsupported arch kind value: {0}", arch);
    }
    }
}
```

This approach does not have the disadvantages of [rejected option](#interface-based-approach-rejected). But it has its downsides:
- A large number of factory methods need to be created. However, this problem can be mitigated by creating some sort of global register, like a DI container in C# or Java.
- Removing the module requires more effort, as the common part link the hardware library(CMake changes), and also need to remove code from factories(see previous point). The problem related to dependencies between libraries can be solved by switching to a plugin system, then we could load the necessary libraries in runtime depending on the arch value.

Please note that despite the dependence of the common part(`npu_compiler_dialect_passes_vpu`) on the HW-specific one(`npu_compiler_vpu37xx`), by design, classes do not depend on it. Here `StrategyManagerPass` depends on interface `IStrategyGetter` and `arch37xx::StrategyGetter` implements this — so both components depend on abstraction and we still follow [DIP](https://en.wikipedia.org/wiki/Dependency_inversion_principle).

This approach is adopted as the main one, as it reduces duplication and decreases the probability of errors in comparison with the [rejected option](#interface-based-approach-rejected).

#### Rewriter-based approach

![Rewriter-based approach scheme](images/rewriter_base.png)

In this example, the different behavior of the pass for different HWs is achieved by adding special rewriters. To do this, we use the interface again: `UnrollDistributedOpsPass` depends on [`IGreedilyPassStrategy`](../../include/vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp). And here is possible implementation of `UnrollDistributedOpsPass::safeRunOnFunc` method:

```C++
void UnrollDistributedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto strategy = createUnrollDistributedOpsStrategy(func, _log);

    mlir::RewritePatternSet patterns(&ctx);
    // add necessary rewriters here
    strategy.addPatterns(patterns);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
```

where `strategy` is `IGreedilyPassStrategy` and it can be implemented in different ways, depending on the version of the device:

```C++

// 37XX
void UnrollDistributedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns) {
    auto module = _func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    patterns.add<VPUIP::ClusterDMARewriter>(&_ctx, dmaPortCount, _log);
    patterns.add<VPUIP::arch37xx::ClusterSWRewriter>(&_ctx, module, _log);
    patterns.add<VPUIP::arch37xx::ClusterNCERewriter>(&ctx, _log);
}

// 40XX
void UnrollDistributedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns) {
    auto module = _func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    patterns.add<VPUIP::ClusterDMARewriter>(&_ctx, dmaPortCount, _log);
    patterns.add<VPUIP::arch37xx::ClusterSWRewriter>(&_ctx, module, _log);
    // Compared to the 37xx, we have specific ClusterNCERewriter here
    patterns.add<VPUIP::arch40xx::ClusterNCERewriter>(&_ctx, _log);
    // Compared to the 37xx, we have also ClusterConvertDMARewriter here
    patterns.add<VPUIP::arch40xx::ClusterConvertDMARewriter>(&ctx, dmaPortCount, _log);
}
```

[`IConversionPassStrategy`](../../include/vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp) also provides a  `markOpLegality` method, useful for setting up operation legality in passes which rely on the dialect conversion driver.

Rewriters can also depend on interfaces to write them in the most general form — kind of combination with [Interface-based approach](#interface-based-approach). In this case, the necessary objects can be created directly in `addPatterns` method.
This approach also helps reducing code duplication since it doesn't require passes to be registered for each device. Then we can use the same name in `vpux-opt` and manage behavior of pass using only `vpu-arch`:

```MLIR
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=NPU40XX allow-custom-values=true" --unroll-distributed-ops  %s | FileCheck %s
```

instead of, for example, duplicating the device version in the pass name:

```MLIR
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=NPU40XX allow-custom-values=true" --unroll-distributed-ops-VPUX40XX  %s | FileCheck %s
```

More detailed information about vpux-opt can be found in the [how-to-test](../../../../guides/how-to-test.md) document.

### Canonicalization

TODO: #-86282

## Pipelines

Compiler has different pipeline for different HW generation. These pipelines are stored in appropriate HW folders: [NPU37XX](../include/vpux/compiler/NPU37XX/pipelines.cpp), etc. To build a pipeline, it is also necessary to implement `IPipelineStrategy` interface for each device:

![Pipeline strategy class diagram](images/pipeline.png)

Then it is used in this way:

```C++
auto pipelineFactory = createPipelineStrategy(arch);
// pm is PassManager
pipelineFactory->buildPipeline(pm, config, rootTiming, log);
```

The main advantage of this approach is that we can easily hide the pipeline for a new device containing HW-specific passes. The consequence of this separation is that there is no need to add passes to the pipeline that do not work with this device. Therefore, the size of the pipeline becomes smaller, only the necessary passes are involved. And it is possible to get rid of such code:

```C++
void MyPass::safeRunOnFunc() {
    // ...
    if (arch != config::ArchKind::NPU37XX) {
        return mlir::failure();
    }
    // ...
}
```

This approach also has a downside. It is not clear why this or that pass participates in one pipeline, but not in another. Are there HW restrictions or did developer forget to add it? A possible solution is to introduce as many sub-pipelines as possible to bring the main pipeline to a similar form:

```C++
// Only sub-pipelines and HW-specific passages should remain in the main pipeline

// 37XX
void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options, Logger log) {
    // ...
    IE::buildName1Pipeline(pm, log);
    IE::buildName2Pipeline(pm, log);
    pm.addPass(IE::arch37xx::createHwSpecific1Pass(log));
    IE::buildName3Pipeline(pm, log);
    // ...
}

// 40XX
void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions40XX& options, Logger log) {
    // ...
    IE::buildName1Pipeline(pm, log);
    pm.addPass(IE::arch40xx::createHwSpecific2Pass(log));
    IE::buildName2Pipeline(pm, log);
    IE::buildName3Pipeline(pm, log);
    // ...
}
```

Some [recommendations](../code_style.md#pipelines-and-passes) are already written in code style.

## Operation interfaces

[Interfaces](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces) and [External models](https://mlir.llvm.org/docs/Interfaces/#external-models-for-attribute-operation-and-type-interfaces) are powerful tools that allow us to add the necessary behavior for operations in runtime. A typical example is the [AdjustLayoutsPass](../../src/dialect/IE/transforms/passes/adjust_layouts.cpp) pass. It works with the [IE::LayoutInfoOpInterface](../../tblgen/vpux/compiler/dialect/IE/ops_interfaces.td) interface. For the same operation from IE dialect we want to have different results depending on the device version. For this purpose, different models can be implemented and then are attached for the same operation depending on device version:

```C++

// 37XX:
IE::SigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
```

Interfaces registration follows the same schema as the pipelines registration:

![Interface registry class diagram](images/interface.png)

```C++
auto interfacesRegistry = createInterfacesRegistry(arch);
interfacesRegistry->registerInterfaces(registry);
```

## Properties

TODO: #-66795. Store properties in module; Handle properties in passes.

## Operations

![Sets of operations](images/operations.png)

TODO: #-86281

There is no complex solution here yet. As a first step, operations are devided between several `ops.td` files depending on the HW version. And the logic of transformations again is based on op-interfaces.

In future we could proceed with HW-specific dialects if necessary:
- VPUIP37XX_SwKernelOp
- VPUIP40XX_ConvertDMAOp
- ..

For example, we already have HW-specific dialects like [VPUMI37XX](../../tblgen/vpux/compiler/dialect/VPUMI37XX/dialect.td).

## Attributes

TODO: #-88494

## Rationale

### "Mixed" passes

#### Interface-based approach (rejected)

![Rejected interface-based approach scheme](images/interface_based_rejected.png)

Here in common part we have `StrategyManagerImplAlgo` class (it can also be a method, but it doesn't really matter), which contains the basic general logic. This class depends on the interface to be specified by HW details, in our case, a specific set of strategies.
This scheme requires the developer to register a pass for each platform:

```MLIR
// src/vpux_compiler/tblgen/vpux/compiler/NPU37XX/dialect/VPU/passes.td
// The same for 40XX
def StrategyManagerPass : PassBase<"strategy-manager", "mlir::OperationPass<mlir::func::FuncOp>"> {
    // ...
    let constructor = "vpux::IE::arch37xx::createStrategyManagerPass()";
    // ...
}
```

The implementation of HW passes is also duplicated for each platform. Possible way:

```C++
void StrategyManagerPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    // in case of 40XX we have to create arch40xx::StrategyGetter
    StrategyManagerImplAlgo algo {func, std::make_unique<arch37xx::StrategyGetter>();}
    algo.foo();
}
```

Then we will have the difference in compilation pipelines:

```C++

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options, Logger log) {
    // ....
    // Accordingly, it will be arch40xx::createStrategyManagerPass for 40XX, etc.
    pm.addPass(VPU::arch37xx::createStrategyManagerPass(log));
    // ...
}
```

The advantage of this approach is that the HW library itself creates the necessary dependencies for the generic algorithms, and therefore minimal changes are required to remove such a library from the repository: platform libraries depend on the generic part, and not vice versa.

At the same time there are several cons:
- Code duplication for declaration and implementation of the pass;
- Impossible to reuse sub-pipelines: we can't have common sub-pipeline for 37xx and 40xx with this pass;
- It is easy to make a mistake when registering passes for vpux-opt. You get an error when trying to register passes for from 37XX and 40XX at the same time, because two passes are registered with the same name.

## Dispatched Inlining

### Motivation

MLIR's inliner will be called as part of the inliner pass. It does a lot behind the scenes, but when it comes to deciding if an operation can be inlined and how it shall be inlined, it is quite simple.

As an example, we take a look at `isLegalToInline()`: If it encounters some operation, it will lookup the dialect. Internally, the inliner saves a mapping from `mlir::Dialect*` to `mlir::DialectInlinerInterface`. If the map contains no such interface for the requested dialect, `false` will be returned. Otherwise the inliner dispatches to that particular inliner interface and return `interface->isLegalToInline(op)`. There are a handful of other functions that work in the same fashion.

We see two important things here: The inliner can only decide on a per-operation basis which interface to choose and the inliner only supports at most one interface per dialect. This means that the inliner **cannot** support multiple different inlining semantics for a particular operation out of the box! This is the motivation for our dispatched inliner interface system.

Our solution is to implement a `func` inliner interface that can then dynamically dispatch to other interfaces. The idea is that the user can add a special attribute, namely `{inliner_dispatch = #MyDialect.MyInlinerDispatchAttr}`, to `func` operations. `UnifiedFuncInlinerInterface` then dispatches to an inliner interface that is associatated with `#MyDialect.MyInlinerDispatchAttr`.

### Tutorial

#### Supporting operations of a custom dialect in the inliner

The common approach here is extending `mlir::DialectInlinerInterface` and implementing `isLegalToInline()`. The most trivial implementation looks like this:
```cpp
struct MyDialectInlinerInterface : public mlir::DialectInlinerInterface {
    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};
```
This then has to be registered during `MyDialect::initialize()`:
```cpp
void MyDialect::initialize() {
    // ...
    addInterface<MyDialectInlinerInterface>();
    // ...
}
```

This is enough to enable inlining in `MyDialect` if the default inlining behaviour (see `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp`) is enough.

#### Supporting custom call (and func, return) operations

Assume we want to implement a custom `MyDialect.Call` operation. It extends `CallOpInterface` and will therefore be handled by `UnifiedFuncInlinerInterface`. If we **don't** want to have the default behaviour (see `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp`) for that kind of operation, we can extend `mlir::DialectInlinerInterface`.
```cpp
struct MyDialectDispatchedInlinerInterface : public mlir::DialectInlinerInterface {
    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        // custom logic
    }

    void processInlinedCallBlocks(mlir::Operation* call,
                                  mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const final {
        // custom logic
    }

    std::tuple<mlir::Block*, mlir::Block::iterator> getInlineBlockAndPoint(mlir::Operation* call) const final {
        // custom logic
    }

    void eraseCall(mlir::Operation* call) const final {
        // custom logic
    }
};
```
Additionally, we have to add an attribute to `MyDialect`. This attribute will be added to the func-like operations in `MyDialect` to tell `UnifiedFuncInlinerInterface` which interface to dispatch to.
```tablegen
def MyDialectInlinerDispatchAttr : InlinerDispatchAttr<MyDialect, "MyDialectInlinerDispatch">;
```
```mlir
MyDialect.Call {inliner_dispatch = #MyDialect.MyInlinerDispatchAttr} @someFunction() -> ()
// or if we just want to have different semantics for func ops
func.call {inliner_dispatch = #MyDialect.MyInlinerDispatchAttr} @someFunction() -> ()
```
We then have to register this interface in the `UnifiedFuncInlinerInterface`:
```cpp
void MyDialect::initialize() {
    // ...

    // support inlining for "normal" ops in MyDialect
    addInterfaces<MyDialectInlinerInterface>();

    // support for func-like ops in MyDialect
    auto funcDialect = getContext()->getLoadedDialect<mlir::func::FuncDialect>();
    assert(funcDialect != nullptr);

    auto interface = funcDialect->getRegisteredInterface<Core::UnifiedFuncInlinerInterface>();
    assert(interface != nullptr);

    interface->registerDispatchedInlinerInterface<MyDialect::MyDialectInlinerDispatchAttr, MyDialect::FuncInlinerInterface>();
}
```

Note: If no dispatched inliner interface is provided via `registerDispatchedInlinerInterface`, a fallback implementation which mirrors `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp` is used! For a lot of use-cases this is enough as the default inlining behaviour is the desired one.

## HostCompile Compilation Pipeline

The HostCompile pipeline is a specialized compilation mode designed to partition a neural network into multiple independently compilable functions. Each function contains NPU code, which is subsequently compiled into separate ELF blobs, along with the main function, which contains CPU host code that manages these compiled blobs using the LevelZero API.

<img src="images/1_ir_levels_HostCompile.drawio.svg" alt="drawing" width="600"/>

### Overview

In the HostCompile pipeline, the network is divided into kernel functions and host functions. Kernel functions contain the NPU-specific code and are compiled into ELF blobs as usual. Host function, on the other hand, consist of operations from MLIR upstream dialects such as tensor, scf, async, memref and arith. This code is later compiled into CPU code responsible for orchestrating the execution of kernel functions, passing various input data, and aggregating their outputs.

### Example

Consider the following example:

```mlir
module @StaticEltwiseNHWC attributes {config.arch = #config.arch_kind<NPU40XX>, config.revisionID = #config.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
  module @Module_1 {
    // function which contains the NPU-specific code and supposed to be compiled into ELF blobs
    func.func private @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
        return %0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
    }
  }
  // Host function which will be compiled into CPU code responsible for orchestrating the execution of kernel functions
  func.func @main(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %dim_0 step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim_0]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = Core.NestedCall @Module_1::@main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }
}
```
In the HostCompile pipeline, as demonstrated in the example, the code is distinctly divided into NPU kernel functions, such as `@Module_1::@main_func0`, and the host function `@main`, which orchestrates their execution.
This contrasts with the `DefaultHW` pipeline, where the compilation process generates a single ELF blob that encapsulates all NPU code, without including any CPU code for orchestration.

### How to use HostCompile mode

#### Locally

To compile a network end to end with `HostCompile` pipeline use one of the following options:

- Provide `NPU_COMPILATION_MODE` environment variable while using compile_tool:

`NPU_COMPILATION_MODE="HostCompile" ./compile_tool -d NPU.4000 -m ./net.onnx -o ./net.blob  -shape [1,3,4..6,7..10]`
- Create config file, specify compilation mode and use this config for compilation:

`./compile_tool -d NPU.4000 -m ./net.onnx -o ./net.blob -c ./extra_config_net.conf -shape [1,3,4..6,7..10]`

Below is the content of the `extra_config_net.conf` file

```plaintext
NPU_COMPILATION_MODE HostCompile
```



#### In CI
Specify "NPU_COMPILATION_MODE": "HostCompile" in `extra_config` section of JSON config file:
```json
  "networks": [
    {
      "name": "Model",
      "ir": "net.onnx",
      "category": "CID/precommit/VPU4000",
      "extra_config": {
        "NPU_PLATFORM": "VPU4000",
        "NPU_COMPILATION_MODE": "HostCompile"
      },
      "Compile": {
        "shape": "[1,3,10..1600,10..2560]"
      },
      "BenchmarkApp": {
        "disabled": true
      },
      "NetTest-CalcRef": {
        "disabled": true
      },
      "NetTest-Validate": {
        "disabled": true
      },
      "AccuracyCheck": {
        "disabled": true
      }
    }
  ]
```
