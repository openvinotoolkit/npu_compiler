# VPU3720 software layer enabling steps

## Introduction
This instruction will guide you through steps of adding a new VPU3720 software layer to the VPUX compiler.
> Be aware, that VPUX compiler is in a rapid development and code snippets might be out of date.

## Requirements

### IE and VPU dialect
The VPUX compiler has to contain a representation of operation in IE and VPU dialect.
If it does not have such then please follow ["MLIR software layer enabling"](sw_layer_enabling.md) instruction to include it.

### VPU op tiling

Software ops on VPU3720 need to have their data fit into NNCMX in order to execute. Therefore, they should be tiled into multiple smaller operations if they do not fit. For this to happen, every operation needs to have:
- the `VPU::TilingBuilderOpInterface` interface attached or inerhited;
- an implementation for the `VPU::TilingBuilderOpInterface::backInferTileInfo` method, which returns the information on the tiles of the input operands when given an output tile (i.e. a smaller part of the output);
- an implementation for the `VPU::TilingBuilderOpInterface::getTilingStrategy` methods, which returns the optimal output tiling scheme
fot this particular operation;
- the `VPU::TilingInfoOpInterface` interface attached or inherited;
- an implementation for the `VPU::TilingInfoOpInterface::isSupportedTiling` method, which returns whether the data used by the operation for a given output tile fits into memory; it generally makes use of the `backInferTileInfo` mentioned above to take the inferred input tiles into account.

The simplest case of enabling tiling for a software operation is when the operation is element-wise: one element in the output corresponds to one element in the input. In such cases, it is enough to have the operation inherit the two following interfaces in [src/vpux_compiler/tblgen/vpux/compiler/dialect/VPU/ops.td](../tblgen/vpux/compiler/dialect/VPU/ops.td):

```
VPU_TilingBuilderOpInterface,
VPU_EltwiseOp
```

The `VPU::EltwiseOp` interface comes with an implementation for the `backInferTileInfo` method that returns the input tile(s) equal to the output tile. Then, `VPU::TilingInfoOpInterface` can be attached to the operation in [src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp). Example:

```cpp
registry.addOpInterface<VPU::SigmoidOp, SwLayerTilingInfoOpModel<VPU::SigmoidOp>>();
```

`SwLayerTilingInfoOpModel` is an implementation of `VPU::TilingInfoOpInterface` for software layer tiling that contains dispatch methods for computing the NNCMX usage based on the operation type. For element-wise operations, a generic method adds up the size of the output and input tiles.

In case your operation is more complex, it might be necessary to provide a dedicated implementation for the `backInferTileInfo` method and/or the dispatch method used by `isSupportedTiling`. See `VPU.MemPermute` as an example.

### Tiling lit-test

To ensure that tiling is functional for your operation, a lit-test should be created. The recommended pass for validating it is `PrefetchTiling`, in the file associated with the VPU3720 platform: [tests/lit/VPUX37XX/dialect/VPU/passes/prefetch_tiling.mlir](../../../tests/lit/VPUX37XX/dialect/VPU/passes/prefetch_tiling.mlir).

### Tiling functional tests

To verify at runtime that the tiling logic is applied, a functional test case with large input values should be added. Example from the [Activation group](../../../tests/functional/shared_tests_instances/single_layer_tests/activation.cpp):

```cpp
std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicTiling = {{{1, 8, 80, 1280}, {{}}}};

```

For groups, such as Eltwise, Activation, Comparison etc. it is not mandatory to have functional test cases on the main developing branch for all of the operators that are enabled, as the tiling logic is the same and also to avoid overloading the CI. They should be tested locally beforehand.

An example would be the ```activationTypesTilingVPU3720 ``` variable, which does not contain all of the Activation operators:

```
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesTilingVPU3720 = {
        {Sigmoid, {{1.0f}}}, {Elu, {{1.0f}}},        {Sqrt, {{1.0f}}}, {Exp, {{1.0f}}},  {Clamp, {{-1.0f, 1.0f}}},
        {Tanh, {{1.0f}}},    {LeakyRelu, {{0.01f}}}, {Log, {{1.0f}}},  {Relu, {{1.0f}}}, {Negative, {{0.01f}}}};
```

 In case of an operator that is standalone, a test case where the tiling logic is tested should always be present. Please see Interpolate [example](../../../tests/functional/shared_tests_instances/single_layer_tests/interpolate.cpp).

### Kernel binaries
[act_shave_bin](../../../sw_runtime_kernels/kernels/prebuild/act_shave_bin) folder should contain the following data:
- sk.<entry point>.<platform>.data
- sk.<entry point>.<platform>.text

If not please follow this instruction: ["How to create act-shave kernel"](../../../sw_runtime_kernels/README.md)

## Single layer test

Current restrictions:
- VPU3720 does not support many important operations, such as Convert, Reorder, etc. which significantly reduces the set of test cases

Therefore, it is proposed to create a new test suite:

```cpp
class KmbActivationLayerTest_VPU3720 : public KmbActivationLayerTest {
};
```

In case of the OV2.0 test framework:

```cpp
class VPUXActivationLayerTest_VPU3720 : public VPUXLayerTestsCommon {
};
```

Call `setPlatformVPU3720` method to limit the scope of the tests to VPU3720 platform only:

```cpp
TEST_P(KmbActivationLayerTest_VPU3720, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceHardwareModeMLIR();
    Run();
}
```

The rest of the actions coincide with the testing of remaining SW/HW layers

## VPUIP Dialect
In VPUIP dialect all VPU3720 software layers are represented via `VPUIP::SwKernelOp` operation:
```cpp
[[SOFTMAX_OUT:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_SoftMax inputs([[VAR1]] : memref<1x1x1x1000xf16, "CMX_NN">) outputs([[VAR2]] : memref<1x1x1x1000xf16, "CMX_NN">) on tile 0 -> memref<1x1x1x1000xf16, "CMX_NN">  {
^bb0(%arg2: memref<1x1x1x1000xf16, "CMX_NN">, %arg3: memref<1x1x1x1000xf16, "CMX_NN">):
  %c3_i64 = arith.constant 3 : i64
  VPUIP.SW.Kernel.run(%arg2, %arg3, %c3_i64) : memref<1x1x1x1000xf16, "CMX_NN">, memref<1x1x1x1000xf16, "CMX_NN">, i64
}
```

This operation differs from the others in that it contains an internal region in which the kernel arguments are stored.
To convert operations from VPU dialect to VPUIP, follow next steps.

### Register interface for VPU
Register `SoftwareLayerOpInterface` interface for new operation:
[src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp)
```cpp
//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    // ...
    registry.addOpInterface<VPU::SoftMaxOp, SoftwareLayerOpModel>();
}
```
Remove corresponding operation from `setupExtraInterfacesAdditional`.
[src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp)

### Add kernel information
To serialize the kernel, you need to provide additional information about the arguments of the kernel, the name of entry point and source file. This information is stored in the structure:

[src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/ops_interfaces.hpp](../include/vpux/compiler/dialect/VPUIP/ops_interfaces.hpp)
```cpp
struct KernelInfo final {
    SmallVector<mlir::Attribute> args;
    SmallString entryName;
    SmallString sourceFileName;
};
```

Provide the necessary information:
[src/vpux_compiler/src/dialect/VPUIP/ops/sw_kernel.cpp](../src/dialect/VPUIP/ops/sw_kernel.cpp)
```cpp
VPUIP::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    return llvm::TypeSwitch<mlir::Operation*, VPUIP::KernelInfo>(origOp)
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp softmax) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{softmax.axisIndAttr()},
                                        {"singleShaveSoftmax"},
                                        {"single_shave_softmax.cpp"}};
            })
            .Default([](mlir::Operation* unknownOp) -> VPUIP::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}
```
