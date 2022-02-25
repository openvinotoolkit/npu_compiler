# MTL software layer enabling steps

## Introduction
This instruction will guide you through steps of adding a new MTL software layer to the VPUX compiler.
> Be aware, that VPUX compiler is in a rapid development and code snippets might be out of date.

## Requirements

### IE and IERT dialect
The VPUX compiler has to contain a representation of operation in IE and IERT dialect.
If it does not have such then please follow ["MLIR software layer enabling"](sw_layer_enabling.md) instruction to include it.

### Kernel binaries
[act_shave_bin](../../../sw_runtime_kernels/kernels/prebuild/act_shave_bin) folder should contain the following data:
- sk.<entry point>.<platform>.data
- sk.<entry point>.<platform>.text

If not please follow this instruction: ["How to create act-shave kernel"](../../../sw_runtime_kernels/README.md)

## Single layer test

Current restrictions:
- MTL does not support many important operations, such as Convert, Reorder, etc. which significantly reduces the set of test cases
- SW kernels are not compiled in precommit because the build system does not provide the location of precompiled kernels
- Single-layer tests are not inferred for MTL

Therefore, it is proposed to create a new test suite:

```cpp
class KmbActivationLayerTest_MTL : public KmbActivationLayerTest {
    void SkipBeforeLoad() override {
        if (std::getenv("OV_BUILD_DIR") == nullptr) {
            throw LayerTestsUtils::KmbSkipTestException(
                    "OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels.");
        }

#if defined(__arm__) || defined(__aarch64__) || defined(_WIN32) || defined(_WIN64)
        throw LayerTestsUtils::KmbSkipTestException("Does not compile on ARM and Windows.");
#endif
    }
    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Runtime issue.");
    }
};
```

In case of the OV2.0 test framework:  
```cpp
class VPUXActivationLayerTest_MTL : public VPUXLayerTestsCommon {
    SkipMessage SkipBeforeLoad() override {
        if (std::getenv("OV_BUILD_DIR") == nullptr) {
            return {"OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels."};
        }

#if defined(__arm__) || defined(__aarch64__) || defined(_WIN32) || defined(_WIN64)
        return {"OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels."};
#endif
    }
    SkipMessage SkipBeforeInfer() override {
        return {"Runtime issue"};
    }
};
```

Call `setPlatformMTL` method to limit the scope of the tests to MTL platform only:

```cpp
TEST_P(KmbActivationLayerTest_MTL, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setPlatformMTL();
    setReferenceHardwareModeMLIR();
    Run();
}
```

The rest of the actions coincide with the testing of remaining SW/HW layers

## VPUIP Dialect
In VPUIP dialect all MTL software layers are represented via `VPUIP::SwKernelOp` operation:  
```cpp
[[SOFTMAX_OUT:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_SoftMax inputs([[VAR1]] : memref<1x1x1x1000xf16, "CMX_NN">) outputs([[VAR2]] : memref<1x1x1x1000xf16, "CMX_NN">) on tile 0 -> memref<1x1x1x1000xf16, "CMX_NN">  {
^bb0(%arg2: memref<1x1x1x1000xf16, "CMX_NN">, %arg3: memref<1x1x1x1000xf16, "CMX_NN">):
  %c3_i64 = arith.constant 3 : i64
  VPUIP.SW.Kernel.run(%arg2, %arg3, %c3_i64) : memref<1x1x1x1000xf16, "CMX_NN">, memref<1x1x1x1000xf16, "CMX_NN">, i64
}
```

This operation differs from the others in that it contains an internal region in which the kernel arguments are stored.
To convert operations from IERT dialect to VPUIP, follow next steps.

### Register interface for IERT
Register `SoftwareLayerOpInterface` interface for new operation:  
[src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp)
```cpp
//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    // ...
    registry.addOpInterface<IERT::SoftMaxOp, SoftwareLayerOpModel>();
}
```

### Add kernel information
To serialize the kernel, you need to provide additional information about the arguments of the kernel, the name of entry point and source file. This information is stored in the structure:

[src/vpux_compiler/include/vpux/compiler/dialect/IERT/ops_interfaces.hpp](../include/vpux/compiler/dialect/IERT/ops_interfaces.hpp)
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
IERT::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    return llvm::TypeSwitch<mlir::Operation*, IERT::KernelInfo>(origOp)
            .Case<IERT::SoftMaxOp>([&](IERT::SoftMaxOp softmax) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{softmax.axisIndAttr()},
                                        {"singleShaveSoftmax"},
                                        {"single_shave_softmax.cpp"}};
            })
            .Default([](mlir::Operation* unknownOp) -> IERT::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}
```