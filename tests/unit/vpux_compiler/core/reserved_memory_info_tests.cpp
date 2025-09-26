//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/reserved_memory_info.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_ArgAllocationInfo = MLIR_UnitBase;

TEST_F(MLIR_ArgAllocationInfo, MultipleCallOps) {
    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
            module @test {
                config.Resources 2 of @NCE at 1.300000e+03 MHz {
                    config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
                    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
                    config.ExecutorResource 2 of @SHAVE_ACT
                    config.ExecutorResource 1 of @SHAVE_NN
                    config.ExecutorResource 1 of @DPU
                }
                config.Resources 1 of @global {
                    config.ExecutorResource 2 of @DMA_NN
                    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
                }
                net.NetworkInfo entryPoint : @main
                inputsInfo : {
                    DataInfo "input" : tensor<1x8x60x60xf16>
                } outputsInfo : {
                    DataInfo "output1" : tensor<1x4x60x60xf16>
                    DataInfo "output2" : tensor<1x2x60x60xf16>
                }

                func.func private @foo1(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
                func.func private @foo2(%arg0: memref<1x4x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
                func.func private @foo3(%arg0: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>, %arg3: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>

                func.func @main(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
                    %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
                    %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x4x60x60xf16, @DDR>>, !async.value<memref<1x2x60x60xf16, @DDR>>)
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
                        %2:2 = func.call @foo1(%arg0, %alloc, %arg2) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
                        async.yield %2#0, %2#1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
                    }

                    %alloc_1 = memref.alloc() : memref<1x3x60x60xf16, @DDR>
                    %alloc_2 = memref.alloc() : memref<1x1x20x60xf16, @DDR>
                    %token_0, %bodyResults_1:2 = async.execute [%token] (%bodyResults#0 as %arg3: !async.value<memref<1x4x60x60xf16, @DDR>>)
                                                    -> (!async.value<memref<1x3x60x60xf16, @DDR>>, !async.value<memref<1x1x20x60xf16, @DDR>>)
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
                        %2:2 = func.call @foo2(%arg3, %alloc_1, %alloc_2) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>,  memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
                        async.yield %2#0, %2#1 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
                    }

                    %token_1, %bodyResults_2 = async.execute [%token_0] (
                                                    %bodyResults_1#0 as %arg4: !async.value<memref<1x3x60x60xf16, @DDR>>,
                                                    %bodyResults_1#1 as %arg5: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 2 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
                        %2 = func.call @foo3(%arg4, %arg5, %arg1) : (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
                        async.yield %2 : memref<1x4x60x60xf16, @DDR>
                    }

                    %0 = async.await %bodyResults#1 : !async.value<memref<1x2x60x60xf16, @DDR>>
                    %1 = async.await %bodyResults_2 : !async.value<memref<1x4x60x60xf16, @DDR>>
                    return %1, %0 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
                }
            }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    ReservedMemInfo argsAllocationInfo{func};

    using ReservedAddressAndSizeVector = SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>;
    const auto checkAllocationInfo = [](const ReservedAddressAndSizeVector& expectedCalleeInfo,
                                        const ReservedAddressAndSizeVector& actualCalleeInfo) {
        EXPECT_EQ(expectedCalleeInfo.size(), actualCalleeInfo.size());

        for (size_t i = 0; i < actualCalleeInfo.size(); i++) {
            EXPECT_EQ(expectedCalleeInfo[i], actualCalleeInfo[i]);
        }
    };

    std::pair<vpux::AddressType, vpux::AddressType> expectAllocAddressAndSize{0, 28800};
    std::pair<vpux::AddressType, vpux::AddressType> expectAlloc1AddressAndSize{28800, 21600};
    std::pair<vpux::AddressType, vpux::AddressType> expectAlloc2AddressAndSize{50432, 2400};

    ReservedAddressAndSizeVector foo1Info{expectAllocAddressAndSize};
    checkAllocationInfo(foo1Info, argsAllocationInfo.getReservedMemInfo("foo1")[VPU::MemoryKind::DDR]);

    ReservedAddressAndSizeVector foo2Info{expectAllocAddressAndSize, expectAlloc1AddressAndSize,
                                          expectAlloc2AddressAndSize};
    checkAllocationInfo(foo2Info, argsAllocationInfo.getReservedMemInfo("foo2")[VPU::MemoryKind::DDR]);

    ReservedAddressAndSizeVector foo3Info{expectAlloc1AddressAndSize, expectAlloc2AddressAndSize};
    checkAllocationInfo(foo3Info, argsAllocationInfo.getReservedMemInfo("foo3")[VPU::MemoryKind::DDR]);
}

TEST_F(MLIR_ArgAllocationInfo, MultipleCallOpsWithMultipleUses) {
    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
            module @test {
                config.Resources 2 of @NCE at 1.300000e+03 MHz {
                    config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
                    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
                    config.ExecutorResource 2 of @SHAVE_ACT
                    config.ExecutorResource 1 of @SHAVE_NN
                    config.ExecutorResource 1 of @DPU
                }
                config.Resources 1 of @global {
                    config.ExecutorResource 2 of @DMA_NN
                    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
                }
                net.NetworkInfo entryPoint : @main
                inputsInfo : {
                    DataInfo "input" : tensor<1x8x60x60xf16>
                } outputsInfo : {
                    DataInfo "output1" : tensor<1x4x60x60xf16>
                    DataInfo "output2" : tensor<1x2x60x60xf16>
                    DataInfo "output3" : tensor<1x4x60x60xf16>
                    DataInfo "output4" : tensor<1x3x60x60xf16>
                }

                func.func private @foo1(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
                func.func private @foo2(%arg0: memref<1x4x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
                func.func private @foo3(%arg0: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>, %arg3: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
                func.func private @foo4(%arg0: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>, %arg3: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
                func.func private @foo5(%arg0: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>

                func.func @main(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>, %arg3: memref<1x4x60x60xf16, @DDR>, %arg4: memref<1x3x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>) {
                    %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
                    %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x4x60x60xf16, @DDR>>, !async.value<memref<1x2x60x60xf16, @DDR>>)
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
                        %2:2 = func.call @foo1(%arg0, %alloc, %arg2) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
                        async.yield %2#0, %2#1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
                    }

                    %alloc_1 = memref.alloc() : memref<1x3x60x60xf16, @DDR>
                    %alloc_2 = memref.alloc() : memref<1x1x20x60xf16, @DDR>
                    %token_0, %bodyResults_1:2 = async.execute [%token] (%bodyResults#0 as %arg5: !async.value<memref<1x4x60x60xf16, @DDR>>)
                                                    -> (!async.value<memref<1x3x60x60xf16, @DDR>>, !async.value<memref<1x1x20x60xf16, @DDR>>)
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
                        %2:2 = func.call @foo2(%arg5, %alloc_1, %alloc_2) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>,  memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
                        async.yield %2#0, %2#1 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
                    }

                    %token_1, %bodyResults_2 = async.execute [%token_0] (
                                                    %bodyResults_1#0 as %arg5: !async.value<memref<1x3x60x60xf16, @DDR>>,
                                                    %bodyResults_1#1 as %arg6: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 2 : i64, cycleBegin = 2 : i64, cycleCost = 1 : i64, cycleEnd = 3 : i64} {
                        %2 = func.call @foo3(%arg5, %arg6, %arg1) : (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
                        async.yield %2 : memref<1x4x60x60xf16, @DDR>
                    }

                    %token_2, %bodyResults_3 = async.execute [%token_1] (
                                                    %bodyResults_1#0 as %arg5: !async.value<memref<1x3x60x60xf16, @DDR>>,
                                                    %bodyResults_1#1 as %arg6: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 3 : i64, cycleBegin = 3 : i64, cycleCost = 1 : i64, cycleEnd = 4 : i64} {
                        %2 = func.call @foo4(%arg5, %arg6, %arg3) : (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
                        async.yield %2 : memref<1x4x60x60xf16, @DDR>
                    }

                    %token_3, %bodyResults_4 = async.execute [%token_2] (%bodyResults#0 as %arg5: !async.value<memref<1x4x60x60xf16, @DDR>>)
                                                    -> !async.value<memref<1x3x60x60xf16, @DDR>>
                                                attributes {VPUIP.executor = @NCE, "async-deps-index" = 4 : i64, cycleBegin = 5 : i64, cycleCost = 1 : i64, cycleEnd = 6 : i64} {
                        %2 = func.call @foo5(%arg5, %arg4) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
                        async.yield %2 : memref<1x3x60x60xf16, @DDR>
                    }

                    %0 = async.await %bodyResults#1 : !async.value<memref<1x2x60x60xf16, @DDR>>
                    %1 = async.await %bodyResults_2 : !async.value<memref<1x4x60x60xf16, @DDR>>
                    %2 = async.await %bodyResults_3 : !async.value<memref<1x4x60x60xf16, @DDR>>
                    %3 = async.await %bodyResults_4 : !async.value<memref<1x3x60x60xf16, @DDR>>
                    return %1, %0, %2, %3: memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>
                }
            }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    ReservedMemInfo argsAllocationInfo{func};

    using ReservedAddressAndSizeVector = SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>;
    const auto checkAllocationInfo = [](const ReservedAddressAndSizeVector& expectedCalleeInfo,
                                        const ReservedAddressAndSizeVector& actualCalleeInfo) {
        EXPECT_EQ(expectedCalleeInfo.size(), actualCalleeInfo.size());

        for (size_t i = 0; i < actualCalleeInfo.size(); i++) {
            EXPECT_EQ(expectedCalleeInfo[i], actualCalleeInfo[i]);
        }
    };

    std::pair<vpux::AddressType, vpux::AddressType> expectAllocAddressAndSize{0, 28800};
    std::pair<vpux::AddressType, vpux::AddressType> expectAlloc1AddressAndSize{28800, 21600};
    std::pair<vpux::AddressType, vpux::AddressType> expectAlloc2AddressAndSize{50432, 2400};

    ReservedAddressAndSizeVector beginEndReserved{expectAllocAddressAndSize};
    ReservedAddressAndSizeVector intermediateReserved{expectAllocAddressAndSize, expectAlloc1AddressAndSize,
                                                      expectAlloc2AddressAndSize};

    checkAllocationInfo(beginEndReserved, argsAllocationInfo.getReservedMemInfo("foo1")[VPU::MemoryKind::DDR]);
    checkAllocationInfo(intermediateReserved, argsAllocationInfo.getReservedMemInfo("foo2")[VPU::MemoryKind::DDR]);
    checkAllocationInfo(intermediateReserved, argsAllocationInfo.getReservedMemInfo("foo3")[VPU::MemoryKind::DDR]);
    checkAllocationInfo(intermediateReserved, argsAllocationInfo.getReservedMemInfo("foo4")[VPU::MemoryKind::DDR]);
    checkAllocationInfo(beginEndReserved, argsAllocationInfo.getReservedMemInfo("foo5")[VPU::MemoryKind::DDR]);
}
