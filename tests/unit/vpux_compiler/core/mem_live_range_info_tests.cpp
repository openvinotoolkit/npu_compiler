//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_MemLiveRangeInfo.*"

using namespace vpux;
using MLIR_MemLiveRangeInfo = MLIR_UnitBase;

TEST_F(MLIR_MemLiveRangeInfo, GetInputOutputAndAllBuffers) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%in: memref<1x32x96x96xf16, {order = #NHWC}>, %out: memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}> {
                %cst0 = const.Declare memref<1x32x96x96xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x96x96xf16>, [#const.Reorder<#NHWC>]
                %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

                %buf0 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
                %buf1 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>
                %buf2 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>
                %buf3 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>
                %buf4 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>
                %buf5 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>

                %t0, %r0 = async.execute -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
                    %0 = VPUIP.NNDMA inputs(%in : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
                    async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
                }

                %t1, %r1 = async.execute -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
                    %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf3 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>
                    async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>
                }

                %t2, %r2:2 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
                        -> (!async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>>, !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>>)
                        attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
                    %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<MAXPOOL>
                        }>
                        input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
                        parent_input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        parent_output(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>)
                        outputs(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>
                        variants :
                        {
                            DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
                        }
                        PPE : {
                        }
                    %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<MAXPOOL>
                        }>
                        input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
                        parent_input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        parent_output(%buf2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>)
                        outputs(%buf2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>
                        variants :
                        {
                            DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
                        }
                        PPE : {
                        }
                    async.yield %1, %2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>, memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>
                }

                %t3, %r3 = async.execute [%t1, %t2] (%r2#0 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>>, %r1 as %arg1 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>>)
                        -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
                    %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                            task_type = #VPUIP.nce_task_type<ELTWISE>
                        }>
                        input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>)
                        weights(%arg1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 3]>)
                        parent_input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 1]>)
                        parent_output(%buf4 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>)
                        outputs(%buf4 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>
                        variants :
                        {
                            DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
                        }
                        PPE : {
                            PPETask {ppe = #VPU.PPEStub<>}
                        }
                    async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>
                }

                %t4, %r4 = async.execute [%t1, %t3] (%r2#1 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>>, %r3 as %arg1 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>>)
                        -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
                    %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                            task_type = #VPUIP.nce_task_type<ELTWISE>
                        }>
                        input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>)
                        weights(%arg1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 4]>)
                        parent_input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 2]>)
                        parent_output(%buf5 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>)
                        outputs(%buf5 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>
                        variants :
                        {
                            DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
                        }
                        PPE : {
                            PPETask {ppe = #VPU.PPEStub<>}
                        }
                    async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>
                }

                %t5, %r5 = async.execute [%t4] (%r4 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>>)
                        -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
                    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 5]>) outputs(%out : memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}>
                }

                %3 = async.await %r5 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}>>
                return %3 : memref<1x32x96x96xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo aliasInfo(func);
    vpux::MemLiveRangeInfo liveRangeInfo(func, aliasInfo);

    auto allBuffers = liveRangeInfo.getAllBuffers();
    EXPECT_EQ(allBuffers.size(), 6) << "Should have 6 allocated buffers, buf0-buf5";

    for (auto buffer : allBuffers) {
        auto defOp = buffer.getDefiningOp();
        ASSERT_TRUE(defOp != nullptr);
        EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(defOp)) << "Each buffer should be defined by memref.alloc";
    }

    llvm::SmallVector<mlir::async::ExecuteOp> execOps;
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOps.push_back(execOp);
    });

    ASSERT_EQ(execOps.size(), 6) << "Should have 6 async.execute operations";

    auto getCmxIdx = [](mlir::Value buffer) -> std::optional<int64_t> {
        auto memrefType = mlir::dyn_cast<NDTypeInterface>(buffer.getType());
        if (!memrefType) {
            return std::nullopt;
        }
        auto memSpace = memrefType.getMemSpace();
        if (!memSpace) {
            return std::nullopt;
        }
        if (auto indexedSymbolAttr = mlir::dyn_cast_or_null<vpux::IndexedSymbolAttr>(memSpace)) {
            return indexedSymbolAttr.getIndex();
        }
        return std::nullopt;
    };

    // t0: DMA_NN - input to buf0
    auto t0Inputs = liveRangeInfo.getInputBuffers(execOps[0]);
    EXPECT_EQ(t0Inputs.size(), 1) << "t0 should have 1 input buffer";
    auto t0Outputs = liveRangeInfo.getOutputBuffers(execOps[0]);
    EXPECT_EQ(t0Outputs.size(), 1) << "t0 should have 1 output buffer";
    EXPECT_EQ(getCmxIdx(*t0Outputs.begin()), 0) << "t0 output should be buf0 [@CMX_NN, 0]";

    // t1: DMA_NN - const to buf3
    auto t1Inputs = liveRangeInfo.getInputBuffers(execOps[1]);
    EXPECT_EQ(t1Inputs.size(), 1) << "t1 should have 1 input buffer";
    auto t1Outputs = liveRangeInfo.getOutputBuffers(execOps[1]);
    EXPECT_EQ(t1Outputs.size(), 1) << "t1 should have 1 output buffer";
    EXPECT_EQ(getCmxIdx(*t1Outputs.begin()), 3) << "t1 output should be buf3 [@CMX_NN, 3]";

    // t2: DPU - reads buf0, writes buf1 and buf2 (2 outputs)
    auto t2Inputs = liveRangeInfo.getInputBuffers(execOps[2]);
    EXPECT_EQ(t2Inputs.size(), 2) << "t2 should have 2 input buffers";
    auto t2InputIt = t2Inputs.begin();
    EXPECT_EQ(getCmxIdx(*t2InputIt), 0) << "t2 first input should be buf0 [@CMX_NN, 0]";
    auto t2Outputs = liveRangeInfo.getOutputBuffers(execOps[2]);
    EXPECT_EQ(t2Outputs.size(), 2) << "t2 should have 2 output buffers";
    auto t2OutputIt = t2Outputs.begin();
    EXPECT_EQ(getCmxIdx(*t2OutputIt), 1) << "t2 first output should be buf1 [@CMX_NN, 1]";
    ++t2OutputIt;
    EXPECT_EQ(getCmxIdx(*t2OutputIt), 2) << "t2 second output should be buf2 [@CMX_NN, 2]";

    // t3: DPU - reads buf1 and buf3, writes buf4
    auto t3Inputs = liveRangeInfo.getInputBuffers(execOps[3]);
    EXPECT_EQ(t3Inputs.size(), 2) << "t3 should have 2 input buffers";
    auto t3InputIt = t3Inputs.begin();
    EXPECT_EQ(getCmxIdx(*t3InputIt), 1) << "t3 first input should be buf1 [@CMX_NN, 1]";
    ++t3InputIt;
    EXPECT_EQ(getCmxIdx(*t3InputIt), 3) << "t3 second input should be buf3 [@CMX_NN, 3]";
    auto t3Outputs = liveRangeInfo.getOutputBuffers(execOps[3]);
    EXPECT_EQ(t3Outputs.size(), 1) << "t3 should have 1 output buffer";
    EXPECT_EQ(getCmxIdx(*t3Outputs.begin()), 4) << "t3 output should be buf4 [@CMX_NN, 4]";

    // t4: DPU - reads buf2 and buf4, writes buf5
    auto t4Inputs = liveRangeInfo.getInputBuffers(execOps[4]);
    EXPECT_EQ(t4Inputs.size(), 2) << "t4 should have 2 input buffers";
    auto t4InputIt = t4Inputs.begin();
    EXPECT_EQ(getCmxIdx(*t4InputIt), 2) << "t4 first input should be buf2 [@CMX_NN, 2]";
    ++t4InputIt;
    EXPECT_EQ(getCmxIdx(*t4InputIt), 4) << "t4 second input should be buf4 [@CMX_NN, 4]";
    auto t4Outputs = liveRangeInfo.getOutputBuffers(execOps[4]);
    EXPECT_EQ(t4Outputs.size(), 1) << "t4 should have 1 output buffer";
    EXPECT_EQ(getCmxIdx(*t4Outputs.begin()), 5) << "t4 output should be buf5 [@CMX_NN, 5]";

    // t5: DMA_NN - reads buf5, writes to output
    auto t5Inputs = liveRangeInfo.getInputBuffers(execOps[5]);
    EXPECT_EQ(t5Inputs.size(), 1) << "t5 should have 1 input buffer";
    EXPECT_EQ(getCmxIdx(*t5Inputs.begin()), 5) << "t5 input should be buf5 [@CMX_NN, 5]";
    auto t5Outputs = liveRangeInfo.getOutputBuffers(execOps[5]);
    EXPECT_EQ(t5Outputs.size(), 1) << "t5 should have 1 output buffer";
}

TEST_F(MLIR_MemLiveRangeInfo, GetUsedBuffers) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x16x32x32xf16, {order = #NHWC}>) -> memref<1x16x32x32xf16, {order = #NHWC}> {
                %buf0 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>
                %buf1 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>

                %token_0, %result_0 = async.execute -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %token_1, %result_1 = async.execute [%token_0]
                        (%result_0 as %input: !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>)
                        -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%input : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf1 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %final_result = async.await %result_1 : !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                return %final_result : memref<1x16x32x32xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo aliasInfo(func);
    vpux::MemLiveRangeInfo liveRangeInfo(func, aliasInfo);

    llvm::SmallVector<mlir::async::ExecuteOp> execOps;
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOps.push_back(execOp);
    });
    ASSERT_EQ(execOps.size(), 2);

    auto usedBuffers0 = liveRangeInfo.getUsedBuffers(execOps[0]);
    EXPECT_EQ(usedBuffers0.size(), 1) << "First exec op should use 1 buffer";

    auto usedBuffers1 = liveRangeInfo.getUsedBuffers(execOps[1]);
    EXPECT_EQ(usedBuffers1.size(), 2) << "Second exec op should use 2 buffers (input from buf0, output to buf1)";
}

TEST_F(MLIR_MemLiveRangeInfo, IsBufferUsedByOp) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x16x32x32xf16, {order = #NHWC}>) -> memref<1x16x32x32xf16, {order = #NHWC}> {
                %buf0 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>
                %buf1 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>

                %token_0, %result_0 = async.execute -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %token_1, %result_1 = async.execute [%token_0]
                        (%result_0 as %input: !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>)
                        -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%input : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf1 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %final_result = async.await %result_1 : !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                return %final_result : memref<1x16x32x32xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo aliasInfo(func);
    vpux::MemLiveRangeInfo liveRangeInfo(func, aliasInfo);

    mlir::Value buf0, buf1;
    func.walk([&](mlir::memref::AllocOp allocOp) {
        if (!buf0) {
            buf0 = allocOp.getResult();
        } else {
            buf1 = allocOp.getResult();
        }
    });

    llvm::SmallVector<mlir::async::ExecuteOp> execOps;
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOps.push_back(execOp);
    });

    ASSERT_TRUE(buf0 && buf1);
    ASSERT_EQ(execOps.size(), 2);
    EXPECT_TRUE(liveRangeInfo.isBufferUsedByOp(buf0, execOps[0])) << "buf0 should be used by first exec op";
    EXPECT_TRUE(liveRangeInfo.isBufferUsedByOp(buf0, execOps[1])) << "buf0 should be used by second exec op (as input)";
    EXPECT_FALSE(liveRangeInfo.isBufferUsedByOp(buf1, execOps[0])) << "buf1 should NOT be used by first exec op";
    EXPECT_TRUE(liveRangeInfo.isBufferUsedByOp(buf1, execOps[1])) << "buf1 should be used by second exec op";
}

TEST_F(MLIR_MemLiveRangeInfo, HasRemainingUsers_ReturnsTrueWhenUsersExist) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x16x32x32xf16, {order = #NHWC}>) -> memref<1x16x32x32xf16, {order = #NHWC}> {
                %buf0 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>
                %buf1 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>

                %token_0, %result_0 = async.execute -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %token_1, %result_1 = async.execute [%token_0]
                        (%result_0 as %input: !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>)
                        -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%input : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf1 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %final_result = async.await %result_1 : !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                return %final_result : memref<1x16x32x32xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo aliasInfo(func);
    vpux::MemLiveRangeInfo liveRangeInfo(func, aliasInfo);

    mlir::Value buf0;
    func.walk([&](mlir::memref::AllocOp allocOp) {
        if (!buf0) {
            buf0 = allocOp.getResult();
        }
    });
    ASSERT_TRUE(buf0);

    // buf0 is used by both exec ops (written by first, read by second)
    EXPECT_TRUE(liveRangeInfo.hasRemainingUsers(buf0)) << "buf0 should have remaining users";
}

TEST_F(MLIR_MemLiveRangeInfo, HasRemainingUsers_ReturnsFalseAfterAllErased) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x16x32x32xf16, {order = #NHWC}>) -> memref<1x16x32x32xf16, {order = #NHWC}> {
                %buf0 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>
                %buf1 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>

                %token_0, %result_0 = async.execute -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %token_1, %result_1 = async.execute [%token_0]
                        (%result_0 as %input: !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>)
                        -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%input : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf1 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %final_result = async.await %result_1 : !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                return %final_result : memref<1x16x32x32xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo aliasInfo(func);
    vpux::MemLiveRangeInfo liveRangeInfo(func, aliasInfo);

    mlir::Value buf0;
    func.walk([&](mlir::memref::AllocOp allocOp) {
        if (!buf0) {
            buf0 = allocOp.getResult();
        }
    });
    ASSERT_TRUE(buf0);

    llvm::SmallVector<mlir::async::ExecuteOp> execOps;
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOps.push_back(execOp);
    });
    ASSERT_EQ(execOps.size(), 2);

    // Erase all users of buf0
    liveRangeInfo.eraseUser(buf0, execOps[0]);
    liveRangeInfo.eraseUser(buf0, execOps[1]);

    EXPECT_FALSE(liveRangeInfo.hasRemainingUsers(buf0)) << "buf0 should have no remaining users after erasing all";
}

TEST_F(MLIR_MemLiveRangeInfo, HasRemainingUsers_PartialErase) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x16x32x32xf16, {order = #NHWC}>) -> memref<1x16x32x32xf16, {order = #NHWC}> {
                %buf0 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>
                %buf1 = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}>

                %token_0, %result_0 = async.execute -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf0 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %token_1, %result_1 = async.execute [%token_0]
                        (%result_0 as %input: !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>)
                        -> !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                        attributes {VPUIP.executor = @DMA_NN} {
                    %0 = VPUIP.NNDMA inputs(%input : memref<1x16x32x32xf16, {order = #NHWC}>)
                                     outputs(%buf1 : memref<1x16x32x32xf16, {order = #NHWC}>)
                        -> memref<1x16x32x32xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x16x32x32xf16, {order = #NHWC}>
                }

                %final_result = async.await %result_1 : !async.value<memref<1x16x32x32xf16, {order = #NHWC}>>
                return %final_result : memref<1x16x32x32xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo aliasInfo(func);
    vpux::MemLiveRangeInfo liveRangeInfo(func, aliasInfo);

    mlir::Value buf0;
    func.walk([&](mlir::memref::AllocOp allocOp) {
        if (!buf0) {
            buf0 = allocOp.getResult();
        }
    });
    ASSERT_TRUE(buf0);

    llvm::SmallVector<mlir::async::ExecuteOp> execOps;
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOps.push_back(execOp);
    });
    ASSERT_EQ(execOps.size(), 2);

    // Erase only one user
    liveRangeInfo.eraseUser(buf0, execOps[0]);

    // Still has the second user
    EXPECT_TRUE(liveRangeInfo.hasRemainingUsers(buf0)) << "buf0 should still have remaining users after partial erase";

    // Erase second user
    liveRangeInfo.eraseUser(buf0, execOps[1]);
    EXPECT_FALSE(liveRangeInfo.hasRemainingUsers(buf0)) << "buf0 should have no remaining users after full erase";
}
