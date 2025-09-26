//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

constexpr vpux::StringRef DPU_NAME = "DPU";

void checkDDRSpace(vpux::IndexedSymbolAttr indexedSymbol) {
    ASSERT_EQ(indexedSymbol.getRootName(), DDR_NAME);
    ASSERT_EQ(indexedSymbol.getLeafName(), DDR_NAME);
    ASSERT_FALSE(indexedSymbol.getIndex().has_value());
    ASSERT_FALSE(indexedSymbol.getNestedReference().has_value());
}

void checkCMXSpace(vpux::IndexedSymbolAttr indexedSymbol, int64_t expIndex) {
    ASSERT_EQ(indexedSymbol.getRootName(), CMX_NAME);
    ASSERT_EQ(indexedSymbol.getLeafName(), CMX_NAME);
    ASSERT_TRUE(indexedSymbol.getIndex().has_value());
    ASSERT_EQ(indexedSymbol.getIndex().value(), expIndex);
    ASSERT_FALSE(indexedSymbol.getNestedReference().has_value());
}

}  // namespace
using MLIR_IndexedSymbolAttr = MLIR_UnitBase;

TEST_F(MLIR_IndexedSymbolAttr, CheckNestedAttr) {
    mlir::MLIRContext ctx(registry);

    const int64_t nestedIdx = 0;
    const auto nestedNameAttr = mlir::FlatSymbolRefAttr::get(&ctx, "@DUMMY");
    const auto nestedAttr = vpux::IndexedSymbolAttr::get(&ctx, {nestedNameAttr, vpux::getIntAttr(&ctx, nestedIdx)});

    const int64_t rootIdx = 1;
    const auto rootNameAttr = mlir::FlatSymbolRefAttr::get(&ctx, CMX_NAME);
    const auto rootAttr =
            vpux::IndexedSymbolAttr::get(&ctx, {rootNameAttr, vpux::getIntAttr(&ctx, rootIdx), nestedAttr});

    const auto fullSymbolAttr = mlir::SymbolRefAttr::get(rootNameAttr.getAttr(), vpux::ArrayRef(nestedNameAttr));

    ASSERT_EQ(rootAttr.getRootReference(), rootNameAttr);
    ASSERT_EQ(rootAttr.getLeafReference(), nestedNameAttr);
    ASSERT_EQ(rootAttr.getFullReference(), fullSymbolAttr);
    ASSERT_TRUE(rootAttr.getIndex().has_value());
    ASSERT_EQ(rootAttr.getIndex().value(), rootIdx);
    ASSERT_TRUE(rootAttr.getNestedReference().has_value());

    const auto actNestedAttr = rootAttr.getNestedReference().value();
    ASSERT_EQ(actNestedAttr, nestedAttr);
    ASSERT_EQ(actNestedAttr.getRootReference(), nestedNameAttr);
    ASSERT_EQ(actNestedAttr.getLeafReference(), nestedNameAttr);
    ASSERT_EQ(actNestedAttr.getFullReference(), nestedNameAttr);
    ASSERT_TRUE(actNestedAttr.getIndex().has_value());
    ASSERT_EQ(actNestedAttr.getIndex().value(), nestedIdx);
    ASSERT_FALSE(actNestedAttr.getNestedReference().has_value());
}

TEST_F(MLIR_IndexedSymbolAttr, CheckMemoryResourceAttr) {
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: memref<1x8x20x20xf16, @DDR>, %arg1: memref<1x8x20x20xf16, @DDR>) -> memref<1x8x20x20xf16, @DDR> {
                %0 = memref.alloc(): memref<1x8x20x20xf16, [@CMX_NN, 0]>
                %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x20x20xf16, @DDR>) outputs(%0 : memref<1x8x20x20xf16, [@CMX_NN, 0]>) -> memref<1x8x20x20xf16, [@CMX_NN, 0]>
                %2 = VPUIP.Copy inputs(%0 : memref<1x8x20x20xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x8x20x20xf16, @DDR>) -> memref<1x8x20x20xf16, @DDR>

                return %2 : memref<1x8x20x20xf16, @DDR>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto checkMemSpace = [](vpux::IndexedSymbolAttr indexedSymAttr) {
        if (indexedSymAttr.getLeafName() == DDR_NAME) {
            checkDDRSpace(indexedSymAttr);
        } else {
            checkCMXSpace(indexedSymAttr, 0);
        }
    };

    for (auto& op : func.getOps()) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto type = mlir::cast<vpux::NDTypeInterface>(allocOp.getMemref().getType());
            auto memSpace = type.getMemSpace();

            checkCMXSpace(memSpace, 0);
        } else if (auto copyOp = mlir::dyn_cast<vpux::VPUIP::CopyOp>(op)) {
            auto inMemSpace = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType()).getMemSpace();
            auto outMemSpace = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType()).getMemSpace();

            ASSERT_NE(inMemSpace, outMemSpace);
            ASSERT_TRUE(inMemSpace.getLeafName() == DDR_NAME || inMemSpace.getLeafName() == CMX_NAME);

            checkMemSpace(inMemSpace);
            checkMemSpace(outMemSpace);
        }
    }
}

TEST_F(MLIR_IndexedSymbolAttr, CheckExecutorResourceAttr) {
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            config.Resources 4 of @NCE at 700.0MHz {
                config.ExecutorResource 5 of @DPU
            }
            config.ExecutorResource 1 of @DMA_NN

            func.func @main(%arg0: memref<1x16x62x62xf16, #NHWC>,
                        %arg1: memref<1x48x60x60xf16, #NHWC>) -> memref<1x48x60x60xf16, #NHWC> {
                %cst = const.Declare memref<48x16x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<48x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
                %0 = VPUIP.StaticAlloc<0> -> memref<1x16x62x62xf16, #NHWC, @CMX_NN>
                %1 = VPUIP.StaticAlloc<468608> -> memref<48x16x3x3xf16, #NHWC, @CMX_NN>
                %2 = VPUIP.StaticAlloc<123008> -> memref<1x48x60x60xf16, #NHWC, @CMX_NN>
                %3 = VPUIP.StaticAlloc<482432> -> memref<48x1x1x4xsi32, @CMX_NN>
                %token, %results = async.execute ->
                                    !async.value<memref<1x16x62x62xf16, #NHWC, @CMX_NN>>
                                        attributes { VPUIP.executor = @DMA_NN } {
                    %5 = VPUIP.Copy inputs(%arg0 : memref<1x16x62x62xf16, #NHWC>)
                                   outputs(%0 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>) -> memref<1x16x62x62xf16, #NHWC, @CMX_NN>
                    async.yield %0 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>
                }
                %token_0, %results_1:2 = async.execute [%token] ->
                                            (!async.value<memref<48x1x1x4xsi32, @CMX_NN>>, !async.value<memref<48x16x3x3xf16, #NHWC, @CMX_NN>>)
                                                attributes { VPUIP.executor = @DMA_NN } {
                  %cst_6 = const.Declare memref<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
                  %5 = VPUIP.Copy inputs(%cst_6 : memref<48x1x1x4xsi32>) outputs(%3 : memref<48x1x1x4xsi32, @CMX_NN>) -> memref<48x1x1x4xsi32, @CMX_NN>
                  %6 = VPUIP.Copy inputs(%cst : memref<48x16x3x3xf16, #NHWC>) outputs(%1 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>) -> memref<48x16x3x3xf16, #NHWC, @CMX_NN>
                  async.yield %3, %1 : memref<48x1x1x4xsi32, @CMX_NN>, memref<48x16x3x3xf16, #NHWC, @CMX_NN>
                }
                %token_2, %results_3 = async.execute [%token_0] (
                                        %results as %arg2: !async.value<memref<1x16x62x62xf16, #NHWC, @CMX_NN>>,
                                        %results_1#1 as %arg3: !async.value<memref<48x16x3x3xf16, #NHWC, @CMX_NN>>,
                                        %results_1#0 as %arg4: !async.value<memref<48x1x1x4xsi32, @CMX_NN>>) ->
                                            !async.value<memref<1x48x60x60xf16, #NHWC, @CMX_NN>>
                                                attributes { VPUIP.executor = [@NCE, 1, [@DPU]] } {
                  %5 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
                                                input(%arg2 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
                                                weights(%arg3 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
                                                weight_table(%arg4 : memref<48x1x1x4xsi32, @CMX_NN>)
                                                parent_input(%arg2 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
                                                parent_output(%2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
                                                outputs(%2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) ->
                memref<1x48x60x60xf16, #NHWC, @CMX_NN> variants :  {
                    DPUTask {outEnd = [59, 11, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
                    DPUTask {outEnd = [59, 23, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 12, 0]}
                    DPUTask {outEnd = [59, 35, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 24, 0]}
                    DPUTask {outEnd = [59, 47, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 36, 0]}
                    DPUTask {outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 48, 0]}
                  } PPE :  {
                  }
                  async.yield %2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
                }
                %token_4, %results_5 = async.execute [%token_2] (%results_3 as %arg2: !async.value<memref<1x48x60x60xf16, #NHWC, @CMX_NN>>) ->
                                        !async.value<memref<1x48x60x60xf16, #NHWC>>
                                            attributes { VPUIP.executor = @DMA_NN } {
                  %5 = VPUIP.Copy inputs(%arg2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x48x60x60xf16, #NHWC>) -> memref<1x48x60x60xf16, #NHWC>
                  async.yield %arg1 : memref<1x48x60x60xf16, #NHWC>
                }
                %4 = async.await %results_5 : !async.value<memref<1x48x60x60xf16, #NHWC>>
                return %4 : memref<1x48x60x60xf16, #NHWC>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    for (auto& op : func.getOps()) {
        if (auto executeOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            const auto executor = vpux::VPUIP::VPUIPDialect::getExecutor(executeOp);
            ASSERT_TRUE(executor != nullptr);

            if (vpux::config::isNceTile(executor.getRootReference())) {
                auto tileRes = vpux::config::getTileExecutor(module.get());
                ASSERT_TRUE(tileRes.hasProcessorFrequency());
                ASSERT_EQ(tileRes.getProcessorFrequency().getValueAsDouble(), 700.0);
                ASSERT_TRUE(executor.getIndex().has_value());
                ASSERT_EQ(executor.getIndex().value(), 1);
                ASSERT_TRUE(executor.getNestedReference().has_value());
                ASSERT_EQ(executor.getLeafName(), DPU_NAME);

                auto nestedExecAttr = executor.getNestedReference().value();
                ASSERT_EQ(nestedExecAttr.getRootName(), DPU_NAME);
                ASSERT_FALSE(nestedExecAttr.getNestedReference().has_value());

                auto nestedExecRes = vpux::config::getAvailableExecutor(module.get(), executor.getFullReference());
                ASSERT_TRUE(nestedExecRes != nullptr);
                ASSERT_EQ(nestedExecRes.getSymName(), DPU_NAME);
            }
        }
    }
}
