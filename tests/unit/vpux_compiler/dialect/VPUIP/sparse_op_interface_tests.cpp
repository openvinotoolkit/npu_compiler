//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

TEST(MLIR_VPUIP_Sparsity, SparseOpInterface) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: memref<1x8x20x20xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x19x19xf16, #NHWC, @CMX_NN>) -> memref<1x16x19x19xf16, #NHWC, @CMX_NN> {
                %0 = const.Declare memref<16x8x2x2xf16, #NHWC, @CMX_NN> = #const.Content<dense<2.0> : tensor<16x8x2x2xf16>, [#const.Reorder<#NHWC>]>

                %1 = VPUIP.WeightsTableOp op_input(%arg0 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>) op_output(%arg1 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
                    weights(%0 : memref<16x8x2x2xf16, #NHWC, @CMX_NN>) bias(#const.Content<dense<2.0> : tensor<1x16x1x1xf16>>) -> memref<16x1x1x4xsi32, @CMX_NN>

                %2 = VPUIP.NCEClusterTask { kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [2, 2], kernel_strides = [1, 1], task_type = "CONV" }
                    input(%arg0 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>) weights(%0 : memref<16x8x2x2xf16, #NHWC, @CMX_NN>) weight_table(%1 : memref<16x1x1x4xsi32, @CMX_NN>)
                    parent_input(%arg0 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>) parent_output(%arg1 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
                    outputs(%arg1 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>) -> memref<1x16x19x19xf16, #NHWC, @CMX_NN>
                    variants : { DPUTask { end = [18, 2, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
                        DPUTask { end = [18, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 3, 0] }
                        DPUTask { end = [18, 8, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0] }
                        DPUTask { end = [18, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 9, 0] }
                        DPUTask { end = [18, 18, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0] }
                    } PPE : {
                    }

                return %2 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::DefaultHW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (mlir::isa<vpux::VPUIP::NCEClusterTaskOp>(op)) {
            ASSERT_TRUE(vpux::VPU::supportsSparseInputs(&op));
            ASSERT_TRUE(vpux::VPU::supportsSparseOutputs(&op));
            ASSERT_TRUE(vpux::VPU::supportsSparseData(&op));
        }
    }
}
