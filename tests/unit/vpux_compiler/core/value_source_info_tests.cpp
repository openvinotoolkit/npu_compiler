//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using MLIR_ValueSourceInfo = MLIR_UnitBase;
auto _log = vpux::Logger::global();

TEST_F(MLIR_ValueSourceInfo, RegionBranch) {
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        !IpOp_Stub = memref<1x64x104x104xf16, #NHWC, [@CMX_NN, 0]>
        !FusedConstantType_DDR = memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
        !FusedConstantType_CMX = memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
        !FusedConstantType_CMX_SubView1 = memref<1x1x1x1024xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
        !FusedConstantType_CMX_SubView2 = memref<1x1x1x4096xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
        !FusedConstantType_CMX_View1 = memref<64x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
        !FusedConstantType_CMX_View2 = memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

        module @test {
            func.func @main() -> !IpOp_Stub {
                %cst = const.Declare !FusedConstantType_DDR = dense<1> : tensor<1x1x1x5120xui8>, [#const.SwizzleConstant<5 : i64, 3 : i64>]

                %in = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> !IpOp_Stub
                %out = VPURT.DeclareBuffer <CMX_NN> [0] <692736> -> !IpOp_Stub

                %copy_ou = VPURT.DeclareBuffer <CMX_NN> [0] <1404928> -> !FusedConstantType_CMX

                %t0, %r0 = async.execute -> !async.value<!FusedConstantType_CMX> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 147 : i64, cycleBegin = 3404528 : i64, cycleEnd = 3404744 : i64} {
                    %1 = VPUIP.Copy inputs(%cst : !FusedConstantType_DDR) outputs(%copy_ou : !FusedConstantType_CMX) -> !FusedConstantType_CMX
                async.yield %1 : !FusedConstantType_CMX
                }

                %t1, %r1 = async.execute (%r0 as %arg7: !async.value<!FusedConstantType_CMX>) -> !async.value<!IpOp_Stub> attributes
                    {VPUIP.executor = @DPU, "async-deps-index" = 155 : i64, cycleBegin = 3499015 : i64, cycleCost = 56892 : i64, cycleEnd = 3555907 : i64} {
                    %4= VPUIP.SubView %arg7 [0, 0, 0, 0] [1, 1, 1, 1024] : !FusedConstantType_CMX to !FusedConstantType_CMX_SubView1
                    %5 = VPUIP.SubView %arg7 [0, 0, 0, 1024] [1, 1, 1, 4096] : !FusedConstantType_CMX to !FusedConstantType_CMX_SubView2
                    %6 = VPUIP.ViewOp %4: !FusedConstantType_CMX_SubView1 to !FusedConstantType_CMX_View1
                    %7 = VPUIP.ViewOp %5: !FusedConstantType_CMX_SubView2 to !FusedConstantType_CMX_View2
                    %8 = VPUIP.NCEClusterTask {constantsFused = true, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 56892 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%in : !IpOp_Stub) weights(%7 : !FusedConstantType_CMX_View2) weight_table(%6 : !FusedConstantType_CMX_View1) parent_input(%in : !IpOp_Stub) parent_output(%out : !IpOp_Stub) outputs(%out : !IpOp_Stub) -> !IpOp_Stub
                    variants : {
                        DPUTask {outEnd = [103, 103, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
                    }
                    PPE : {
                        PPETask {ppe = #VPU.PPEStub<>}
                    }
                async.yield %8 : !IpOp_Stub
                }

                %5 = async.await %r1 : !async.value<memref<1x64x104x104xf16, #NHWC, [@CMX_NN, 0]>>
                return %5 : memref<1x64x104x104xf16, #NHWC, [@CMX_NN, 0]>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    func.walk([&](mlir::Operation* op) {
        if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
            const auto returnRes = returnOp.getOperand(0);
            _log.trace("returnRes: {0} ", returnRes);

            const auto returnSourcesExpected = info.getSources(returnRes);
            EXPECT_TRUE(returnSourcesExpected.size() == 1);
            const auto& returnRootsExpected = info.getRoots(returnRes);
            EXPECT_TRUE(returnRootsExpected.size() == 1);
            auto rootOpExpected = (*returnRootsExpected.begin()).getDefiningOp();
            EXPECT_TRUE(mlir::isa<vpux::VPURT::DeclareBufferOp>(rootOpExpected));

            vpux::ValueSourceInfo valueInfo(returnRes);
            const auto returnSourcesActual = valueInfo.getSources(returnRes);
            EXPECT_TRUE(*returnSourcesExpected.begin() == *returnSourcesActual.begin());

            auto outAlloc = returnRes.getDefiningOp()->getOperand(0);
            EXPECT_TRUE(outAlloc == *returnSourcesActual.begin());
            _log.trace("returnRes's source : {0} ", *returnSourcesExpected.begin());

            const auto returnRootsActual = valueInfo.getRoots(returnRes);
            EXPECT_TRUE(*returnRootsExpected.begin() == *returnRootsActual.begin());
            _log.trace("returnRes's root : {0} ", *returnRootsExpected.begin());
        }
    });
}

TEST_F(MLIR_ValueSourceInfo, FuncArguments) {
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            module @VPU.SW  {
                func.func private @builtin_Softmax(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }
            func.func @main(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
                %0 = memref.alloc() : memref<1x512xf16>
                %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_Softmax inputs(%arg0 as %arg2: memref<1x512xf16>) outputs(%0 as %arg3: memref<1x512xf16>) on tile 0 -> memref<1x512xf16>  {
                    VPUIP.SW.Kernel.run {attrs = [1]}(%arg2, %arg3) : memref<1x512xf16>, memref<1x512xf16>
                }
                %2 = VPUIP.Copy inputs(%1 : memref<1x512xf16>) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
                memref.dealloc %0 : memref<1x512xf16>
                return %2 : memref<1x512xf16>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto funcArgSourceExpected = info.getSource(funcArg);
    EXPECT_TRUE(funcArgSourceExpected == nullptr);

    const auto funcArgRootExpected = info.getRoot(funcArg);
    EXPECT_TRUE(funcArgRootExpected == funcArg);

    vpux::ValueSourceInfo valueInfo(funcArg);
    const auto funcArgSourceActual = valueInfo.getSource(funcArg);
    EXPECT_TRUE(funcArgSourceActual == funcArgSourceExpected);
    const auto funcArgRootActual = valueInfo.getRoot(funcArg);
    EXPECT_TRUE(funcArgRootActual == funcArgRootExpected);
}
