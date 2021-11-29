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

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

TEST(MLIR_VPUIP_LayerInfo, AsyncLayerOpInterface) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) -> memref<1x512xf32> {
                %0 = memref.alloc() : memref<1x512xf32>
                %1 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x512xf32>) outputs(%0 : memref<1x512xf32>) -> memref<1x512xf32>
                %2 = IERT.Copy inputs(%1 : memref<1x512xf32>) outputs(%arg1 : memref<1x512xf32>) -> memref<1x512xf32>
                memref.dealloc %0 : memref<1x512xf32>
                return %2 : memref<1x512xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::ReferenceSW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (mlir::isa<vpux::IERT::SoftMaxOp>(op)) {
            auto iface = mlir::dyn_cast<vpux::IERT::AsyncLayerOpInterface>(op);
            ASSERT_NE(iface, nullptr);

            uint32_t numUnits = 0;
            auto kind = iface.getExecutor(numUnits);

            ASSERT_TRUE(kind != nullptr);
            ASSERT_TRUE(kind.isa<vpux::VPU::ExecutorKindAttr>());
            EXPECT_EQ(kind.cast<vpux::VPU::ExecutorKindAttr>().getValue(), vpux::VPU::ExecutorKind::SHAVE_UPA);
            EXPECT_GE(numUnits, 1u);
        } else if (mlir::isa<vpux::IERT::CopyOp>(op)) {
            auto iface = mlir::dyn_cast<vpux::IERT::AsyncLayerOpInterface>(op);
            ASSERT_NE(iface, nullptr);

            uint32_t numUnits = 0;
            auto kind = iface.getExecutor(numUnits);

            ASSERT_TRUE(kind != nullptr);
            ASSERT_TRUE(kind.isa<vpux::VPU::ExecutorKindAttr>());
            EXPECT_EQ(kind.cast<vpux::VPU::ExecutorKindAttr>().getValue(), vpux::VPU::ExecutorKind::DMA_NN);
            EXPECT_EQ(numUnits, 1u);
        }
    }
}
