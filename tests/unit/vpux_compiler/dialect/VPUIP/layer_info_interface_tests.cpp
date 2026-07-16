//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

namespace {

void checkExecutorKind(mlir::Operation* op, vpux::config::ExecutorKind expectedKind) {
    auto iface = mlir::dyn_cast<vpux::VPUIP::AsyncLayerOpInterface>(op);
    ASSERT_NE(iface, nullptr);

    auto kindAttr = iface.getExecutor();
    ASSERT_TRUE(kindAttr != nullptr);
    ASSERT_TRUE(mlir::isa<mlir::SymbolRefAttr>(kindAttr));

    auto kind = vpux::config::symbolizeEnum<vpux::config::ExecutorKind>(kindAttr.getLeafName());
    EXPECT_EQ(kind.value(), expectedKind);
}

}  // namespace

using MLIR_VPUIP_LayerInfo = MLIR_UnitBase;

TEST_F(MLIR_VPUIP_LayerInfo, AsyncLayerOpInterface) {
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

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions =
            vpux::VPU::InitCompilerOptions(vpux::config::Platform::NPU3720, vpux::config::CompilationMode::ReferenceSW);

    vpux::VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (mlir::isa<vpux::VPUIP::CopyOp>(op)) {
            ::checkExecutorKind(&op, vpux::config::ExecutorKind::DMA_NN);
        } else if (mlir::isa<vpux::VPUIP::SwKernelOp>(op)) {
            ::checkExecutorKind(&op, vpux::config::ExecutorKind::SHAVE_ACT);
        }
    }
}
