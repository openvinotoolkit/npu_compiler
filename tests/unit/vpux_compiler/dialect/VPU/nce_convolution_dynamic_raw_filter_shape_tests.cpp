//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Run cmd: npuUnitTests --gtest_filter="MLIR_NCEConvolutionDynamicRawFilterShape.*"

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

// rawFilterShape's OC dim (%tile_sz) comes from an scf.for-tiled affine.min chain, the same
// shape SCF tiling/multiclustering produces in practice. KY/KX must stay static: NCE
// verification rejects a dynamic kernel size outright.
constexpr llvm::StringLiteral CONV_DYNAMIC_OC_FROM_SCF_TILING = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x100x16x16xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<100x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<100x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c22 = arith.constant 22 : index
        %out = tensor.empty() : tensor<1x100x16x16xf16, {order = #NHWC}>

        %result = scf.for %k = %c0 to %c100 step %c22 iter_args(%acc = %out) -> (tensor<1x100x16x16xf16, {order = #NHWC}>) {
            %tile_sz = affine.min affine_map<(d0) -> (-d0 + 100, 22)>(%k)

            %w_slice = tensor.extract_slice %cst[%k, 0, 0, 0] [%tile_sz, 16, 1, 1] [1, 1, 1, 1]
                : tensor<100x16x1x1xf16, {order = #NHWC}>
                to tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[100, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

            %conv = VPU.NCE.Convolution(%arg0, %w_slice) rawFilterShape [%tile_sz, 16, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[100, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>, index
                -> tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 100, 16, 16]> : tensor<4xsi64>, order = #NHWC}>

            %insert = tensor.insert_slice %conv into %acc[0, %k, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
                : tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 100, 16, 16]> : tensor<4xsi64>, order = #NHWC}>
                into tensor<1x100x16x16xf16, {order = #NHWC}>
            scf.yield %insert : tensor<1x100x16x16xf16, {order = #NHWC}>
        }

        return %result : tensor<1x100x16x16xf16, {order = #NHWC}>
    }
}
)";

}  // namespace

class MLIR_NCEConvolutionDynamicRawFilterShape : public testing::Test {
public:
    void SetUp() override {
        constexpr auto platform = config::Platform::NPU4000;

        registry = vpux::createDialectRegistry();
        auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
        interfacesRegistry->registerInterfaces(registry);
        VPU::initializeSingletons(registry, platform);

        ctx = std::make_unique<mlir::MLIRContext>(registry);
    }

    mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringLiteral ir) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, ctx.get());
        if (!module) {
            return {};
        }

        mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        auto opts = VPU::InitCompilerOptions(config::Platform::NPU4000, config::CompilationMode::DefaultHW);
        VPU::buildInitCompilerPipeline(pm, opts, Logger::global());

        if (mlir::failed(pm.run(module.get()))) {
            return {};
        }
        return module;
    }

    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;
};

// No caller reads the OC element getConstRawFilterShape() returns here (the sole caller,
// getKernelSize(), only uses KY/KX) — this test exists purely to pin down that the fold
// itself is correct, since no lit test can observe it.
TEST_F(MLIR_NCEConvolutionDynamicRawFilterShape, DynamicOCFoldsToUpperBound) {
    auto module = parse(CONV_DYNAMIC_OC_FROM_SCF_TILING);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_EQ(op.getConstRawFilterShape(), (SmallVector<int64_t>{22, 16, 1, 1}));
    });
    EXPECT_TRUE(found);
}
