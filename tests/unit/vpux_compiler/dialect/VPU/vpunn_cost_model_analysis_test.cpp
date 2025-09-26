//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <vpu_layer_cost_model.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace CostModelAnalysisTests {

/**
 * Pass to check if the cost model analysis is preserved
 */
class CheckCachePass : public mlir::PassWrapper<CheckCachePass, vpux::FunctionPass> {
public:
    ::llvm::StringRef getName() const override {
        return "CheckCachePass";
    }
    void safeRunOnFunc() final {
        auto func = getOperation();
        auto module = func->getParentOfType<mlir::ModuleOp>();
        auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
        VPUX_THROW_UNLESS(maybeCostModelAnalysis.has_value(),
                          "Expect to have a preserved cost model analysis, but not");
        auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
        VPUX_THROW_UNLESS(maybeLayerCostModelAnalysis.has_value(),
                          "Expect to have a preserved layer cost model analysis, but not");
    }
};

/**
 * Pass to check the cost model is successfully destroyed
 */
class CheckNoCachePass : public mlir::PassWrapper<CheckNoCachePass, vpux::FunctionPass> {
public:
    ::llvm::StringRef getName() const override {
        return "CheckNoCachePass";
    }
    void safeRunOnFunc() final {
        auto func = getOperation();
        auto module = func->getParentOfType<mlir::ModuleOp>();
        auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
        VPUX_THROW_WHEN(maybeCostModelAnalysis.has_value(), "Dtor pass ran but cost model analysis is still preserved");
        auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
        VPUX_THROW_WHEN(maybeLayerCostModelAnalysis.has_value(),
                        "Dtor pass ran but cost model analysis is still preserved");
    }
};

/**
 * Pass to check that the cost model analysis and layer cost model analysis share the same VPUCostModel instance
 */
class CheckSharedCostModelPass : public mlir::PassWrapper<CheckSharedCostModelPass, vpux::FunctionPass> {
public:
    ::llvm::StringRef getName() const override {
        return "CheckSharedCostModelPass";
    }
    void safeRunOnFunc() final {
        auto func = getOperation();
        auto module = func->getParentOfType<mlir::ModuleOp>();
        const auto arch = config::getArch(module);

        const auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
        auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);

        const auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
        auto layerCostModel =
                VPU::LayerCostModelAnalysis::getOrCreateLayerCostModel(maybeLayerCostModelAnalysis, arch, _log)
                        ->get_cost_model_shared();

        VPUX_THROW_UNLESS(costModel != nullptr, "CostModelAnalysis must have a valid VPUCostModel instance");
        VPUX_THROW_UNLESS(layerCostModel != nullptr, "LayerCostModelAnalysis must have a valid VPUCostModel instance");
        VPUX_THROW_UNLESS(costModel == layerCostModel,
                          "CostModelAnalysis and LayerCostModelAnalysis must share the same VPUCostModel instance");
    }
};

}  // namespace CostModelAnalysisTests

using MLIR_CostModelAnalysisTest = MLIR_UnitBase;

const static llvm::StringLiteral inputIR = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        func.func @main(%arg0: tensor<1x128x32x32xf16, {order = #NHWC}>) -> tensor<1x64x32x32xf16, {order = #NHWC}> {
            %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
            %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                -> tensor<1x64x32x32xf16, {order = #NHWC}>
            return %0 : tensor<1x64x32x32xf16, {order = #NHWC}>
        }
})";

TEST_F(MLIR_CostModelAnalysisTest, CostModelAnalysisBehavior) {
    auto registry = vpux::createDialectRegistry();
    const auto arch = config::ArchKind::NPU40XX;
    auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU40XX);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createCostModelAnalysisConstructPass(vpux::Logger::global()));
    pm.addPass(std::make_unique<CostModelAnalysisTests::CheckCachePass>());
    pm.addPass(VPU::createCostModelAnalysisDestroyPass(vpux::Logger::global()));
    pm.addPass(std::make_unique<CostModelAnalysisTests::CheckNoCachePass>());

    // No exception thrown during pm run
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));
}
