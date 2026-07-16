//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/minimal_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using vpux::config::Platform;
using namespace vpux;

using MLIR_VPU_VFScheduling = vpux::VPU::arch40xx::UnitTest;

namespace {

class TestVFScheduling : public VPU::VF::v2::MinimalRequirementsVFScheduling {
public:
    TestVFScheduling(Logger log): MinimalRequirementsVFScheduling(log, true) {
    }

    VPU::StrategyCost getPrefetchCost(mlir::Operation* operation, VPU::VF::v2::VFConfig& config,
                                      const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                      const VPU::TilingOperationStorage::UPtr& tilingInfo, const int64_t tileIdx,
                                      const bool isInput) const {
        auto params = fillInCostParam(operation, tilingInfo, tileIdx);
        return getPrefetchingCost(operation, config, costFunction, params, isInput, tilingInfo, tileIdx);
    }

    std::optional<VPU::StrategyCost> getViewLikeOpDMACost(
            mlir::Operation* operation, VPU::VF::v2::VFConfig& config,
            const VPU::TilingOperationStorage::UPtr& tilingInfo, size_t tileIdx,
            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
        return VFScheduling::getViewLikeOpDMACost(operation, config, tilingInfo, tileIdx, costFunction);
    }
};
}  // namespace

TEST_F(MLIR_VPU_VFScheduling, PrefetchDMAUsesVFBlockArgumentThroughViewLikeOp) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @main {
    func.func @main(%arg0: tensor<1x48x160x16xf16, {order = #NHWC}>, %argW: tensor<64x48x1x1xf16, {order = #NHWC}>) -> tensor<1x64x160x16xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

        %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x48x160x16xf16, {order = #NHWC}>, %argW as %arg2: tensor<64x48x1x1xf16, {order = #NHWC}>, %cst as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 8, 1]}
            -> tensor<1x64x160x16xf16, {order = #NHWC}> {
            %1 = VPU.SoftMax(%arg1)
                {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x48x160x16xf16, {order = #NHWC}>
                -> tensor<1x48x160x16xf16, {order = #NHWC}>
            %2 = VPU.ShapeCast {shape = [64, 48, 1, 1]} inputs(%arg2 : tensor<64x48x1x1xf16, {order = #NHWC}>)
                -> tensor<64x48x1x1xf16, {order = #NHWC}>
            %3 = VPU.NCE.Convolution(%1, %2, %arg3) rawFilterShape [64, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                 strides = [1, 1]}
                : tensor<1x48x160x16xf16, {order = #NHWC}>, tensor<64x48x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x160x16xf16, {order = #NHWC}>
            VPU.Yield %3
        }
        return %0 : tensor<1x64x160x16xf16, {order = #NHWC}>
    }
}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto vfOps = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOps.size(), 1);

    auto vfOp = vfOps.front();
    auto operationStorage = std::make_unique<VPU::TilingOperationStorage>();
    restoreTilingRegions(vfOp, vpux::Logger::global(), operationStorage);

    auto convOps = to_small_vector(vfOp.getBody()->getOps<VPU::NCEOpInterface>());
    ASSERT_EQ(convOps.size(), 1);

    auto conv = convOps.front();
    auto weightsOperand = conv.getWeightsOperand();
    ASSERT_TRUE(weightsOperand != nullptr);
    EXPECT_TRUE(VPU::VF::v2::getVFBlockArgument(weightsOperand) != nullptr);

    auto config = VPU::VF::v2::VFConfig(vfOp);
    auto layerCost = std::make_unique<VPU::LayerVPUNNCost>(func);
    auto scheduling = TestVFScheduling(vpux::Logger::global());

    const auto prefetchCost =
            scheduling.getPrefetchCost(conv, config, layerCost, operationStorage, 0, /*isInput=*/false);
    EXPECT_GT(prefetchCost, 0U);
}

TEST_F(MLIR_VPU_VFScheduling, ViewLikeOpDMACostWithGroupSparseTensor) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0:0>

module @main {
    func.func @main(%arg0: tensor<1x256x128x128x!qElemType, {order = #NHWC}>,
                    %sparsityMap: tensor<1x256x257x257xi1, {order = #NHWC}>,
                    %preWeights: tensor<256x256x1x1x!qElemType, {order = #NHWC}>,
                    %preScale: tensor<256x1x1x4xsi32>,
                    %weights: tensor<64x256x2x2x!qElemType, {order = #NHWC}>,
                    %scale: tensor<64x1x1x4xsi32>) -> tensor<1x64x256x256x!qElemType, {order = #NHWC}> {

        %storageElemTable = VPU.StorageElementTable {
            dataElemType = !qElemType,
            dataShape = [1, 256, 128, 128],
            seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
            seDepth = 1 : i64,
            seSize = [256]
        } -> tensor<1x1x257x257xi32, {order = #NHWC}>

        %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x128x128x!qElemType, {order = #NHWC}>,
                                 %sparsityMap as %arg2: tensor<1x256x257x257xi1, {order = #NHWC}>,
                                 %storageElemTable as %arg3: tensor<1x1x257x257xi32, {order = #NHWC}>,
                                 %preWeights as %arg4: tensor<256x256x1x1x!qElemType, {order = #NHWC}>,
                                 %preScale as %arg5: tensor<256x1x1x4xsi32>,
                                 %weights as %arg6: tensor<64x256x2x2x!qElemType, {order = #NHWC}>,
                                 %scale as %arg7: tensor<64x1x1x4xsi32>)
            attributes {tilingStrategy = [1, 1, 1, 5]} -> tensor<1x64x256x256x!qElemType, {order = #NHWC}> {
            %preConv = VPU.NCE.Convolution(%arg1, %arg4, %arg5) rawFilterShape [256, 256, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                 strides = [1, 1]}
                : tensor<1x256x128x128x!qElemType, {order = #NHWC}>,
                  tensor<256x256x1x1x!qElemType, {order = #NHWC}>,
                  tensor<256x1x1x4xsi32>
                -> tensor<1x256x128x128x!qElemType, {order = #NHWC}>
            %sparse = VPU.GroupSparseTensor(%preConv, %arg2, %arg3) {
                seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>}
                -> !VPU.SparseTensor<data=tensor<1x256x128x128x!qElemType, {order = #NHWC}>,
                                     sparsity_map=tensor<1x256x257x257xi1, {order = #NHWC}>,
                                     storage_element_table=tensor<1x1x257x257xi32, {order = #NHWC}>,
                                     #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>
            %conv = VPU.NCE.Convolution(%sparse, %arg6, %arg7) rawFilterShape [64, 256, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                 strides = [1, 1]}
                : !VPU.SparseTensor<data=tensor<1x256x128x128x!qElemType, {order = #NHWC}>,
                                     sparsity_map=tensor<1x256x257x257xi1, {order = #NHWC}>,
                                     storage_element_table=tensor<1x1x257x257xi32, {order = #NHWC}>,
                                     #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>,
                  tensor<64x256x2x2x!qElemType, {order = #NHWC}>,
                  tensor<64x1x1x4xsi32>
                -> tensor<1x64x256x256x!qElemType, {order = #NHWC}>
            VPU.Yield %conv
        }
        return %0 : tensor<1x64x256x256x!qElemType, {order = #NHWC}>
    }
}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU5010, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto vfOps = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOps.size(), 1);

    auto vfOp = vfOps.front();
    auto operationStorage = std::make_unique<VPU::TilingOperationStorage>();
    restoreTilingRegions(vfOp, vpux::Logger::global(), operationStorage);
    // Find the GroupSparseTensor operation
    auto groupSparseOps = to_small_vector(vfOp.getBody()->getOps<VPU::GroupSparseTensorOp>());
    ASSERT_EQ(groupSparseOps.size(), 1);

    auto groupSparseOp = groupSparseOps.front();
    auto config = VPU::VF::v2::VFConfig(vfOp);
    auto layerCost = std::make_unique<VPU::LayerVPUNNCost>(func);
    auto scheduling = TestVFScheduling(vpux::Logger::global());

    const auto dmaCost = scheduling.getViewLikeOpDMACost(groupSparseOp, config, operationStorage, 0, layerCost);
    EXPECT_TRUE(dmaCost.has_value());
}
