//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <mlir/Parser/Parser.h>
#include "common/utils.hpp"

#include <gtest/gtest.h>

using vpux::VPU::ArchKind;
using namespace vpux;
using MLIR_VPU_doesTopKLayerFitIntoCMX = MLIR_UnitBase;

const int64_t numDPUs = 5;

TEST(MLIR_VPU_TilingUtils, BackInferPadsTile) {
    const auto compareInferredPads = [&](ShapeRef inputShape, PadInfo padInfo, ArrayRef<int64_t> kernelSize,
                                         ArrayRef<int64_t> kernelStrides, ShapeRef tileShape, ShapeRef tileOffsets,
                                         PadInfo expectedPads) {
        TileInfo outTile(tileShape);
        outTile.offsets = Shape(tileOffsets.raw());
        outTile.axis[Dims4D::Act::H] = numDPUs;
        const auto inferredPads = backInferPadsTile(outTile, inputShape, padInfo, kernelSize, kernelStrides);
        EXPECT_EQ(inferredPads, expectedPads);
    };

    {
        const Shape inShape{1, 16, 7, 7};
        const PadInfo padInfo{0, 0, 0, 0};
        const SmallVector<int64_t> kernelSize{1, 1};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{0, 0, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{0, 0, 0, 0});
    }

    {
        const Shape inShape{1, 16, 9, 9};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{0, 0, 0, 0};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{0, 0, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{0, 0, 0, 0});
    }

    {
        const Shape inShape{1, 16, 7, 7};
        const PadInfo padInfo{1, 1, 1, 1};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{1, 1, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{1, 1, 0, 1});
    }

    {
        const Shape inShape{1, 16, 13, 13};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{1, 1, 1, 1};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{2, 2};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{1, 1, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{1, 1, 0, 1});
    }

    {
        const Shape inShape{1, 16, 7, 7};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{2, 2, 2, 2};
        const SmallVector<int64_t> kernelSize{5, 5};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 2, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{2, 2, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 2, 0}, /*expectedPads=*/{2, 2, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 2, 0, 1});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{2, 2, 0, 2});

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 2, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 2, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 2, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 2, 0, 2});
    }

    {
        const Shape inShape{1, 16, 14, 14};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{2, 2, 2, 2};
        const SmallVector<int64_t> kernelSize{5, 5};
        const SmallVector<int64_t> kernelStrides{2, 2};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 1, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{2, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{2, 1, 0, 1});
    }
}

TEST_F(MLIR_VPU_doesTopKLayerFitIntoCMX, TopKfitsCMX) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        module @main {
            func.func @main(%arg0: tensor<1x1x1x100xf16>) -> tensor<1x1x1x1xsi32> {
                %cst = const.Declare tensor<1x1x1x640000xui8> = dense<0> : tensor<1x1x1x640000xui8>
                %output_values, %target_shape = VPU.TopK(%arg0, %cst)
                {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort =
                #IE.topk_sort_type<NONE>} : tensor<1x1x1x100xf16>, tensor<1x1x1x640000xui8> -> tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
            return %target_shape : tensor<1x1x1x1xsi32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, VPU::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto siblingsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    func->walk([&](VPU::TopKOp topk) {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        auto reservedMem = Byte(0);
        auto doesLayerFitIntoCMX = topk.doesLayerFitIntoCMX(strategy, siblingsAnalysis, reservedMem);
        EXPECT_EQ(doesLayerFitIntoCMX, true);
    });
}

TEST_F(MLIR_VPU_doesTopKLayerFitIntoCMX, TopKdoesNotFitCMX) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        module @main {
            func.func @main(%arg0: tensor<1x1x200x32000xf16>) -> tensor<1x1x200x1xsi32> {
                %cst = const.Declare tensor<1x1x1x640000xui8> = dense<0> : tensor<1x1x1x640000xui8>
                %output_values, %target_shape = VPU.TopK(%arg0, %cst)
                {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort =
                #IE.topk_sort_type<NONE>} : tensor<1x1x200x32000xf16>, tensor<1x1x1x640000xui8> -> tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>
            return %target_shape : tensor<1x1x200x1xsi32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, VPU::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto siblingsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    func->walk([&](VPU::TopKOp topk) {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        auto reservedMem = Byte(0);
        auto doesLayerFitIntoCMX = topk.doesLayerFitIntoCMX(strategy, siblingsAnalysis, reservedMem);
        EXPECT_EQ(doesLayerFitIntoCMX, false);
    });
}
