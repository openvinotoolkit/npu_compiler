//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Run cmd: npuUnitTests --gtest_filter="MLIR_LoopRegionBuilder.*"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/loop_region.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

class MLIR_LoopRegionBuilder : public MLIR_UnitBase {
public:
    MLIR_LoopRegionBuilder() {
        _ctx.appendDialectRegistry(registry);
        _ctx.loadAllAvailableDialects();
    }

protected:
    mlir::MLIRContext _ctx;
    Logger _log = Logger::global();
    mlir::OwningOpRef<mlir::ModuleOp> _module;

    mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef ir) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &_ctx);
        if (module.get() == nullptr) {
            ADD_FAILURE() << "Failed to parse test IR";
            return {};
        }
        return module;
    }

    mlir::func::FuncOp lookupMainFunction(mlir::ModuleOp module) {
        auto mainFunction = module.lookupSymbol<mlir::func::FuncOp>("main");
        if (mainFunction == nullptr) {
            ADD_FAILURE() << "Module has no @main function";
            return {};
        }
        return mainFunction;
    }

    VPUIP::DmaOptimization::LoopRegionBoundaryInfo buildRegions(llvm::StringRef ir,
                                                                VPUIP::DmaOptimization::LoopRegionBuilder& builder) {
        _module = parseModule(ir);
        if (_module.get() == nullptr) {
            return VPUIP::DmaOptimization::LoopRegionBoundaryInfo();
        }
        auto func = lookupMainFunction(_module.get());
        if (func == nullptr) {
            return VPUIP::DmaOptimization::LoopRegionBoundaryInfo();
        }
        return builder.build(func);
    }
};

void expectValueInfo(
        const VPUIP::DmaOptimization::LoopRegionBoundaryInfo& boundaries, mlir::Value value,
        VPUIP::DmaOptimization::LoopRegionKind kind, int64_t loopIndex, int64_t tileIndex, bool hasPort = false,
        VPUIP::DmaOptimization::LoopRegionPortKind portKind = VPUIP::DmaOptimization::LoopRegionPortKind::Input,
        uint32_t portIdx = 0) {
    const auto infoIt = boundaries.valueToLoopRegions.find(value);
    ASSERT_NE(infoIt, boundaries.valueToLoopRegions.end());
    EXPECT_TRUE(llvm::any_of(infoIt->second, [&](const auto& info) {
        return info.kind == kind && info.loopIndex == loopIndex && info.tileCoordinate == nullptr &&
               info.tileIndex == tileIndex && info.hasPort == hasPort &&
               (!hasPort || (info.portKind == portKind && info.portIdx == portIdx));
    }));
}

bool hasTileCoordinate(const VPUIP::DmaOptimization::LoopRegionInfo& info, llvm::ArrayRef<int64_t> axes,
                       llvm::ArrayRef<int64_t> indices) {
    if (info.tileCoordinate == nullptr) {
        return false;
    }
    const auto axesAttr = info.tileCoordinate.getAs<mlir::DenseI64ArrayAttr>("axes");
    const auto indicesAttr = info.tileCoordinate.getAs<mlir::DenseI64ArrayAttr>("indices");
    return axesAttr != nullptr && indicesAttr != nullptr && axesAttr.asArrayRef() == axes &&
           indicesAttr.asArrayRef() == indices;
}

void expectTilingInfo(
        const VPUIP::DmaOptimization::LoopRegionInfo& info, int64_t loopIndex, llvm::ArrayRef<int64_t> axes,
        llvm::ArrayRef<int64_t> indices, bool hasPort = false,
        VPUIP::DmaOptimization::LoopRegionPortKind portKind = VPUIP::DmaOptimization::LoopRegionPortKind::Input,
        uint32_t portIdx = 0) {
    EXPECT_EQ(info.kind, VPUIP::DmaOptimization::LoopRegionKind::Tiling);
    EXPECT_EQ(info.loopIndex, loopIndex);
    EXPECT_TRUE(hasTileCoordinate(info, axes, indices));
    EXPECT_EQ(info.tileIndex, -1);
    EXPECT_EQ(info.hasPort, hasPort);
    if (hasPort) {
        EXPECT_EQ(info.portKind, portKind);
        EXPECT_EQ(info.portIdx, portIdx);
    }
}

void expectTilingValueInfo(
        const VPUIP::DmaOptimization::LoopRegionBoundaryInfo& boundaries, mlir::Value value, int64_t loopIndex,
        llvm::ArrayRef<int64_t> axes, llvm::ArrayRef<int64_t> indices, bool hasPort = false,
        VPUIP::DmaOptimization::LoopRegionPortKind portKind = VPUIP::DmaOptimization::LoopRegionPortKind::Input,
        uint32_t portIdx = 0) {
    const auto infoIt = boundaries.valueToLoopRegions.find(value);
    ASSERT_NE(infoIt, boundaries.valueToLoopRegions.end());
    EXPECT_TRUE(llvm::any_of(infoIt->second, [&](const auto& info) {
        return info.kind == VPUIP::DmaOptimization::LoopRegionKind::Tiling && info.loopIndex == loopIndex &&
               hasTileCoordinate(info, axes, indices) && info.hasPort == hasPort &&
               (!hasPort || (info.portKind == portKind && info.portIdx == portIdx));
    }));
}

void expectNoTilingValuePortInfo(const VPUIP::DmaOptimization::LoopRegionBoundaryInfo& boundaries, mlir::Value value,
                                 int64_t loopIndex, llvm::ArrayRef<int64_t> axes, llvm::ArrayRef<int64_t> indices,
                                 VPUIP::DmaOptimization::LoopRegionPortKind portKind, uint32_t portIdx = 0) {
    const auto infoIt = boundaries.valueToLoopRegions.find(value);
    ASSERT_NE(infoIt, boundaries.valueToLoopRegions.end());
    EXPECT_FALSE(llvm::any_of(infoIt->second, [&](const auto& info) {
        return info.kind == VPUIP::DmaOptimization::LoopRegionKind::Tiling && info.loopIndex == loopIndex &&
               hasTileCoordinate(info, axes, indices) && info.hasPort && info.portKind == portKind &&
               info.portIdx == portIdx;
    }));
}

constexpr llvm::StringLiteral SINGLE_TILING_REGION_IR = R"(
    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

    func.func @main(%input: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
            -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]> {
        %output_buf = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
        %compute = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>},
                        resultSegmentSizes = array<i32: 1, 0, 0>, tiling_loop_index = 1 : i64}
                        @VPU.SW::@builtin_LogSoftmax
            inputs(%input as %compute_input: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
            outputs(%output_buf as %compute_output: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
            on tile 0 -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]> {
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%compute_input, %compute_output)
                    : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                      memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            }
        %concat_out = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
        %out_subview = VPUIP.SubView %concat_out [0, 0, 0, 0] [1, 1, 1, 64]
            : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            to memref<1x1x1x64xf16, {order = #NCHW, strides = [64, 64, 64, 1]}, [@CMX_NN, 0]>
        %copy = VPUIP.Copy
            inputs(%compute : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
            outputs(%out_subview : memref<1x1x1x64xf16, {order = #NCHW, strides = [64, 64, 64, 1]}, [@CMX_NN, 0]>)
            -> memref<1x1x1x64xf16, {order = #NCHW, strides = [64, 64, 64, 1]}, [@CMX_NN, 0]>
        %concat = VPUIP.ConcatView
            inputs(%copy : memref<1x1x1x64xf16, {order = #NCHW, strides = [64, 64, 64, 1]}, [@CMX_NN, 0]>)
            outputs(%concat_out : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
            -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
        return %concat : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
    }
)";

}  // namespace

TEST(LoopRegionInfoValidity, RequiresKindAndLoopIndex) {
    VPUIP::DmaOptimization::LoopRegionInfo info;
    EXPECT_FALSE(info.isValid());

    info.hasPort = true;
    EXPECT_FALSE(info.isValid());

    info.kind = VPUIP::DmaOptimization::LoopRegionKind::VF;
    EXPECT_FALSE(info.isValid());

    info.loopIndex = 0;
    EXPECT_TRUE(info.isValid());

    info.hasPort = false;
    EXPECT_TRUE(info.isValid());
}

// This test verifies that LoopRegionBuilder reconstructs one two-tile Tiling
// region from structured coordinates rather than IR order. The first tile has
// a two-operation compute chain sharing coordinate {1, 0}; the latter operation
// has the lexicographically earlier coordinate {0, 1}. Both are 2-D coordinates.
//
// Topology:
//   %input -> %sv0 -> %tile0_mid -> %tile0
//   %input -> %sv1 = VPUIP.SubView(%input) -> %tile1 = VPUIP.SW.Kernel(%sv1, %tile1_out)
//   %tile0, %tile1 -> %concat = VPUIP.ConcatView(%tile0, %tile1, %out_buf) -> func.return
TEST_F(MLIR_LoopRegionBuilder, TwoTileTilingRegion) {
    constexpr llvm::StringLiteral IR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        func.func @main(%input: memref<1x2x4x4xf16, #NCHW, @DDR>,
                        %out_buf: memref<1x2x4x4xf16, #NCHW, @DDR>)
                -> memref<1x2x4x4xf16, #NCHW, @DDR> {
            %sv0 = VPUIP.SubView %input [0, 0, 0, 0] [1, 1, 4, 4]
                : memref<1x2x4x4xf16, #NCHW, @DDR>
                to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 16, 4, 1]}, @DDR>
            %tile0_mid_out = memref.alloc() : memref<1x1x4x4xf16, #NCHW, @DDR>
            %tile0_mid = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 0>},
                            resultSegmentSizes = array<i32: 1, 0, 0>,
                            tiling_loop_index = 7 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%sv0 as %tile0_input: memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 16, 4, 1]}, @DDR>)
                outputs(%tile0_mid_out as %tile0_mid_output: memref<1x1x4x4xf16, #NCHW, @DDR>)
                on tile 0 -> memref<1x1x4x4xf16, #NCHW, @DDR> {
                    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%tile0_input, %tile0_mid_output)
                        : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 16, 4, 1]}, @DDR>,
                          memref<1x1x4x4xf16, #NCHW, @DDR>
                }

            %tile0_out = memref.alloc() : memref<1x1x4x4xf16, #NCHW, @DDR>
            %tile0 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 0>},
                            resultSegmentSizes = array<i32: 1, 0, 0>, tiling_loop_index = 7 : i64}
                            @VPU.SW::@builtin_LogSoftmax
                inputs(%tile0_mid as %tile0_chain_input: memref<1x1x4x4xf16, #NCHW, @DDR>)
                outputs(%tile0_out as %tile0_output: memref<1x1x4x4xf16, #NCHW, @DDR>)
                on tile 0 -> memref<1x1x4x4xf16, #NCHW, @DDR> {
                    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%tile0_chain_input, %tile0_output)
                        : memref<1x1x4x4xf16, #NCHW, @DDR>, memref<1x1x4x4xf16, #NCHW, @DDR>
                }

            %sv1 = VPUIP.SubView %input [0, 1, 0, 0] [1, 1, 4, 4]
                : memref<1x2x4x4xf16, #NCHW, @DDR>
                to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 16, 4, 1]}, @DDR>
            %tile1_out = memref.alloc() : memref<1x1x4x4xf16, #NCHW, @DDR>
            %tile1 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 1>},
                            resultSegmentSizes = array<i32: 1, 0, 0>,
                            tiling_loop_index = 7 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%sv1 as %tile1_input: memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 16, 4, 1]}, @DDR>)
                outputs(%tile1_out as %tile1_output: memref<1x1x4x4xf16, #NCHW, @DDR>)
                on tile 1 -> memref<1x1x4x4xf16, #NCHW, @DDR> {
                    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%tile1_input, %tile1_output)
                        : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 16, 4, 1]}, @DDR>,
                          memref<1x1x4x4xf16, #NCHW, @DDR>
                }

            %concat = VPUIP.ConcatView
                inputs(%tile0, %tile1 :
                    memref<1x1x4x4xf16, #NCHW, @DDR>,
                    memref<1x1x4x4xf16, #NCHW, @DDR>)
                outputs(%out_buf : memref<1x2x4x4xf16, #NCHW, @DDR>)
                -> memref<1x2x4x4xf16, #NCHW, @DDR>

            return %concat : memref<1x2x4x4xf16, #NCHW, @DDR>
        }
    )";

    VPUIP::DmaOptimization::LoopRegionBuilder builder(_log);
    const auto boundaries = buildRegions(IR, builder);
    const auto& regions = builder.getRegions();

    ASSERT_EQ(regions.size(), 1u);
    const auto& region = regions.front();
    EXPECT_EQ(region.kind, VPUIP::DmaOptimization::LoopRegionKind::Tiling);
    EXPECT_EQ(region.loopIndex, 7);

    SmallVector<VPUIP::SubViewOp> subViews;
    SmallVector<VPUIP::SwKernelOp> computeOps;
    VPUIP::ConcatViewOp concatOp;
    _module->walk([&](VPUIP::SubViewOp op) {
        subViews.push_back(op);
    });
    _module->walk([&](VPUIP::SwKernelOp op) {
        computeOps.push_back(op);
    });
    _module->walk([&](VPUIP::ConcatViewOp op) {
        concatOp = op;
    });
    ASSERT_EQ(subViews.size(), 2u);
    ASSERT_EQ(computeOps.size(), 3u);
    ASSERT_NE(concatOp.getOperation(), nullptr);

    // Verify the logical Input and Output boundaries shared by both tiles.
    expectValueInfo(boundaries, _module->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0),
                    VPUIP::DmaOptimization::LoopRegionKind::Tiling, 7, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::Input, 0);
    expectValueInfo(boundaries, concatOp.getOutput(), VPUIP::DmaOptimization::LoopRegionKind::Tiling, 7, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::Output, 0);
    expectValueInfo(boundaries, concatOp.getOutputBuff(), VPUIP::DmaOptimization::LoopRegionKind::Tiling, 7, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::DestBuffer, 0);

    const SmallVector<SmallVector<int64_t>> expectedIndices = {{1, 0}, {1, 0}, {0, 1}};
    for (size_t index = 0; index < computeOps.size(); ++index) {
        const auto opIt = boundaries.opToLoopRegion.find(computeOps[index].getOperation());
        ASSERT_NE(opIt, boundaries.opToLoopRegion.end());
        expectTilingInfo(opIt->second, 7, {1, 2}, expectedIndices[index]);

        expectTilingValueInfo(boundaries, computeOps[index].getResult(0), 7, {1, 2}, expectedIndices[index]);
        expectNoTilingValuePortInfo(boundaries, computeOps[index].getResult(0), 7, {1, 2}, expectedIndices[index],
                                    VPUIP::DmaOptimization::LoopRegionPortKind::Output, 0);
        auto& outputOperand = computeOps[index]->getOpOperand(1);
        EXPECT_EQ(boundaries.operandToLoopRegion.find(&outputOperand), boundaries.operandToLoopRegion.end());
        EXPECT_EQ(boundaries.valueToLoopRegions.find(outputOperand.get()), boundaries.valueToLoopRegions.end());
    }

    EXPECT_EQ(boundaries.opToLoopRegion.at(computeOps[0].getOperation()),
              boundaries.opToLoopRegion.at(computeOps[1].getOperation()));
    EXPECT_FALSE(boundaries.opToLoopRegion.at(computeOps[0].getOperation()) ==
                 boundaries.opToLoopRegion.at(computeOps[2].getOperation()));

    expectTilingValueInfo(boundaries, subViews[0].getResult(), 7, {1, 2}, {1, 0}, true,
                          VPUIP::DmaOptimization::LoopRegionPortKind::Input, 0);
    expectTilingValueInfo(boundaries, subViews[1].getResult(), 7, {1, 2}, {0, 1}, true,
                          VPUIP::DmaOptimization::LoopRegionPortKind::Input, 0);

    const auto firstInputIt = boundaries.operandToLoopRegion.find(&computeOps[0]->getOpOperand(0));
    const auto chainedInputIt = boundaries.operandToLoopRegion.find(&computeOps[1]->getOpOperand(0));
    const auto secondTileInputIt = boundaries.operandToLoopRegion.find(&computeOps[2]->getOpOperand(0));
    ASSERT_NE(firstInputIt, boundaries.operandToLoopRegion.end());
    ASSERT_NE(chainedInputIt, boundaries.operandToLoopRegion.end());
    ASSERT_NE(secondTileInputIt, boundaries.operandToLoopRegion.end());
    expectTilingInfo(firstInputIt->second, 7, {1, 2}, {1, 0}, true, VPUIP::DmaOptimization::LoopRegionPortKind::Input,
                     0);
    expectTilingInfo(chainedInputIt->second, 7, {1, 2}, {1, 0});
    expectTilingInfo(secondTileInputIt->second, 7, {1, 2}, {0, 1}, true,
                     VPUIP::DmaOptimization::LoopRegionPortKind::Input, 0);

    expectTilingValueInfo(boundaries, computeOps[1].getResult(0), 7, {1, 2}, {1, 0});
    expectTilingValueInfo(boundaries, computeOps[2].getResult(0), 7, {1, 2}, {0, 1});
}

TEST_F(MLIR_LoopRegionBuilder, TilingRegionRejectsMissingTileCoordinate) {
    auto module = parseModule(SINGLE_TILING_REGION_IR);
    auto func = lookupMainFunction(module.get());
    VPUIP::SwKernelOp computeOp;
    func.walk([&](VPUIP::SwKernelOp op) {
        computeOp = op;
    });
    ASSERT_NE(computeOp, nullptr);
    computeOp->removeAttr(LOOP_TILE_COORD);

    VPUIP::DmaOptimization::LoopRegionBuilder builder(_log);
    EXPECT_THROW(static_cast<void>(builder.build(func)), vpux::Exception);
}

TEST_F(MLIR_LoopRegionBuilder, TilingRegionFindsLogicalOutputThroughCopyAndSubView) {
    VPUIP::DmaOptimization::LoopRegionBuilder builder(_log);
    const auto boundaries = buildRegions(SINGLE_TILING_REGION_IR, builder);
    auto func = lookupMainFunction(_module.get());

    VPUIP::SwKernelOp computeOp;
    VPUIP::ConcatViewOp concatOp;
    func.walk([&](mlir::Operation* op) {
        if (auto swKernel = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            computeOp = swKernel;
        } else if (auto concatView = mlir::dyn_cast<VPUIP::ConcatViewOp>(op)) {
            concatOp = concatView;
        }
    });
    ASSERT_NE(computeOp, nullptr);
    ASSERT_NE(concatOp, nullptr);

    expectValueInfo(boundaries, concatOp.getOutput(), VPUIP::DmaOptimization::LoopRegionKind::Tiling, 1, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::Output, 0);
    expectValueInfo(boundaries, concatOp.getOutputBuff(), VPUIP::DmaOptimization::LoopRegionKind::Tiling, 1, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::DestBuffer, 0);
    expectTilingValueInfo(boundaries, computeOp.getResult(0), 1, {2}, {0});
    expectNoTilingValuePortInfo(boundaries, computeOp.getResult(0), 1, {2}, {0},
                                VPUIP::DmaOptimization::LoopRegionPortKind::Output, 0);

    auto& outputOperand = computeOp->getOpOperand(1);
    EXPECT_EQ(boundaries.valueToLoopRegions.find(outputOperand.get()), boundaries.valueToLoopRegions.end());
    EXPECT_EQ(boundaries.operandToLoopRegion.find(&outputOperand), boundaries.operandToLoopRegion.end());
}

// This test verifies that LoopRegionBuilder preserves multiple roles for one
// SSA value. When `%input` enters two independent VF regions, it must have one
// Input-port information per region instead of one region overwriting the other. Each
// compute operation, result, destination operand, and logical Output must keep
// the matching region identity.
//
// Topology:
//   %input +-> %compute0 = VPUIP.SW.Kernel(%input, %output0_buf) -> func.return operand 0
//          +-> %compute1 = VPUIP.SW.Kernel(%input, %output1_buf) -> func.return operand 1
TEST_F(MLIR_LoopRegionBuilder, SharedInputAcrossVfRegions) {
    constexpr llvm::StringLiteral IR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        func.func @main(%input: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                -> (memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                    memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>) {
            %output0_buf = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %compute0 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>,
                            vf_loop_index = 1 : i64, vf_loop_tile_index = 0 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%input as %compute0_input: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%output0_buf as %compute0_output: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]> {
                    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%compute0_input, %compute0_output)
                        : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                          memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
                }
            %concat0_buf = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %copy0 = VPUIP.Copy
                inputs(%compute0 : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%concat0_buf : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %concat0 = VPUIP.ConcatView
                inputs(%copy0 : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%concat0_buf : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>

            %output1_buf = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %compute1 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>,
                            vf_loop_index = 2 : i64, vf_loop_tile_index = 0 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%input as %compute1_input: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%output1_buf as %compute1_output: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]> {
                    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%compute1_input, %compute1_output)
                        : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                          memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
                }
            %concat1_buf = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %copy1 = VPUIP.Copy
                inputs(%compute1 : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%concat1_buf : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %concat1 = VPUIP.ConcatView
                inputs(%copy1 : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%concat1_buf : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            return %concat0, %concat1 : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                                        memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);
    auto func = lookupMainFunction(module.get());
    VPUIP::DmaOptimization::LoopRegionBuilder builder(_log);
    const auto boundaries = builder.build(func);

    ASSERT_EQ(builder.getRegions().size(), 2u);
    EXPECT_EQ(builder.getRegions()[0].kind, VPUIP::DmaOptimization::LoopRegionKind::VF);
    EXPECT_EQ(builder.getRegions()[0].loopIndex, 1);
    EXPECT_EQ(builder.getRegions()[1].kind, VPUIP::DmaOptimization::LoopRegionKind::VF);
    EXPECT_EQ(builder.getRegions()[1].loopIndex, 2);

    // Verify that the value-level map preserves both region memberships.
    expectValueInfo(boundaries, func.getArgument(0), VPUIP::DmaOptimization::LoopRegionKind::VF, 1, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::Input, 0);
    expectValueInfo(boundaries, func.getArgument(0), VPUIP::DmaOptimization::LoopRegionKind::VF, 2, -1, true,
                    VPUIP::DmaOptimization::LoopRegionPortKind::Input, 0);

    SmallVector<VPUIP::SwKernelOp> computeOps;
    func.walk([&](VPUIP::SwKernelOp op) {
        computeOps.push_back(op);
    });
    SmallVector<VPUIP::ConcatViewOp> concatOps;
    func.walk([&](VPUIP::ConcatViewOp op) {
        concatOps.push_back(op);
    });
    ASSERT_EQ(computeOps.size(), 2u);
    ASSERT_EQ(concatOps.size(), 2u);
    const auto expectVfTileInfo = [&](const VPUIP::DmaOptimization::LoopRegionInfo& info, int64_t loopIndex,
                                      bool hasPort,
                                      VPUIP::DmaOptimization::LoopRegionPortKind portKind =
                                              VPUIP::DmaOptimization::LoopRegionPortKind::Input) {
        EXPECT_EQ(info.kind, VPUIP::DmaOptimization::LoopRegionKind::VF);
        EXPECT_EQ(info.loopIndex, loopIndex);
        EXPECT_EQ(info.tileIndex, 0);
        EXPECT_TRUE(hasTileCoordinate(info, {2}, {0}));
        EXPECT_EQ(info.hasPort, hasPort);
        if (hasPort) {
            EXPECT_EQ(info.portKind, portKind);
            EXPECT_EQ(info.portIdx, 0u);
        }
    };
    const auto expectVfTileValueInfo = [&](mlir::Value value, int64_t loopIndex, bool hasPort,
                                           VPUIP::DmaOptimization::LoopRegionPortKind portKind =
                                                   VPUIP::DmaOptimization::LoopRegionPortKind::Input) {
        const auto infoIt = boundaries.valueToLoopRegions.find(value);
        ASSERT_NE(infoIt, boundaries.valueToLoopRegions.end());
        EXPECT_TRUE(llvm::any_of(infoIt->second, [&](const auto& info) {
            return info.kind == VPUIP::DmaOptimization::LoopRegionKind::VF && info.loopIndex == loopIndex &&
                   info.tileIndex == 0 && hasTileCoordinate(info, {2}, {0}) && info.hasPort == hasPort &&
                   (!hasPort || (info.portKind == portKind && info.portIdx == 0));
        }));
    };
    for (size_t index = 0; index < computeOps.size(); ++index) {
        const auto loopIndex = static_cast<int64_t>(index + 1);
        const auto opIt = boundaries.opToLoopRegion.find(computeOps[index].getOperation());
        ASSERT_NE(opIt, boundaries.opToLoopRegion.end());
        expectVfTileInfo(opIt->second, loopIndex, false);

        auto& inputOperand = computeOps[index]->getOpOperand(0);
        const auto operandIt = boundaries.operandToLoopRegion.find(&inputOperand);
        ASSERT_NE(operandIt, boundaries.operandToLoopRegion.end());
        expectVfTileInfo(operandIt->second, loopIndex, true, VPUIP::DmaOptimization::LoopRegionPortKind::Input);

        expectVfTileValueInfo(computeOps[index].getResult(0), loopIndex, false);
        expectValueInfo(boundaries, concatOps[index].getOutput(), VPUIP::DmaOptimization::LoopRegionKind::VF, loopIndex,
                        -1, true, VPUIP::DmaOptimization::LoopRegionPortKind::Output, 0);
        expectValueInfo(boundaries, concatOps[index].getOutputBuff(), VPUIP::DmaOptimization::LoopRegionKind::VF,
                        loopIndex, -1, true, VPUIP::DmaOptimization::LoopRegionPortKind::DestBuffer, 0);
    }
}
