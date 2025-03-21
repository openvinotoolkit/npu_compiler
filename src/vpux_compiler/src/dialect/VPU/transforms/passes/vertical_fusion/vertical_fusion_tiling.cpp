//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_VFTILING
#define GEN_PASS_DEF_VFTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// VerticalFusionTilingRewriter
//

typedef std::function<void(int64_t, mlir::Operation*, mlir::Value&, Shape&)> TilingFunction;

class VerticalFusionTilingRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    VerticalFusionTilingRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _vpunnCostFunction(costFunction),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    void adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation, InputTiling& inputTiling,
                          mlir::IRMapping& mapper, TilingStorage& tilingStorage,
                          const TilingOperationStorage::UPtr& opStorage, int64_t tilingIndex, Dim axis) const;
    void processOffset(mlir::Value operand, TilingStorage& tilingStorage, const TilingOperationStorage::UPtr& opStorage,
                       TileInfo& originalTiling, int64_t tilingIndex, Dim axis, ShapeRef expectedShape) const;
    void applyLinearTiling(const int64_t numTiles, VFConfig& config, SmallVector<mlir::Value>& resultTileVals,
                           SmallVector<Shape>& resultTileOffsets, const TilingFunction& tilingProcedure) const;
    void applyPipelinedTiling(const int64_t numTiles, VFConfig& config, SmallVector<mlir::Value>& resultTileVals,
                              SmallVector<Shape>& resultTileOffsets, const TilingFunction& tilingProcedure,
                              const TilingOperationStorage::UPtr& storage) const;

    bool _enableVerticalFusionPipelining;
    const std::unique_ptr<VPU::LayerVPUNNCost>& _vpunnCostFunction;
    Logger _log;
};

void VerticalFusionTilingRewriter::processOffset(mlir::Value operand, TilingStorage& tilingStorage,
                                                 const TilingOperationStorage::UPtr& opStorage,
                                                 TileInfo& originalTiling, int64_t tilingIndex, Dim axis,
                                                 ShapeRef expectedShape) const {
    auto& offset = originalTiling.offsets[axis];
    if (offset == 0) {
        return;
    }

    const auto shiftOffsetBlockArg = [&](mlir::BlockArgument blockArg) {
        // in case previous operation is outside the block and
        // operand is block argument, correct offset on its offset from tiling info
        if (blockArg == nullptr) {
            return;
        }

        const auto storageInfo = tilingStorage.get(blockArg.getArgNumber(), tilingIndex);
        VPUX_THROW_WHEN(!storageInfo.has_value(), "Tiling info for argument {0} with index {1} not found", blockArg,
                        tilingIndex);

        auto tileInfo = storageInfo.value();

        VPUX_THROW_UNLESS(static_cast<size_t>(axis.ind()) < tileInfo.shape.size(), "Got invalid tiling shape size {0}",
                          tileInfo.shape.size());
        const auto inputOffset = tileInfo.offsets[axis];
        const auto inputDimShape = tileInfo.shape[axis];
        const auto origDimSize = originalTiling.shape[axis];

        _log.trace("Input Offset {0}, shape {1} ==> offset: {2}, shape: {3} ", inputOffset, inputDimShape, offset,
                   origDimSize);

        VPUX_THROW_WHEN((inputOffset > offset) || ((inputOffset + inputDimShape) < (offset + origDimSize)),
                        "Got invalid offsets");
        offset -= inputOffset;
    };

    if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
        shiftOffsetBlockArg(blockArg);
        return;
    }

    auto operandOp = operand.getDefiningOp();
    if (operandOp != nullptr) {
        VPUX_THROW_WHEN(operandOp == nullptr, "Can not get defining op for '{0}'", operand);
        auto inputOutputTiling = opStorage->get(operandOp, tilingIndex);
        VPUX_THROW_UNLESS(inputOutputTiling.has_value(), "Couldn't find tiling info at {0}", operandOp->getLoc());
        const auto inputOutputTilingPair = inputOutputTiling.value();
        auto& outTile = inputOutputTilingPair.second;
        offset -= outTile.offsets[axis];
        return;
    }

    offset = expectedShape[axis] - originalTiling.shape[axis];
}

/*
 This function slice to original tile shape in case bigger tile size was chosen
 during backpropagation process.
 In this case adjust shapes to original one by slicing
*/
void VerticalFusionTilingRewriter::adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation,
                                                    InputTiling& inputTiling, mlir::IRMapping& mapper,
                                                    TilingStorage& tilingStorage,
                                                    const TilingOperationStorage::UPtr& opStorage, int64_t tilingIndex,
                                                    Dim axis) const {
    VPUX_THROW_WHEN(inputTiling.tiles.size() < operation->getOperands().size(),
                    "Number of operands {0} is more than number of operand tiles {1}", operation->getOperands().size(),
                    inputTiling.tiles.size());
    for (auto op : operation->getOperands() | indexed) {
        auto operand = op.value();
        auto opIndex = op.index();

        auto expectedOp = mapper.lookupOrNull(operand);
        if (expectedOp == nullptr) {
            continue;
        }

        auto originalTiling = inputTiling.tiles[opIndex];
        auto expectedShape = getShape(expectedOp);
        auto expectedOpSize = expectedShape.totalSize();
        const auto originalOpSize = originalTiling.shape.totalSize();
        if (expectedOpSize == originalOpSize) {
            continue;
        }

        //
        // For below pattern, the Eltwise3 may be tiled before the Eltwise2.
        // Then the Operand has been mapped to the new "SliceOp1" instead of "Eltwise1".
        // While tiling "Eltwise2", it throw exception of "expectedOpSize < originalOpSize".
        // Need to update this branch operand for this case.
        //
        // VF tilingStrategy: [1, 1, 1, 4]
        //                |                                 |
        //           Eltwise1: 1x64x72x128       Conv: 1x64x72x128
        //                |                 X               |
        //           Eltwise2: 1x64x72x128       Eltwise3: 1x64x72x128
        //                |                                 |
        //             Conv: 1x64x72x128                    |
        //                |                                 |
        //             Conv: 1x64x72x128                    |
        //                           \                     /
        //                             Eltwise4: 1x64x72x128
        //                                     |
        //
        // tiling into:
        //
        //                |                                 |
        //           Eltwise1: 1x64x72x36       Conv: 1x64x72x36
        //                |                 X               |
        //                |               /  SliceOp1    SliceOp2
        //                |             /         \         |
        //           Eltwise2: 1x64x72x36       Eltwise3: 1x64x72x32
        //                |                                 |
        //             Conv: 1x64x72x34                     |
        //                |                                 |
        //             Conv: 1x64x72x32                     |
        //                            \                    /
        //                             Eltwise4: 1x64x72x32
        //                                     |
        if (expectedOpSize < originalOpSize) {
            if (auto insertSliceOp = mlir::dyn_cast<VPU::SliceOp>(expectedOp.getDefiningOp())) {
                expectedOp = insertSliceOp.getInputs().front();
                expectedShape = getShape(expectedOp);
                expectedOpSize = expectedShape.totalSize();
            }
        }

        VPUX_THROW_WHEN(
                expectedOpSize < originalOpSize,
                "Original shape size for operand {0} is bigger than current one. Current size {1}, original size {2}",
                operand, expectedOpSize, originalOpSize);

        VPUX_THROW_WHEN(expectedShape.size() != originalTiling.shape.size(),
                        "Expected shape {0} and original one {1} must have same rank", expectedShape,
                        originalTiling.shape);

        // correct offset of operations based on offsets of block argument
        // In case the output of previous operation is bigger than expected
        // which might happen when bigger tile was chosen for same block argument
        // slice operation is needed after the output with correct offsets
        // calculated based on tiling information of current operation and previous one
        _log.trace("Offset before {0}, shape {1}", originalTiling.offsets, originalTiling.shape);

        processOffset(operand, tilingStorage, opStorage, originalTiling, tilingIndex, axis, expectedShape);
        _log.trace("Offset after {0}", originalTiling.offsets);

        const auto valName = printToString("input {0}", opIndex);
        auto opSlice = makeTile(rewriter, operation->getLoc(), expectedOp, originalTiling, valName);

        mapper.map(operand, opSlice);
    }
}

void VerticalFusionTilingRewriter::applyLinearTiling(const int64_t numTiles, VFConfig& config,
                                                     SmallVector<mlir::Value>& resultTileVals,
                                                     SmallVector<Shape>& resultTileOffsets,
                                                     const TilingFunction& tilingProcedure) const {
    auto operations = config.getVFOperations();

    for (auto index : irange(numTiles)) {
        mlir::Value currentResult;
        Shape currentTile;
        for (auto* op : operations) {
            tilingProcedure(index, op, currentResult, currentTile);
        }

        resultTileVals.push_back(currentResult);
        resultTileOffsets.push_back(currentTile);
    }
}

void VerticalFusionTilingRewriter::applyPipelinedTiling(const int64_t numTiles, VFConfig& config,
                                                        SmallVector<mlir::Value>& resultTileVals,
                                                        SmallVector<Shape>& resultTileOffsets,
                                                        const TilingFunction& tilingProcedure,
                                                        const TilingOperationStorage::UPtr& storage) const {
    auto scheduling = config.getSubgraph().getScenario();
    VPUX_THROW_WHEN(!scheduling.has_value(), "Cannot get scheduling scenario from VF {0}", config.getSubgraph());

    VFSchedulingFactory costFactory(/*prefetching=*/true);
    auto scenario = costFactory.createVFScenario(scheduling.value(), _log);

    if (auto pipelinedScenario = std::dynamic_pointer_cast<IVFPipelinedScheduling>(scenario)) {
        auto pipelining = pipelinedScenario->getPipelining(config, numTiles, storage, _vpunnCostFunction);

        mlir::Value currentResult;
        Shape currentTile;
        for (auto& [index, operation] : pipelining.getTimeLine()) {
            // currentResult and currentTiles keep result from previous call tilingProcedure
            tilingProcedure(index, operation, currentResult, currentTile);

            if (llvm::find(config.getOutputs(), operation) != config.getOutputs().end()) {
                resultTileVals.push_back(currentResult);
                resultTileOffsets.push_back(currentTile);
            }
        }
    } else {
        applyLinearTiling(numTiles, config, resultTileVals, resultTileOffsets, tilingProcedure);
    }
}

mlir::LogicalResult VerticalFusionTilingRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy().cast<mlir::ArrayAttr>());

    const auto numTiledAxis = llvm::count_if(tilingStrategy, [](auto num) {
        return num > 1;
    });

    VPUX_THROW_WHEN(numTiledAxis != 1, "VF tiling is supported only for one axis");

    auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());

    if (maxTiledLen == tilingStrategy.end()) {
        return mlir::failure();
    }

    VPUX_THROW_WHEN(*maxTiledLen <= 1, "There is no tiling for VF");

    auto operationStorage = std::make_unique<TilingOperationStorage>();
    auto tilingStorage = restoreTilingRegions(vfOp, _log, operationStorage);

    auto vfConfig = VFConfig(vfOp, _enableVerticalFusionPipelining);

    SmallVector<mlir::Value> resultTileVals;
    resultTileVals.reserve(*maxTiledLen);
    SmallVector<Shape> resultTileOffsets;
    mlir::IRMapping mapper;

    auto dim = Dim(std::distance(tilingStrategy.begin(), maxTiledLen));

    const auto tilingProcedure = [&](int64_t index, mlir::Operation* op, mlir::Value& currentResult,
                                     Shape& currentTile) {
        for (auto operand : op->getOperands()) {
            if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                const auto valName = printToString("input {0}", index);
                auto origInput = vfOp.getOperand(blockArg.getArgNumber());
                auto tileInfo = tilingStorage.get(blockArg.getArgNumber(), index);

                VPUX_THROW_WHEN(!tileInfo.has_value(), "Couldn't find tile information for argument {0} and tile {1}",
                                blockArg.getArgNumber(), index);
                auto operandTile = VPU::makeTile(rewriter, op->getLoc(), origInput, tileInfo.value(), valName);

                mapper.map(operand, operandTile);
            }
        }

        auto inputTiling = operationStorage->get(op, index);

        VPUX_THROW_WHEN(!inputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}", *op,
                        index);

        const auto inputTilingPair = inputTiling.value();
        auto inputTilingInfo = inputTilingPair.first;
        adjustInputShape(rewriter, op, inputTilingInfo, mapper, tilingStorage, operationStorage, index, dim);

        auto* copiedOp = rewriter.clone(*op, mapper);
        currentResult = copiedOp->getResult(0);

        currentTile = inputTilingPair.second.offsets;
        const auto baseResType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
        if (auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(copiedOp)) {
            tiledBuilderOp.adjustAttrs(inputTilingInfo, inputTilingPair.second);
        } else if (auto tiledViewOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(copiedOp)) {
            tiledViewOp.adjustAttrs(inputTilingInfo, inputTilingPair.second, baseResType.getShape());
        }
        const auto tiledResType =
                baseResType.extractDenseTile(inputTilingPair.second.offsets, inputTilingPair.second.shape);

        currentResult.setType(tiledResType);
        mapper.map(op->getResult(0), currentResult);
    };

    if (vfConfig.isPipelined()) {
        applyPipelinedTiling(*maxTiledLen, vfConfig, resultTileVals, resultTileOffsets, tilingProcedure,
                             operationStorage);
    } else {
        applyLinearTiling(*maxTiledLen, vfConfig, resultTileVals, resultTileOffsets, tilingProcedure);
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(vfOp, vfOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                               ArrayRef(resultTileOffsets));

    return mlir::success();
}

//
// VfTilingPass
//

class VfTilingPass final : public VPU::impl::VfTilingBase<VfTilingPass> {
public:
    explicit VfTilingPass(bool enableVerticalFusionPipelining, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
};

mlir::LogicalResult VfTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading VfTilingPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void VfTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto costFunction = std::make_unique<VPU::LayerVPUNNCost>(func);

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::VerticalFusionOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();

    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VerticalFusionTilingRewriter>(&ctx, _enableVerticalFusionPipelining, costFunction, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createVfTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createVfTilingPass(bool enableVerticalFusionPipelining, Logger log) {
    return std::make_unique<VfTilingPass>(enableVerticalFusionPipelining, log);
}
