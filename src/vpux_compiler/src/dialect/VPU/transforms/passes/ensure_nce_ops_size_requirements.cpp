//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ENSURENCEOPSSIZEREQUIREMENTS
#define GEN_PASS_DEF_ENSURENCEOPSSIZEREQUIREMENTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

SmallVector<Dim> getDimsOverKHWLimit(ShapeRef shape, ArrayRef<int64_t> dimThresholds) {
    SmallVector<Dim> wrongDims = {};
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = Dim(i);
        if (shape[dim] > dimThresholds[i]) {
            wrongDims.push_back(dim);
        }
    }
    return wrongDims;
}

bool hasSplitOverKernelStrategy(mlir::Operation* op) {
    if (mlir::isa<VPU::ClusteredOpInterface>(op)) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
        const auto strategy = clusteredOp.getMultiClusterStrategy();
        if (!strategy.has_value()) {
            return false;
        }

        return strategy.value() == VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return false;
}

class EnsureNCEOpSizeRequirements final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    EnsureNCEOpSizeRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("EnsureNCEOpSizeRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureNCEOpSizeRequirements::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    rewriter.setInsertionPoint(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto log = _log.nest();
    const auto tilingMode = TilingMode::ISOLATED;
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    _log.nest(4).trace("Tile Dim order is {0}", tileDimOrder);
    const auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = config::getTileExecutor(moduleOp).getCount();

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, Dim dimToTile, ArrayRef<int64_t> dimThresholds) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        for (auto tile : tiles.value()) {
            if (tile.shape[dimToTile] > dimThresholds[dimToTile.ind()]) {
                return false;
            }
            auto inputTiling = origOp.backInferTileInfo(tile, log);
            auto& inTiles = inputTiling.tiles;
            if ((dimToTile != Dims4D::Act::C) &&
                (inTiles.begin()->shape[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                return false;
            }
        }
        return true;
    };

    // Construct dim-specific thresholds for input and output shapes
    // In our test, extending the threshold on Dim C can improve performance by reducing workloads for SOK NCE
    // operations when the number of clusters is greater than 2
    SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
    if (hasSplitOverKernelStrategy(op) && numClusters > 2) {
        outputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
    }

    for (auto tileDimIter = tileDimOrder.begin(); tileDimIter < tileDimOrder.end(); ++tileDimIter) {
        auto dimToTile = *tileDimIter;
        while (!isSupportedTileSize(nTilesOnDim, dimToTile, outputDimThresholds)) {
            _log.nest(1).trace("Failed to tile {0} at {1} with {2}", op->getName(), dimToTile, nTilesOnDim);
            ++nTilesOnDim[dimToTile];
        }
    }

    // In case of single tile scheduled there is no need for tiling
    if (llvm::none_of(nTilesOnDim, [](int64_t tiles) {
            return tiles > 1;
        })) {
        return mlir::failure();
    }

    const auto tilesNew = fillDividedTiles(op, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Apply Tiling Strategy for {0} with {1}", op->getName(), nTilesOnDim);
    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, log.nest());
}

//
//  EnsureConvICRequirements
//

class EnsureConvICRequirements final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    EnsureConvICRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _log(log) {
        this->setDebugName("EnsureConvICRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureConvICRequirements::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    // Split over IC supported only for NCEConvolutionOp
    // TODO: E#70421

    // Get some of the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getShape(origOp.getInput());
    auto inputC = inputShape[Dims4D::Act::C];

    if (inputC <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }

    const auto kernelShape = getShape(origOp.getFilter());
    auto kernelW = kernelShape[Dims4D::Filter::KX];
    auto kernelH = kernelShape[Dims4D::Filter::KY];

    auto maxTiles = vpux::divUp(inputC, VPU::NCEInvariant::VPU_DIMENSION_LIMIT);

    if (maxTiles == 1) {
        return mlir::failure();
    }

    Shape nTilesOnDim(inputShape.size(), 1);
    nTilesOnDim[Dims4D::Act::C] = maxTiles;
    SmallVector<int64_t> alignment(inputShape.size(), 1);
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    auto inAlignment = VPU::NCEInvariant::getAlignment(inType.getElementType());
    auto weightsAlignment = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    // Weights alignment requirement is IC * KH * KW aligned with weightsAlignment. For
    // int4 case, weightsAlignment = 32, if KH = 2, then IC = 16 can meet the requirement.
    // So here we fist check if inAlignment can meet the requirement or not.
    if ((inAlignment * kernelW * kernelH) % weightsAlignment == 0) {
        alignment[Dims4D::Act::C.ind()] = inAlignment;
    } else {
        alignment[Dims4D::Act::C.ind()] = weightsAlignment;
    }

    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    const auto tiles = fillDividedTiles(nTilesOnDim, inputShape, optionalAlignment);

    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    auto weightInput = origOp.getFilter();
    // check for parent weight shave dequantize op
    auto weightDequantizeOp = weightInput.getDefiningOp<VPU::DequantizeOp>();
    if (weightDequantizeOp != nullptr) {
        weightInput = weightDequantizeOp.getInput();
    }

    SmallVector<VPU::NCEConvolutionOp> convOps;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    SmallVector<VPU::DequantizeOp> dequantizeOps;
    mlir::Value result = VPU::splitNCEConvolutionOverIC(origOp, weightInput, convOps, addOps, dequantizeOps,
                                                        tiles.value(), weightDequantizeOp, rewriter, _log.nest());

    rewriter.replaceOp(origOp, result);

    return mlir::success();
}

//
// EnsureNCEOpsSizeRequirementsPass
//

class EnsureNCEOpsSizeRequirementsPass final :
        public VPU::impl::EnsureNCEOpsSizeRequirementsBase<EnsureNCEOpsSizeRequirementsPass> {
public:
    explicit EnsureNCEOpsSizeRequirementsPass(bool enableOutputEnsurance,
                                              bool enableDequantWeightEnsuranceBeforeStrategy, bool skipNonConvOC,
                                              Logger log) {
        this->enableOutputEnsurance = enableOutputEnsurance;
        this->enableDequantWeightEnsuranceBeforeStrategy = enableDequantWeightEnsuranceBeforeStrategy;
        this->skipNonConvOC = skipNonConvOC;
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void EnsureNCEOpsSizeRequirementsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEConvolutionOp>(op)) {
            return true;
        }

        const auto inputShape = getShape(op->getOperand(0));
        return inputShape[Dims4D::Act::C] <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    });

    patterns.add<EnsureConvICRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }

    // If output shape ensurance is disabled, skip the rest of the pass
    // OC will be split at multi-cluster and tiling pass if needed
    if (!enableOutputEnsurance) {
        return;
    }

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op)) {
            return true;
        }

        if (mlir::isa<VPU::TilingInfoOpInterface>(op)) {
            const auto inputShape = getShape(op->getOperand(0));
            const auto outputShape = getShape(op->getResult(0));
            const auto numClusters = config::getTileExecutor(moduleOp).getCount();

            // Construct dim-specific thresholds for input and output shapes
            // In our test, extending the threshold on Dim C can improve performance by reducing workloads for SOK NCE
            // operations when the number of clusters is greater than 2
            SmallVector<int64_t> inputDimThresholds(inputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
            SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
            if (hasSplitOverKernelStrategy(op) && numClusters > 2) {
                inputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
                outputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
            }

            if (skipNonConvOC && !mlir::isa<VPU::NCEConvolutionOp>(op)) {
                // OC limit will be handled in multi-cluster and tiling pass
                // Limit to non-conv to avoid regression of memory fragmentation
                return true;
            }

            auto inSizeWrongDims = getDimsOverKHWLimit(inputShape, inputDimThresholds);
            if (!inSizeWrongDims.empty()) {
                _log.nest(2).debug("Input size has dims greater than HW requirements: {0}", inSizeWrongDims);
            }
            const auto outSizeWrongDims = getDimsOverKHWLimit(outputShape, outputDimThresholds);
            if (!outSizeWrongDims.empty()) {
                _log.nest(2).debug("Output size has dims greater than HW requirements: {0}", outSizeWrongDims);
            }
            // Skip slicing conv with dequant weight input before strategy is assigned : this allows for more vertical
            // fusion for large convs
            if (enableDequantWeightEnsuranceBeforeStrategy) {
                if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
                    auto weightDequantizeOp = convOp.getFilter().getDefiningOp<VPU::DequantizeOp>();
                    if (weightDequantizeOp != nullptr) {
                        _log.nest(2).debug("Allow op {0} with dequant weights to skip dimension check before strategy "
                                           "assignment",
                                           op->getLoc());
                        return true;
                    }
                }
            }

            return inSizeWrongDims.empty() && outSizeWrongDims.empty();
        }

        return true;
    });

    patterns.clear();
    patterns.add<EnsureNCEOpSizeRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createEnsureNCEOpsSizeRequirementsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEnsureNCEOpsSizeRequirementsPass(
        bool enableOutputEnsurance, bool enableDequantWeightEnsuranceBeforeStrategy, bool skipNonConvOC, Logger log) {
    return std::make_unique<EnsureNCEOpsSizeRequirementsPass>(
            enableOutputEnsurance, enableDequantWeightEnsuranceBeforeStrategy, skipNonConvOC, log);
}
