//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVOLUTIONSPLITOVERINPUTCHANNEL
#define GEN_PASS_DEF_CONVOLUTIONSPLITOVERINPUTCHANNEL
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//                       Input1(1x8192x256x4xf16)     Input2(3072x8192x1x1xf16)
//                                      \                           /
//                  VPU.NCE.Convolution(1x3072x256x4xf16), SoK, tilingStrategy = [1, 16, 43, 1]

// To:

//                       Input1(1x8192x256x4xf16)          Input2(3072x8192x1x1xf16)
//                                      |                           |
//                  VPU::Slice(1x512x256x4xf16)x16      VPU::Slice(3072x512x1x1xf16)x16
//                                      \                           /
//                  (VPU.NCE.Convolution(1x3072x256x4xf16), SoH, tilingStrategy = [1, 5, 1, 1])x16
//                                                      |
//                  (VPU.NCE.Eltwise.ADD(1x3072x256x4xf16), SoH, tilingStrategy = [1, 5, 1, 1])x15

class NCEConvolutionSplitOverInputChannel final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    NCEConvolutionSplitOverInputChannel(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _numClusters(numClusters), _log(log) {
        this->setDebugName("NCEConvolutionSplitOverInputChannel");
    }

    bool getStrategies(mlir::Operation* op, llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies,
                       std::optional<vpux::Shape>& tilingStrategy,
                       std::optional<VPU::MultiClusterStrategy>& clusterStrategy, Logger log) const;
    template <typename OpType>
    void setStrategies(llvm::ArrayRef<OpType> newOps,
                       llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies = {}) const;
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

// TODO: need to investigate if we can find a better logic to identify the optimal tiling/clustering strategy for new
// Conv/Add/DequantizeOps, and reuse logic from MultiClusterStrategyAssignment. E#178645
bool NCEConvolutionSplitOverInputChannel::getStrategies(mlir::Operation* op,
                                                        llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies,
                                                        std::optional<vpux::Shape>& tilingStrategy,
                                                        std::optional<VPU::MultiClusterStrategy>& clusterStrategy,
                                                        Logger log) const {
    llvm::SmallVector<DimArr> dimOrders = {DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W},
                                           DimArr{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C}};
    tilingStrategy = std::nullopt;
    TilingMode currentMode = TilingMode::ISOLATED;

    auto updateStrategy = [&](vpux::Shape& candidateStrategy, TilingMode candidateMode) -> bool {
        if (!tilingStrategy.has_value()) {
            tilingStrategy = candidateStrategy;
            currentMode = candidateMode;
            return true;
        } else if (candidateMode == TilingMode::PIPELINING && currentMode == TilingMode::ISOLATED) {
            tilingStrategy = candidateStrategy;
            currentMode = candidateMode;
            return true;
        } else if (candidateMode == currentMode) {
            auto currentTilingIndex = getNonOneDim(tilingStrategy.value()).size() == 0
                                              ? vpux::Dim(0)
                                              : *getNonOneDim(tilingStrategy.value()).begin();
            auto candidateIndex = getNonOneDim(candidateStrategy).size() == 0
                                          ? vpux::Dim(0)
                                          : *getNonOneDim(candidateStrategy).begin();
            auto currentTilingNumber = tilingStrategy.value()[currentTilingIndex];
            auto candidateTilingNumber = candidateStrategy[candidateIndex];
            if (candidateTilingNumber < currentTilingNumber) {
                tilingStrategy = candidateStrategy;
                return true;
            }
        }
        return false;
    };

    auto strategyIter = clusterStrategies.begin();
    do {
        std::optional<VPU::MultiClusterStrategy> strategy = std::nullopt;
        if (_numClusters != 1 && strategyIter != clusterStrategies.end()) {
            strategy = *strategyIter;
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
            clusteredOp.setMultiClusterStrategy(strategy.value());
        }
        for (auto& dimOrder : dimOrders) {
            auto tilingMode = TilingMode::PIPELINING;
            auto tempOptionalTilingStrategyOutputTiling =
                    getSWLayerTilingStrategyWithTileDimOrder(op, tilingMode, dimOrder, log);
            if (mlir::failed(tempOptionalTilingStrategyOutputTiling)) {
                continue;
            }
            vpux::Shape tempTilingStrategy = tempOptionalTilingStrategyOutputTiling.value()[0].axis;
            log.trace("Attempt tilingStrategy for op {0}: {1}", tempTilingStrategy, op->getName());
            if (getNonOneDim(tempTilingStrategy).size() <= 1) {
                if (updateStrategy(tempTilingStrategy, tilingMode)) {
                    clusterStrategy = strategy;
                }
            }
        }
        if (strategyIter != clusterStrategies.end()) {
            ++strategyIter;
        }
    } while (strategyIter != clusterStrategies.end() && _numClusters != 1);

    return tilingStrategy.has_value();
}

template <typename OpType>
void NCEConvolutionSplitOverInputChannel::setStrategies(
        llvm::ArrayRef<OpType> newOps, llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies) const {
    if (newOps.empty()) {
        return;
    }
    auto firstOp = newOps[0];
    std::optional<VPU::MultiClusterStrategy> opMultiClusterStrategy = std::nullopt;
    std::optional<vpux::Shape> opTilingStrategy = std::nullopt;
    if (!getStrategies(firstOp.getOperation(), clusterStrategies, opTilingStrategy, opMultiClusterStrategy, _log)) {
        // TODO: iterate all supported multiClusterStrategy, E#178645
        // TODO: generate warning, clear new ops and return failure, or retry with a different tiling number,
        // E#178732
        VPUX_THROW("Failed to find legal tilingStrategy for new ops: {0}", firstOp);
    }

    _log.debug("[{0}] Set '{1}' at '{2}': '{3}' '{4}'", this->getDebugName(), firstOp->getName(), firstOp->getLoc(),
               opTilingStrategy.value(), firstOp);

    for (auto newOp : newOps) {
        newOp.getOperation()->setAttr(vpux::tilingStrategy,
                                      getIntArrayAttr(newOp.getOperation()->getContext(), opTilingStrategy.value()));
        if (_numClusters > 1 && opMultiClusterStrategy.has_value()) {
            newOp.setMultiClusterStrategy(opMultiClusterStrategy.value());
        }
    }
}

// TODO: support for 5D MatMul, E#179189
mlir::LogicalResult NCEConvolutionSplitOverInputChannel::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    const auto ndInputs = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto ndWeights = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    const auto ndOutputs = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    // Skip if origOp has quantized inputs/weights/outputs, which would cause accuracy regressions in CI.
    // TODO: figure out the root cause and try to extend the transformation on this case, E#178621
    if (mlir::isa<mlir::quant::QuantizedType>(ndInputs.getElementType()) ||
        mlir::isa<mlir::quant::QuantizedType>(ndWeights.getElementType()) ||
        mlir::isa<mlir::quant::QuantizedType>(ndOutputs.getElementType())) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel for quantized op {1}", this->getDebugName(),
                   origOp);
        return mlir::failure();
    }
    // Skip if origOp has sparsified inputs/weights, which would cause compilation failures. Need to reorder passes.
    // TODO: investigate if we can address this conflict by reordering passes, E#178623
    if (mlir::isa<vpux::VPU::SparseTensorType>(origOp.getInput().getType()) ||
        mlir::isa<vpux::VPU::SparseTensorType>(origOp.getFilter().getType())) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel for sparse op {1}", this->getDebugName(),
                   origOp);
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.getInput());
    const auto kernelShape = getShape(origOp.getFilter());
    const auto kernelW = kernelShape[Dims4D::Filter::KX];
    const auto kernelH = kernelShape[Dims4D::Filter::KY];

    const auto op = origOp.getOperation();
    if (!op->hasAttr(vpux::tilingStrategy)) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel for op without tilingStrategy {1}",
                   this->getDebugName(), origOp);
        return mlir::failure();
    }

    const vpux::Shape originalTilingStrategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy))));
    const auto tilingOnOutChannel = originalTilingStrategy[Dims4D::Act::C];

    // Tiling on both height and width is embodied on inputs only, so does not introduce extra reloading for inputs or
    // weights.
    const auto tilingOnHeightAndWidth = originalTilingStrategy[Dims4D::Act::H] * originalTilingStrategy[Dims4D::Act::W];
    if (tilingOnOutChannel == 1 || tilingOnHeightAndWidth == 1) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel for op with 1D tilingStrategy {1}",
                   this->getDebugName(), origOp);
        return mlir::failure();
    }

    // Skip if origOp has been pipelined, because it would introduce performance regression
    // TODO: investigate the root cause and try to extend the transformation to this case, E#178629
    if (mlir::succeeded(isSupportedTileSize(origOp, originalTilingStrategy, TilingMode::PIPELINING, _log))) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel for op already pipelining {1}",
                   this->getDebugName(), origOp);
        return mlir::failure();
    }

    const auto f16Type = mlir::Float16Type::get(rewriter.getContext());
    // used as Eltwise Add inputs
    const auto f16TypeOutputs = ndOutputs.changeElemType(f16Type);

    const auto inputsTotalSize = ndInputs.getTotalAllocSize().count();
    const auto weightsTotalSize = ndWeights.getTotalAllocSize().count();
    const auto outputsTotalSizeInF16 = f16TypeOutputs.getTotalAllocSize().count();

    auto reloadParameterSize = tilingOnHeightAndWidth < tilingOnOutChannel ? weightsTotalSize : inputsTotalSize;

    const auto maxTiles = std::min(tilingOnHeightAndWidth, tilingOnOutChannel);

    // This formula works to guarantee this transformation won't result in DMA increase in theory.
    // The overhead of 2D tiling is reloading reloadParameterSize by (maxTiles - 1) times.
    // The cost of splitting over input channel is round-trip of outputsTotalSizeInF16 by maxTiles times;
    // TODO: identify a better criterion to extend this transformation, E#178648
    // Although a looser determination could cause more DMA demands, the benefit from pipelining/VF may be able to cover
    // it.
    auto const roundTripFactor = 2;
    if ((reloadParameterSize * (maxTiles - 1)) < (outputsTotalSizeInF16 * maxTiles * roundTripFactor)) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel for op with large outputs: ({1} * ({2} - 1)) < "
                   "({3} * {4} * {5}), Op: {6}",
                   this->getDebugName(), reloadParameterSize, maxTiles, outputsTotalSizeInF16, maxTiles,
                   roundTripFactor, origOp);
        return mlir::failure();
    }

    _log.debug("[{0}] Got '{1}' at '{2}': '{3}' '{4}'", this->getDebugName(), origOp->getName(), origOp->getLoc(),
               originalTilingStrategy, origOp);

    Shape nTilesOnDim(inputShape.size(), 1);
    nTilesOnDim[Dims4D::Act::C] = maxTiles;
    SmallVector<int64_t> alignment(inputShape.size(), 1);
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inType.getElementType());
    const auto weightsAlignment = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    // Weights alignment requirement is IC * KH * KW aligned with weightsAlignment. For
    // int4 case, weightsAlignment = 32, if KH = 2, then IC = 16 can meet the requirement.
    // So here we fist check if inAlignment can meet the requirement or not.
    if ((inAlignment * kernelW * kernelH) % weightsAlignment == 0) {
        alignment[Dims4D::Act::C.ind()] = inAlignment;
    } else {
        alignment[Dims4D::Act::C.ind()] = weightsAlignment;
    }

    const auto tiles = fillDividedTiles(nTilesOnDim, inputShape, std::optional<ArrayRef<int64_t>>(alignment));

    // TODO: investigate if we can apply a different nTilesOnDim for this case, E#178728
    if (mlir::failed(tiles)) {
        _log.debug("[{0}] Failed to split over input channel by {1} at '{2}': '{3}'", this->getDebugName(), nTilesOnDim,
                   origOp->getLoc(), origOp);
        return mlir::failure();
    }

    auto weightInput = origOp.getFilter();
    // check for parent weight shave dequantize op
    auto weightDequantizeOp = weightInput.getDefiningOp<VPU::DequantizeOp>();
    if (weightDequantizeOp != nullptr) {
        weightInput = weightDequantizeOp.getInput();
    }

    bool isNonConstWTorBias = [](VPU::NCEConvolutionOp origOp) {
        auto weightsOrBiasTable =
                origOp.getWeightsTable() != nullptr ? origOp.getWeightsTable() : origOp.getWeightTableBias();
        if (weightsOrBiasTable != nullptr) {
            if (weightsOrBiasTable.getDefiningOp<Const::DeclareOp>() == nullptr) {
                return true;
            }
        }
        return false;
    }(origOp);

    if (isNonConstWTorBias) {
        _log.debug("[{0}] Failed to split over input channel since the WT or bias is not a constant at '{1}': '{2}'",
                   this->getDebugName(), origOp->getLoc(), origOp);
        return mlir::failure();
    }

    SmallVector<VPU::NCEConvolutionOp> convOps;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    SmallVector<VPU::DequantizeOp> dequantizeOps;
    mlir::Value result = VPU::splitNCEConvolutionOverIC(origOp, weightInput, convOps, addOps, dequantizeOps,
                                                        tiles.value(), weightDequantizeOp, rewriter, _log.nest());

    setStrategies<VPU::NCEConvolutionOp>(
            convOps, {VPU::MultiClusterStrategy::SplitOverKernel, VPU::MultiClusterStrategy::SplitOverHeight});
    setStrategies<VPU::NCEEltwiseOp>(addOps, {VPU::MultiClusterStrategy::SplitOverHeight});
    setStrategies<VPU::DequantizeOp>(dequantizeOps);

    rewriter.replaceOp(origOp, result);

    return mlir::success();
}

//
// ConvolutionSplitOverInputChannelPass
//
// TODO: extract redundant functions from this rewriter and EnsureNCEOpsSizeRequirementsPass::EnsureConvICRequirements,
// E#178729
class ConvolutionSplitOverInputChannelPass final :
        public VPU::impl::ConvolutionSplitOverInputChannelBase<ConvolutionSplitOverInputChannelPass> {
public:
    explicit ConvolutionSplitOverInputChannelPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult ConvolutionSplitOverInputChannelPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvolutionSplitOverInputChannelPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto tileOp = config::getTileExecutor(func);
    const auto numClusters = tileOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<NCEConvolutionSplitOverInputChannel>(&ctx, numClusters, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvolutionSplitOverInputChannelPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvolutionSplitOverInputChannelPass(Logger log) {
    return std::make_unique<ConvolutionSplitOverInputChannelPass>(log);
}
