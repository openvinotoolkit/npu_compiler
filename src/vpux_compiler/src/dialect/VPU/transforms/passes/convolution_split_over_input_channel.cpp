//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
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
    bool splitNCEConvolutionWithLegacyWeightTable(VPU::NCEConvolutionOp& origOp, mlir::Value& weightInput,
                                                  SmallVector<VPU::NCEConvolutionOp>& convOps,
                                                  const OutputTiling& tiles, int64_t maxTiles,
                                                  VPU::DequantizeOp& weightDequantizeOp, VPU::PPEAttr& strippedPpeAttr,
                                                  mlir::PatternRewriter& rewriter) const;
    bool splitNCEConvolutionWithNewWeightTable(VPU::NCEConvolutionOp& origOp, mlir::Value& weightInput,
                                               SmallVector<VPU::NCEConvolutionOp>& convOps, const OutputTiling& tiles,
                                               int64_t maxTiles, VPU::DequantizeOp& weightDequantizeOp,
                                               VPU::PPEAttr& strippedPpeAttr, mlir::PatternRewriter& rewriter) const;
    bool getStrategies(mlir::Operation* op, llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies,
                       std::optional<vpux::Shape>& tilingStrategy,
                       std::optional<VPU::MultiClusterStrategy>& clusterStrategy, Logger log) const;
    template <typename OpType>
    void setStrategies(llvm::ArrayRef<OpType> newOps,
                       llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies) const;
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

bool NCEConvolutionSplitOverInputChannel::splitNCEConvolutionWithLegacyWeightTable(
        VPU::NCEConvolutionOp& origOp, mlir::Value& weightInput, SmallVector<VPU::NCEConvolutionOp>& convOps,
        const OutputTiling& tiles, int64_t maxTiles, VPU::DequantizeOp& weightDequantizeOp,
        VPU::PPEAttr& strippedPpeAttr, mlir::PatternRewriter& rewriter) const {
    // Get the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getShape(origOp.getInput());
    auto inputW = inputShape[Dims4D::Act::W];
    auto inputH = inputShape[Dims4D::Act::H];
    auto inputN = inputShape[Dims4D::Act::N];

    const auto kernelShape = getShape(origOp.getFilter());
    auto kernelW = kernelShape[Dims4D::Filter::KX];
    auto kernelH = kernelShape[Dims4D::Filter::KY];
    auto kernelN = kernelShape[Dims4D::Filter::OC];

    auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    auto filterElemType = filterType.getElementType();

    auto weightsTable = origOp.getWeightsTable();
    auto weightsTableConst = weightsTable.getDefiningOp<Const::DeclareOp>();
    if (weightsTableConst == nullptr) {
        _log.trace("Could not extract constant from weights table.");
        return false;
    }
    auto weightsTableContent = weightsTableConst.getContent();
    auto weightsTableValues = weightsTableContent.getValues<int32_t>();
    auto weightsTableVecSize = weightsTableValues.size();
    std::vector<int32_t> weightsTableVec(weightsTableVecSize);
    std::copy(weightsTableValues.begin(), weightsTableValues.end(), weightsTableVec.begin());

    auto origDataType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
    auto f16Type = mlir::FloatType::getF16(rewriter.getContext());
    // used as Eltwise inputs
    // the outputs of ConvOps are used as intermediate data between ConvOps and AddOps
    // convert it to fp16 to make allowance for both precision and compatibility with Eltwise
    const auto f16TypeOutputs = origDataType.changeElemType(f16Type);

    for (auto tile = 0; tile < maxTiles; tile++) {
        auto offsetIC = tiles[tile].offsets[Dims4D::Act::C];
        auto sizeIC = tiles[tile].shape[Dims4D::Act::C];
        _log.nest().trace("Slicing channels {0} - {1}", offsetIC, sizeIC);

        // Slice inputs
        const Shape inSliceOffsets{0, offsetIC, 0, 0};
        const Shape inSliceShape{inputN, sizeIC, inputH, inputW};
        auto convInput = rewriter.create<VPU::SliceOp>(origOp->getLoc(), origOp.getInput(),
                                                       getIntArrayAttr(rewriter, inSliceOffsets.raw()),
                                                       getIntArrayAttr(rewriter, inSliceShape.raw()));

        // Slice kernels
        const Shape kernelSliceOffsets{0, offsetIC, 0, 0};
        const Shape kernelSliceShape{kernelN, sizeIC, kernelH, kernelW};
        const auto rawKernelSliceShape = getIntArrayAttr(rewriter, kernelSliceShape);
        auto weightSlice = rewriter.create<VPU::SliceOp>(origOp.getLoc(), weightInput,
                                                         getIntArrayAttr(rewriter, kernelSliceOffsets.raw()),
                                                         getIntArrayAttr(rewriter, kernelSliceShape.raw()));
        auto weightSliceResult = weightSlice.getResult();
        if (weightDequantizeOp != nullptr) {
            // Slice over VPU.DequantizeOp
            // TODO: This logic may also be practicable to other element-wise operaions, E#178703
            // input1            input2
            //                     |
            //    \             VPU.DequantizeOp
            //                     /
            //    VPU.NCE.Convolution

            // To:

            //  input1                input2
            //      |                   |
            // VPU.SlicexN          VPU.SlicexN
            //                          |
            //       \          VPU.DequantizeOpxN
            //                          /
            //         VPU.NCE.ConvolutionxN
            //                  |
            //          VPU.NCE.Eltwise.ADDx(N-1)

            auto dequantizeSlice = rewriter.create<VPU::DequantizeOp>(weightDequantizeOp->getLoc(), weightSliceResult,
                                                                      weightDequantizeOp.getDstElemTypeAttr(),
                                                                      weightDequantizeOp.getMultiClusterStrategyAttr());
            weightSliceResult = dequantizeSlice.getResult();
        }

        // Adjust the weights table pointers to correspond to the new offsets of the slices
        const auto noOfBits = vpux::getElemTypeSize(filterElemType);
        const auto weightSetSize = alignMemSize(kernelH * kernelW * sizeIC * noOfBits,
                                                Byte(VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT))
                                           .to<Byte>()
                                           .count();
        const auto sparsitySetSize =
                alignValUp(divUp(kernelH * kernelW * sizeIC, CHAR_BIT * getValuesPerSparsityBit(filterElemType)),
                           static_cast<int64_t>(VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT));

        auto const weightsOffset = 0;
        auto const sparsityOffset = 1;
        auto const biasOffset = 3;
        for (size_t i = 0; i < weightsTableVecSize / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC; ++i) {
            auto index = VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * i;
            // Apply bias for the first convolution only
            // originalConvOp+bias = (convOp0+convOp1+...+convOpN)+bias = (convOp0+bias)+convOp1+...+convOpN
            if (tile != 0) {
                weightsTableVec[index + biasOffset] = checked_cast<int32_t>(0);
            }
            weightsTableVec[index + weightsOffset] = checked_cast<int32_t>(i * weightSetSize);
            weightsTableVec[index + sparsityOffset] = checked_cast<int32_t>(i * sparsitySetSize);
        }

        auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                origOp.getLoc(), f16TypeOutputs, convInput.getResult(), weightSliceResult, weightsTable,
                origOp.getWeightTableDataPtr(), origOp.getWeightTableSpPtr(), origOp.getWeightTableScale(),
                origOp.getWeightTableBias(), origOp.getWeightZeroPoints(), origOp.getStrides(), origOp.getPad(),
                strippedPpeAttr, origOp.getMpeEngineAttr(), rawKernelSliceShape, origOp.getMultiClusterStrategyAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

        convOps.push_back(convOp);
    }

    return true;
}

bool NCEConvolutionSplitOverInputChannel::splitNCEConvolutionWithNewWeightTable(
        VPU::NCEConvolutionOp& origOp, mlir::Value& weightInput, SmallVector<VPU::NCEConvolutionOp>& convOps,
        const OutputTiling& tiles, int64_t maxTiles, VPU::DequantizeOp& weightDequantizeOp,
        VPU::PPEAttr& strippedPpeAttr, mlir::PatternRewriter& rewriter) const {
    // Get the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getShape(origOp.getInput());
    auto inputW = inputShape[Dims4D::Act::W];
    auto inputH = inputShape[Dims4D::Act::H];
    auto inputN = inputShape[Dims4D::Act::N];

    const auto kernelShape = getShape(origOp.getFilter());
    auto kernelW = kernelShape[Dims4D::Filter::KX];
    auto kernelH = kernelShape[Dims4D::Filter::KY];
    auto kernelN = kernelShape[Dims4D::Filter::OC];

    auto biasTable = origOp.getWeightTableBias();
    std::vector<float> biasTableVec;
    auto biasTableVecSize = biasTableVec.size();
    if (biasTable != nullptr) {
        auto biasTableConst = biasTable.getDefiningOp<Const::DeclareOp>();
        if (biasTableConst == nullptr) {
            _log.trace("Could not extract constant from bias table.");
            return false;
        }
        auto biasTableContent = biasTableConst.getContent();
        auto biasTableValues = biasTableContent.getValues<float>();
        biasTableVecSize = biasTableValues.size();
        biasTableVec.resize(biasTableVecSize, 0);
        std::copy(biasTableValues.begin(), biasTableValues.end(), biasTableVec.begin());
    }

    auto origDataType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
    auto f16Type = mlir::FloatType::getF16(rewriter.getContext());
    // used as Eltwise inputs
    const auto f16TypeOutputs = origDataType.changeElemType(f16Type);

    for (auto tile = 0; tile < maxTiles; tile++) {
        auto offsetIC = tiles[tile].offsets[Dims4D::Act::C];
        auto sizeIC = tiles[tile].shape[Dims4D::Act::C];
        _log.nest().trace("Slicing channels {0} - {1}", offsetIC, sizeIC);

        // Slice inputs
        const Shape inSliceOffsets{0, offsetIC, 0, 0};
        const Shape inSliceShape{inputN, sizeIC, inputH, inputW};
        auto convInput = rewriter.create<VPU::SliceOp>(origOp->getLoc(), origOp.getInput(),
                                                       getIntArrayAttr(rewriter, inSliceOffsets.raw()),
                                                       getIntArrayAttr(rewriter, inSliceShape.raw()));

        // Slice kernels
        const Shape kernelSliceOffsets{0, offsetIC, 0, 0};
        const Shape kernelSliceShape{kernelN, sizeIC, kernelH, kernelW};
        const auto rawKernelSliceShape = getIntArrayAttr(rewriter, kernelSliceShape);
        auto weightSlice = rewriter.create<VPU::SliceOp>(origOp.getLoc(), weightInput,
                                                         getIntArrayAttr(rewriter, kernelSliceOffsets.raw()),
                                                         getIntArrayAttr(rewriter, kernelSliceShape.raw()));
        auto weightSliceResult = weightSlice.getResult();
        if (weightDequantizeOp != nullptr) {
            auto dequantizeSlice = rewriter.create<VPU::DequantizeOp>(weightDequantizeOp->getLoc(), weightSliceResult,
                                                                      weightDequantizeOp.getDstElemTypeAttr(),
                                                                      weightDequantizeOp.getMultiClusterStrategyAttr());
            weightSliceResult = dequantizeSlice.getResult();
        }

        // Apply bias for the first convolution only
        if (tile != 0) {
            // Set the bias values to 0
            std::fill(biasTableVec.begin(), biasTableVec.end(), checked_cast<float>(0));
        }

        auto newBiasTable = biasTable == nullptr
                                    ? origOp.getWeightTableBias()
                                    : VPU::createNewWeightsTableTensor<float>(rewriter, origOp->getLoc(), biasTableVec,
                                                                              rewriter.getF32Type());
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                origOp.getLoc(), f16TypeOutputs, convInput.getResult(), weightSliceResult, origOp.getWeightsTable(),
                origOp.getWeightTableDataPtr(), origOp.getWeightTableSpPtr(), origOp.getWeightTableScale(),
                newBiasTable, origOp.getWeightZeroPoints(), origOp.getStrides(), origOp.getPad(), strippedPpeAttr,
                origOp.getMpeEngineAttr(), rawKernelSliceShape, origOp.getMultiClusterStrategyAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

        convOps.push_back(convOp);
    }

    return true;
}

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

    for (auto& strategy : clusterStrategies) {
        if (_numClusters != 1) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
            clusteredOp.setMultiClusterStrategy(strategy);
        }
        for (auto& dimOrder : dimOrders) {
            auto tilingMode = TilingMode::PIPELINING;
            auto tempOptionalTilingStrategyOutputTiling =
                    getSWLayerTilingStrategyWithTileDimOrder(op, tilingMode, dimOrder, log);
            if (mlir::failed(tempOptionalTilingStrategyOutputTiling)) {
                continue;
            }
            vpux::Shape tempTilingStrategy = tempOptionalTilingStrategyOutputTiling.value()[0].axis;
            log.trace("Attempt tilingStrategy for op {0}: {1}, {2}", tempTilingStrategy, strategy, op->getName());
            if (getNonOneDim(tempTilingStrategy).size() <= 1) {
                if (updateStrategy(tempTilingStrategy, tilingMode)) {
                    clusterStrategy = strategy;
                }
            }
        }
        if (_numClusters == 1) {
            break;
        }
    }
    return tilingStrategy.has_value();
}

template <typename OpType>
void NCEConvolutionSplitOverInputChannel::setStrategies(
        llvm::ArrayRef<OpType> newOps, llvm::ArrayRef<VPU::MultiClusterStrategy> clusterStrategies) const {
    auto firstOp = newOps[0];
    std::optional<VPU::MultiClusterStrategy> opMultiClusterStrategy = std::nullopt;
    std::optional<vpux::Shape> opTilingStrategy = std::nullopt;
    if (!getStrategies(firstOp.getOperation(), clusterStrategies, opTilingStrategy, opMultiClusterStrategy, _log)) {
        // TODO: iterate all supported multiClusterStrategy, E#178645
        // TODO: generate warning, clear new ops and return failure, or retry with a different tiling number,
        // E#178732
        VPUX_THROW("Failed to find legal tilingStrategy for new ops: {0}", firstOp);
    }

    VPUX_THROW_UNLESS(opMultiClusterStrategy.has_value() || _numClusters == 1,
                      "Operation '{0}' doesn't get a valid multiClusterStrategy: {1}", firstOp->getName(), firstOp);

    _log.debug("[{0}] Set '{1}' at '{2}': '{3}' '{4}' '{5}'", this->getDebugName(), firstOp->getName(),
               firstOp->getLoc(), opTilingStrategy.value(), opMultiClusterStrategy, firstOp);

    for (auto newOp : newOps) {
        newOp.getOperation()->setAttr(vpux::tilingStrategy,
                                      getIntArrayAttr(newOp.getOperation()->getContext(), opTilingStrategy.value()));
        if (_numClusters > 1) {
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

    const auto f16Type = mlir::FloatType::getF16(rewriter.getContext());
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

    // A stripped PPE is generated, ignoring post-op's and per-tensor scale/bias (since NCEConvolutionOp is not a
    // LayerWithPostOp and scale/bias info is discarded)
    auto strippedPpeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    // The original PPE attribute of the convolution (containing post-op and per-tensor scale/bias info), ends up in the
    // final Add
    auto finalPpeAttr = origOp.getPpeAttr();

    auto weightInput = origOp.getFilter();
    // check for parent weight shave dequantize op
    auto weightDequantizeOp = weightInput.getDefiningOp<VPU::DequantizeOp>();
    if (weightDequantizeOp != nullptr) {
        weightInput = weightDequantizeOp.getInput();
    }

    SmallVector<VPU::NCEConvolutionOp> convOps;
    if (origOp.getWeightsTable() != nullptr) {
        if (!splitNCEConvolutionWithLegacyWeightTable(origOp, weightInput, convOps, tiles.value(), maxTiles,
                                                      weightDequantizeOp, strippedPpeAttr, rewriter)) {
            _log.debug("[{0}] Failed to split over input channel with legacy weight table at '{1}': '{2}'",
                       this->getDebugName(), origOp->getLoc(), origOp);
            return mlir::failure();
        }
    } else if (!splitNCEConvolutionWithNewWeightTable(origOp, weightInput, convOps, tiles.value(), maxTiles,
                                                      weightDequantizeOp, strippedPpeAttr, rewriter)) {
        _log.debug("[{0}] Failed to split over input channel with new weight table at '{1}': '{2}'",
                   this->getDebugName(), origOp->getLoc(), origOp);
        return mlir::failure();
    }

    // Add the outputs of the convolutions with NCEEltwise Add operations. This is needed because NCEConvolutionOp
    // accumulates all its input channels into 1 output channel. Splitting the Convolutions into smaller Convolutions,
    // the outputs have to be added together.

    const auto opType = VPU::EltwiseType::ADD;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    VPU::NCEEltwiseOp addResult;

    // Elwise-ops do not have a weights table, thus per-channel scale/bias need to be applied through Convolutions. The
    // PPE for the generated Add's must reflect this by setting neutral values to scale and bias.
    // TODO: E#150106, a similar logic is also needed for IntPPE
    if (const auto wtInfoAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterWeightsTableInfo*>()) {
        finalPpeAttr = wtInfoAdapter->discardWeightsTableIfPresent(finalPpeAttr);
        strippedPpeAttr = wtInfoAdapter->discardWeightsTableIfPresent(strippedPpeAttr);
    }

    for (size_t index = 0; index < convOps.size() - 1; index++) {
        const auto addOperand = index == 0 ? convOps[index].getOutput() : addResult.getOutput();

        // NCEEltwise inType and outType are always same with convOp outType
        // TODO: check how in-place works here, E#178731
        addResult = rewriter.create<VPU::NCEEltwiseOp>(
                appendLoc(origOp->getLoc(), "_accumulator"), (index == convOps.size() - 2 ? ndOutputs : f16TypeOutputs),
                addOperand, convOps[index + 1].getOutput(), opType,
                (index == convOps.size() - 2 ? finalPpeAttr : strippedPpeAttr), nullptr, nullptr,
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

        // change NCEConv's output layout to supported NCEEltwise input layout
        // Eg: if NCEConv (inL=NHWC,outL=NCHW) splits into 3 small NCEConv:
        //   NCEConv (inL=NHWC,out=NHWC)    NCEConv (inL=NHWC,out=NHWC)     NCEConv (inL=NHWC,out=NHWC)
        //              \                         /                                     /
        //               NCEElt (inL=NHWC,out=NHWC)                                    /
        //                             \                                              /
        //                                         NCEElt (inL=NHWC,out=NCHW)
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(addResult.getOperation())) {
            auto orderInfo = iface.getLayoutInfo();
            iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
            const auto supportOrder1 = orderInfo.getInput(0);
            const auto supportOrder2 = orderInfo.getInput(1);
            const auto inputOrder1 = DimsOrder::fromValue(addResult.getInput1());
            const auto inputOrder2 = DimsOrder::fromValue(addResult.getInput2());

            if (supportOrder1 != inputOrder1 && supportOrder2 != inputOrder2) {
                const auto newInput1Type = mlir::dyn_cast<vpux::NDTypeInterface>(addResult.getInput1().getType())
                                                   .changeDimsOrder(supportOrder1);
                const auto newInput2Type = mlir::dyn_cast<vpux::NDTypeInterface>(addResult.getInput2().getType())
                                                   .changeDimsOrder(supportOrder2);

                auto input1Op = addResult.getInput1().getDefiningOp();
                auto input2Op = addResult.getInput2().getDefiningOp();
                input1Op->getResult(0).setType(newInput1Type);
                input2Op->getResult(0).setType(newInput2Type);

                addResult.getOperation()->setOperands({input1Op->getResult(0), input2Op->getResult(0)});
            }
        }
        addOps.push_back(addResult);
    }

    setStrategies<VPU::NCEConvolutionOp>(
            convOps, {VPU::MultiClusterStrategy::SplitOverKernel, VPU::MultiClusterStrategy::SplitOverHeight});
    setStrategies<VPU::NCEEltwiseOp>(addOps, {VPU::MultiClusterStrategy::SplitOverHeight});

    rewriter.replaceOp(origOp, addResult.getOutput());

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
