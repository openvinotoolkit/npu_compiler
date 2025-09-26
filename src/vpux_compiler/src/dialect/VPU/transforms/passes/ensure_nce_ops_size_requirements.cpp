//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
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
    bool splitNCEConvolutionWithLegacyWeightTable(VPU::NCEConvolutionOp& origOp, mlir::Value& weightInput,
                                                  SmallVector<VPU::NCEConvolutionOp>& convOps,
                                                  const OutputTiling& tiles, int64_t maxTiles,
                                                  VPU::DequantizeOp& weightDequantizeOp, VPU::PPEAttr& strippedPpeAttr,
                                                  mlir::PatternRewriter& rewriter) const;
    bool splitNCEConvolutionWithNewWeightTable(VPU::NCEConvolutionOp& origOp, mlir::Value& weightInput,
                                               SmallVector<VPU::NCEConvolutionOp>& convOps, const OutputTiling& tiles,
                                               int64_t maxTiles, VPU::DequantizeOp& weightDequantizeOp,
                                               VPU::PPEAttr& strippedPpeAttr, mlir::PatternRewriter& rewriter) const;
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool EnsureConvICRequirements::splitNCEConvolutionWithLegacyWeightTable(
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

    // TODO: E#70371 - Remaining opens for InputChannels 8K size
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
                                                                      weightDequantizeOp.getDstElemTypeAttr());
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

        // Apply bias for the first convolution only
        if (tile != 0) {
            // Set the bias values to 0
            for (size_t i = 3; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
                weightsTableVec[i] = checked_cast<int32_t>(0);
            }
        }

        // Adjust the weight pointers
        for (size_t i = 0; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
            weightsTableVec[i] =
                    checked_cast<int32_t>((i / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) * weightSetSize);
        }

        // Adjust the sparsity pointers
        for (size_t i = 1; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
            weightsTableVec[i] =
                    checked_cast<int32_t>((i / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) * sparsitySetSize);
        }

        auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                origOp.getLoc(), origOp.getType(), convInput.getResult(), weightSliceResult, weightsTable,
                origOp.getWeightTableDataPtr(), origOp.getWeightTableSpPtr(), origOp.getWeightTableScale(),
                origOp.getWeightTableBias(), origOp.getWeightZeroPoints(), origOp.getStrides(), origOp.getPad(),
                strippedPpeAttr, origOp.getMpeEngineAttr(), rawKernelSliceShape, origOp.getMultiClusterStrategyAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

        convOps.push_back(convOp);
    }

    return true;
}

bool EnsureConvICRequirements::splitNCEConvolutionWithNewWeightTable(
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

    // TODO: E#70371 - Remaining opens for InputChannels 8K size
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
                                                                      weightDequantizeOp.getDstElemTypeAttr());
            weightSliceResult = dequantizeSlice.getResult();
        }

        // Apply bias for the first convolution only
        if (tile != 0) {
            // Set the bias values to 0
            for (size_t i = 0; i < biasTableVecSize; i++) {
                biasTableVec[i] = checked_cast<float>(0);
            }
        }

        auto newBiasTable = biasTable == nullptr
                                    ? origOp.getWeightTableBias()
                                    : VPU::createNewWeightsTableTensor<float>(rewriter, origOp->getLoc(), biasTableVec,
                                                                              rewriter.getF32Type());
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                origOp.getLoc(), origOp.getType(), convInput.getResult(), weightSliceResult, origOp.getWeightsTable(),
                origOp.getWeightTableDataPtr(), origOp.getWeightTableSpPtr(), origOp.getWeightTableScale(),
                newBiasTable, origOp.getWeightZeroPoints(), origOp.getStrides(), origOp.getPad(), strippedPpeAttr,
                origOp.getMpeEngineAttr(), rawKernelSliceShape, origOp.getMultiClusterStrategyAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

        convOps.push_back(convOp);
    }

    return true;
}

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
    if (origOp.getWeightsTable()) {
        if (!splitNCEConvolutionWithLegacyWeightTable(origOp, weightInput, convOps, tiles.value(), maxTiles,
                                                      weightDequantizeOp, strippedPpeAttr, rewriter)) {
            return mlir::failure();
        }
    } else {
        if (!splitNCEConvolutionWithNewWeightTable(origOp, weightInput, convOps, tiles.value(), maxTiles,
                                                   weightDequantizeOp, strippedPpeAttr, rewriter)) {
            return mlir::failure();
        }
    }

    // Add the outputs of the convolutions with NCEEltwise Add operations. This is needed because NCEConvolutionOp
    // accumulates all its input channels into 1 output channel. Splitting the Convolutions into smaller Convolutions,
    // the outputs have to be added together.
    auto output = origOp->getResult(0);
    auto targetEltwiseOutputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
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
        auto addOperand = index == 0 ? convOps[index].getOutput() : addResult.getOutput();

        // NCEEltwise inType and outType are always same with ConvOp outType
        addResult = rewriter.create<VPU::NCEEltwiseOp>(
                origOp->getLoc(), targetEltwiseOutputType, addOperand, convOps[index + 1].getOutput(), opType,
                (index == convOps.size() - 2 ? finalPpeAttr : strippedPpeAttr), nullptr, nullptr,
                origOp.getOutputPaddingAttr(), origOp.getOutputPaddingAttr());

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

    rewriter.replaceOp(origOp, addResult.getOutput());

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
